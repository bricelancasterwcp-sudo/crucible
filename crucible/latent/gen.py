"""Corpus generator + AST validator for the B-lite corpus (prereg §4).

CRUXEval-style generation: a proposer (the Phase-A frozen 1.5B, or any object
matching `crucible.proposer.client.Proposer` -- `.generate(prompt, *, n,
seed, ...)` returning objects with `.text`) proposes short self-contained
`def f(...)` functions plus a handful of example argument tuples in a single
completion. Each candidate is parsed, validated, and every accepted
(function, input) pair is run through Task 1's `harvest` (which itself runs
the sample twice and reports determinism/truncation). Every rejection at
every stage is COUNTED, never silently dropped -- this module's whole job is
an honest corpus, not a big one.

Containment (read before trusting anything downstream of this module, and
read precisely -- this paragraph states what is actually held, not an
aspiration): `crucible.latent.harvest` does NOT sandbox filesystem writes --
a sample body can still `open()`/`shutil.rmtree`/etc a real file on this
box, unbounded, for as long as `EXEC_TIMEOUT_S` allows (see harvest.py's own
"Containment" note). `validate()` in this module is the actual containment
for the corpus, and it holds exactly these lines, no more:

- Import/ImportFrom, Global/Nonlocal, and an oversized AST are rejected
  outright.
- Any `.__something` ATTRIBUTE access is rejected (`dunder-attribute`) --
  closes the `().__class__.__bases__[0].__subclasses__()` sandbox-escape
  family reached via real `ast.Attribute` nodes.
- Any bare NAME-level Load of a builtin from a small deny-list (open, exec,
  eval, compile, __import__, globals, vars, getattr, setattr, delattr,
  input, breakpoint) is rejected (`banned-builtin:<name>`), per the
  controller ruling that STRENGTHENS the brief's call-level check to close
  the aliasing hole: `g = open; g(path, "w")` never contains an `ast.Call`
  node whose func is literally `open`, but it does contain a Load of the
  name `open`.
- Any STRING CONSTANT containing the substring `"__"` is rejected
  (`dunder-in-string`) -- a second controller ruling, added after a live
  falsification: `"{0.__class__}".format(a)` reaches a dunder attribute via
  `str.format`'s PEP 3101 field-name mini-language, which is parsed out of
  the string's own text at RUNTIME, never producing an `ast.Attribute` node
  at all -- so the attribute-level rule above cannot see it. This
  string-contents check is a coarse heuristic (it will also reject an
  innocent string that merely happens to contain `"__"`), accepted
  deliberately as the cheap fix for a hole that is otherwise invisible to
  AST attribute walking.

Residual, stated plainly rather than implied: this is a STATIC check over
the literal source text. A sample that builds a dunder name at RUNTIME
(string concatenation, `chr()` arithmetic, reading `"_" + "_class" + "_"`
from a loop, etc.) is out of scope for what `validate()` can see and is not
covered by either dunder rule above. This is accepted because the threat
model documented throughout this module and harvest.py is "our own
generator's output, occasionally buggy," not an adversarial sample actively
trying to evade a known static check -- if the corpus ever has to ingest
untrusted/adversarial code, this validator is not sufficient on its own.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Callable

from crucible.latent.config import BALANCE_GUARD_MIN_SAMPLES, MAX_AST_NODES, SKEW_LIMIT, STALL_CALLS
from crucible.latent.harvest import HarvestError, Snapshot, harvest

# Balance guard (spec §4): the corpus's binary outcome must not be skewed
# past SKEW_LIMIT once it has enough samples (BALANCE_GUARD_MIN_SAMPLES) to
# make "skew" a meaningful statement -- see generate_corpus. Both knobs now
# live in config.py (final review MEDIUM); `BALANCE_GUARD_MIN_SAMPLES` is
# imported into this module's own namespace, so
# `monkeypatch.setattr(gen, "BALANCE_GUARD_MIN_SAMPLES", ...)` still works
# exactly as before -- it patches THIS module's binding, which is what
# `_balance_guard_rejects` reads.

# Name-level deny-list (controller ruling, strengthens the brief's call-level
# ban). Any bare reference (ast.Name, Load context) to one of these is a
# rejection, in addition to the brief's Import/ImportFrom, `__`-attribute,
# Global/Nonlocal, and node-count rules.
_BANNED_BUILTIN_NAMES = frozenset({
    "open", "exec", "eval", "compile", "__import__", "globals", "vars",
    "getattr", "setattr", "delattr", "input", "breakpoint",
})

# Live-visibility knob (controller ruling, round-3 live-fire fix): every
# this many ACCEPTED functions, generate_corpus flushes gen_stats.json to
# disk early (in addition to the unconditional write when the run ends) --
# an unattended multi-hour run that stalls or crashes silently should not
# leave a 0-byte/absent stats file with no way to tell how far it got. An
# ops/UX knob, not a prereg-cited number -- kept local to this module (not
# config.py) and directly monkeypatchable by tests.
GEN_STATS_FLUSH_INTERVAL = 25

# Same live-visibility knob as GEN_STATS_FLUSH_INTERVAL, but for the minority-
# input second pass's own minority_stats.json (spec S12 pre-lock amendment,
# see generate_minority_inputs) -- kept as its own name/value so the two
# passes' flush cadences can diverge and be monkeypatched independently.
MINORITY_STATS_FLUSH_INTERVAL = 50

GEN_PROMPT = (
    "Write one short, self-contained Python function for a code-execution "
    "dataset.\n\n"
    "Rules:\n"
    "- Exactly one top-level function, named exactly `f`.\n"
    "- At most 30 lines long.\n"
    "- No import statements -- use only names already in scope.\n"
    "- No file, network, subprocess, or environment access.\n"
    "- Deterministic: no randomness, no clock/time reads, no global state.\n\n"
    "After the function, on its own line, give 3 to 5 example argument "
    "tuples the function could be called with, as a Python list literal:\n"
    "INPUTS: [(<args for call 1>), (<args for call 2>), ...]\n\n"
    "Example:\n"
    "def f(a, b):\n"
    "    if a > b:\n"
    "        return a - b\n"
    "    return b - a\n"
    "INPUTS: [(1, 2), (5, 5), (-3, 4), (10, -10), (0, 0)]\n"
)

# Minority-input second pass (spec S12 pre-lock amendment -- see
# generate_minority_inputs). The proposer's SELF-PROPOSED inputs run ~92%
# clean-return, which starves the pre-registered 80/20 balance guard of
# exception/timeout samples once accepted_samples is large; this prompt asks
# the SAME proposer for adversarial inputs to an ALREADY-ACCEPTED function
# instead of a new function -- the function source is appended verbatim after
# this prompt's own text (`MINORITY_PROMPT + function_src`), so all
# instructions live before the code, not interleaved with a template
# placeholder. Same `INPUTS: [...]` reply contract as GEN_PROMPT: a reply is
# parsed by `_extract_inputs_list` (the same bracket/quote-aware extraction
# `parse_candidate` uses), which does NOT require any text -- function source
# or otherwise -- to precede the marker, since the function is already known
# here and a terse reply that skips repeating it must not be penalized as a
# parse failure.
MINORITY_PROMPT = (
    "Below is a Python function from a code-execution dataset. Its normal "
    "example inputs already exist elsewhere in this dataset -- do not "
    "repeat them.\n\n"
    "Find 3 to 5 NEW argument tuples that make this function CRASH (raise "
    "an exception) or exercise a genuine edge case: empty containers, "
    "zero, huge integers, negative numbers where a positive one is "
    "expected, or mismatched-but-plausible types for the same parameters. "
    "Keep the SAME number of arguments the function actually takes -- do "
    "not change arity unless the function itself accepts a variable "
    "number of arguments (e.g. *args).\n\n"
    "Give ONLY the argument tuples, as a Python list literal, on their own "
    "line:\n"
    "INPUTS: [(<args for call 1>), (<args for call 2>), ...]\n\n"
    "Example, for `def f(a, b): return a / b`:\n"
    "INPUTS: [(1, 0), (0, 0), (10**18, 1), (-1, -1), (1.5, 0)]\n\n"
    "Function:\n"
)


# -- candidate parsing --------------------------------------------------------
#
# Round-3 controller ruling (live-fire finding): the original regex-only
# extractor (`^INPUTS:\s*(\[.*\])\s*$`) rejected most real model output --
# `INPUTS = [...]` (assignment, not a labeled line) parse-failed outright,
# and any trailing prose/punctuation after the closing bracket on the same
# line ALSO parse-failed it (the `\s*$` anchor demanded nothing else on the
# line). Both are fixed below by separating "find the marker" from "find the
# literal": the marker regex only locates `INPUTS` followed by `:` or `=`;
# the literal itself is then extracted by bracket/quote-aware scanning
# (`_matching_bracket_end`), not a greedy regex, so trailing text after the
# closing bracket is simply never looked at, on any line.

_FENCE_RE = re.compile(r"```[a-zA-Z0-9_+-]*[ \t]*\r?\n?(.*?)```", re.DOTALL)
_INPUTS_MARKER_RE = re.compile(r"^INPUTS\s*[:=]\s*", re.MULTILINE)


def parse_candidate(text: str) -> tuple[str, list[str]] | None:
    """Extract `(function_src, args_literals)` from one raw completion, or
    `None` if the completion does not carry the shape the prompt asked for.

    This is intentionally a LOOSE extraction step -- it does not judge
    whether `function_src` is a well-formed, safe function (that is
    `validate`'s job, run separately by the caller). Accepts EITHER
    `INPUTS:` or `INPUTS =` (optional whitespace around the `:`/`=`), with
    optional markdown code fences around the whole candidate (stripped
    first, see `_unfence`). Fails closed (`None`) on: no `INPUTS` marker at
    all, no `[` immediately following it, an unparseable literal, or an
    empty list -- ANYTHING after the literal's closing bracket (a trailing
    period, more prose, anything) is simply never inspected, so it cannot
    cause a rejection.

    Normalizes each entry of the parsed list: an entry that is not already
    a `tuple` (a bare literal like `5`, or `(5)` -- which Python itself
    evaluates as the int `5`, NOT a 1-tuple, since a single parenthesized
    value with no trailing comma is just grouping) is wrapped as a 1-tuple.
    A single-argument call is a legitimate call shape; this only papers
    over the model's punctuation, it does not change what gets called.
    `args_literals` are `repr()`s of the NORMALIZED tuples -- ready to pass
    straight to `harvest`'s `args_literal` parameter, which itself decodes
    them with `ast.literal_eval`.
    """
    body = _unfence(text)
    marker = _INPUTS_MARKER_RE.search(body)
    if marker is None:
        return None
    function_src = body[: marker.start()].rstrip() + "\n"
    if not function_src.strip():
        return None

    normalized = _extract_inputs_list(body, marker)
    if normalized is None:
        return None
    return function_src, [repr(item) for item in normalized]


def _extract_inputs_list(body: str, marker: re.Match[str] | None = None) -> list[tuple] | None:
    """The bracket/quote-aware `INPUTS` literal extraction shared by
    `parse_candidate` and `generate_minority_inputs` -- factored out
    (spec S12 pre-lock amendment) so the minority pass can reuse the exact
    same robust extraction on a reply that carries ONLY the `INPUTS: [...]`
    line, with no function source (or anything else) before the marker --
    `parse_candidate` itself additionally REQUIRES non-blank text before the
    marker (that text becomes `function_src`), a constraint that has nothing
    to do with locating or parsing the literal itself, so it stays in
    `parse_candidate` rather than moving down into this function.

    `marker` may be passed in by a caller (`parse_candidate`) that already
    located the `INPUTS` marker via `_INPUTS_MARKER_RE`, avoiding a second
    regex search over the same text; a bare caller omits it and this
    function searches fresh. Returns the NORMALIZED list of argument tuples
    (a bare non-tuple entry wrapped as a 1-tuple -- see `parse_candidate`'s
    docstring on why), or `None` on any of: no marker, no `[` immediately
    after it, an unterminated/unparseable literal, or an empty list.
    """
    if marker is None:
        marker = _INPUTS_MARKER_RE.search(body)
        if marker is None:
            return None

    bracket_start = marker.end()
    if bracket_start >= len(body) or body[bracket_start] != "[":
        return None
    bracket_end = _matching_bracket_end(body, bracket_start)
    if bracket_end is None:
        return None

    try:
        raw_inputs = ast.literal_eval(body[bracket_start:bracket_end])
    except (SyntaxError, ValueError):
        return None
    if not isinstance(raw_inputs, list) or not raw_inputs:
        return None

    return [item if isinstance(item, tuple) else (item,) for item in raw_inputs]


def _matching_bracket_end(text: str, open_index: int) -> int | None:
    """Index just PAST the bracket that closes `text[open_index]` (one of
    `([{`), tracking nesting depth across all three bracket kinds together
    (sufficient here -- a malformed mismatch like `[1, 2)` still balances to
    depth 0 at the same place a matched one would, and `ast.literal_eval`
    is the real correctness check downstream). Skips characters inside
    single/double-quoted string literals (respecting `\\` escapes) so a
    bracket character inside a string entry does not corrupt the count.
    `None` if the text ends before the bracket closes.
    """
    depth = 0
    quote: str | None = None
    i = open_index
    while i < len(text):
        ch = text[i]
        if quote is not None:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = None
        elif ch in ("'", '"'):
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return None


def _unfence(text: str) -> str:
    """Strip a single ``` fenced block (with or without a language tag) if
    the completion has one."""
    match = _FENCE_RE.search(text)
    return match.group(1) if match else text


# -- validator ------------------------------------------------------------------

def validate(function_src: str) -> str | None:
    """`None` if `function_src` is an acceptable corpus candidate, else a
    short machine-readable rejection reason (`generate_corpus` buckets by
    this string in `gen_stats.json["validate_fail"]`).

    Every rule below is independently mutation-testable (see
    `tests/latent/test_gen.py`, one test per rule):

    - `syntax-error`: does not even `ast.parse`.
    - `not-single-statement` / `not-a-function-def` / `wrong-function-name`:
      the module body must be EXACTLY one `def f(...):`.
    - `node-count-exceeded`: `ast.walk` count over `config.MAX_AST_NODES`.
    - `import`: any `Import`/`ImportFrom` anywhere in the body.
    - `global-nonlocal`: any `Global`/`Nonlocal` statement.
    - `dunder-attribute`: any `.__something` attribute access (closes the
      `().__class__.__bases__[0].__subclasses__()` sandbox-escape family).
    - `banned-builtin:<name>`: any bare Load of a name in the containment
      deny-list -- see the module docstring for why this is NAME-level, not
      call-level.
    - `dunder-in-string`: any string constant containing `"__"` -- closes
      the `str.format` field-name traversal hole (`"{0.__class__}".format
      (a)`), see the module docstring's containment paragraph.
    """
    try:
        tree = ast.parse(function_src)
    except SyntaxError:
        return "syntax-error"

    shape_reason = _check_shape(tree)
    if shape_reason is not None:
        return shape_reason

    if sum(1 for _ in ast.walk(tree)) > MAX_AST_NODES:
        return "node-count-exceeded"

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return "import"
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            return "global-nonlocal"
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return "dunder-attribute"
        if (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
                and node.id in _BANNED_BUILTIN_NAMES):
            return f"banned-builtin:{node.id}"
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and "__" in node.value):
            return "dunder-in-string"
    return None


def _check_shape(tree: ast.Module) -> str | None:
    """The module body must be exactly one `def f(...):`."""
    if len(tree.body) != 1:
        return "not-single-statement"
    node = tree.body[0]
    if not isinstance(node, ast.FunctionDef):
        return "not-a-function-def"
    if node.name != "f":
        return "wrong-function-name"
    return None


# -- outcome -> binary label (shared with Tasks 3/8) -----------------------------

def binary_label(outcome: str) -> int:
    """The binary reduction the B-lite head predicts (prereg §4): `1` for a
    clean return, `0` for anything else (`exception:<Type>` or `timeout`)."""
    return 1 if outcome == "return" else 0


# -- fn_id ------------------------------------------------------------------------

def _fn_id(function_src: str) -> str:
    """A stable, short identifier for a function's exact source text: the
    first 16 hex characters of its sha256. Used to group a function's
    samples together (Task 3's function-level split) without storing the
    whole source as a join key."""
    return hashlib.sha256(function_src.encode("utf-8")).hexdigest()[:16]


# -- corpus generation --------------------------------------------------------------

def generate_corpus(
    proposer, target_functions: int, out_dir: Path, *, seed: int,
    log: Callable[[str], None] = print,
) -> dict:
    """Drive `proposer` until `target_functions` accepted functions are
    collected, harvesting every accepted (function, input) sample and
    writing the corpus + stats to `out_dir`. The caller controls the overall
    budget window (prereg §4's "budgeted generation window") -- this loop
    itself only stops once the target is reached.

    Two independent accounting levels, each internally conserved:

    - FUNCTION level: every candidate text pulled from the proposer is
      exactly one of `parse_fail`, one `validate_fail` reason,
      `duplicate_rejected`, or an accepted function
      (`parse_fail + sum(validate_fail.values()) + duplicate_rejected +
      accepted_functions == candidates`). `duplicate_rejected` counts a
      candidate whose `fn_id` (sha256/16 of its source) has ALREADY been
      accepted earlier in this run -- a low-diversity proposer repeating
      itself does not inflate `accepted_functions` with the same function
      counted twice; the earlier live-fire bug this fixes cycled the same
      handful of functions forever with no dedup at all.
    - SAMPLE level: every (accepted function, input) pair harvested is
      exactly one of `truncated_rejected`, `nondet_rejected`,
      `balance_rejected`, `harvest_error`, or an accepted sample. A function
      is accepted (written to `functions.jsonl`) independently of how many
      of its own samples survive this stage -- **the balance guard rejects
      SAMPLES, never the function they belong to**.

    `harvest_error` counts a `HarvestError`/`OSError` raised BY `harvest()`
    itself (e.g. environment problems -- no `sensorium` console script,
    a disk write failure) for one (function, input) pair; the pair is
    skipped and generation continues. Any OTHER exception is not swallowed:
    it propagates out of this function, but `gen_stats.json` is still
    written first (in a `finally`) with `"complete": False` and whatever
    counts were accumulated up to that point -- a mid-run crash loses no
    accounting for the candidates already processed, even though it does
    abort the run.

    Each `proposer.generate()` call gets its OWN seed, deterministically
    derived from the base `seed` and a per-call counter:
    `call_seed = seed + call_index`, where `call_index` is `0` for the
    FIRST call and increments by one every call after
    (`seed+0, seed+1, seed+2, ...`) -- reproducible from the base seed alone,
    and gives a real-server proposer (temperature > 0) a fresh sampling seed
    every batch instead of resampling the same one repeatedly.

    Stall guard: if `config.STALL_CALLS` CONSECUTIVE `generate()` calls each
    accept zero NEW functions (every candidate in the call was a parse
    failure, a validate failure, or a duplicate), the loop stops early and
    `gen_stats["stalled"]` is `True` -- protects against a degenerate/
    low-diversity proposer that can never reach `target_functions` on its
    own. `stalled` is `False` whenever the target was reached (or the run
    crashed -- see `complete` above) before the guard tripped.

    Live visibility (round-3 live-fire fix): `gen_stats.json` is flushed to
    disk early, in addition to the final write, every
    `GEN_STATS_FLUSH_INTERVAL` accepted functions -- `gen_stats["
    flushed_at_functions"]` records the `accepted_functions` count as of
    that last periodic checkpoint (`0` if the run ended before the first
    one). Every `samples.jsonl`/`functions.jsonl` line is also flushed to
    disk immediately after it is written (not buffered until the file
    closes) -- an unattended run that stalls or is killed leaves real,
    readable partial output, not a 0-byte file.

    Each `functions.jsonl` record now also carries `args_literals` (the
    NORMALIZED argument-tuple literals `parse_candidate` extracted for this
    function -- see its docstring on normalization) and `samples_kept` (how
    many of those inputs actually survived harvesting into `samples.jsonl`
    for this function) -- an audit trail for the case a function is
    accepted but yields zero kept samples (every input timed out, was
    nondeterministic, etc.), which was previously invisible.

    Writes `samples.jsonl` (one JSON line per accepted sample), `functions
    .jsonl` (one JSON line per accepted function, written AFTER that
    function's samples are processed so `samples_kept` is known), and
    `gen_stats.json` (this function's return value, verbatim, plus
    `"complete"`/`"stalled"`/`"flushed_at_functions"`).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch = out_dir / "_harvest_scratch"

    stats = {
        "target_functions": target_functions,
        "seed": seed,
        "candidates": 0,
        "parse_fail": 0,
        "validate_fail": {},
        "duplicate_rejected": 0,
        "nondet_rejected": 0,
        "truncated_rejected": 0,
        "balance_rejected": 0,
        "harvest_error": 0,
        "accepted_functions": 0,
        "accepted_samples": 0,
        "complete": False,
        "stalled": False,
        "flushed_at_functions": 0,
    }
    class_counts = {0: 0, 1: 0}
    accepted_fn_ids: set[str] = set()
    call_index = 0
    consecutive_stalls = 0

    samples_path = out_dir / "samples.jsonl"
    functions_path = out_dir / "functions.jsonl"
    gen_stats_path = out_dir / "gen_stats.json"
    try:
        with samples_path.open("w") as samples_f, functions_path.open("w") as functions_f:
            while stats["accepted_functions"] < target_functions:
                call_seed = seed + call_index
                call_index += 1
                accepted_before_call = stats["accepted_functions"]
                candidates = proposer.generate(GEN_PROMPT, n=8, seed=call_seed)
                for candidate in candidates:
                    stats["candidates"] += 1
                    parsed = parse_candidate(candidate.text)
                    if parsed is None:
                        stats["parse_fail"] += 1
                        continue
                    function_src, args_literals = parsed
                    reason = validate(function_src)
                    if reason is not None:
                        stats["validate_fail"][reason] = stats["validate_fail"].get(reason, 0) + 1
                        continue
                    fn_id = _fn_id(function_src)
                    if fn_id in accepted_fn_ids:
                        stats["duplicate_rejected"] += 1
                        continue

                    samples_kept = _harvest_and_write_samples(
                        function_src, args_literals, scratch, samples_f, stats, class_counts,
                    )
                    accepted_fn_ids.add(fn_id)
                    functions_f.write(json.dumps({
                        "fn_id": fn_id,
                        "function_src": function_src,
                        "args_literals": args_literals,
                        "samples_kept": samples_kept,
                    }) + "\n")
                    functions_f.flush()
                    stats["accepted_functions"] += 1
                    log(f"accepted function {stats['accepted_functions']}/{target_functions} "
                        f"(fn_id={fn_id}, samples_kept={samples_kept})")

                    if stats["accepted_functions"] % GEN_STATS_FLUSH_INTERVAL == 0:
                        stats["flushed_at_functions"] = stats["accepted_functions"]
                        _write_gen_stats(gen_stats_path, stats)

                    if stats["accepted_functions"] >= target_functions:
                        break

                if stats["accepted_functions"] == accepted_before_call:
                    consecutive_stalls += 1
                else:
                    consecutive_stalls = 0
                if consecutive_stalls >= STALL_CALLS:
                    stats["stalled"] = True
                    break
        stats["complete"] = True
    finally:
        _write_gen_stats(gen_stats_path, stats)
    return stats


def _harvest_and_write_samples(
    function_src: str, args_literals: list[str], scratch: Path,
    samples_f, stats: dict, class_counts: dict[int, int],
) -> int:
    """Harvest every (function, input) pair for one accepted function,
    filter+count each, and write the survivors to `samples_f`. Returns the
    number of samples actually KEPT (written) for this function -- the
    caller records this as `functions.jsonl`'s `samples_kept` audit field,
    since a function can be accepted (its SOURCE passed `validate()`) while
    still contributing zero kept samples (every input timed out, was
    nondeterministic, etc.) -- previously invisible without this count.

    `HarvestError`/`OSError` from `harvest()` itself (environment problems,
    not a sample-execution outcome -- see harvest.py's `HarvestError`
    docstring) are caught here and counted in `stats["harvest_error"]`; the
    pair is skipped, the function's remaining inputs are still attempted.
    """
    fn_id = _fn_id(function_src)
    kept = 0
    for args_literal in args_literals:
        try:
            result = harvest(function_src, args_literal, scratch)
        except (HarvestError, OSError):
            stats["harvest_error"] += 1
            continue

        if result.truncated:
            stats["truncated_rejected"] += 1
            continue
        if not result.deterministic:
            stats["nondet_rejected"] += 1
            continue

        label = binary_label(result.outcome)
        if _balance_guard_rejects(label, class_counts):
            stats["balance_rejected"] += 1
            continue

        class_counts[label] += 1
        stats["accepted_samples"] += 1
        kept += 1
        samples_f.write(json.dumps({
            "fn_id": fn_id,
            "function_src": function_src,
            "args": args_literal,
            "outcome": result.outcome,
            "return_repr": result.return_repr,
            "snapshots": [_snapshot_to_json(s) for s in result.snapshots],
        }) + "\n")
        samples_f.flush()
    return kept


def _balance_guard_rejects(label: int, class_counts: dict[int, int]) -> bool:
    """Spec §4 balance guard: once at least `BALANCE_GUARD_MIN_SAMPLES`
    samples have been accepted, reject further samples of whichever binary
    class is already the majority while its running fraction exceeds
    `SKEW_LIMIT`. Evaluated against the counts BEFORE this sample -- a
    minority-class sample is never rejected by this guard, so accepting it
    is exactly what lets the corpus recover toward balance over time
    (rejection sampling on the majority class, prereg §4)."""
    total = class_counts[0] + class_counts[1]
    if total < BALANCE_GUARD_MIN_SAMPLES:
        return False
    majority_label = 0 if class_counts[0] >= class_counts[1] else 1
    balance = class_counts[majority_label] / total
    return balance > SKEW_LIMIT and label == majority_label


def _snapshot_to_json(snapshot: Snapshot) -> dict:
    return {"line": snapshot.line, "locals": [list(row) for row in snapshot.locals]}


def _write_gen_stats(path: Path, stats: dict) -> None:
    """The single gen_stats.json write path -- used both for the periodic
    live-visibility flush (every `GEN_STATS_FLUSH_INTERVAL` accepted
    functions) and the unconditional final write in `generate_corpus`'s
    `finally` block, so there is exactly one place that decides HOW the
    stats dict is serialized."""
    _dump_stats_json(path, stats)


def _dump_stats_json(path: Path, stats: dict) -> None:
    """The actual serialization both `_write_gen_stats` and
    `_write_minority_stats` delegate to -- kept as one shared body (DRY: the
    two passes must serialize identically) while `_write_gen_stats` and
    `_write_minority_stats` stay separate, independently-monkeypatchable
    NAMES, each scoped to its own stats file and its own test suite."""
    path.write_text(json.dumps(stats, indent=2))


# -- minority-input second pass (spec S12 pre-lock amendment) -----------------
#
# Context recorded by the controller: the 1.5B's SELF-proposed inputs
# (generate_corpus, above) run ~92% clean-return, so the pre-registered 80/20
# balance guard (_balance_guard_rejects) has been rejecting nearly every new
# majority-class sample once BALANCE_GUARD_MIN_SAMPLES is reached -- the
# corpus is minority-starved, not undersized. This pass does not touch the
# guard, the endpoints, or the model: it asks the SAME proposer, for each
# function the FIRST pass already accepted, to propose BREAKING inputs
# instead of a new function -- enriching exception/timeout samples for
# functions already in the corpus, one generate() call per function.


def _scan_existing_corpus_state(samples_path: Path) -> tuple[dict[int, int], dict[str, set[str]]]:
    """Scan `samples.jsonl` ONCE, before this pass writes anything, for the
    two things it needs to seed itself correctly against whatever the corpus
    already holds:

    - `class_counts`: the REAL current binary-outcome balance (via
      `binary_label`), read off every sample already on disk -- so
      `_balance_guard_rejects` evaluates this pass's candidates against the
      corpus as it ACTUALLY stands (first pass's samples included), never
      against an empty-corpus assumption. This is the one property that
      makes calling this a genuine SECOND pass rather than a fresh guard
      reset: a corpus already sitting at the 80/20 skew limit must reject a
      majority-class candidate on this pass's very first sample, not after
      accumulating BALANCE_GUARD_MIN_SAMPLES of its own from zero.
    - `args_by_fn`: every `(fn_id -> {args_literal, ...})` already written to
      `samples.jsonl`, folded into this pass's per-function duplicate-input
      check ALONGSIDE `functions.jsonl`'s original `args_literals` (see
      `generate_minority_inputs`). This closes a crash-and-retry gap the
      launcher's one-pass refusal (keyed off `minority_stats.json`'s
      existence) cannot cover on its own: if a PRIOR minority-pass attempt
      appended some samples before crashing, before `minority_stats.json`
      ever got written, a second attempt over the same `corpus_dir` must not
      re-harvest and duplicate those already-written samples.

    Returns `({0: 0, 1: 0}, {})` if `samples.jsonl` does not exist yet.
    """
    class_counts = {0: 0, 1: 0}
    args_by_fn: dict[str, set[str]] = {}
    if not samples_path.exists():
        return class_counts, args_by_fn
    with samples_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            class_counts[binary_label(row["outcome"])] += 1
            args_by_fn.setdefault(row["fn_id"], set()).add(row["args"])
    return class_counts, args_by_fn


def generate_minority_inputs(
    proposer, corpus_dir: Path, *, seed: int, log: Callable[[str], None] = print,
) -> dict:
    """Second pass over an ALREADY-GENERATED corpus (spec S12 pre-lock
    amendment): for every function `generate_corpus` already accepted (read
    from `functions.jsonl`, in file order), ask `proposer` for BREAKING
    inputs to that exact function instead of a new function, harvest every
    genuinely new one, and append the survivors to the SAME `samples.jsonl`
    the first pass wrote. Data generation only -- no threshold, endpoint, or
    model change; `corpus_dir`'s `functions.jsonl` and `gen_stats.json` are
    never opened for writing by this function.

    One `proposer.generate()` call PER FUNCTION, `n=1`, temperature left at
    the proposer's own default: `call_seed = seed + index`, `index` the
    function's position in `functions.jsonl` starting at 0 -- the same
    `seed + call_index` contract `generate_corpus` uses, applied here per
    FUNCTION rather than per generate() batch. The prompt sent is
    `MINORITY_PROMPT + function_src` (see `MINORITY_PROMPT`'s own docstring
    on why the function is appended rather than templated in). The reply is
    parsed with `_extract_inputs_list` -- NOT `parse_candidate` -- since a
    minority reply need not repeat the function source at all.

    A reply that fails to parse counts `parse_fail` for that FUNCTION (like
    `generate_corpus`'s `parse_fail`, this is a per-candidate-text bucket,
    not a per-input one) and moves on to the next function. Every input in a
    reply that DOES parse lands in exactly one of these buckets (conservation
    invariant, mutation-tested):

    - `invalid_literal`: `repr(item)` does not itself round-trip back through
      `ast.literal_eval` -- e.g. the proposer legitimately answering this
      pass's own "huge integers" prompt with `1e400` (a valid literal that
      overflows to `float('inf')` at parse time) produces `repr(inf) ==
      "inf"`, a bare NAME that `ast.literal_eval` rejects. `harvest()`
      decodes `args_literal` with `ast.literal_eval` at runtime (see
      `harvest.py`'s `_write_runner_script`), so an input that cannot survive
      that round trip can never actually be run -- caught here, before a
      harvest is even attempted, rather than surfacing as an opaque
      `harvest_error` later.
    - `duplicate_input`: `repr(item)` is already in this function's
      `args_literals` (from `functions.jsonl`) OR already in `samples.jsonl`
      for this `fn_id` (from `_scan_existing_corpus_state` -- see its
      docstring on the crash-and-retry case this also covers) OR was already
      ACCEPTED earlier in this same reply (a proposer that repeats itself
      within one list must not write the same sample twice either). No
      duplicate sample is ever written.
    - `harvest_error` / `truncated_rejected` / `nondet_rejected` /
      `balance_rejected` / an accepted sample: identical meaning and
      identical bucket-priority ordering to `generate_corpus`'s own
      `_harvest_and_write_samples` (see `_harvest_and_write_minority_samples`,
      which mirrors it). The balance guard (`_balance_guard_rejects`) is the
      SAME function `generate_corpus` uses, evaluated against `class_counts`
      seeded from the corpus's REAL current balance (see
      `_scan_existing_corpus_state`) -- this pass never resets the guard to
      zero, and never touches `config.SKEW_LIMIT` / `config.
      BALANCE_GUARD_MIN_SAMPLES` itself.
    - `accepted_minority` is incremented alongside `accepted_samples`
      whenever the accepted sample's `binary_label` is `0` (`outcome !=
      "return"`) -- this pass's entire purpose, made countable.

    Every accepted sample is appended to `samples.jsonl` in the exact same
    JSON-line shape `generate_corpus` writes (opened in APPEND mode, never
    truncated -- this pass can only ever grow the corpus, never shrink or
    rewrite it) and flushed immediately, same discipline as `generate_corpus`.
    `minority_stats.json` (this function's return value, verbatim, plus
    `"complete"`) is written to `corpus_dir` -- its OWN file, distinct from
    `gen_stats.json`, so the two passes' provenance stays distinguishable --
    flushed early every `MINORITY_STATS_FLUSH_INTERVAL` functions in addition
    to the unconditional final write in a `finally` block (mirrors
    `generate_corpus`'s live-visibility discipline exactly: a crash mid-run
    still leaves `complete: False` and the real counts accumulated so far,
    never an absent or stale stats file).
    """
    corpus_dir = Path(corpus_dir)
    scratch = corpus_dir / "_minority_harvest_scratch"
    functions_path = corpus_dir / "functions.jsonl"
    samples_path = corpus_dir / "samples.jsonl"
    minority_stats_path = corpus_dir / "minority_stats.json"

    functions = [
        json.loads(line) for line in functions_path.read_text().splitlines() if line.strip()
    ]

    stats = {
        "seed": seed,
        "functions_total": len(functions),
        "functions_processed": 0,
        "generate_calls": 0,
        "parse_fail": 0,
        "invalid_literal": 0,
        "duplicate_input": 0,
        "harvest_error": 0,
        "nondet_rejected": 0,
        "truncated_rejected": 0,
        "balance_rejected": 0,
        "accepted_samples": 0,
        "accepted_minority": 0,
        "complete": False,
    }
    class_counts, args_by_fn = _scan_existing_corpus_state(samples_path)

    try:
        with samples_path.open("a") as samples_f:
            for index, fn in enumerate(functions):
                function_src = fn["function_src"]
                fn_id = fn["fn_id"]
                existing_literals = set(fn.get("args_literals", ()))
                existing_literals |= args_by_fn.get(fn_id, set())

                call_seed = seed + index
                stats["generate_calls"] += 1
                candidates = proposer.generate(MINORITY_PROMPT + function_src, n=1, seed=call_seed)
                reply = candidates[0].text
                normalized = _extract_inputs_list(_unfence(reply))
                stats["functions_processed"] += 1

                if normalized is None:
                    stats["parse_fail"] += 1
                else:
                    _harvest_and_write_minority_samples(
                        fn_id, function_src, normalized, existing_literals,
                        scratch, samples_f, stats, class_counts,
                    )

                log(f"minority pass {stats['functions_processed']}/{len(functions)} "
                    f"(fn_id={fn_id})")

                if stats["functions_processed"] % MINORITY_STATS_FLUSH_INTERVAL == 0:
                    _write_minority_stats(minority_stats_path, stats)
        stats["complete"] = True
    finally:
        _write_minority_stats(minority_stats_path, stats)
    return stats


def _harvest_and_write_minority_samples(
    fn_id: str, function_src: str, inputs: list[tuple], existing_literals: set[str],
    scratch: Path, samples_f, stats: dict, class_counts: dict[int, int],
) -> None:
    """Harvest every NEW candidate input for one already-accepted function,
    bucket each into `stats`, and append survivors to `samples_f`. Mirrors
    `_harvest_and_write_samples`'s bucket-priority ordering (harvest_error ->
    truncated -> nondet -> balance -> accepted) with two checks ahead of it
    that have no equivalent in the first pass, because the first pass never
    sees a candidate input twice: `invalid_literal` and `duplicate_input`
    (both explained on `generate_minority_inputs`'s own docstring) are
    resolved BEFORE `harvest()` is ever called -- no point spending a harvest
    on an input already in the corpus or one that cannot even round-trip
    through `ast.literal_eval`.

    `existing_literals` is MUTATED in place: an input accepted earlier in
    this same call is immediately visible to the rest of this function's
    inputs, so a proposer that repeats itself within one reply cannot cause
    the same sample to be written twice.
    """
    for item in inputs:
        literal = repr(item)
        try:
            ast.literal_eval(literal)
        except (SyntaxError, ValueError):
            stats["invalid_literal"] += 1
            continue
        if literal in existing_literals:
            stats["duplicate_input"] += 1
            continue

        try:
            result = harvest(function_src, literal, scratch)
        except (HarvestError, OSError):
            stats["harvest_error"] += 1
            continue

        if result.truncated:
            stats["truncated_rejected"] += 1
            continue
        if not result.deterministic:
            stats["nondet_rejected"] += 1
            continue

        label = binary_label(result.outcome)
        if _balance_guard_rejects(label, class_counts):
            stats["balance_rejected"] += 1
            continue

        class_counts[label] += 1
        existing_literals.add(literal)
        stats["accepted_samples"] += 1
        if label == 0:
            stats["accepted_minority"] += 1
        samples_f.write(json.dumps({
            "fn_id": fn_id,
            "function_src": function_src,
            "args": literal,
            "outcome": result.outcome,
            "return_repr": result.return_repr,
            "snapshots": [_snapshot_to_json(s) for s in result.snapshots],
        }) + "\n")
        samples_f.flush()


def _write_minority_stats(path: Path, stats: dict) -> None:
    """The single minority_stats.json write path -- mirrors `_write_gen_stats`
    exactly (delegates to the same `_dump_stats_json`), kept as its OWN name
    so the two passes' periodic-flush test doubles
    (`monkeypatch.setattr(gen, "_write_gen_stats"/"_write_minority_stats",
    ...)`) can be swapped independently without one pass's spy intercepting
    the other pass's writes."""
    _dump_stats_json(path, stats)
