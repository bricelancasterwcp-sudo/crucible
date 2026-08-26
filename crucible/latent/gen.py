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

    normalized = [item if isinstance(item, tuple) else (item,) for item in raw_inputs]
    return function_src, [repr(item) for item in normalized]


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
    path.write_text(json.dumps(stats, indent=2))
