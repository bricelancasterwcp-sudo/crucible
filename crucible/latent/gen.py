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

Containment (read before trusting anything downstream of this module):
`crucible.latent.harvest` does NOT sandbox filesystem writes -- a sample body
can still `open()`/`shutil.rmtree`/etc a real file on this box, unbounded,
for as long as `EXEC_TIMEOUT_S` allows (see harvest.py's own "Containment"
note). `validate()` in this module is the actual containment for the corpus:
it rejects Import/ImportFrom, Global/Nonlocal, any `__`-attribute access,
oversized ASTs, and -- per the controller ruling that STRENGTHENS the
brief -- any bare NAME-level Load of a builtin from a small deny-list
(open/exec/eval/compile/__import__/globals/vars/getattr/setattr/delattr/
input/breakpoint), not merely a *call* to one of them. The name-level ban
closes the aliasing hole a call-only check would miss: `g = open; g(path,
"w")` never contains an `ast.Call` node whose func is literally `open`, but
it does contain a Load of the name `open`, which is banned regardless of
what the code goes on to do with the reference. Every sample this module
accepts has already been screened this way before it ever reaches
`harvest()`; the threat model is "our own generator's output, occasionally
buggy," not an adversarial sample (same threat model harvest.py documents).
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Callable

from crucible.latent.config import MAX_AST_NODES, SKEW_LIMIT
from crucible.latent.harvest import Snapshot, harvest

# Balance guard (spec §4): the corpus's binary outcome must not be skewed
# past SKEW_LIMIT once it has enough samples to make "skew" a meaningful
# statement. Below this many ACCEPTED samples, an early lucky/unlucky run of
# one class would trip the guard on noise alone -- see generate_corpus.
# Module-level (not config.py) since it is a mechanism knob of THIS
# generator, not a corpus-wide constant other tasks read; tests monkeypatch
# it directly to exercise the guard without generating 1000 real samples.
BALANCE_GUARD_MIN_SAMPLES = 1000

# Name-level deny-list (controller ruling, strengthens the brief's call-level
# ban). Any bare reference (ast.Name, Load context) to one of these is a
# rejection, in addition to the brief's Import/ImportFrom, `__`-attribute,
# Global/Nonlocal, and node-count rules.
_BANNED_BUILTIN_NAMES = frozenset({
    "open", "exec", "eval", "compile", "__import__", "globals", "vars",
    "getattr", "setattr", "delattr", "input", "breakpoint",
})

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

_FENCE_RE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)
_INPUTS_RE = re.compile(r"^INPUTS:\s*(\[.*\])\s*$", re.MULTILINE)


def parse_candidate(text: str) -> tuple[str, list[str]] | None:
    """Extract `(function_src, args_literals)` from one raw completion, or
    `None` if the completion does not carry the shape the prompt asked for.

    This is intentionally a LOOSE extraction step -- it does not judge
    whether `function_src` is a well-formed, safe function (that is
    `validate`'s job, run separately by the caller). It fails closed (`None`)
    on: no `INPUTS:` line at all, an unparseable or empty inputs list, or an
    inputs list whose entries are not tuples. `args_literals` are `repr()`s
    of each input tuple -- ready to pass straight to `harvest`'s
    `args_literal` parameter, which itself decodes them with
    `ast.literal_eval`.
    """
    body = _unfence(text)
    match = _INPUTS_RE.search(body)
    if match is None:
        return None
    function_src = body[: match.start()].rstrip() + "\n"
    if not function_src.strip():
        return None
    try:
        inputs = ast.literal_eval(match.group(1))
    except (SyntaxError, ValueError):
        return None
    if not isinstance(inputs, list) or not inputs:
        return None
    if not all(isinstance(item, tuple) for item in inputs):
        return None
    return function_src, [repr(item) for item in inputs]


def _unfence(text: str) -> str:
    """Strip a single ```python fenced block if the completion has one."""
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
      exactly one of `parse_fail`, one `validate_fail` reason, or an
      accepted function (`parse_fail + sum(validate_fail.values()) +
      accepted_functions == candidates`).
    - SAMPLE level: every (accepted function, input) pair harvested is
      exactly one of `truncated_rejected`, `nondet_rejected`,
      `balance_rejected`, or an accepted sample. A function is accepted
      (written to `functions.jsonl`) independently of how many of its own
      samples survive this stage -- **the balance guard rejects SAMPLES,
      never the function they belong to**.

    Writes `samples.jsonl` (one JSON line per accepted sample), `functions
    .jsonl` (one JSON line per accepted function), and `gen_stats.json`
    (this function's return value, verbatim).
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
        "nondet_rejected": 0,
        "truncated_rejected": 0,
        "balance_rejected": 0,
        "accepted_functions": 0,
        "accepted_samples": 0,
    }
    class_counts = {0: 0, 1: 0}
    call_index = 0

    samples_path = out_dir / "samples.jsonl"
    functions_path = out_dir / "functions.jsonl"
    with samples_path.open("w") as samples_f, functions_path.open("w") as functions_f:
        while stats["accepted_functions"] < target_functions:
            call_index += 1
            candidates = proposer.generate(GEN_PROMPT, n=8, seed=seed + call_index)
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

                _harvest_and_write_samples(
                    function_src, args_literals, scratch, samples_f, stats, class_counts,
                )
                fn_id = _fn_id(function_src)
                functions_f.write(json.dumps({"fn_id": fn_id, "function_src": function_src}) + "\n")
                stats["accepted_functions"] += 1
                log(f"accepted function {stats['accepted_functions']}/{target_functions} "
                    f"(fn_id={fn_id})")

                if stats["accepted_functions"] >= target_functions:
                    break

    (out_dir / "gen_stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def _harvest_and_write_samples(
    function_src: str, args_literals: list[str], scratch: Path,
    samples_f, stats: dict, class_counts: dict[int, int],
) -> None:
    """Harvest every (function, input) pair for one accepted function,
    filter+count each, and write the survivors to `samples_f`."""
    fn_id = _fn_id(function_src)
    for args_literal in args_literals:
        result = harvest(function_src, args_literal, scratch)

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
        samples_f.write(json.dumps({
            "fn_id": fn_id,
            "function_src": function_src,
            "args": args_literal,
            "outcome": result.outcome,
            "return_repr": result.return_repr,
            "snapshots": [_snapshot_to_json(s) for s in result.snapshots],
        }) + "\n")


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
