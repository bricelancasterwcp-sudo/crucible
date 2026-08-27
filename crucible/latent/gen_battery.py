"""Deterministic adversarial battery: the REPLACEMENT for the LLM-proposed
minority pass (spec §12 amendment, controller ruling round 2).

Live-fire on `crucible.latent.gen.generate_minority_inputs` failed after
200/5000 functions: 93% `parse_fail` (the 1.5B would not reliably follow the
"reply with just an INPUTS line" shape) and 0 `accepted_minority` (the
handful of inputs that DID parse were not actually adversarial against these
mostly-total tiny functions). Evidence preserved at
`runs/blite-corpus/minority_stats.llm-attempt.json`. This module drops the
proposer entirely: for every function `generate_corpus` already accepted, it
enumerates a FIXED, ordered set of adversarial argument tuples
(`config.BATTERY_VALUES`) from the function's arity alone, no sampling, no
model call, no randomness anywhere.

This is a SIBLING to `generate_minority_inputs`, not a rewrite of it -- same
corpus-directory contract (reads `functions.jsonl`, appends to
`samples.jsonl`, never touches either of those or `gen_stats.json`), same
balance-guard-aware harvesting, same crash-and-retry dedup safety net. Both
passes are additive and independent; nothing here removes or supersedes the
LLM pass's code, which stays as the documented, live-fire-falsified first
attempt (spec §12's own amendment trail).

Round 3 (controller ruling, sizing): sequential harvesting does not scale --
~5000 functions x ~12 post-dedup candidates is ~60k `harvest()` calls, each
running two `sensorium run` subprocesses, ~11h sequential. `generate_battery_
inputs` now PARALLELIZES the harvest step, mirroring `crucible.latent.
reharvest.reharvest_samples`'s reviewed `ThreadPoolExecutor` pattern EXACTLY
(see that module's docstring for why this is safe post the round-3 CRITICAL
fix: every `harvest()` call gets its own `call-<uuid4>` subdirectory, so many
calls may run concurrently against one shared scratch directory). What stays
SEQUENTIAL and deterministic, unchanged: candidate ENUMERATION
(`_battery_candidates`) and dedup-then-cap selection (`_select_after_dedup`)
-- the exact SET of `(fn_id, function_src, args_literal)` work items handed
to the executor never depends on `jobs`, only their harvesting does. On
success, `samples.jsonl` (the WHOLE file -- this pass APPENDS to whatever a
prior pass already wrote) is rewritten sorted, atomically, via
`crucible.latent.reharvest._rewrite_samples_sorted` (reused, not
duplicated) -- `corpus.build_manifest` hashes the raw file bytes, and the
order threads finish writing in is a scheduling accident, not reproducible
run to run.

Reuses `crucible.latent.gen`'s pure, stateless pieces rather than
duplicating them: `_scan_existing_corpus_state` (real on-disk balance +
crash-retry dedup map -- UNLIKE `reharvest_samples`, which rebuilds the
guard from zero, this pass still seeds it from the corpus's real existing
balance, since enrichment is meant to respect what is already there),
`binary_label`, `_balance_guard_rejects`, `_snapshot_to_json`, and
`_dump_stats_json`. It does NOT reuse `gen._harvest_and_write_minority_
samples` any more (round 2's design) -- that helper is not thread-safe (no
locking around its `class_counts`/file-write mutations), exactly the same
reason `reharvest.py` never reused it either; `_battery_harvest_one` below
mirrors `reharvest._reharvest_one`'s locking discipline instead: `harvest()`
itself runs UNLOCKED (the expensive, subprocess-bound step), and only the
balance-guard check, `class_counts`, `stats`, and the `samples.jsonl` write
are serialized under one `threading.Lock`.

Monkeypatching `crucible.latent.gen_battery.harvest` (THIS module's own
global, imported directly from `crucible.latent.harvest` -- not `gen`'s) is
the patch target for tests, same convention `reharvest.py` established and
for the same reason: `_battery_harvest_one` calls the bare name `harvest`
looked up in `gen_battery`'s own module globals.
"""
from __future__ import annotations

import ast
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from crucible.latent import gen, reharvest
from crucible.latent.config import BATTERY_MAX_PER_FN, BATTERY_VALUES
from crucible.latent.harvest import HarvestError, harvest

# Live-visibility knob, same role as gen.py's GEN_STATS_FLUSH_INTERVAL /
# MINORITY_STATS_FLUSH_INTERVAL / reharvest.py's REHARVEST_STATS_FLUSH_
# INTERVAL -- an ops/UX cadence, not a prereg-cited number, kept local here
# (not config.py) and directly monkeypatchable. Counted in COMPLETED work
# items (harvest calls finishing), not functions, since round 3 moved
# harvesting off the per-function loop entirely -- mirrors reharvest.py's
# own choice of unit exactly, for the same reason (the per-function loop is
# now the cheap, fully-sequential enumeration phase; the executor is where
# wall-clock time is actually spent).
BATTERY_STATS_FLUSH_INTERVAL = 100


# -- deterministic candidate enumeration -------------------------------------


def _function_arity(function_src: str) -> int:
    """Positional-callable arity of `def f(...)` in `function_src`: the count
    of positional-only PLUS positional-or-keyword parameters. `*args`,
    keyword-only parameters, and `**kwargs` are not counted -- they cannot be
    filled by a plain positional call, the only calling convention this
    battery (or `harvest`'s own runner script) ever uses. `function_src` is
    always something `crucible.latent.gen.validate()` already accepted
    (exactly one top-level `def f(...):`), so `tree.body[0]` is always that
    one `FunctionDef`.
    """
    tree = ast.parse(function_src)
    node = tree.body[0]
    return len(node.args.posonlyargs) + len(node.args.args)


def _battery_candidates(arity: int) -> list[tuple]:
    """The full, deterministic, PRE-dedup enumeration of adversarial argument
    tuples for a function of this arity. ORDER IS PINNED (mutation-tested by
    `tests/latent/test_gen_battery.py`):

    1. HOMOGENEOUS probes, one per `config.BATTERY_VALUES` entry IN THAT
       ORDER: `(v,) * arity`.
    2. Only when `arity >= 2`, HETEROGENEOUS probes, position-major then
       probe-value-minor: for each argument position `i` in `range(arity)`,
       for each of `(None, [], "")` in that order, a tuple of `1`s with
       position `i` replaced by that probe value.

    `arity == 0` yields `config.BATTERY_VALUES`-many copies of the empty
    tuple (every homogeneous probe collapses to `()` when there are no
    positions to fill) -- harmless: the caller's dedup-before-cap step
    reduces those to the one candidate that actually gets harvested.
    """
    homogeneous = [(v,) * arity for v in BATTERY_VALUES]
    if arity < 2:
        return homogeneous
    heterogeneous = [
        tuple(probe if position == i else 1 for position in range(arity))
        for i in range(arity)
        for probe in (None, [], "")
    ]
    return homogeneous + heterogeneous


def _select_after_dedup(
    raw_candidates: list[tuple], existing_literals: set[str], stats: dict,
) -> list[str]:
    """The first `config.BATTERY_MAX_PER_FN` entries of `raw_candidates`
    (returned as their `repr()` literals, ready to hand straight to
    `harvest`) that round-trip through `ast.literal_eval` and are NOT
    already in `existing_literals` (this function's original `args_literals`
    union whatever `samples.jsonl` already holds for it) or a repeat of an
    earlier survivor in this same call. Fully SEQUENTIAL and deterministic --
    called once per function, before any harvesting starts, so the exact SET
    of work items handed to the executor never depends on `jobs`.

    - `invalid_literal`: `repr(item)` does not itself round-trip back
      through `ast.literal_eval` -- no `BATTERY_VALUES`-derived tuple
      actually triggers this in practice, but the check (and its counter)
      stay here rather than silently assuming every enumerated tuple is
      harvestable.
    - `duplicate_input`: already known, from either source above, or a
      repeat within this same enumeration.

    Iteration stops the moment `config.BATTERY_MAX_PER_FN` survivors have
    been collected; candidates after that point are never inspected -- "the
    first `BATTERY_MAX_PER_FN` candidates AFTER dedup" means exactly this:
    a raw candidate beyond what was needed to fill the cap is neither a
    counted duplicate nor a harvested sample, simply out of this run's
    budget.
    """
    selected: list[str] = []
    seen: set[str] = set()
    for item in raw_candidates:
        if len(selected) >= BATTERY_MAX_PER_FN:
            break
        literal = repr(item)
        try:
            ast.literal_eval(literal)
        except (SyntaxError, ValueError):
            stats["invalid_literal"] += 1
            continue
        if literal in existing_literals or literal in seen:
            stats["duplicate_input"] += 1
            continue
        selected.append(literal)
        seen.add(literal)
    return selected


# -- ordering guard (round-3 CRITICAL fix follow-up, controller ruling) -------


def _refuse_if_reharvest_incomplete(corpus_dir: Path) -> None:
    """Refuse to enrich a corpus that is mid-repair.

    If `samples.jsonl.replay-corrupt` exists, a `crucible.latent.reharvest.
    reharvest_samples` run was STARTED against this `corpus_dir` at some
    point -- `samples.jsonl` right now is THAT reharvest's own (possibly
    still-incomplete) output, not the archived pre-fix data, since
    `reharvest_samples` archives-then-truncates in that order. Enriching on
    top of an interrupted or still-running reharvest's `samples.jsonl`
    means building on a file that reharvest may still truncate and rewrite
    out from under this pass, or that never finished being measured in the
    first place.

    Refuses UNLESS a COMPLETE `reharvest_stats.json` also exists (`complete
    == True`), confirming that reharvest actually finished. Deliberately
    narrow: a `corpus_dir` that was NEVER reharvested at all (no
    `.replay-corrupt` file) is NOT refused here, even though its
    `samples.jsonl` could in principle still be pre-round-3-fix-corrupted --
    detecting that would need dating or version-stamping the corpus, which
    this guard does not attempt; it only closes the specific "reharvest in
    progress or abandoned partway" ordering hazard.
    """
    corrupt_marker = corpus_dir / "samples.jsonl.replay-corrupt"
    if not corrupt_marker.exists():
        return
    reharvest_stats_path = corpus_dir / "reharvest_stats.json"
    if reharvest_stats_path.exists():
        stats = json.loads(reharvest_stats_path.read_text())
        if stats.get("complete"):
            return
    raise RuntimeError(
        f"{corpus_dir} has samples.jsonl.replay-corrupt but no COMPLETE "
        "reharvest_stats.json -- run reharvest first "
        "(crucible.latent.reharvest.reharvest_samples) before enriching "
        "this corpus further"
    )


# -- battery pass -------------------------------------------------------------


def generate_battery_inputs(
    corpus_dir: Path, *, seed: int, jobs: int = 8, log: Callable[[str], None] = print,
) -> dict:
    """Deterministic replacement for `generate_minority_inputs` (spec §12
    amendment, controller ruling round 2), now PARALLELIZED (round 3) --
    see this module's docstring for both the live-fire evidence that
    motivated round 2 and the sizing problem (~60k harvests, ~11h
    sequential) that motivated round 3.

    No proposer is passed or used: every candidate input comes from
    `_battery_candidates`, a fixed function of the function's own arity
    alone. `seed` is accepted ONLY for interface symmetry with
    `generate_minority_inputs` -- it is never read, never mixed into
    anything, and no RNG is ever constructed or consulted anywhere in this
    module. `jobs` controls ONLY how many `harvest()` calls run
    concurrently (`ThreadPoolExecutor(max_workers=jobs)`, mirroring
    `crucible.latent.reharvest.reharvest_samples` exactly) -- it changes
    wall-clock time and on-disk write ORDER, never which candidates are
    considered or which are ultimately accepted. Two calls against the same
    `corpus_dir` state, with any two `seed`/`jobs` values, accept the exact
    same SET of samples (mutation/regression-tested), and `samples.jsonl`
    is byte-identical after the final sort-rewrite regardless of `jobs`.

    Two phases:

    1. SEQUENTIAL enumeration (unchanged from round 2's design, still fully
       deterministic): for every function in `functions.jsonl` (file
       order), compute `arity` and the full candidate list
       (`_battery_candidates(arity)`) -- added to `stats["candidates"]`
       UNCONDITIONALLY, before any filtering -- then `_select_after_dedup`
       picks up to `config.BATTERY_MAX_PER_FN` fresh candidates (see its own
       docstring for the exact dedup/cap/invalid-literal rules), each
       becoming one `(fn_id, function_src, args_literal)` work item.
       `stats["functions_processed"]` counts progress through THIS phase
       only -- it therefore reaches `functions_total` before any harvesting
       starts; per-item harvest progress is tracked separately (see below).
    2. PARALLEL harvest: every work item is submitted to a `ThreadPoolExecutor
       (jobs)` (`_battery_harvest_one`, mirroring `reharvest._reharvest_one`):
       `HarvestError`/`OSError` -> `harvest_error`; `result.truncated` ->
       `truncated_rejected`; not `result.deterministic` -> `nondet_rejected`;
       the balance guard (`gen._balance_guard_rejects`, evaluated against
       `class_counts` SEEDED from `_scan_existing_corpus_state` -- unlike
       `reharvest_samples`, this pass does not rebuild the guard from zero,
       since enrichment is meant to respect the corpus's real existing
       balance) rejects -> `balance_rejected`; otherwise -> appended to
       `samples.jsonl` and counted in `accepted_samples` (plus
       `accepted_minority` when `binary_label(outcome) == 0`, i.e. `outcome
       != "return"` -- this pass's entire purpose, made countable). Only the
       balance-guard check, `class_counts`, `stats`, and the file write are
       serialized under one `threading.Lock`; `harvest()` itself always
       runs unlocked.

    CONSERVATION (test-enforced) holds over work items actually submitted:
    `harvest_error + nondet_rejected + truncated_rejected + balance_rejected
    + accepted_samples == len(work_items)`. `duplicate_input` +
    `invalid_literal` are a SEPARATE, phase-1-only conservation equation
    against `stats["candidates"]` (the raw, pre-cap total) -- when the cap
    binds, `candidates` can exceed `duplicate_input + invalid_literal +
    len(work_items)`, by construction, since the whole point of the cap is
    to leave some already-fresh candidates unexamined.

    On the success path only, `samples.jsonl` (the WHOLE file, not just this
    run's additions -- this pass APPENDS to whatever a prior pass already
    wrote) is rewritten sorted and atomically
    (`reharvest._rewrite_samples_sorted`, reused verbatim) BEFORE `stats[
    "complete"]` is ever set `True` -- if the rewrite itself fails, `samples
    .jsonl` is left exactly as harvesting wrote it and `complete` stays
    `False`, same discipline `reharvest_samples` established.
    `battery_stats.json` (this function's return value, verbatim, plus
    `"jobs"`/`"complete"`) is flushed every `BATTERY_STATS_FLUSH_INTERVAL`
    COMPLETED work items, and unconditionally in a `finally` block.
    `functions.jsonl`, `gen_stats.json`, and `minority_stats.json` are never
    opened for writing.
    """
    corpus_dir = Path(corpus_dir)
    _refuse_if_reharvest_incomplete(corpus_dir)
    scratch = corpus_dir / "_battery_harvest_scratch"
    functions_path = corpus_dir / "functions.jsonl"
    samples_path = corpus_dir / "samples.jsonl"
    battery_stats_path = corpus_dir / "battery_stats.json"

    functions = [
        json.loads(line) for line in functions_path.read_text().splitlines() if line.strip()
    ]

    stats = {
        "jobs": jobs,
        "seed": seed,
        "functions_total": len(functions),
        "functions_processed": 0,
        "candidates": 0,
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
    class_counts, args_by_fn = gen._scan_existing_corpus_state(samples_path)

    # Phase 1 -- SEQUENTIAL, deterministic: enumerate + dedup every
    # function's candidates BEFORE any harvesting begins. The resulting
    # work_items list (and therefore what gets harvested and what CAN be
    # accepted) never depends on `jobs`.
    work_items: list[tuple[str, str, str]] = []
    for fn in functions:
        function_src = fn["function_src"]
        fn_id = fn["fn_id"]
        existing_literals = set(fn.get("args_literals", ()))
        existing_literals |= args_by_fn.get(fn_id, set())

        arity = _function_arity(function_src)
        raw_candidates = _battery_candidates(arity)
        stats["candidates"] += len(raw_candidates)

        for literal in _select_after_dedup(raw_candidates, existing_literals, stats):
            work_items.append((fn_id, function_src, literal))
        stats["functions_processed"] += 1

    lock = threading.Lock()
    try:
        with samples_path.open("a") as samples_f, ThreadPoolExecutor(max_workers=jobs) as ex:
            futures = [
                ex.submit(_battery_harvest_one, fn_id, function_src, args_literal,
                         scratch, samples_f, stats, class_counts, lock)
                for fn_id, function_src, args_literal in work_items
            ]
            for i, fut in enumerate(as_completed(futures), start=1):
                fut.result()   # propagate any bug that isn't HarvestError/OSError
                log(f"battery pass {i}/{len(work_items)}")
                if i % BATTERY_STATS_FLUSH_INTERVAL == 0:
                    with lock:
                        _write_battery_stats(battery_stats_path, stats)
        # Sort-rewrite FIRST, mark complete only AFTER it succeeds -- see
        # this function's own docstring and reharvest._rewrite_samples_
        # sorted's docstring for why (atomic temp-file + os.replace swap).
        reharvest._rewrite_samples_sorted(samples_path)
        stats["complete"] = True
    finally:
        _write_battery_stats(battery_stats_path, stats)
    return stats


def _battery_harvest_one(
    fn_id: str, function_src: str, args_literal: str, scratch: Path,
    samples_f, stats: dict, class_counts: dict[int, int], lock: threading.Lock,
) -> None:
    """Harvest ONE battery candidate, on a worker thread, and -- if it
    survives every filter -- append it to `samples_f`. Mirrors `crucible.
    latent.reharvest._reharvest_one`'s locking discipline exactly, plus this
    pass's own `accepted_minority` count.

    `harvest()` itself runs UNLOCKED: it is subprocess-bound and, post the
    round-3 CRITICAL fix, fully isolated per call (its own `call-<uuid4>`
    scratch subdirectory), so many of these run genuinely concurrently.
    Only the parts that touch STATE SHARED across threads -- `stats`,
    `class_counts`, and the file -- run under `lock`, and only after the
    expensive work is already done, so the lock is held for a dict update
    and a line write, never for a harvest.
    """
    try:
        result = harvest(function_src, args_literal, scratch)
    except (HarvestError, OSError):
        with lock:
            stats["harvest_error"] += 1
        return

    if result.truncated:
        with lock:
            stats["truncated_rejected"] += 1
        return
    if not result.deterministic:
        with lock:
            stats["nondet_rejected"] += 1
        return

    label = gen.binary_label(result.outcome)
    # Serialized OUTSIDE the lock -- json.dumps touches only this call's own
    # local `result`, never shared state, so there is nothing to protect by
    # doing it while holding the lock.
    row = json.dumps({
        "fn_id": fn_id,
        "function_src": function_src,
        "args": args_literal,
        "outcome": result.outcome,
        "return_repr": result.return_repr,
        "snapshots": [gen._snapshot_to_json(s) for s in result.snapshots],
    })
    with lock:
        if gen._balance_guard_rejects(label, class_counts):
            stats["balance_rejected"] += 1
            return
        class_counts[label] += 1
        stats["accepted_samples"] += 1
        if label == 0:
            stats["accepted_minority"] += 1
        samples_f.write(row + "\n")
        samples_f.flush()


def _write_battery_stats(path: Path, stats: dict) -> None:
    """The single battery_stats.json write path -- delegates to
    `gen._dump_stats_json` (DRY: identical serialization to every other
    pass's stats file) but kept as its own name so this pass's
    periodic-flush test double can be monkeypatched independently of
    `gen`'s own `_write_gen_stats` / `_write_minority_stats` and
    `reharvest`'s own `_write_reharvest_stats` spies."""
    gen._dump_stats_json(path, stats)
