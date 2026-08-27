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

Reuses `crucible.latent.gen`'s internals rather than duplicating them:
`_scan_existing_corpus_state` (real on-disk balance + crash-retry dedup
map), `_harvest_and_write_minority_samples` (harvest -> deterministic ∧
non-truncated -> balance-guard -> append, with its own duplicate_input /
harvest_error / truncated_rejected / nondet_rejected / balance_rejected /
accepted_samples / accepted_minority bucketing), and `_dump_stats_json` (the
shared stats-file serialization both passes' stats writers delegate to).
Monkeypatching `crucible.latent.gen.harvest` in a test intercepts this
module's harvesting too, since `_harvest_and_write_minority_samples` looks
`harvest` up in `gen`'s own module globals regardless of which module calls
it.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Callable

from crucible.latent import gen
from crucible.latent.config import BATTERY_MAX_PER_FN, BATTERY_VALUES

# Live-visibility knob, same role as gen.py's GEN_STATS_FLUSH_INTERVAL /
# MINORITY_STATS_FLUSH_INTERVAL -- an ops/UX cadence, not a prereg-cited
# number, kept local here (not config.py) and directly monkeypatchable.
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
    corpus_dir: Path, *, seed: int, log: Callable[[str], None] = print,
) -> dict:
    """Deterministic replacement for `generate_minority_inputs` (spec §12
    amendment, controller ruling round 2 -- see this module's docstring for
    the live-fire evidence that motivated it). No proposer is passed or
    used: every candidate input comes from `_battery_candidates`, a fixed
    function of the function's own arity alone. `seed` is accepted ONLY for
    interface symmetry with `generate_minority_inputs` -- it is never read,
    never mixed into anything, and no RNG is ever constructed or consulted
    anywhere in this module. Two calls against the same `corpus_dir` state,
    with any two `seed` values, produce byte-identical `samples.jsonl`
    output (mutation/regression-tested as a determinism property, not just
    asserted).

    For every function in `functions.jsonl` (file order):

    1. Compute `arity` (`_function_arity`) and the full candidate list
       (`_battery_candidates(arity)`) -- added to `stats["candidates"]`
       UNCONDITIONALLY, before any filtering, mirroring
       `generate_corpus`'s own `candidates` field (every candidate the
       generation step produced, not just the ones actually attempted).
    2. Walk that list in order, skipping (and counting in
       `duplicate_input`) any candidate whose `repr()` is already in this
       function's `args_literals` (from `functions.jsonl`), already in
       `samples.jsonl` for this `fn_id` (crash-and-retry safety net, see
       `gen._scan_existing_corpus_state`), or already selected earlier in
       THIS walk -- collecting up to `config.BATTERY_MAX_PER_FN` survivors,
       then stopping (this is "the first `BATTERY_MAX_PER_FN` candidates
       AFTER dedup": candidates beyond whatever was needed to fill that cap
       are never even inspected, so a raw candidate can be neither a
       counted duplicate nor a harvested sample -- simply out of this run's
       budget).
    3. Hand the (at most `BATTERY_MAX_PER_FN`) survivors to
       `gen._harvest_and_write_minority_samples`, which does the harvest ->
       deterministic ∧ non-truncated -> balance-guard -> append work and
       its own bucket counting (`harvest_error`, `truncated_rejected`,
       `nondet_rejected`, `balance_rejected`, `accepted_samples`,
       `accepted_minority`) -- identical to what `generate_minority_inputs`
       uses. Its `invalid_literal` check never actually fires for battery
       values (every entry of `BATTERY_VALUES` and every heterogeneous
       probe round-trips cleanly through `repr()`/`ast.literal_eval`), but
       the counter is still initialized so that shared function never sees
       a missing key.

    CONSERVATION (test-enforced) holds over candidates this pass actually
    examines: `duplicate_input + harvest_error + nondet_rejected +
    truncated_rejected + balance_rejected + accepted_samples ==
    (duplicate_input + len(survivors))` for any run where the cap does not
    bind (every raw candidate gets examined) -- when the cap DOES bind,
    `stats["candidates"]` (the raw, pre-cap total) can exceed that sum, by
    construction, since the whole point of the cap is to leave some
    already-fresh candidates unexamined.

    The balance guard (`gen._balance_guard_rejects`) evaluates against
    `class_counts` seeded from a fresh scan of the corpus's REAL current
    `samples.jsonl` (`gen._scan_existing_corpus_state`) -- this pass never
    resets it to zero and never touches `config.SKEW_LIMIT` /
    `config.BALANCE_GUARD_MIN_SAMPLES` itself.

    Writes ONLY to `samples.jsonl` (opened in APPEND mode, flushed per
    line -- append-only, can only grow the corpus) and `battery_stats.json`
    (this function's return value, verbatim, plus `"complete"`; flushed
    early every `BATTERY_STATS_FLUSH_INTERVAL` functions, and unconditionally
    in a `finally` block so a mid-run crash still leaves `complete: False`
    and the real counts accumulated so far). `functions.jsonl`,
    `gen_stats.json`, and `minority_stats.json` are never opened for
    writing.
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

    try:
        with samples_path.open("a") as samples_f:
            for fn in functions:
                function_src = fn["function_src"]
                fn_id = fn["fn_id"]
                existing_literals = set(fn.get("args_literals", ()))
                existing_literals |= args_by_fn.get(fn_id, set())

                arity = _function_arity(function_src)
                raw_candidates = _battery_candidates(arity)
                stats["candidates"] += len(raw_candidates)

                selected = _select_after_dedup(raw_candidates, existing_literals, stats)

                gen._harvest_and_write_minority_samples(
                    fn_id, function_src, selected, existing_literals,
                    scratch, samples_f, stats, class_counts,
                )
                stats["functions_processed"] += 1
                log(f"battery pass {stats['functions_processed']}/{len(functions)} "
                    f"(fn_id={fn_id}, arity={arity}, selected={len(selected)})")

                if stats["functions_processed"] % BATTERY_STATS_FLUSH_INTERVAL == 0:
                    _write_battery_stats(battery_stats_path, stats)
        stats["complete"] = True
    finally:
        _write_battery_stats(battery_stats_path, stats)
    return stats


def _write_battery_stats(path: Path, stats: dict) -> None:
    """The single battery_stats.json write path -- delegates to
    `gen._dump_stats_json` (DRY: identical serialization to the other two
    passes' stats files) but kept as its own name so this pass's
    periodic-flush test double can be monkeypatched independently of
    `gen`'s own `_write_gen_stats` / `_write_minority_stats` spies."""
    gen._dump_stats_json(path, stats)


def _select_after_dedup(
    raw_candidates: list[tuple], existing_literals: set[str], stats: dict,
) -> list[tuple]:
    """The first `config.BATTERY_MAX_PER_FN` entries of `raw_candidates` that
    are NOT already in `existing_literals` (this function's original
    `args_literals` union whatever `samples.jsonl` already holds for it) and
    not a repeat of an earlier survivor in this same call -- every skipped
    duplicate along the way is counted in `stats["duplicate_input"]`.
    Iteration stops the moment `config.BATTERY_MAX_PER_FN` survivors have
    been collected; candidates after that point are never inspected (see
    `generate_battery_inputs`'s docstring on why that is the correct reading
    of "cap applied AFTER dedup").
    """
    selected: list[tuple] = []
    selected_literals: set[str] = set()
    for item in raw_candidates:
        if len(selected) >= BATTERY_MAX_PER_FN:
            break
        literal = repr(item)
        if literal in existing_literals or literal in selected_literals:
            stats["duplicate_input"] += 1
            continue
        selected.append(item)
        selected_literals.add(literal)
    return selected
