"""Corpus assembly, hash splits, and floor verdicts for the B-lite corpus
(prereg §4).

This module reads what Task 2 (`crucible.latent.gen`) already wrote --
`samples.jsonl` (one accepted (function, input) measurement per line),
`functions.jsonl` (one accepted function per line), `gen_stats.json`
(generation-time bucket counts) -- and turns it into two things downstream
tasks depend on:

* `assign_split` / `load_split`: a deterministic, FUNCTION-level train/val/
  test partition. Splitting is done on `fn_id`, never on a sample identity,
  so every sample of one function always lands in the same split -- the
  property that keeps the test set honest (no function trains on some of its
  own inputs and gets evaluated on the rest).
* `build_manifest`: counts, class balance, split sizes, and floor verdicts,
  written to `manifest.json` AND returned. Floor verdicts are computed HERE
  ONLY -- Task 10 (ops) reads `"floor_functions"` / `"nondet_kill"` /
  `"skew_ok"` off the manifest and must never recompute them ad hoc; if a
  floor's meaning changes, it changes in exactly one place.

Honesty note on `nondet_rate`'s denominator, AMENDED by the controller ruling
at spec §12 (pre-lock): it is `nondet_rejected / (nondet_rejected +
truncated_rejected + balance_rejected + accepted_samples)` -- every
(function, input) pair `harvest()` actually returned a determinism verdict
for. Reading `crucible.latent.harvest.harvest`: `deterministic` is computed
UNCONDITIONALLY, before `HarvestResult` is even constructed (the two runs'
`outcome` and state-hash are compared regardless of either run's
`truncated`) -- so a truncated sample still carries a real True/False
determinism verdict, it is simply `continue`d into `truncated_rejected` by
`crucible.latent.gen._harvest_and_write_samples`'s bucket-priority ordering
(truncation is checked before determinism there) rather than ALSO being
tallied under `nondet_rejected` -- each sample lands in exactly one bucket,
per that module's own sample-level conservation invariant. Per the §12
amendment, that verdict counts toward the denominator regardless of which
bucket carries it. Only `harvest_error` is excluded: a pair where
`harvest()` itself raised (`HarvestError`/`OSError`) never produced a
`HarvestResult`, so there is no verdict, computed or otherwise, to count.
`candidates` / `parse_fail` / `validate_fail` remain excluded as
FUNCTION-level buckets (whole candidates rejected before any sample-level
harvesting happens) -- a different level entirely, not a determinism verdict
at all.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from crucible.latent.config import (
    FLOOR_FUNCTIONS,
    NONDET_REJECT_KILL,
    SKEW_LIMIT,
    SPLIT_FRACTIONS,
    SPLIT_SEED,
)
from crucible.latent.gen import binary_label
from crucible.latent.harvest import Snapshot

_VALID_SPLITS = ("train", "val", "test")


# -- hash split -----------------------------------------------------------------


def assign_split(fn_id: str, seed: int) -> str:
    """Deterministic function-level train/val/test assignment.

    `sha256(f"{seed}:{fn_id}")`'s first 8 bytes, read big-endian, scaled to
    `[0, 1)` by dividing by 2**64, then bucketed by `SPLIT_FRACTIONS`'
    cumulative boundaries (`(0.8, 0.1, 0.1)` by default: `u < 0.8` ->
    "train", `u < 0.9` -> "val", else "test"). Seed is mixed INTO the hashed
    text, not applied as a separate transform, specifically so a corpus
    re-split under a different `SPLIT_SEED` is not a trivial permutation of
    the same ordering.

    Pure function of `(fn_id, seed)` -- the same fn_id always lands in the
    same split for a given seed, which is what makes grouping every sample
    of one function into a single split (never split-mixed) automatic: every
    caller of this module computes a function's split by calling this
    function with that function's `fn_id`, nothing else.
    """
    digest = hashlib.sha256(f"{seed}:{fn_id}".encode("utf-8")).digest()
    u = int.from_bytes(digest[:8], "big") / 2**64
    train_frac, val_frac, _test_frac = SPLIT_FRACTIONS
    if u < train_frac:
        return "train"
    if u < train_frac + val_frac:
        return "val"
    return "test"


# -- Sample ------------------------------------------------------------------


@dataclass(frozen=True)
class Sample:
    """One (function, input) measurement, deserialized from a `samples.jsonl`
    line written by `crucible.latent.gen.generate_corpus`."""

    fn_id: str
    function_src: str
    args: str
    outcome: str
    return_repr: str | None
    snapshots: tuple[Snapshot, ...]


def _snapshot_from_json(row: dict) -> Snapshot:
    return Snapshot(line=row["line"], locals=tuple(tuple(entry) for entry in row["locals"]))


def _sample_from_json(row: dict) -> Sample:
    return Sample(
        fn_id=row["fn_id"],
        function_src=row["function_src"],
        args=row["args"],
        outcome=row["outcome"],
        return_repr=row.get("return_repr"),
        snapshots=tuple(_snapshot_from_json(s) for s in row["snapshots"]),
    )


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


# -- load_split -----------------------------------------------------------------


def load_split(corpus_dir: Path, split: str) -> list[Sample]:
    """Every `Sample` whose function hashes into `split`.

    `split` has NO default deliberately -- silently falling back to some
    split (e.g. "train") would turn a caller's typo into a corrupted
    experiment rather than a loud failure.
    """
    if split not in _VALID_SPLITS:
        raise ValueError(f"unknown split {split!r}, expected one of {_VALID_SPLITS}")
    corpus_dir = Path(corpus_dir)
    rows = _read_jsonl(corpus_dir / "samples.jsonl")
    return [
        _sample_from_json(row) for row in rows
        if assign_split(row["fn_id"], SPLIT_SEED) == split
    ]


# -- sample-provenance stats reconciliation (round-3 CRITICAL fix follow-up) -----

_SAMPLE_LEVEL_BUCKETS = (
    "nondet_rejected", "truncated_rejected", "balance_rejected", "accepted_samples",
)


def _stat_or_raise(stats: dict, source_name: str, key: str) -> int:
    """`stats[key]`, or a clear `KeyError` naming BOTH the missing bucket
    and which stats file it was expected in -- NEVER `.get(key, 0)` (final
    review MEDIUM, extended to every stats source this function reads): a
    malformed or truncated stats file missing a required bucket must not
    silently read as zero, which could flip `nondet_kill` (or
    `floor_functions`, transitively, via a misleadingly-empty `nondet_rate`
    denominator) to a false PASS. A malformed stats file must fail loud,
    never read as a clean corpus."""
    try:
        return stats[key]
    except KeyError as exc:
        raise KeyError(
            f"{source_name} is missing required bucket {key!r} -- a "
            "malformed stats file must not silently read as 0 and risk a "
            "false floor PASS"
        ) from exc


def _load_stats_if_complete(path: Path) -> dict:
    """One stats JSON file, hard-failing if it exists but `complete` is not
    true. An in-progress or crashed pass's partial bucket counts are not a
    trustworthy sample-provenance source -- reading them as if the run had
    finished could silently understate (or overstate) the corpus's real
    determinism/truncation rate. Never a silent fallback to a different
    source: the caller decided THIS file is the source of record, and an
    incomplete file at that path is a hard error, not a cue to look
    elsewhere."""
    stats = json.loads(path.read_text())
    if not stats.get("complete"):
        raise RuntimeError(
            f"{path} exists but its complete field is not true -- refusing "
            "to use an incomplete pass as a sample-provenance source"
        )
    return stats


def _sample_stats_and_sources(corpus_dir: Path) -> tuple[dict[str, int], list[str]]:
    """The four §12 sample-level buckets (`_SAMPLE_LEVEL_BUCKETS`) this
    corpus's floors are computed from, and which file(s) they came from
    (`manifest["stats_sources"]` -- names the lens for anyone reading the
    manifest later, since the answer depends on which stats file(s) exist).

    `reharvest_stats.json`, when present, IS the sample-provenance source
    (round-3 CRITICAL fix: `harvest()`'s pre-fix scratch-dir-reuse bug meant
    every call after the first one sharing a scratch directory silently
    replayed that first call's result, so `gen_stats.json`'s OWN bucket
    counts from before a reharvest describe a run that was never a real
    measurement). Hard-fails (via `_load_stats_if_complete`) if it exists
    but is not `complete`.

    `battery_stats.json`, when it ALSO exists alongside `reharvest_stats
    .json`, contributes its own buckets into the SAME sums -- its accepted
    rows live in the CURRENT `samples.jsonl` too, and its own `harvest()`
    calls are always post-fix (`generate_battery_inputs` only ever runs
    against an already-generated corpus), so there is no reason to exclude
    them. Also hard-fails if present but incomplete.

    Falls back to `gen_stats.json` ONLY when `reharvest_stats.json` does
    not exist at all -- the ORIGINAL, single-source behavior, byte-for-byte
    unchanged for a corpus that was never reharvested (regression-tested).
    `battery_stats.json` is NEVER combined into this fallback path: mixing
    a still-fixed-only battery pass into a `gen_stats.json` whose OWN
    `generate_corpus` samples may still be replay-corrupted would
    understate, not correct, the problem.
    """
    reharvest_path = corpus_dir / "reharvest_stats.json"
    if not reharvest_path.exists():
        gen_stats = json.loads((corpus_dir / "gen_stats.json").read_text())
        totals = {
            key: _stat_or_raise(gen_stats, "gen_stats.json", key)
            for key in _SAMPLE_LEVEL_BUCKETS
        }
        return totals, ["gen_stats.json"]

    sources = [("reharvest_stats.json", _load_stats_if_complete(reharvest_path))]
    battery_path = corpus_dir / "battery_stats.json"
    if battery_path.exists():
        sources.append(("battery_stats.json", _load_stats_if_complete(battery_path)))

    totals = {key: 0 for key in _SAMPLE_LEVEL_BUCKETS}
    for name, stats in sources:
        for key in _SAMPLE_LEVEL_BUCKETS:
            totals[key] += _stat_or_raise(stats, name, key)
    return totals, [name for name, _stats in sources]


# -- build_manifest ---------------------------------------------------------------


def build_manifest(corpus_dir: Path) -> dict:
    """Assemble `manifest.json` from `functions.jsonl`, `samples.jsonl`, and
    `gen_stats.json` under `corpus_dir`; write it and return the same dict.

    Schema (this IS the pre-registered interface Task 10 reads):
    ```
    {
      "accepted_functions": int,
      "accepted_samples": int,
      "class_balance": {"binary": {"0": int, "1": int}, "multiclass": {<outcome>: int, ...}},
      "split_sizes": {"functions": {"train"|"val"|"test": int, ...},
                       "samples":   {"train"|"val"|"test": int, ...}},
      "floors": {
        "floor_functions": "PASS"|"FAIL",   # accepted_functions >= FLOOR_FUNCTIONS
        "nondet_rate": float,               # see module docstring for the denominator
        "nondet_kill": "PASS"|"FAIL",       # nondet_rate <= NONDET_REJECT_KILL
        "balance": float,                   # majority binary class fraction
        "skew_ok": bool,                    # balance <= SKEW_LIMIT
      },
      "stats_sources": [str, ...],
      "samples_sha256": str,
    }
    ```
    `accepted_functions`/`accepted_samples` are counted from the files
    actually on disk (ground truth), not trusted from any stats file's
    self-reported totals -- stats files are used ONLY for the
    `nondet_rejected`/`truncated_rejected`/`balance_rejected` buckets, which
    exist nowhere else (rejected samples are never written to
    `samples.jsonl`).

    Stats-source reconciliation (round-3 CRITICAL fix follow-up): a corpus
    whose `harvest()` calls predate the round-3 fix has a `samples.jsonl`
    that may be entirely replay-corrupted -- `gen_stats.json`'s OWN bucket
    counts, from before a reharvest, describe that same corrupted run. If
    `reharvest_stats.json` exists in `corpus_dir`, it is THE
    sample-provenance source for the four §12 buckets, not `gen_stats.json`
    -- see `_sample_stats_and_sources` for the exact rule (hard-fails if it
    exists but is not `complete`; adds `battery_stats.json`'s buckets into
    the same sums when that ALSO exists, since its rows live in the current
    `samples.jsonl` and its own `harvest()` calls are never pre-fix).
    `gen_stats.json` remains the source ONLY when no `reharvest_stats.json`
    exists at all. `manifest["stats_sources"]` names exactly which file(s)
    the four buckets came from, so a manifest reader never has to guess
    which lens produced `nondet_rate`. Function-level floors
    (`floor_functions`) are unaffected either way -- they are already
    counted directly from `functions.jsonl`, never from any stats file.
    """
    corpus_dir = Path(corpus_dir)
    samples_path = corpus_dir / "samples.jsonl"
    functions = _read_jsonl(corpus_dir / "functions.jsonl")
    samples = _read_jsonl(samples_path)

    accepted_functions = len(functions)
    accepted_samples = len(samples)

    sample_stats, stats_sources = _sample_stats_and_sources(corpus_dir)

    binary_counts = {"0": 0, "1": 0}
    multiclass_counts: dict[str, int] = {}
    for row in samples:
        label = str(binary_label(row["outcome"]))
        binary_counts[label] += 1
        multiclass_counts[row["outcome"]] = multiclass_counts.get(row["outcome"], 0) + 1

    split_function_counts = {"train": 0, "val": 0, "test": 0}
    for fn in functions:
        split_function_counts[assign_split(fn["fn_id"], SPLIT_SEED)] += 1

    split_sample_counts = {"train": 0, "val": 0, "test": 0}
    for row in samples:
        split_sample_counts[assign_split(row["fn_id"], SPLIT_SEED)] += 1

    floor_functions_pass = accepted_functions >= FLOOR_FUNCTIONS

    # Determinism-verdict denominator (spec §12 amendment) -- see module
    # docstring: every bucket a harvested pair with a real determinism
    # verdict can land in, EXCEPT harvest_error (no HarvestResult at all).
    nondet_rejected = sample_stats["nondet_rejected"]
    truncated_rejected = sample_stats["truncated_rejected"]
    balance_rejected = sample_stats["balance_rejected"]
    gen_accepted_samples = sample_stats["accepted_samples"]
    screened = nondet_rejected + truncated_rejected + balance_rejected + gen_accepted_samples
    nondet_rate = (nondet_rejected / screened) if screened else 0.0
    nondet_kill_pass = nondet_rate <= NONDET_REJECT_KILL

    total_binary = binary_counts["0"] + binary_counts["1"]
    majority = max(binary_counts["0"], binary_counts["1"])
    balance = (majority / total_binary) if total_binary else 0.0
    skew_ok = balance <= SKEW_LIMIT

    manifest = {
        "accepted_functions": accepted_functions,
        "accepted_samples": accepted_samples,
        "class_balance": {
            "binary": binary_counts,
            "multiclass": multiclass_counts,
        },
        "split_sizes": {
            "functions": split_function_counts,
            "samples": split_sample_counts,
        },
        "floors": {
            "floor_functions": "PASS" if floor_functions_pass else "FAIL",
            "nondet_rate": nondet_rate,
            "nondet_kill": "PASS" if nondet_kill_pass else "FAIL",
            "balance": balance,
            "skew_ok": skew_ok,
        },
        "stats_sources": stats_sources,
        "samples_sha256": hashlib.sha256(samples_path.read_bytes()).hexdigest(),
    }

    (corpus_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
