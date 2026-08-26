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
      "samples_sha256": str,
    }
    ```
    `accepted_functions`/`accepted_samples` are counted from the files
    actually on disk (ground truth), not trusted from `gen_stats.json`'s
    self-reported totals -- `gen_stats.json` is used ONLY for the
    `nondet_rejected`/`balance_rejected` buckets, which exist nowhere else
    (rejected samples are never written to `samples.jsonl`).
    """
    corpus_dir = Path(corpus_dir)
    samples_path = corpus_dir / "samples.jsonl"
    functions = _read_jsonl(corpus_dir / "functions.jsonl")
    samples = _read_jsonl(samples_path)
    gen_stats = json.loads((corpus_dir / "gen_stats.json").read_text())

    accepted_functions = len(functions)
    accepted_samples = len(samples)

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
    nondet_rejected = gen_stats.get("nondet_rejected", 0)
    truncated_rejected = gen_stats.get("truncated_rejected", 0)
    balance_rejected = gen_stats.get("balance_rejected", 0)
    gen_accepted_samples = gen_stats.get("accepted_samples", 0)
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
        "samples_sha256": hashlib.sha256(samples_path.read_bytes()).hexdigest(),
    }

    (corpus_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
