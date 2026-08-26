"""RED/GREEN tests for corpus assembly, hash splits, and floor verdicts
(prereg §4).

`assign_split` decides test-set integrity for the whole B-lite experiment --
every test in the first section pins exactly the properties that make that
true: pure determinism, function-level grouping (no sample-level leakage
across splits), the ±3pp fraction contract, and genuine seed-sensitivity (a
mutant that drops `seed` from the hash must be caught here, not downstream).

`build_manifest` is the ONLY place floor verdicts are computed; Task 4 (ops)
reads them off `manifest.json` and never recomputes them -- so this file also
pins each floor's PASS/FAIL boundary independently (monkeypatched tighter),
and pins the nondet_rate denominator: `nondet_rejected /
(nondet_rejected + balance_rejected + accepted_samples)` -- the set of
(function, input) pairs gen.py's `_harvest_and_write_samples` actually
reached its determinism check for, per gen.py's own bucket ordering (harvest
errors and truncated pairs never reach that check; see corpus.py's
docstring).

No subprocess, no real proposer, no real sensorium trace anywhere in this
file -- synthetic corpora are built by writing `samples.jsonl` /
`functions.jsonl` / `gen_stats.json` directly in Task 2's on-disk format.
"""
from __future__ import annotations

import json

import pytest

from crucible.latent import corpus
from crucible.latent.harvest import Snapshot


# -- helpers ------------------------------------------------------------------


def _write_jsonl(path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _sample_row(fn_id: str, args: str = "()", outcome: str = "return",
                 return_repr: str | None = "1", snapshots: list[dict] | None = None) -> dict:
    return {
        "fn_id": fn_id,
        "function_src": f"def f():\n    return 1  # {fn_id}\n",
        "args": args,
        "outcome": outcome,
        "return_repr": return_repr,
        "snapshots": snapshots if snapshots is not None else [],
    }


def _fn_id_for_split(split: str, seed: int = 0, prefix: str = "fn") -> str:
    """Search synthetic fn_ids until one hashes into `split` -- avoids
    hardcoding a hash value while still exercising the real assign_split."""
    for i in range(10_000):
        candidate = f"{prefix}-{i}"
        if corpus.assign_split(candidate, seed) == split:
            return candidate
    raise AssertionError(f"no synthetic fn_id landed in split={split!r} within 10000 tries")


def _write_corpus(corpus_dir, *, functions: list[dict], samples: list[dict],
                   gen_stats: dict | None = None) -> None:
    _write_jsonl(corpus_dir / "functions.jsonl", functions)
    _write_jsonl(corpus_dir / "samples.jsonl", samples)
    stats = {
        "target_functions": len(functions),
        "seed": 0,
        "candidates": len(functions),
        "parse_fail": 0,
        "validate_fail": {},
        "nondet_rejected": 0,
        "truncated_rejected": 0,
        "balance_rejected": 0,
        "harvest_error": 0,
        "accepted_functions": len(functions),
        "accepted_samples": len(samples),
        "complete": True,
    }
    if gen_stats:
        stats.update(gen_stats)
    (corpus_dir / "gen_stats.json").write_text(json.dumps(stats))


# -- assign_split: determinism, function-level integrity, fractions, seed -----


def test_assign_split_is_deterministic():
    assert corpus.assign_split("some-fn-id", 0) == corpus.assign_split("some-fn-id", 0)


def test_assign_split_returns_only_known_split_names():
    for i in range(200):
        assert corpus.assign_split(f"fn-{i}", 0) in {"train", "val", "test"}


def test_assign_split_function_level_integrity_two_samples_same_fn_id():
    """Two Samples of the SAME function must land in the same split -- this
    is the whole point of hashing on fn_id, not on a per-sample identity."""
    fn_id = "shared-function-id"
    split_a = corpus.assign_split(fn_id, 0)
    split_b = corpus.assign_split(fn_id, 0)
    assert split_a == split_b


def test_assign_split_fractions_within_3pp_over_10k_synthetic_ids():
    """Mutation pin: a boundary mutant (e.g. the 0.8 train cutoff drifting to
    0.9) shifts the train fraction by ~10pp, far outside this ±3pp band."""
    n = 10_000
    counts = {"train": 0, "val": 0, "test": 0}
    for i in range(n):
        counts[corpus.assign_split(f"synthetic-{i}", 0)] += 1

    assert abs(counts["train"] / n - 0.8) <= 0.03
    assert abs(counts["val"] / n - 0.1) <= 0.03
    assert abs(counts["test"] / n - 0.1) <= 0.03


def test_assign_split_changes_with_seed():
    """Mutation pin: if `seed` were dropped from the hash, every id would
    land in the identical split for every seed -- 0% would differ. The real
    hash mixes seed in, so a large, non-flaky fraction differs instead."""
    n = 1000
    differ = sum(
        corpus.assign_split(f"seed-sens-{i}", 0) != corpus.assign_split(f"seed-sens-{i}", 1)
        for i in range(n)
    )
    assert differ / n > 0.2


# -- Sample deserialization ----------------------------------------------------


def test_load_split_deserializes_snapshots_as_snapshot_tuples(tmp_path):
    fn_id = _fn_id_for_split("test")
    snap = {"line": 3, "locals": [["a", "int", "1"], ["b", "str", "'x'"]]}
    _write_corpus(
        tmp_path,
        functions=[{"fn_id": fn_id, "function_src": "def f():\n    return 1\n"}],
        samples=[_sample_row(fn_id, snapshots=[snap])],
    )
    samples = corpus.load_split(tmp_path, "test")
    assert len(samples) == 1
    sample = samples[0]
    assert isinstance(sample, corpus.Sample)
    assert sample.fn_id == fn_id
    assert sample.snapshots == (Snapshot(line=3, locals=(("a", "int", "1"), ("b", "str", "'x'"))),)


# -- load_split purity + no-default-split signature pin ------------------------


def test_load_split_test_returns_only_test_split_samples(tmp_path):
    train_fn = _fn_id_for_split("train", prefix="tr")
    val_fn = _fn_id_for_split("val", prefix="va")
    test_fn = _fn_id_for_split("test", prefix="te")

    _write_corpus(
        tmp_path,
        functions=[
            {"fn_id": train_fn, "function_src": "def f():\n    return 1\n"},
            {"fn_id": val_fn, "function_src": "def f():\n    return 2\n"},
            {"fn_id": test_fn, "function_src": "def f():\n    return 3\n"},
        ],
        samples=[
            _sample_row(train_fn, args="(1,)"),
            _sample_row(train_fn, args="(2,)"),
            _sample_row(val_fn, args="(1,)"),
            _sample_row(test_fn, args="(1,)"),
            _sample_row(test_fn, args="(2,)"),
        ],
    )

    test_samples = corpus.load_split(tmp_path, "test")
    assert len(test_samples) == 2
    assert all(s.fn_id == test_fn for s in test_samples)
    # purity: no train/val fn_id leaked into the "test" result
    assert all(corpus.assign_split(s.fn_id, corpus.SPLIT_SEED) == "test" for s in test_samples)


def test_load_split_has_no_default_split_argument(tmp_path):
    """Signature pin: `split` must be required, not defaulted -- a caller
    that forgets to pass it is a bug, not a silent fallback to some split."""
    (tmp_path / "samples.jsonl").write_text("")
    (tmp_path / "functions.jsonl").write_text("")
    with pytest.raises(TypeError):
        corpus.load_split(tmp_path)  # type: ignore[call-arg]


# -- build_manifest: exact counts ----------------------------------------------


def test_build_manifest_exact_counts_and_class_balance(tmp_path):
    fn_a = "fn-a-manifest"
    fn_b = "fn-b-manifest"
    _write_corpus(
        tmp_path,
        functions=[
            {"fn_id": fn_a, "function_src": "def f():\n    return 1\n"},
            {"fn_id": fn_b, "function_src": "def f():\n    return 2\n"},
        ],
        samples=[
            _sample_row(fn_a, outcome="return"),
            _sample_row(fn_a, outcome="return"),
            _sample_row(fn_a, outcome="exception:ValueError", return_repr=None),
            _sample_row(fn_b, outcome="return"),
            _sample_row(fn_b, outcome="return"),
        ],
        gen_stats={"nondet_rejected": 2, "balance_rejected": 1, "accepted_samples": 5},
    )

    manifest = corpus.build_manifest(tmp_path)

    assert manifest["accepted_functions"] == 2
    assert manifest["accepted_samples"] == 5
    assert manifest["class_balance"]["binary"] == {"0": 1, "1": 4}
    assert manifest["class_balance"]["multiclass"] == {"return": 4, "exception:ValueError": 1}

    split_fns = manifest["split_sizes"]["functions"]
    split_samples = manifest["split_sizes"]["samples"]
    assert sum(split_fns.values()) == 2
    assert sum(split_samples.values()) == 5

    on_disk = json.loads((tmp_path / "manifest.json").read_text())
    assert on_disk == manifest


def test_build_manifest_nondet_rate_uses_the_determinism_screened_denominator(tmp_path):
    """nondet_rate = nondet_rejected / (nondet_rejected + balance_rejected +
    accepted_samples) -- NOT divided by candidates, harvest_error, or
    truncated_rejected, none of which ever reached the determinism check in
    gen.py's `_harvest_and_write_samples` (harvest_error: harvest() raised,
    no HarvestResult to check; truncated_rejected: `continue`d before the
    determinism check runs)."""
    fn_id = "fn-nondet-rate"
    _write_corpus(
        tmp_path,
        functions=[{"fn_id": fn_id, "function_src": "def f():\n    return 1\n"}],
        samples=[_sample_row(fn_id)],
        gen_stats={
            "candidates": 999,          # must NOT appear in the denominator
            "harvest_error": 50,        # must NOT appear in the denominator
            "truncated_rejected": 50,   # must NOT appear in the denominator
            "nondet_rejected": 2,
            "balance_rejected": 1,
            "accepted_samples": 5,
        },
    )
    manifest = corpus.build_manifest(tmp_path)
    assert manifest["floors"]["nondet_rate"] == pytest.approx(2 / 8)


def test_build_manifest_samples_sha256_matches_the_file(tmp_path):
    fn_id = "fn-sha"
    _write_corpus(
        tmp_path,
        functions=[{"fn_id": fn_id, "function_src": "def f():\n    return 1\n"}],
        samples=[_sample_row(fn_id)],
    )
    import hashlib
    expected = hashlib.sha256((tmp_path / "samples.jsonl").read_bytes()).hexdigest()
    manifest = corpus.build_manifest(tmp_path)
    assert manifest["samples_sha256"] == expected


# -- build_manifest: floor verdicts, one mutation pin per floor ----------------


def _tiny_corpus(tmp_path, *, binary_counts=(4, 1), nondet_rejected=2, balance_rejected=1,
                  gen_accepted_samples=5):
    """A 2-function, 5-sample corpus: 4 "return" + 1 "exception" -> balance
    4/5 = 0.8. Reused across the floor-flip tests below."""
    fn_a, fn_b = "floor-fn-a", "floor-fn-b"
    n_return, n_exc = binary_counts
    samples = [_sample_row(fn_a, outcome="return") for _ in range(n_return - 1)]
    samples.append(_sample_row(fn_b, outcome="return"))
    samples.append(_sample_row(fn_b, outcome="exception:ValueError", return_repr=None))
    # trim/pad to exactly n_exc exceptions if caller varies binary_counts
    samples = samples[: n_return + n_exc]
    _write_corpus(
        tmp_path,
        functions=[
            {"fn_id": fn_a, "function_src": "def f():\n    return 1\n"},
            {"fn_id": fn_b, "function_src": "def f():\n    return 2\n"},
        ],
        samples=samples,
        gen_stats={
            "nondet_rejected": nondet_rejected,
            "balance_rejected": balance_rejected,
            "accepted_samples": gen_accepted_samples,
        },
    )


def test_floor_functions_verdict_flips_under_a_tighter_monkeypatched_floor(tmp_path, monkeypatch):
    _tiny_corpus(tmp_path)  # 2 accepted functions

    monkeypatch.setattr(corpus, "FLOOR_FUNCTIONS", 2)
    assert corpus.build_manifest(tmp_path)["floors"]["floor_functions"] == "PASS"

    monkeypatch.setattr(corpus, "FLOOR_FUNCTIONS", 3)  # tighter: now above the corpus's count
    assert corpus.build_manifest(tmp_path)["floors"]["floor_functions"] == "FAIL"


def test_nondet_kill_verdict_flips_under_a_tighter_monkeypatched_limit(tmp_path, monkeypatch):
    _tiny_corpus(tmp_path, nondet_rejected=2, balance_rejected=1, gen_accepted_samples=5)
    # nondet_rate = 2 / (2 + 1 + 5) = 0.25

    monkeypatch.setattr(corpus, "NONDET_REJECT_KILL", 0.5)
    assert corpus.build_manifest(tmp_path)["floors"]["nondet_kill"] == "PASS"

    monkeypatch.setattr(corpus, "NONDET_REJECT_KILL", 0.1)  # tighter than the 0.25 rate
    assert corpus.build_manifest(tmp_path)["floors"]["nondet_kill"] == "FAIL"


def test_skew_ok_verdict_flips_under_a_tighter_monkeypatched_skew_limit(tmp_path, monkeypatch):
    _tiny_corpus(tmp_path)  # binary balance = 4/5 = 0.8

    monkeypatch.setattr(corpus, "SKEW_LIMIT", 0.9)
    manifest = corpus.build_manifest(tmp_path)
    assert manifest["floors"]["balance"] == pytest.approx(0.8)
    assert manifest["floors"]["skew_ok"] is True

    monkeypatch.setattr(corpus, "SKEW_LIMIT", 0.5)  # tighter than the 0.8 balance
    manifest = corpus.build_manifest(tmp_path)
    assert manifest["floors"]["skew_ok"] is False
