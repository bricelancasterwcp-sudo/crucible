"""RED/GREEN tests for the parallel re-harvest pass (round-3 CRITICAL fix):
`crucible.latent.reharvest`.

Every real `harvest()` call is stubbed here -- no subprocess ever runs in
this file, same house rule as every other gen*/reharvest test in this
package (`crucible.latent.reharvest.harvest` is the patch target, since
`_reharvest_one` calls the bare name `harvest` looked up in THIS module's
own globals, not `gen`'s). What's under test is `reharvest.py`'s OWN
orchestration: archive-then-fresh `samples.jsonl` handling (never deleting
the corrupt evidence, never overwriting it on a second run), conservation
across buckets under a real `ThreadPoolExecutor`, the balance guard rebuilt
from zero and evaluated correctly under genuine thread concurrency, and the
final sort-rewrite that makes `corpus.build_manifest`'s raw-byte
`samples_sha256` reproducible regardless of which thread's write landed
first.
"""
from __future__ import annotations

import json
import time

import pytest

from crucible.latent import gen, reharvest
from crucible.latent.harvest import HarvestError
from tests.latent.test_gen import _result
from tests.latent.test_gen_minority import _fn_record, _write_jsonl


def _read_samples(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# --- archive-then-fresh -------------------------------------------------------


def test_reharvest_archives_existing_samples_before_fresh_write(tmp_path, monkeypatch):
    corrupt_rows = [
        {"fn_id": "deadbeefdeadbeef", "function_src": "def f(a):\n    return a\n",
         "args": "(1,)", "outcome": "return", "return_repr": "999", "snapshots": []},
        {"fn_id": "deadbeefdeadbeef", "function_src": "def f(a):\n    return a\n",
         "args": "(2,)", "outcome": "return", "return_repr": "999", "snapshots": []},
    ]
    _write_jsonl(tmp_path / "samples.jsonl", corrupt_rows)
    corrupt_bytes_before = (tmp_path / "samples.jsonl").read_bytes()

    fn = _fn_record("def f(a):\n    return a\n", ["(1,)", "(2,)"])
    _write_jsonl(tmp_path / "functions.jsonl", [fn])
    monkeypatch.setattr(reharvest, "harvest",
                        lambda src, args, workdir: _result(outcome="return", return_repr=args))

    stats = reharvest.reharvest_samples(tmp_path, jobs=1)

    corrupt_path = tmp_path / "samples.jsonl.replay-corrupt"
    assert corrupt_path.exists()
    assert corrupt_path.read_bytes() == corrupt_bytes_before   # byte-identical evidence

    fresh = _read_samples(tmp_path / "samples.jsonl")
    assert {row["return_repr"] for row in fresh} == {"(1,)", "(2,)"}   # NOT the old "999"
    assert stats["accepted_samples"] == 2


def test_reharvest_never_overwrites_an_existing_corrupt_archive(tmp_path, monkeypatch):
    """A SECOND reharvest run over a corpus_dir must not clobber the
    original corrupt evidence with whatever the FIRST reharvest wrote --
    that would destroy the very evidence this archiving step exists to
    keep. Simulated directly: seed `samples.jsonl.replay-corrupt` with a
    sentinel and `samples.jsonl` with different (non-corrupt-looking)
    content, as if a prior reharvest had already run once."""
    sentinel_rows = [{"fn_id": "sentinel", "function_src": "def f():\n    return 0\n",
                      "args": "()", "outcome": "return", "return_repr": "0", "snapshots": []}]
    _write_jsonl(tmp_path / "samples.jsonl.replay-corrupt", sentinel_rows)
    sentinel_bytes = (tmp_path / "samples.jsonl.replay-corrupt").read_bytes()

    _write_jsonl(tmp_path / "samples.jsonl", [
        {"fn_id": "y", "function_src": "def f():\n    return 2\n",
         "args": "()", "outcome": "return", "return_repr": "2", "snapshots": []},
    ])

    fn = _fn_record("def f():\n    return 3\n", ["()"])
    _write_jsonl(tmp_path / "functions.jsonl", [fn])
    monkeypatch.setattr(reharvest, "harvest",
                        lambda src, args, workdir: _result(outcome="return", return_repr="3"))

    reharvest.reharvest_samples(tmp_path, jobs=1)

    assert (tmp_path / "samples.jsonl.replay-corrupt").read_bytes() == sentinel_bytes


def test_reharvest_starts_fresh_with_no_prior_samples_file(tmp_path, monkeypatch):
    fn = _fn_record("def f():\n    return 1\n", ["()"])
    _write_jsonl(tmp_path / "functions.jsonl", [fn])
    monkeypatch.setattr(reharvest, "harvest",
                        lambda src, args, workdir: _result(outcome="return", return_repr="1"))

    stats = reharvest.reharvest_samples(tmp_path, jobs=1)

    assert not (tmp_path / "samples.jsonl.replay-corrupt").exists()
    assert stats["accepted_samples"] == 1
    assert _read_samples(tmp_path / "samples.jsonl")[0]["return_repr"] == "1"


def test_reharvest_never_modifies_functions_jsonl(tmp_path, monkeypatch):
    fn = _fn_record("def f():\n    return 1\n", ["()"])
    _write_jsonl(tmp_path / "functions.jsonl", [fn])
    before = (tmp_path / "functions.jsonl").read_bytes()
    monkeypatch.setattr(reharvest, "harvest",
                        lambda src, args, workdir: _result(outcome="return", return_repr="1"))

    reharvest.reharvest_samples(tmp_path, jobs=1)

    assert (tmp_path / "functions.jsonl").read_bytes() == before


# --- conservation --------------------------------------------------------------


def test_reharvest_conservation_across_buckets(tmp_path, monkeypatch):
    fn = _fn_record("def f(a):\n    return a\n", ["(1,)", "(2,)", "(3,)", "(4,)", "(5,)"])
    _write_jsonl(tmp_path / "functions.jsonl", [fn])

    def stub_harvest(src, args, workdir):
        if args == "(1,)":
            raise HarvestError("boom")
        if args == "(2,)":
            return _result(truncated=True)
        if args == "(3,)":
            return _result(deterministic=False)
        return _result(outcome="return", return_repr=args)

    monkeypatch.setattr(reharvest, "harvest", stub_harvest)
    stats = reharvest.reharvest_samples(tmp_path, jobs=2)

    assert stats["candidates"] == 5
    assert stats["harvest_error"] == 1
    assert stats["truncated_rejected"] == 1
    assert stats["nondet_rejected"] == 1
    assert stats["accepted_samples"] == 2   # (4,) and (5,)
    total = (stats["harvest_error"] + stats["truncated_rejected"] + stats["nondet_rejected"]
            + stats["balance_rejected"] + stats["accepted_samples"])
    assert total == stats["candidates"]
    assert stats["complete"] is True


# --- parallel determinism of RESULTS -------------------------------------------


def test_reharvest_jobs_1_vs_jobs_4_produce_the_same_accepted_set(tmp_path, monkeypatch):
    """Same underlying (deterministic, guard-inert) harvest results, run
    with jobs=1 and jobs=4 in two SEPARATE corpus dirs: the ACCEPTED SET
    must match. Row ORDER is explicitly NOT compared -- `samples.jsonl`'s
    on-disk order under jobs>1 depends on thread completion timing (which
    is why `reharvest_samples` sorts the file at the end,
    `_rewrite_samples_sorted`); this test sorts before comparing regardless,
    documenting that non-determinism plainly rather than leaning on the
    sort-rewrite alone to hide it (that rewrite is checked separately,
    below). Kept well under `config.BALANCE_GUARD_MIN_SAMPLES` (10 items)
    so the balance guard never fires regardless of processing order --
    job-count-dependent GUARD decisions are a separate property, covered by
    the concurrency test further down, not this one.
    """
    literals = [f"({i},)" for i in range(10)]
    fn = _fn_record("def f(a):\n    return a\n", literals)

    def make_dir(name):
        d = tmp_path / name
        d.mkdir()
        _write_jsonl(d / "functions.jsonl", [fn])
        return d

    dir1, dir4 = make_dir("jobs1"), make_dir("jobs4")
    monkeypatch.setattr(reharvest, "harvest",
                        lambda src, args, workdir: _result(outcome="return", return_repr=args))

    stats1 = reharvest.reharvest_samples(dir1, jobs=1)
    stats4 = reharvest.reharvest_samples(dir4, jobs=4)

    set1 = sorted(_read_samples(dir1 / "samples.jsonl"), key=lambda r: r["args"])
    set4 = sorted(_read_samples(dir4 / "samples.jsonl"), key=lambda r: r["args"])
    assert set1 == set4
    assert stats1["accepted_samples"] == stats4["accepted_samples"] == 10


def test_reharvest_final_samples_file_is_sorted_and_hash_reproducible(tmp_path, monkeypatch):
    """The final-sort-rewrite property directly: `samples.jsonl` on disk is
    line-sorted after a run, and two separate jobs=4 runs over identical
    input produce BYTE-IDENTICAL output -- the property
    `corpus.build_manifest`'s raw-byte `samples_sha256` needs."""
    literals = [f"({i},)" for i in range(8)]
    fn = _fn_record("def f(a):\n    return a\n", literals)

    def make_dir(name):
        d = tmp_path / name
        d.mkdir()
        _write_jsonl(d / "functions.jsonl", [fn])
        return d

    dir_a, dir_b = make_dir("a"), make_dir("b")
    monkeypatch.setattr(reharvest, "harvest",
                        lambda src, args, workdir: _result(outcome="return", return_repr=args))

    reharvest.reharvest_samples(dir_a, jobs=4)
    reharvest.reharvest_samples(dir_b, jobs=4)

    lines_a = (dir_a / "samples.jsonl").read_text().splitlines()
    assert lines_a == sorted(lines_a)
    assert (dir_a / "samples.jsonl").read_bytes() == (dir_b / "samples.jsonl").read_bytes()


# --- balance guard under concurrency -------------------------------------------


def test_reharvest_balance_guard_holds_conservation_under_real_concurrency(tmp_path, monkeypatch):
    """A stub harvest that sleeps briefly forces genuine thread overlap
    (jobs=8, 40 items, EVERY item the same binary label) against the
    balance guard's running `class_counts`, under real concurrent access to
    shared state -- and an artificial delay INSIDE `gen._balance_guard_
    rejects` itself (patched, not the real function) to widen the
    check-then-act race window the shared `threading.Lock` exists to close.

    With every item sharing one label, the correct outcome is exact and
    ORDER-INDEPENDENT, not just "some guard activity happened": the first
    `BALANCE_GUARD_MIN_SAMPLES` (5) items are accepted unconditionally
    (`total < BALANCE_GUARD_MIN_SAMPLES` short-circuits the guard before it
    ever compares a fraction), and because no OTHER label is ever accepted,
    `class_counts` freezes at `{1: 5}` from then on -- so `balance ==
    5/5 == 1.0 > SKEW_LIMIT (0.5)` for every one of the remaining 35 items,
    every time, regardless of which thread checks it or in what order.
    `accepted_samples == 5` and `balance_rejected == 35` are therefore the
    ONLY correct result under ANY interleaving of a correctly-locked
    implementation -- a lost update (two threads both reading `total < 5`
    before either increments `class_counts`) would push `accepted_samples`
    above 5, which this test would catch directly, not just infer from
    conservation (conservation across the five buckets holds regardless of
    locking correctness -- ThreadPoolExecutor loses no futures either way --
    so it alone cannot prove the lock does anything).
    """
    monkeypatch.setattr(gen, "BALANCE_GUARD_MIN_SAMPLES", 5)
    monkeypatch.setattr(gen, "SKEW_LIMIT", 0.5)
    real_guard = gen._balance_guard_rejects

    def slow_guard(label, class_counts):
        # Read the decision from class_counts AS IT IS RIGHT NOW, THEN
        # sleep -- widening the window between that read and this thread's
        # own class_counts[label] += 1 (which happens AFTER this call
        # returns) so other threads can run their own read-decide-update
        # sequence in between. This is exactly the window `lock` in
        # `_reharvest_one` exists to make atomic; sleeping BEFORE the read
        # instead would just let every other thread finish first and hand
        # this thread an up-to-date value, which proves nothing.
        decision = real_guard(label, class_counts)
        time.sleep(0.002)
        return decision

    monkeypatch.setattr(gen, "_balance_guard_rejects", slow_guard)

    literals = [f"({i},)" for i in range(40)]
    fn = _fn_record("def f(a):\n    return a\n", literals)
    _write_jsonl(tmp_path / "functions.jsonl", [fn])

    def stub_harvest(src, args, workdir):
        time.sleep(0.002)
        return _result(outcome="return", return_repr=args)   # every item the SAME label

    monkeypatch.setattr(reharvest, "harvest", stub_harvest)
    stats = reharvest.reharvest_samples(tmp_path, jobs=8)

    total = (stats["harvest_error"] + stats["truncated_rejected"] + stats["nondet_rejected"]
            + stats["balance_rejected"] + stats["accepted_samples"])
    assert total == stats["candidates"] == 40
    assert stats["complete"] is True
    assert stats["accepted_samples"] == 5      # exact, order-independent (see docstring)
    assert stats["balance_rejected"] == 35

    written = _read_samples(tmp_path / "samples.jsonl")
    assert len(written) == stats["accepted_samples"]
    assert len({row["args"] for row in written}) == len(written)   # no duplicate writes


# --- stats file discipline ------------------------------------------------------


def test_reharvest_periodic_and_final_stats_flush(tmp_path, monkeypatch):
    monkeypatch.setattr(reharvest, "REHARVEST_STATS_FLUSH_INTERVAL", 3)
    literals = [f"({i},)" for i in range(10)]
    fn = _fn_record("def f(a):\n    return a\n", literals)
    _write_jsonl(tmp_path / "functions.jsonl", [fn])
    monkeypatch.setattr(reharvest, "harvest",
                        lambda src, args, workdir: _result(outcome="return", return_repr=args))

    calls = []
    orig_write = reharvest._write_reharvest_stats

    def spy(path, stats):
        calls.append(dict(stats))
        orig_write(path, stats)

    monkeypatch.setattr(reharvest, "_write_reharvest_stats", spy)

    stats = reharvest.reharvest_samples(tmp_path, jobs=1)

    # jobs=1 processes work items one at a time in submission order, so this
    # is exact, not a lower bound: periodic flushes at i=3,6,9 plus the one
    # unconditional final write in the `finally` block == 4 total calls.
    assert len(calls) == 4
    assert calls[-1]["complete"] is True
    final_on_disk = json.loads((tmp_path / "reharvest_stats.json").read_text())
    assert final_on_disk == stats


def test_reharvest_sort_rewrite_is_atomic_and_gates_complete(tmp_path, monkeypatch):
    """Round-3 follow-up fix: the completion sort-rewrite writes a temp
    file and swaps it in via `os.replace` -- so a failure DURING that
    replace (disk full, permissions, killed mid-rename) must leave
    `samples.jsonl` exactly as harvesting wrote it (never a half-written
    reordering) and `complete` False in `reharvest_stats.json` (never
    marked True until AFTER the rewrite actually succeeds).
    """
    literals = [f"({i},)" for i in range(5)]
    fn = _fn_record("def f(a):\n    return a\n", literals)
    _write_jsonl(tmp_path / "functions.jsonl", [fn])
    monkeypatch.setattr(reharvest, "harvest",
                        lambda src, args, workdir: _result(outcome="return", return_repr=args))

    pre_rewrite_bytes_holder = {}

    def failing_replace(src_path, dst_path):
        # Capture what samples.jsonl looked like the instant BEFORE the
        # (about to fail) replace -- this is what must still be on disk
        # afterward.
        pre_rewrite_bytes_holder["bytes"] = dst_path.read_bytes()
        raise OSError("simulated replace failure")

    monkeypatch.setattr(reharvest.os, "replace", failing_replace)

    with pytest.raises(OSError, match="simulated replace failure"):
        reharvest.reharvest_samples(tmp_path, jobs=1)

    assert (tmp_path / "samples.jsonl").read_bytes() == pre_rewrite_bytes_holder["bytes"]
    rows = _read_samples(tmp_path / "samples.jsonl")
    assert len(rows) == 5   # every accepted sample is still there, just unsorted

    stats_on_disk = json.loads((tmp_path / "reharvest_stats.json").read_text())
    assert stats_on_disk["complete"] is False
    assert stats_on_disk["accepted_samples"] == 5


def test_reharvest_partial_stats_preserved_on_crash(tmp_path, monkeypatch):
    """An unexpected bug (NOT `HarvestError`/`OSError`) must propagate --
    never be silently swallowed as just another `harvest_error` -- and must
    still leave `reharvest_stats.json` on disk with `complete: False`."""
    literals = ["(1,)", "(2,)", "(3,)"]
    fn = _fn_record("def f(a):\n    return a\n", literals)
    _write_jsonl(tmp_path / "functions.jsonl", [fn])

    def stub_harvest(src, args, workdir):
        if args == "(2,)":
            raise RuntimeError("unexpected bug, not a HarvestError")
        return _result(outcome="return", return_repr=args)

    monkeypatch.setattr(reharvest, "harvest", stub_harvest)

    with pytest.raises(RuntimeError, match="unexpected bug"):
        reharvest.reharvest_samples(tmp_path, jobs=1)

    stats_on_disk = json.loads((tmp_path / "reharvest_stats.json").read_text())
    assert stats_on_disk["complete"] is False
