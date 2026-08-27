"""RED/GREEN tests for the deterministic adversarial battery (spec S12
amendment, controller ruling round 2): `crucible.latent.gen_battery`.

This REPLACES the LLM-proposed minority pass (`generate_minority_inputs`,
`tests/latent/test_gen_minority.py`), which live-fire-failed -- 93%
parse_fail, 0 accepted_minority after 200/5000 functions, evidence at
`runs/blite-corpus/minority_stats.llm-attempt.json`. No proposer exists in
this module at all: every candidate comes from a FIXED enumeration over the
function's own arity. Same house rule as the other gen* test files:
`crucible.latent.gen.harvest` is always monkeypatched with an in-process
stub -- no subprocess ever runs here.
"""
from __future__ import annotations

import copy
import json

import pytest

from crucible.latent import config, gen, gen_battery
from crucible.latent.harvest import HarvestError
from tests.latent.test_gen import _result
from tests.latent.test_gen_minority import _fn_record, _write_jsonl

ARITY_1_SRC = "def f(a):\n    return a\n"
ARITY_2_SRC = "def f(a, b):\n    return a + b\n"


# --- _function_arity ---------------------------------------------------------


def test_function_arity_counts_positional_params():
    assert gen_battery._function_arity(ARITY_1_SRC) == 1
    assert gen_battery._function_arity(ARITY_2_SRC) == 2
    assert gen_battery._function_arity("def f():\n    return 1\n") == 0


# --- _battery_candidates: exact enumeration + order (mutation pins) --------


def test_battery_candidates_arity_1_exact_order():
    expected = [
        (0,), (-1,), (10**9,), (None,), ("",), ("x",), ([],), ([0],), ({},), (True,),
    ]
    assert gen_battery._battery_candidates(1) == expected


def test_battery_candidates_arity_2_exact_order():
    expected = [
        # homogeneous: one (v, v) per BATTERY_VALUES entry, in order
        (0, 0), (-1, -1), (10**9, 10**9), (None, None), ("", ""), ("x", "x"),
        ([], []), ([0], [0]), ({}, {}), (True, True),
        # heterogeneous: position-major (i=0 then i=1), probe-minor (None, [], "")
        (None, 1), ([], 1), ("", 1),
        (1, None), (1, []), (1, ""),
    ]
    assert gen_battery._battery_candidates(2) == expected


def test_battery_candidates_arity_0_collapses_to_the_empty_tuple():
    result = gen_battery._battery_candidates(0)
    assert result == [()] * len(config.BATTERY_VALUES)


def test_battery_candidates_match_config_battery_values_length():
    # sanity: exactly one homogeneous probe per configured battery value,
    # regardless of the module under test hardcoding its own copy of them.
    assert len(gen_battery._battery_candidates(1)) == len(config.BATTERY_VALUES)


# --- cap honored post-dedup (mutation pin: cap value / dedup-before-cap) ---


def test_generate_battery_inputs_caps_selection_after_dedup(tmp_path, monkeypatch):
    """arity-2 produces 16 raw candidates (10 homogeneous + 6 heterogeneous),
    none of which duplicate each other or any pre-existing input -- only the
    first BATTERY_MAX_PER_FN (12) may ever reach harvest(), in enumeration
    order, and the trailing 4 must never be inspected at all."""
    fn = _fn_record(ARITY_2_SRC, [])
    _write_jsonl(tmp_path / "functions.jsonl", [fn])

    harvest_calls: list[str] = []

    def spy_harvest(src, args, workdir):
        harvest_calls.append(args)
        return _result()

    monkeypatch.setattr(gen, "harvest", spy_harvest)

    stats = gen_battery.generate_battery_inputs(tmp_path, seed=0)

    expected_selected = gen_battery._battery_candidates(2)[: config.BATTERY_MAX_PER_FN]
    assert harvest_calls == [repr(c) for c in expected_selected]
    assert len(harvest_calls) == config.BATTERY_MAX_PER_FN == 12
    assert stats["candidates"] == 16
    assert stats["duplicate_input"] == 0   # nothing skipped -- the cap alone bounds this
    assert stats["accepted_samples"] == 12


def test_generate_battery_inputs_dedup_before_cap_lets_a_later_candidate_through(tmp_path, monkeypatch):
    """If the very first raw candidate is ALREADY a known input, it must be
    skipped (counted as duplicate_input) and the cap window slides to admit
    one more later candidate -- proving the cap is applied AFTER dedup, not
    against the raw list's first 12 positions regardless of duplicates."""
    first_candidate_literal = repr(gen_battery._battery_candidates(2)[0])   # "(0, 0)"
    fn = _fn_record(ARITY_2_SRC, [first_candidate_literal])
    _write_jsonl(tmp_path / "functions.jsonl", [fn])

    harvest_calls: list[str] = []

    def spy_harvest(src, args, workdir):
        harvest_calls.append(args)
        return _result()

    monkeypatch.setattr(gen, "harvest", spy_harvest)
    stats = gen_battery.generate_battery_inputs(tmp_path, seed=0)

    all_candidates = gen_battery._battery_candidates(2)
    # skip index 0 (duplicate), take the next 12 -> indices 1..12
    expected_selected = all_candidates[1:1 + config.BATTERY_MAX_PER_FN]
    assert harvest_calls == [repr(c) for c in expected_selected]
    assert stats["duplicate_input"] == 1


# --- exception outcomes count as accepted_minority --------------------------


def test_generate_battery_inputs_counts_exception_outcomes_as_minority(tmp_path, monkeypatch):
    fn = _fn_record(ARITY_1_SRC, [])
    _write_jsonl(tmp_path / "functions.jsonl", [fn])
    monkeypatch.setattr(
        gen, "harvest",
        lambda src, args, workdir: _result(outcome="exception:ZeroDivisionError", return_repr=None),
    )

    stats = gen_battery.generate_battery_inputs(tmp_path, seed=0)

    n = len(config.BATTERY_VALUES)
    assert stats["candidates"] == n
    assert stats["accepted_samples"] == n
    assert stats["accepted_minority"] == n

    samples = [json.loads(line) for line in (tmp_path / "samples.jsonl").read_text().splitlines()]
    assert len(samples) == n
    assert all(s["outcome"] == "exception:ZeroDivisionError" for s in samples)
    assert all(s["fn_id"] == fn["fn_id"] for s in samples)


def test_generate_battery_inputs_does_not_count_clean_return_as_minority(tmp_path, monkeypatch):
    fn = _fn_record(ARITY_1_SRC, [])
    _write_jsonl(tmp_path / "functions.jsonl", [fn])
    monkeypatch.setattr(gen, "harvest", lambda src, args, workdir: _result(outcome="return"))

    stats = gen_battery.generate_battery_inputs(tmp_path, seed=0)

    assert stats["accepted_samples"] == len(config.BATTERY_VALUES)
    assert stats["accepted_minority"] == 0


# --- balance guard reads the REAL existing balance (reuse pattern) ---------


def test_generate_battery_inputs_balance_guard_reads_real_existing_balance(tmp_path, monkeypatch):
    """Same pattern as generate_minority_inputs's own guard-init test: seed
    samples.jsonl with pre-existing majority-class samples for an unrelated
    fn_id, lower the guard's threshold/limit, and show the guard rejects this
    pass's candidates on the strength of that pre-existing file alone."""
    monkeypatch.setattr(gen, "BALANCE_GUARD_MIN_SAMPLES", 2)
    monkeypatch.setattr(gen, "SKEW_LIMIT", 0.5)

    other_src = "def f(x):\n    return x\n"
    existing_samples = [
        {"fn_id": "deadbeefdeadbeef", "function_src": other_src, "args": "(1,)",
         "outcome": "return", "return_repr": "1", "snapshots": []},
        {"fn_id": "deadbeefdeadbeef", "function_src": other_src, "args": "(2,)",
         "outcome": "return", "return_repr": "2", "snapshots": []},
    ]
    _write_jsonl(tmp_path / "samples.jsonl", existing_samples)

    fn = _fn_record(ARITY_1_SRC, [])
    _write_jsonl(tmp_path / "functions.jsonl", [fn])
    monkeypatch.setattr(gen, "harvest", lambda src, args, workdir: _result(outcome="return"))

    stats = gen_battery.generate_battery_inputs(tmp_path, seed=0)

    assert stats["balance_rejected"] == len(config.BATTERY_VALUES)
    assert stats["accepted_samples"] == 0
    # append-only: still exactly the 2 seeded lines
    assert len((tmp_path / "samples.jsonl").read_text().splitlines()) == 2


# --- conservation: every candidate examined lands in exactly one bucket ----


def test_generate_battery_inputs_conservation_across_buckets(tmp_path, monkeypatch):
    fn = _fn_record(ARITY_1_SRC, [])   # arity 1 -> exactly 10 raw candidates, cap never binds
    _write_jsonl(tmp_path / "functions.jsonl", [fn])

    outcomes = iter([
        HarvestError("boom"),
        _result(truncated=True),
        _result(deterministic=False),
        _result(outcome="exception:ValueError"),
        _result(outcome="exception:ValueError"),
        _result(outcome="exception:ValueError"),
        _result(outcome="return"),
        _result(outcome="return"),
        _result(outcome="return"),
        _result(outcome="return"),
    ])

    def stub_harvest(src, args, workdir):
        item = next(outcomes)
        if isinstance(item, BaseException):
            raise item
        return item

    monkeypatch.setattr(gen, "harvest", stub_harvest)

    stats = gen_battery.generate_battery_inputs(tmp_path, seed=0)

    total = (
        stats["duplicate_input"] + stats["harvest_error"] + stats["nondet_rejected"]
        + stats["truncated_rejected"] + stats["balance_rejected"] + stats["accepted_samples"]
    )
    assert stats["candidates"] == len(config.BATTERY_VALUES) == 10
    assert total == stats["candidates"]
    assert stats["duplicate_input"] == 0
    assert stats["harvest_error"] == 1
    assert stats["truncated_rejected"] == 1
    assert stats["nondet_rejected"] == 1
    assert stats["accepted_samples"] == 7
    assert stats["accepted_minority"] == 3


# --- determinism: seed is accepted but unused -------------------------------


def test_generate_battery_inputs_is_deterministic_regardless_of_seed(tmp_path_factory, monkeypatch):
    fn = _fn_record(ARITY_2_SRC, [])
    dir1 = tmp_path_factory.mktemp("battery_det_1")
    dir2 = tmp_path_factory.mktemp("battery_det_2")
    _write_jsonl(dir1 / "functions.jsonl", [fn])
    _write_jsonl(dir2 / "functions.jsonl", [fn])

    def stub_harvest(src, args, workdir):
        # purely a function of the literal's own text -- no hashing/RNG
        outcome = "exception:ValueError" if "None" in args else "return"
        return _result(outcome=outcome)

    monkeypatch.setattr(gen, "harvest", stub_harvest)

    stats1 = gen_battery.generate_battery_inputs(dir1, seed=0)
    stats2 = gen_battery.generate_battery_inputs(dir2, seed=999999)

    args1 = [json.loads(line)["args"] for line in (dir1 / "samples.jsonl").read_text().splitlines()]
    args2 = [json.loads(line)["args"] for line in (dir2 / "samples.jsonl").read_text().splitlines()]
    assert args1 == args2

    stats1_no_seed = {k: v for k, v in stats1.items() if k != "seed"}
    stats2_no_seed = {k: v for k, v in stats2.items() if k != "seed"}
    assert stats1_no_seed == stats2_no_seed
    assert stats1["seed"] == 0
    assert stats2["seed"] == 999999


# --- stats file + periodic flush --------------------------------------------


def test_generate_battery_inputs_writes_battery_stats_json(tmp_path, monkeypatch):
    fn = _fn_record(ARITY_1_SRC, [])
    _write_jsonl(tmp_path / "functions.jsonl", [fn])
    monkeypatch.setattr(gen, "harvest", lambda src, args, workdir: _result())

    stats = gen_battery.generate_battery_inputs(tmp_path, seed=0)

    on_disk = json.loads((tmp_path / "battery_stats.json").read_text())
    assert on_disk == stats
    assert on_disk["complete"] is True


def test_generate_battery_inputs_flushes_stats_periodically(tmp_path, monkeypatch):
    monkeypatch.setattr(gen_battery, "BATTERY_STATS_FLUSH_INTERVAL", 2)
    monkeypatch.setattr(gen, "harvest", lambda src, args, workdir: _result())
    functions = [_fn_record(f"def f(a):\n    return a + {i}\n", []) for i in range(3)]
    _write_jsonl(tmp_path / "functions.jsonl", functions)

    calls: list[dict] = []
    real_write = gen_battery._write_battery_stats

    def spy_write(path, stats):
        calls.append(copy.deepcopy(stats))
        real_write(path, stats)

    monkeypatch.setattr(gen_battery, "_write_battery_stats", spy_write)

    final_stats = gen_battery.generate_battery_inputs(tmp_path, seed=0)

    assert len(calls) == 2   # periodic @2, unconditional final @3
    assert calls[0]["functions_processed"] == 2
    assert calls[0]["complete"] is False
    assert calls[1]["functions_processed"] == 3
    assert calls[1]["complete"] is True

    on_disk = json.loads((tmp_path / "battery_stats.json").read_text())
    assert on_disk == final_stats


def test_generate_battery_inputs_writes_partial_stats_with_complete_false_on_unhandled_error(tmp_path, monkeypatch):
    fn = _fn_record(ARITY_1_SRC, [])
    _write_jsonl(tmp_path / "functions.jsonl", [fn])

    def stub_harvest(src, args, workdir):
        raise ValueError("unexpected bug, not HarvestError/OSError")

    monkeypatch.setattr(gen, "harvest", stub_harvest)

    import pytest
    with pytest.raises(ValueError):
        gen_battery.generate_battery_inputs(tmp_path, seed=0)

    on_disk = json.loads((tmp_path / "battery_stats.json").read_text())
    assert on_disk["complete"] is False
    assert on_disk["accepted_samples"] == 0


# --- append-only guarantees; never touches sibling files --------------------


def test_generate_battery_inputs_never_touches_other_stats_files(tmp_path, monkeypatch):
    fn = _fn_record(ARITY_1_SRC, [])
    _write_jsonl(tmp_path / "functions.jsonl", [fn])
    before = (tmp_path / "functions.jsonl").read_text()
    monkeypatch.setattr(gen, "harvest", lambda src, args, workdir: _result())

    gen_battery.generate_battery_inputs(tmp_path, seed=0)

    assert (tmp_path / "functions.jsonl").read_text() == before
    assert not (tmp_path / "gen_stats.json").exists()
    assert not (tmp_path / "minority_stats.json").exists()
    assert (tmp_path / "battery_stats.json").exists()


def test_generate_battery_inputs_appends_without_truncating_existing_samples(tmp_path, monkeypatch):
    sentinel = {"fn_id": "sentinelfnid0000", "function_src": "def f(a):\n    return a\n",
                "args": "(1,)", "outcome": "return", "return_repr": "1", "snapshots": []}
    _write_jsonl(tmp_path / "samples.jsonl", [sentinel])

    fn = _fn_record(ARITY_1_SRC, [])
    _write_jsonl(tmp_path / "functions.jsonl", [fn])
    monkeypatch.setattr(gen, "harvest", lambda src, args, workdir: _result(outcome="exception:ValueError"))

    gen_battery.generate_battery_inputs(tmp_path, seed=0)

    lines = (tmp_path / "samples.jsonl").read_text().splitlines()
    assert len(lines) == 1 + len(config.BATTERY_VALUES)
    assert json.loads(lines[0]) == sentinel   # original line untouched, still first


def test_generate_battery_inputs_dedups_against_samples_jsonl_from_a_prior_partial_run(tmp_path, monkeypatch):
    fn = _fn_record(ARITY_1_SRC, [])
    _write_jsonl(tmp_path / "functions.jsonl", [fn])
    prior = {"fn_id": fn["fn_id"], "function_src": ARITY_1_SRC, "args": "(0,)",
             "outcome": "return", "return_repr": "0", "snapshots": []}
    _write_jsonl(tmp_path / "samples.jsonl", [prior])

    harvest_calls: list[str] = []

    def spy_harvest(src, args, workdir):
        harvest_calls.append(args)
        return _result()

    monkeypatch.setattr(gen, "harvest", spy_harvest)
    stats = gen_battery.generate_battery_inputs(tmp_path, seed=0)

    assert "(0,)" not in harvest_calls
    assert stats["duplicate_input"] == 1


# --- ordering guard: refuse a corpus mid-repair (round-3 fix follow-up) -----


def test_generate_battery_inputs_refuses_when_replay_corrupt_marker_has_no_reharvest_stats(tmp_path, monkeypatch):
    """samples.jsonl.replay-corrupt existing with NO reharvest_stats.json
    at all means a reharvest never even reached its final stats write --
    refuse rather than enrich a corpus that may still be mid-repair."""
    fn = _fn_record(ARITY_1_SRC, [])
    _write_jsonl(tmp_path / "functions.jsonl", [fn])
    (tmp_path / "samples.jsonl.replay-corrupt").write_text("")
    monkeypatch.setattr(gen, "harvest", lambda src, args, workdir: _result(outcome="return"))

    with pytest.raises(RuntimeError, match="run reharvest first"):
        gen_battery.generate_battery_inputs(tmp_path, seed=0)


def test_generate_battery_inputs_refuses_when_reharvest_stats_present_but_incomplete(tmp_path, monkeypatch):
    fn = _fn_record(ARITY_1_SRC, [])
    _write_jsonl(tmp_path / "functions.jsonl", [fn])
    (tmp_path / "samples.jsonl.replay-corrupt").write_text("")
    (tmp_path / "reharvest_stats.json").write_text(json.dumps({"complete": False}))
    monkeypatch.setattr(gen, "harvest", lambda src, args, workdir: _result(outcome="return"))

    with pytest.raises(RuntimeError, match="run reharvest first"):
        gen_battery.generate_battery_inputs(tmp_path, seed=0)


def test_generate_battery_inputs_runs_when_reharvest_is_complete(tmp_path, monkeypatch):
    """The guard is scoped to the ordering hazard only -- once reharvest
    genuinely finished, battery must run normally on top of it."""
    fn = _fn_record(ARITY_1_SRC, [])
    _write_jsonl(tmp_path / "functions.jsonl", [fn])
    (tmp_path / "samples.jsonl.replay-corrupt").write_text("")
    (tmp_path / "reharvest_stats.json").write_text(json.dumps({"complete": True}))
    monkeypatch.setattr(gen, "harvest", lambda src, args, workdir: _result(outcome="return"))

    stats = gen_battery.generate_battery_inputs(tmp_path, seed=0)

    assert stats["complete"] is True


def test_generate_battery_inputs_runs_when_no_replay_corrupt_marker_at_all(tmp_path, monkeypatch):
    """A corpus that was NEVER reharvested (no .replay-corrupt file at all)
    is explicitly NOT refused by this guard -- deliberately narrow scope,
    see _refuse_if_reharvest_incomplete's own docstring."""
    fn = _fn_record(ARITY_1_SRC, [])
    _write_jsonl(tmp_path / "functions.jsonl", [fn])
    monkeypatch.setattr(gen, "harvest", lambda src, args, workdir: _result(outcome="return"))

    stats = gen_battery.generate_battery_inputs(tmp_path, seed=0)

    assert stats["complete"] is True
