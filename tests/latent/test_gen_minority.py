"""RED/GREEN tests for the minority-input second pass (spec S12 pre-lock
amendment): `crucible.latent.gen.generate_minority_inputs`.

Kept in its OWN file, deliberately, mirroring the amendment's own provenance
requirement -- the first pass (`generate_corpus`, `tests/latent/test_gen.py`)
and this second pass are two separate write paths into the same corpus
directory, and keeping their tests apart makes that separation obvious at a
glance rather than interleaved through one already-680-line file.

Same house rules as `test_gen.py`: no subprocess ever runs here --
`crucible.latent.gen.harvest` is always monkeypatched with an in-process
stub, and every rejection bucket is asserted, never assumed.
"""
from __future__ import annotations

import copy
import json

from crucible.latent import gen
from crucible.latent.harvest import HarvestError
from crucible.run.types import Candidate
from tests.latent.test_gen import FakeProposer, _result

FN_SRC = "def f(a, b):\n    return a + b\n"


def _fn_record(function_src: str, args_literals: list[str]) -> dict:
    return {
        "fn_id": gen._fn_id(function_src),
        "function_src": function_src,
        "args_literals": args_literals,
        "samples_kept": len(args_literals),
    }


def _write_jsonl(path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


# --- MINORITY_PROMPT --------------------------------------------------------


def test_minority_prompt_is_a_deterministic_static_string():
    assert isinstance(gen.MINORITY_PROMPT, str)
    assert "INPUTS:" in gen.MINORITY_PROMPT
    assert "CRASH" in gen.MINORITY_PROMPT
    for banned in ("datetime.now", "time.time", "random.", "uuid."):
        assert banned not in gen.MINORITY_PROMPT
    assert gen.MINORITY_PROMPT == gen.MINORITY_PROMPT


# --- _extract_inputs_list (refactored out of parse_candidate) ---------------


def test_extract_inputs_list_parses_a_bare_reply_with_no_function_source():
    # THE reason this helper exists separately from parse_candidate: a
    # minority reply need not repeat the function source at all -- the
    # function is already known, only new inputs are being asked for.
    normalized = gen._extract_inputs_list("INPUTS: [(1, 0), (0, 0)]\n")
    assert normalized == [(1, 0), (0, 0)]


def test_extract_inputs_list_returns_none_when_marker_missing():
    assert gen._extract_inputs_list("no marker anywhere in this text\n") is None


def test_extract_inputs_list_normalizes_non_tuple_entries():
    assert gen._extract_inputs_list("INPUTS: [1, 2]\n") == [(1,), (2,)]


# --- generate_minority_inputs: acceptance + minority counting ---------------


def test_generate_minority_inputs_accepts_and_counts_minority_outcome(tmp_path, monkeypatch):
    fn = _fn_record(FN_SRC, ["(1, 2)", "(3, 4)"])
    _write_jsonl(tmp_path / "functions.jsonl", [fn])

    monkeypatch.setattr(
        gen, "harvest",
        lambda src, args, workdir: _result(outcome="exception:ZeroDivisionError", return_repr=None),
    )
    proposer = FakeProposer(["INPUTS: [(0, 0)]\n"])

    stats = gen.generate_minority_inputs(proposer, tmp_path, seed=0)

    assert stats["accepted_samples"] == 1
    assert stats["accepted_minority"] == 1   # outcome != clean-return
    assert stats["complete"] is True

    samples = [json.loads(line) for line in (tmp_path / "samples.jsonl").read_text().splitlines()]
    assert len(samples) == 1
    assert samples[0]["fn_id"] == fn["fn_id"]
    assert samples[0]["args"] == "(0, 0)"
    assert samples[0]["outcome"] == "exception:ZeroDivisionError"


def test_generate_minority_inputs_does_not_count_a_clean_return_as_minority(tmp_path, monkeypatch):
    fn = _fn_record(FN_SRC, ["(1, 2)"])
    _write_jsonl(tmp_path / "functions.jsonl", [fn])
    monkeypatch.setattr(gen, "harvest", lambda src, args, workdir: _result(outcome="return"))
    proposer = FakeProposer(["INPUTS: [(9, 9)]\n"])

    stats = gen.generate_minority_inputs(proposer, tmp_path, seed=0)

    assert stats["accepted_samples"] == 1
    assert stats["accepted_minority"] == 0


# --- balance guard: MUST read the REAL current balance (mutation pin) -------


def test_generate_minority_inputs_balance_guard_reads_real_existing_balance(tmp_path, monkeypatch):
    """Seed samples.jsonl with 2 pre-existing MAJORITY ('return') samples for an
    unrelated function BEFORE this pass ever runs. With BALANCE_GUARD_MIN_SAMPLES
    lowered to 2 and SKEW_LIMIT to 0.5, the guard must see total=2 (not 0) and
    reject a brand-new, non-duplicate majority-class candidate outright.

    MUTATION PIN: if the guard's class_counts were seeded from an empty start
    instead of scanning the existing file, total would read 0 (< 2), the guard
    would never fire, and this candidate would be wrongly accepted -- this test
    fails exactly that way if the seed-from-file logic is dropped.
    """
    monkeypatch.setattr(gen, "BALANCE_GUARD_MIN_SAMPLES", 2)
    monkeypatch.setattr(gen, "SKEW_LIMIT", 0.5)

    other_fn_src = "def f(a):\n    return a\n"
    existing_samples = [
        {"fn_id": "deadbeefdeadbeef", "function_src": other_fn_src, "args": "(1,)",
         "outcome": "return", "return_repr": "1", "snapshots": []},
        {"fn_id": "deadbeefdeadbeef", "function_src": other_fn_src, "args": "(2,)",
         "outcome": "return", "return_repr": "2", "snapshots": []},
    ]
    _write_jsonl(tmp_path / "samples.jsonl", existing_samples)

    fn = _fn_record(FN_SRC, ["(1, 2)"])
    _write_jsonl(tmp_path / "functions.jsonl", [fn])

    monkeypatch.setattr(gen, "harvest", lambda src, args, workdir: _result(outcome="return"))
    proposer = FakeProposer(["INPUTS: [(9, 9)]\n"])   # not a duplicate of (1, 2)

    stats = gen.generate_minority_inputs(proposer, tmp_path, seed=0)

    assert stats["balance_rejected"] == 1
    assert stats["accepted_samples"] == 0
    # append-only: still exactly the 2 seeded lines, nothing added, nothing lost
    assert len((tmp_path / "samples.jsonl").read_text().splitlines()) == 2


def test_generate_minority_inputs_still_accepts_minority_class_past_the_skew(tmp_path, monkeypatch):
    """Companion sanity: with the same seeded skew as above, a MINORITY-class
    ('exception') candidate is still accepted -- the guard rejects the majority
    class only, never the minority class this whole pass exists to enrich."""
    monkeypatch.setattr(gen, "BALANCE_GUARD_MIN_SAMPLES", 2)
    monkeypatch.setattr(gen, "SKEW_LIMIT", 0.5)

    other_fn_src = "def f(a):\n    return a\n"
    existing_samples = [
        {"fn_id": "deadbeefdeadbeef", "function_src": other_fn_src, "args": "(1,)",
         "outcome": "return", "return_repr": "1", "snapshots": []},
        {"fn_id": "deadbeefdeadbeef", "function_src": other_fn_src, "args": "(2,)",
         "outcome": "return", "return_repr": "2", "snapshots": []},
    ]
    _write_jsonl(tmp_path / "samples.jsonl", existing_samples)

    fn = _fn_record(FN_SRC, ["(1, 2)"])
    _write_jsonl(tmp_path / "functions.jsonl", [fn])

    monkeypatch.setattr(gen, "harvest", lambda src, args, workdir: _result(outcome="exception:ValueError"))
    proposer = FakeProposer(["INPUTS: [(9, 9)]\n"])

    stats = gen.generate_minority_inputs(proposer, tmp_path, seed=0)

    assert stats["balance_rejected"] == 0
    assert stats["accepted_samples"] == 1
    assert stats["accepted_minority"] == 1


# --- duplicate-input skip (mutation pin) ------------------------------------


def test_generate_minority_inputs_skips_a_duplicate_input(tmp_path, monkeypatch):
    """(1, 2) is already one of this function's args_literals -- proposing it
    again must be SKIPPED, never re-harvested.

    MUTATION PIN: if the duplicate-input skip were dropped, harvest_calls would
    be non-empty and accepted_samples would be 1 instead of 0.
    """
    fn = _fn_record(FN_SRC, ["(1, 2)"])
    _write_jsonl(tmp_path / "functions.jsonl", [fn])

    harvest_calls: list[str] = []

    def spy_harvest(src, args, workdir):
        harvest_calls.append(args)
        return _result(outcome="return")

    monkeypatch.setattr(gen, "harvest", spy_harvest)
    proposer = FakeProposer(["INPUTS: [(1, 2)]\n"])

    stats = gen.generate_minority_inputs(proposer, tmp_path, seed=0)

    assert stats["duplicate_input"] == 1
    assert stats["accepted_samples"] == 0
    assert harvest_calls == []
    assert (tmp_path / "samples.jsonl").read_text() == ""


def test_generate_minority_inputs_dedups_against_samples_jsonl_from_a_prior_partial_run(tmp_path, monkeypatch):
    """A prior (interrupted) minority-pass attempt already appended a sample for
    this fn_id/args pair to samples.jsonl WITHOUT functions.jsonl ever
    reflecting it (args_literals there only ever holds the FIRST pass's
    original inputs) -- e.g. the process crashed before minority_stats.json was
    ever written, so a second attempt ran over the same corpus_dir. That prior
    sample must still be treated as a duplicate, not re-harvested and written
    twice -- closing the gap the launcher's one-pass refusal (keyed off
    minority_stats.json's existence) cannot cover on its own.
    """
    fn = _fn_record(FN_SRC, ["(1, 2)"])
    _write_jsonl(tmp_path / "functions.jsonl", [fn])
    prior = {"fn_id": fn["fn_id"], "function_src": FN_SRC, "args": "(9, 9)",
             "outcome": "exception:ValueError", "return_repr": None, "snapshots": []}
    _write_jsonl(tmp_path / "samples.jsonl", [prior])

    harvest_calls: list[str] = []

    def spy_harvest(src, args, workdir):
        harvest_calls.append(args)
        return _result()

    monkeypatch.setattr(gen, "harvest", spy_harvest)
    proposer = FakeProposer(["INPUTS: [(9, 9)]\n"])

    stats = gen.generate_minority_inputs(proposer, tmp_path, seed=0)

    assert stats["duplicate_input"] == 1
    assert harvest_calls == []
    lines = (tmp_path / "samples.jsonl").read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == prior


# --- conservation: every parsed input lands in exactly one bucket ----------


def test_generate_minority_inputs_conservation_across_buckets(tmp_path, monkeypatch):
    src = "def f(a):\n    return a\n"
    fn = _fn_record(src, ["(5,)"])   # (5,) is the pre-existing duplicate below
    _write_jsonl(tmp_path / "functions.jsonl", [fn])

    outcomes = iter([
        HarvestError("boom"),
        _result(truncated=True),
        _result(deterministic=False),
        _result(outcome="return"),
    ])

    def stub_harvest(src_, args, workdir):
        item = next(outcomes)
        if isinstance(item, BaseException):
            raise item
        return item

    monkeypatch.setattr(gen, "harvest", stub_harvest)
    # (5,) -> duplicate; (1e400,) -> invalid_literal (repr "inf" doesn't round-trip
    # through ast.literal_eval); the remaining 4 reach harvest, one per outcome above.
    reply = "INPUTS: [(5,), (1e400,), (10,), (20,), (30,), (40,)]\n"
    proposer = FakeProposer([reply])

    stats = gen.generate_minority_inputs(proposer, tmp_path, seed=0)

    total = (
        stats["invalid_literal"] + stats["duplicate_input"] + stats["harvest_error"]
        + stats["nondet_rejected"] + stats["truncated_rejected"] + stats["balance_rejected"]
        + stats["accepted_samples"]
    )
    assert total == 6
    assert stats["duplicate_input"] == 1
    assert stats["invalid_literal"] == 1
    assert stats["harvest_error"] == 1
    assert stats["truncated_rejected"] == 1
    assert stats["nondet_rejected"] == 1
    assert stats["accepted_samples"] == 1


# --- per-call seed contract --------------------------------------------------


def test_generate_minority_inputs_derives_a_fresh_seed_per_call(tmp_path, monkeypatch):
    """`call_seed = seed + index`, `index` starting at 0 for the FIRST function in
    functions.jsonl's file order -- same contract as generate_corpus's per-call
    seed, applied here per FUNCTION instead of per generate() batch."""
    monkeypatch.setattr(gen, "harvest", lambda src, args, workdir: _result())
    functions = [_fn_record(f"def f(a):\n    return a + {i}\n", ["(1,)"]) for i in range(3)]
    _write_jsonl(tmp_path / "functions.jsonl", functions)

    seen_seeds: list[int] = []

    class SeedSpyProposer:
        model = "spy"

        def generate(self, prompt, *, n, seed, max_tokens=1024, temperature=0.7):
            seen_seeds.append(seed)
            return [Candidate("INPUTS: [(2,)]\n", None, 1.0) for _ in range(n)]

    gen.generate_minority_inputs(SeedSpyProposer(), tmp_path, seed=100)

    assert seen_seeds == [100, 101, 102]


def test_generate_minority_inputs_calls_generate_with_n_1(tmp_path, monkeypatch):
    monkeypatch.setattr(gen, "harvest", lambda src, args, workdir: _result())
    fn = _fn_record(FN_SRC, ["(1, 2)"])
    _write_jsonl(tmp_path / "functions.jsonl", [fn])
    proposer = FakeProposer(["INPUTS: [(9, 9)]\n"])

    gen.generate_minority_inputs(proposer, tmp_path, seed=0)

    assert len(proposer.calls) == 1
    assert proposer.calls[0]["n"] == 1


# --- stats file: separate from gen_stats.json, complete=True on success ----


def test_generate_minority_inputs_writes_minority_stats_json(tmp_path, monkeypatch):
    fn = _fn_record(FN_SRC, ["(1, 2)"])
    _write_jsonl(tmp_path / "functions.jsonl", [fn])
    monkeypatch.setattr(gen, "harvest", lambda src, args, workdir: _result())
    proposer = FakeProposer(["INPUTS: [(9, 9)]\n"])

    stats = gen.generate_minority_inputs(proposer, tmp_path, seed=0)

    on_disk = json.loads((tmp_path / "minority_stats.json").read_text())
    assert on_disk == stats
    assert on_disk["complete"] is True
    assert not (tmp_path / "gen_stats.json").exists()   # first pass's file untouched


def test_generate_minority_inputs_flushes_stats_periodically(tmp_path, monkeypatch):
    monkeypatch.setattr(gen, "harvest", lambda src, args, workdir: _result())
    monkeypatch.setattr(gen, "MINORITY_STATS_FLUSH_INTERVAL", 2)
    functions = [_fn_record(f"def f(a):\n    return a + {i}\n", ["(1,)"]) for i in range(3)]
    _write_jsonl(tmp_path / "functions.jsonl", functions)
    proposer = FakeProposer(["INPUTS: [(2,)]\n"])

    calls: list[dict] = []
    real_write = gen._write_minority_stats

    def spy_write(path, stats):
        calls.append(copy.deepcopy(stats))
        real_write(path, stats)

    monkeypatch.setattr(gen, "_write_minority_stats", spy_write)

    final_stats = gen.generate_minority_inputs(proposer, tmp_path, seed=0)

    assert len(calls) == 2   # periodic @2, unconditional final @3
    assert calls[0]["functions_processed"] == 2
    assert calls[0]["complete"] is False
    assert calls[1]["functions_processed"] == 3
    assert calls[1]["complete"] is True

    on_disk = json.loads((tmp_path / "minority_stats.json").read_text())
    assert on_disk == final_stats


def test_generate_minority_inputs_counts_parse_failure(tmp_path, monkeypatch):
    fn = _fn_record(FN_SRC, ["(1, 2)"])
    _write_jsonl(tmp_path / "functions.jsonl", [fn])
    monkeypatch.setattr(gen, "harvest", lambda src, args, workdir: _result())
    proposer = FakeProposer(["sorry, I cannot help with that\n"])

    stats = gen.generate_minority_inputs(proposer, tmp_path, seed=0)

    assert stats["parse_fail"] == 1
    assert stats["functions_processed"] == 1
    assert stats["generate_calls"] == 1
    assert stats["accepted_samples"] == 0


def test_generate_minority_inputs_writes_partial_stats_with_complete_false_on_unhandled_error(tmp_path, monkeypatch):
    fn = _fn_record(FN_SRC, ["(1, 2)"])
    _write_jsonl(tmp_path / "functions.jsonl", [fn])

    def stub_harvest(src, args, workdir):
        raise ValueError("unexpected bug, not HarvestError/OSError")

    monkeypatch.setattr(gen, "harvest", stub_harvest)
    proposer = FakeProposer(["INPUTS: [(9, 9)]\n"])

    import pytest
    with pytest.raises(ValueError):
        gen.generate_minority_inputs(proposer, tmp_path, seed=0)

    on_disk = json.loads((tmp_path / "minority_stats.json").read_text())
    assert on_disk["complete"] is False
    assert on_disk["accepted_samples"] == 0


# --- append-only guarantees (self-review: this pass must never REDUCE data) -


def test_generate_minority_inputs_does_not_modify_functions_jsonl(tmp_path, monkeypatch):
    fn = _fn_record(FN_SRC, ["(1, 2)"])
    _write_jsonl(tmp_path / "functions.jsonl", [fn])
    before = (tmp_path / "functions.jsonl").read_text()

    monkeypatch.setattr(gen, "harvest", lambda src, args, workdir: _result())
    proposer = FakeProposer(["INPUTS: [(9, 9)]\n"])
    gen.generate_minority_inputs(proposer, tmp_path, seed=0)

    assert (tmp_path / "functions.jsonl").read_text() == before


def test_generate_minority_inputs_appends_without_truncating_existing_samples(tmp_path, monkeypatch):
    sentinel = {"fn_id": "sentinelfnid0000", "function_src": "def f(a):\n    return a\n",
                "args": "(1,)", "outcome": "return", "return_repr": "1", "snapshots": []}
    _write_jsonl(tmp_path / "samples.jsonl", [sentinel])

    fn = _fn_record(FN_SRC, ["(1, 2)"])
    _write_jsonl(tmp_path / "functions.jsonl", [fn])

    monkeypatch.setattr(gen, "harvest", lambda src, args, workdir: _result(outcome="exception:ValueError"))
    proposer = FakeProposer(["INPUTS: [(9, 9)]\n"])
    gen.generate_minority_inputs(proposer, tmp_path, seed=0)

    lines = (tmp_path / "samples.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == sentinel   # original line untouched, still first
