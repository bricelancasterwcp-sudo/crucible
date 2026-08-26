"""RED/GREEN tests for the corpus generator + AST validator (prereg §4).

Honesty rules apply here same as everywhere in this repo: every rejection is
COUNTED, never silently dropped (`generate_corpus`'s stats dict). No
subprocess ever runs in this file -- `crucible.latent.gen.harvest` is always
monkeypatched with an in-process stub, mirroring the house `FakeProposer`
pattern from `tests/run/test_arm.py` (a local fake here, same SHAPE:
scripted `.generate(prompt, *, n, seed, ...)` returning objects with
`.text`).
"""
from __future__ import annotations

import json

import pytest

from crucible.latent import gen
from crucible.latent.harvest import HarvestResult, Snapshot
from crucible.run.types import Candidate

# --- the local FakeProposer (house pattern, tests/run/test_arm.py) -----------


class FakeProposer:
    """In-process proposer that returns scripted candidate texts, cycling."""

    def __init__(self, texts: list[str]) -> None:
        self.model = "fake-gen-proposer"
        self._texts = texts
        self.calls: list[dict] = []

    def generate(self, prompt, *, n, seed, max_tokens=1024, temperature=0.7):
        self.calls.append({"n": n, "seed": seed})
        return [Candidate(self._texts[i % len(self._texts)], None, 1.0) for i in range(n)]


def _result(outcome="return", return_repr="1", deterministic=True, truncated=False,
            snapshots=()):
    return HarvestResult(
        outcome=outcome, return_repr=return_repr, snapshots=snapshots,
        truncated=truncated, deterministic=deterministic,
    )


CLEAN_SRC = "def f(a, b):\n    return a + b\n"
CLEAN_TEXT = f"{CLEAN_SRC}INPUTS: [(1, 2), (3, 4), (5, 6)]\n"


# --- GEN_PROMPT ----------------------------------------------------------------


def test_gen_prompt_is_a_deterministic_static_string():
    assert isinstance(gen.GEN_PROMPT, str)
    assert "INPUTS:" in gen.GEN_PROMPT
    assert "def f(" in gen.GEN_PROMPT
    # no clock/random content that would make two reads disagree
    for banned in ("datetime.now", "time.time", "random.", "uuid."):
        assert banned not in gen.GEN_PROMPT
    assert gen.GEN_PROMPT == gen.GEN_PROMPT  # same object every import, trivially stable


# --- parse_candidate -----------------------------------------------------------


def test_parse_candidate_round_trips_function_and_inputs():
    parsed = gen.parse_candidate(CLEAN_TEXT)
    assert parsed is not None
    function_src, args_literals = parsed
    assert function_src == CLEAN_SRC
    assert args_literals == ["(1, 2)", "(3, 4)", "(5, 6)"]


def test_parse_candidate_handles_a_fenced_code_block():
    text = f"Sure, here you go:\n```python\n{CLEAN_TEXT}```\n"
    parsed = gen.parse_candidate(text)
    assert parsed is not None
    function_src, args_literals = parsed
    assert function_src == CLEAN_SRC
    assert args_literals == ["(1, 2)", "(3, 4)", "(5, 6)"]


def test_parse_candidate_none_when_no_inputs_line():
    assert gen.parse_candidate(CLEAN_SRC) is None


def test_parse_candidate_none_when_inputs_not_a_literal():
    text = f"{CLEAN_SRC}INPUTS: [not, valid, python]\n"
    assert gen.parse_candidate(text) is None


def test_parse_candidate_none_when_inputs_is_empty():
    text = f"{CLEAN_SRC}INPUTS: []\n"
    assert gen.parse_candidate(text) is None


def test_parse_candidate_none_when_inputs_entries_are_not_tuples():
    text = f"{CLEAN_SRC}INPUTS: [1, 2, 3]\n"
    assert gen.parse_candidate(text) is None


def test_parse_candidate_none_when_function_body_is_blank():
    text = "INPUTS: [(1,)]\n"
    assert gen.parse_candidate(text) is None


# --- validate: one rule per test (mutation pins) --------------------------------


def test_validate_accepts_a_clean_function():
    assert gen.validate(CLEAN_SRC) is None


def test_validate_rejects_syntax_error():
    assert gen.validate("def f(a, b:\n    return a\n") == "syntax-error"


def test_validate_rejects_more_than_one_top_level_statement():
    src = "x = 1\ndef f(a):\n    return a\n"
    assert gen.validate(src) == "not-single-statement"


def test_validate_rejects_a_non_function_top_level_statement():
    assert gen.validate("x = 1\n") == "not-a-function-def"


def test_validate_rejects_a_function_not_named_f():
    assert gen.validate("def g(a):\n    return a\n") == "wrong-function-name"


def test_validate_rejects_import():
    # `import os` as its own top-level statement would be caught by the shape
    # rule first (more than one top-level statement) -- so pin the Import
    # rule with the import INSIDE the function body, where the shape rule
    # does not fire and only the Import rule can explain the rejection.
    src = "def f(a):\n    import os\n    return a\n"
    assert gen.validate(src) == "import"


def test_validate_rejects_import_from():
    src = "def f(a):\n    from os import path\n    return a\n"
    assert gen.validate(src) == "import"


def test_validate_rejects_global():
    src = "def f(a):\n    global x\n    x = a\n    return x\n"
    assert gen.validate(src) == "global-nonlocal"


def test_validate_rejects_nonlocal():
    src = "def f(a):\n    nonlocal x\n    return x\n"
    assert gen.validate(src) == "global-nonlocal"


def test_validate_rejects_dunder_attribute():
    src = "def f(a):\n    return a.__class__\n"
    assert gen.validate(src) == "dunder-attribute"


def test_validate_rejects_node_count_exceeded(monkeypatch):
    monkeypatch.setattr(gen, "MAX_AST_NODES", 5)
    assert gen.validate(CLEAN_SRC) == "node-count-exceeded"


def test_validate_accepts_under_the_default_node_cap():
    # sanity: the real (non-monkeypatched) cap does not reject a tiny function
    assert gen.validate(CLEAN_SRC) is None


_BANNED_BUILTINS = (
    "open", "exec", "eval", "compile", "__import__", "globals", "vars",
    "getattr", "setattr", "delattr", "input", "breakpoint",
)


@pytest.mark.parametrize("name", _BANNED_BUILTINS)
def test_validate_rejects_each_banned_builtin_by_bare_name(name):
    src = f"def f(a):\n    return {name}\n"
    assert gen.validate(src) == f"banned-builtin:{name}"


def test_validate_rejects_banned_builtin_via_name_level_alias():
    # THE controller-ruling pin: `g = open` is a Load of `open` (aliasing),
    # not a call -- must be caught at the NAME level, before any call ever
    # happens.
    src = "def f(a):\n    g = open\n    return a\n"
    assert gen.validate(src) == "banned-builtin:open"


def test_validate_does_not_reject_a_bare_store_of_a_banned_name():
    # Assigning TO `open` (Store context) with no later Load of it is not,
    # by itself, a load of the real builtin -- the rule fires on Load only.
    src = "def f(a):\n    open = a\n    return a\n"
    assert gen.validate(src) is None


# --- binary_label ----------------------------------------------------------------


def test_binary_label_return_is_one():
    assert gen.binary_label("return") == 1


@pytest.mark.parametrize("outcome", ["exception:ValueError", "exception:IndexError", "timeout"])
def test_binary_label_non_return_is_zero(outcome):
    assert gen.binary_label(outcome) == 0


# --- fn_id stability -------------------------------------------------------------


def test_fn_id_is_stable_and_16_hex_chars():
    fid = gen._fn_id(CLEAN_SRC)
    assert fid == gen._fn_id(CLEAN_SRC)
    assert len(fid) == 16
    int(fid, 16)  # valid hex


def test_fn_id_differs_for_different_source():
    assert gen._fn_id(CLEAN_SRC) != gen._fn_id("def f(a, b):\n    return a - b\n")


# --- generate_corpus ---------------------------------------------------------------


def test_generate_corpus_writes_accepted_samples_and_functions(tmp_path, monkeypatch):
    monkeypatch.setattr(gen, "harvest", lambda src, args, workdir: _result())
    proposer = FakeProposer([CLEAN_TEXT])
    stats = gen.generate_corpus(proposer, target_functions=1, out_dir=tmp_path, seed=0)

    assert stats["accepted_functions"] == 1
    assert stats["accepted_samples"] == 3   # 3 input tuples in CLEAN_TEXT

    samples = [json.loads(line) for line in (tmp_path / "samples.jsonl").read_text().splitlines()]
    assert len(samples) == 3
    fn_id = gen._fn_id(CLEAN_SRC)
    for row in samples:
        assert row["fn_id"] == fn_id
        assert row["function_src"] == CLEAN_SRC
        assert row["outcome"] == "return"
        assert row["return_repr"] == "1"
        assert row["snapshots"] == []
    assert sorted(row["args"] for row in samples) == ["(1, 2)", "(3, 4)", "(5, 6)"]

    functions = [json.loads(line) for line in (tmp_path / "functions.jsonl").read_text().splitlines()]
    assert len(functions) == 1
    assert functions[0]["fn_id"] == fn_id
    assert functions[0]["function_src"] == CLEAN_SRC

    on_disk_stats = json.loads((tmp_path / "gen_stats.json").read_text())
    assert on_disk_stats == stats


def test_generate_corpus_serializes_snapshots(tmp_path, monkeypatch):
    snap = Snapshot(line=2, locals=(("a", "int", "1"), ("b", "int", "2")))
    monkeypatch.setattr(gen, "harvest", lambda src, args, workdir: _result(snapshots=(snap,)))
    proposer = FakeProposer([f"{CLEAN_SRC}INPUTS: [(1, 2)]\n"])
    gen.generate_corpus(proposer, target_functions=1, out_dir=tmp_path, seed=0)

    row = json.loads((tmp_path / "samples.jsonl").read_text().splitlines()[0])
    assert row["snapshots"] == [{"line": 2, "locals": [["a", "int", "1"], ["b", "int", "2"]]}]


def test_generate_corpus_counts_parse_and_validate_failures(tmp_path, monkeypatch):
    monkeypatch.setattr(gen, "harvest", lambda src, args, workdir: _result())
    texts = [
        "not a candidate at all, no INPUTS line",              # parse-fail
        "x = 1\nINPUTS: [(1,)]\n",                              # validate-fail: not-a-function-def
        CLEAN_TEXT,                                             # accepted
    ]
    proposer = FakeProposer(texts)
    stats = gen.generate_corpus(proposer, target_functions=1, out_dir=tmp_path, seed=0)

    assert stats["parse_fail"] >= 1
    assert stats["validate_fail"].get("not-a-function-def", 0) >= 1
    assert stats["accepted_functions"] == 1


def test_generate_corpus_stats_conserve_candidates_processed(tmp_path, monkeypatch):
    """accepted_functions + every validate_fail bucket + parse_fail == candidates."""
    monkeypatch.setattr(gen, "harvest", lambda src, args, workdir: _result())
    texts = [
        "garbage, no inputs line",                              # parse-fail
        "x = 1\nINPUTS: [(1,)]\n",                              # validate-fail
        "def g(a):\n    return a\nINPUTS: [(1,)]\n",            # validate-fail (wrong name)
        CLEAN_TEXT,                                             # accepted
    ]
    proposer = FakeProposer(texts)
    stats = gen.generate_corpus(proposer, target_functions=1, out_dir=tmp_path, seed=0)

    conserved = (
        stats["parse_fail"]
        + sum(stats["validate_fail"].values())
        + stats["accepted_functions"]
    )
    assert conserved == stats["candidates"]


def test_generate_corpus_excludes_truncated_and_nondeterministic_samples(tmp_path, monkeypatch):
    outcomes = iter([
        _result(truncated=True),        # truncated -> rejected, counted
        _result(deterministic=False),   # nondet -> rejected, counted
        _result(),                      # accepted
    ])
    monkeypatch.setattr(gen, "harvest", lambda src, args, workdir: next(outcomes))
    proposer = FakeProposer([CLEAN_TEXT])
    stats = gen.generate_corpus(proposer, target_functions=1, out_dir=tmp_path, seed=0)

    assert stats["truncated_rejected"] == 1
    assert stats["nondet_rejected"] == 1
    assert stats["accepted_samples"] == 1
    # the function is still accepted even though 2 of its 3 samples were dropped
    assert stats["accepted_functions"] == 1


def test_generate_corpus_accepts_a_function_even_if_every_sample_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(gen, "harvest", lambda src, args, workdir: _result(truncated=True))
    proposer = FakeProposer([CLEAN_TEXT])
    stats = gen.generate_corpus(proposer, target_functions=1, out_dir=tmp_path, seed=0)

    assert stats["accepted_functions"] == 1
    assert stats["accepted_samples"] == 0
    assert stats["truncated_rejected"] == 3
    functions = (tmp_path / "functions.jsonl").read_text().splitlines()
    assert len(functions) == 1


# --- balance guard (spec §4, mutation pin: monkeypatched threshold) ---------------


def test_balance_guard_rejects_further_majority_class_samples(tmp_path, monkeypatch):
    """With the guard's min-sample threshold lowered to 2 and SKEW_LIMIT at 0.80:
    two accepted majority-class ("return") samples put balance at 1.0 (> 0.80),
    so a THIRD majority-class sample must be rejected and counted, while a
    minority-class ("exception") sample is still accepted -- the guard rejects
    SAMPLES, not the function they belong to."""
    monkeypatch.setattr(gen, "BALANCE_GUARD_MIN_SAMPLES", 2)
    monkeypatch.setattr(gen, "SKEW_LIMIT", 0.80)

    results = iter([
        _result(outcome="return"),               # 1: total<2, accepted, class 1 -> {1:1}
        _result(outcome="return"),                # 2: total<2 still (guard checks BEFORE), accepted -> {1:2}
        _result(outcome="return"),                # 3: total=2>=2, balance=2/2=1.0>0.80, majority(1)==label -> REJECTED
        _result(outcome="exception:ValueError"),  # 4: minority class -> accepted -> {0:1,1:2}
    ])
    monkeypatch.setattr(gen, "harvest", lambda src, args, workdir: next(results))
    text = f"{CLEAN_SRC}INPUTS: [(1, 2), (3, 4), (5, 6), (7, 8)]\n"
    proposer = FakeProposer([text])
    stats = gen.generate_corpus(proposer, target_functions=1, out_dir=tmp_path, seed=0)

    assert stats["balance_rejected"] == 1
    assert stats["accepted_samples"] == 3
    assert stats["accepted_functions"] == 1  # the function itself is still accepted


def test_balance_guard_does_not_fire_below_the_min_sample_threshold(tmp_path, monkeypatch):
    monkeypatch.setattr(gen, "BALANCE_GUARD_MIN_SAMPLES", 1000)  # real default: never fires here
    monkeypatch.setattr(gen, "harvest", lambda src, args, workdir: _result(outcome="return"))
    proposer = FakeProposer([CLEAN_TEXT])
    stats = gen.generate_corpus(proposer, target_functions=1, out_dir=tmp_path, seed=0)

    assert stats["balance_rejected"] == 0
    assert stats["accepted_samples"] == 3
