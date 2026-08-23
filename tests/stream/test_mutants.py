"""Tests for mutant enumeration, application, and content-hash keying."""

import random

from crucible.stream import mutants
from crucible.stream.mutants import Mutant, apply_spec, enumerate_specs, make_mutant, sample_specs
from crucible.stream.units import Unit, sha256_text

SRC = "def f(a, b):\n    if a < b:\n        return a + b\n    return a - b\n"
UNIT = Unit("HumanEval/0", "unit_humaneval_0", "f", SRC, "", "", sha256_text(SRC), 1, 0, ())


def test_enumerate_add_sub_and_lt_gte():
    specs = enumerate_specs(SRC, ["ReplaceBinaryOperator_Add_Sub", "ReplaceComparisonOperator_Lt_GtE", "StatementDeletion"])
    ops = [(s.operator, s.occurrence, s.family) for s in specs]
    assert ("ReplaceBinaryOperator_Add_Sub", 0, "ARITH") in ops
    assert ("ReplaceComparisonOperator_Lt_GtE", 0, "CMP") in ops
    # Only ``simple_stmt``s inside an indented suite are deletable: ``return a + b``
    # (inside the ``if`` suite) and ``return a - b`` (inside ``f``'s suite). The
    # ``if`` statement itself is compound, so it is not a candidate.
    assert sum(1 for s in specs if s.family == "SDL") == 2


def test_enumerate_skips_operators_that_carry_no_family():
    # An A2-excluded name and an unknown one both carry no family, so they have no class
    # to belong to and cannot enter the stream. They must be skipped before instantiation:
    # VariableReplacer raises TypeError (it needs constructor args our call cannot supply)
    # and an unknown name raises KeyError out of the plugin registry.
    assert enumerate_specs(SRC, ["VariableReplacer", "NotAnOperator"]) == []


def test_apply_and_make_mutant_key_is_content_hash():
    spec = enumerate_specs(SRC, ["ReplaceBinaryOperator_Add_Sub"])[0]
    assert apply_spec(SRC, spec) == SRC.replace("a + b", "a - b")
    m = make_mutant(UNIT, spec)
    assert isinstance(m, Mutant) and m.key == sha256_text(UNIT.src_hash + "\n" + m.diff)
    assert m.diff.startswith("--- a/unit_humaneval_0.py\n+++ b/unit_humaneval_0.py\n")
    assert Mutant.from_dict(m.to_dict()) == m


def test_make_mutant_returns_none_when_unchanged_or_invalid():
    # ``Mul_Div`` has no '*' to match in SRC, so it contributes no positions at all.
    specs = enumerate_specs(SRC, ["ReplaceBinaryOperator_Mul_Div"])
    assert specs == []


def test_make_mutant_returns_none_when_the_mutation_changes_nothing():
    # ExceptionReplacer rewrites the caught exception to cosmic-ray's sentinel name, so on
    # a source that already names it the "mutation" is byte-identical. An empty diff would
    # key a task that is not a bug, so this must not become a Mutant.
    src = "def h(x):\n    try:\n        return x()\n    except CosmicRayTestingException:\n        return 0\n"
    unit = Unit("HumanEval/1", "unit_humaneval_1", "h", src, "", "", sha256_text(src), 1, 0, ())
    spec = enumerate_specs(src, ["ExceptionReplacer"])[0]
    assert spec.family == "EXC"
    assert apply_spec(src, spec) == src
    assert make_mutant(unit, spec) is None


def test_make_mutant_returns_none_when_the_result_does_not_compile(monkeypatch):
    # No mapped operator is known to emit invalid syntax, so the failure is injected: a
    # module that will not compile fails every test for the wrong reason and would score
    # as a kill the agent never earned.
    spec = enumerate_specs(SRC, ["ReplaceBinaryOperator_Add_Sub"])[0]
    monkeypatch.setattr(mutants, "apply_spec", lambda src, spec: "def f(a, b:\n    return a\n")
    assert make_mutant(UNIT, spec) is None


def test_sample_specs_caps_per_family_and_is_seeded():
    specs = enumerate_specs(SRC, ["ReplaceBinaryOperator_Add_Sub", "ReplaceBinaryOperator_Sub_Add",
                                  "ReplaceComparisonOperator_Lt_GtE", "ReplaceComparisonOperator_Lt_Gt", "StatementDeletion"])
    s1 = sample_specs(specs, per_family=1, rng=random.Random(7))
    s2 = sample_specs(specs, per_family=1, rng=random.Random(7))
    assert s1 == s2 and len(s1) == 3 and len({s.family for s in s1}) == 3
    # Equal output alone is a weak detector here (each family pool holds only two specs,
    # so an *unseeded* generator would still agree by chance about one run in eight).
    # Consuming the caller's rng is the property that actually makes the stream
    # reproducible from the seed in the run record, so assert its state advanced.
    rng = random.Random(7)
    before = rng.getstate()
    assert sample_specs(specs, per_family=1, rng=rng) == s1
    assert rng.getstate() != before
