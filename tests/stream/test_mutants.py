"""Tests for mutant enumeration, application, and content-hash keying."""

import random
import warnings

from crucible.stream import mutants
from crucible.stream.mutants import Mutant, apply_spec, enumerate_specs, make_mutant, sample_specs
from crucible.stream.units import Unit, sha256_text

SRC = "def f(a, b):\n    if a < b:\n        return a + b\n    return a - b\n"
UNIT = Unit("HumanEval/0", "unit_humaneval_0", "f", SRC, "", "", sha256_text(SRC), 1, 0, ())

# Two '+' tokens in one expression: the fixture that tells global occurrence numbering
# apart from per-node numbering.
MULTI_SRC = "def g(a, b, c):\n    return a + b + c\n"
MULTI_UNIT = Unit("HumanEval/2", "unit_humaneval_2", "g", MULTI_SRC, "", "", sha256_text(MULTI_SRC), 1, 0, ())

# The exact diff for the Add_Sub mutant of SRC. The key is sha256 over this text, so it is
# pinned in full -- headers, hunk header, context width and direction all change the key.
EXPECTED_DIFF = (
    "--- a/unit_humaneval_0.py\n"
    "+++ b/unit_humaneval_0.py\n"
    "@@ -1,4 +1,4 @@\n"
    " def f(a, b):\n"
    "     if a < b:\n"
    "-        return a + b\n"
    "+        return a - b\n"
    "     return a - b\n"
)


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


def test_occurrences_are_numbered_globally_not_per_node():
    # Each '+' is its own node, so numbering that restarted per node would label both
    # positions 0 and make every spec apply the *first* '+' -- silently breaking the
    # span-to-content correspondence the whole record depends on.
    specs = enumerate_specs(MULTI_SRC, ["ReplaceBinaryOperator_Add_Sub"])
    assert [s.occurrence for s in specs] == [0, 1]
    assert [s.span for s in specs] == [((2, 13), (2, 14)), ((2, 17), (2, 18))]
    # Each occurrence must change the token at *its own* span: the first '+' then the second.
    assert apply_spec(MULTI_SRC, specs[0]) == "def g(a, b, c):\n    return a - b + c\n"
    assert apply_spec(MULTI_SRC, specs[1]) == "def g(a, b, c):\n    return a + b - c\n"
    assert make_mutant(MULTI_UNIT, specs[1]).mutated_src == "def g(a, b, c):\n    return a + b - c\n"


def test_apply_and_make_mutant_key_is_content_hash():
    spec = enumerate_specs(SRC, ["ReplaceBinaryOperator_Add_Sub"])[0]
    assert apply_spec(SRC, spec) == SRC.replace("a + b", "a - b")
    m = make_mutant(UNIT, spec)
    assert isinstance(m, Mutant)
    # The mutated source must be the *mutated* text, not the original passed through.
    assert m.mutated_src == apply_spec(SRC, spec) == SRC.replace("a + b", "a - b")
    assert m.diff == EXPECTED_DIFF
    assert m.diff.startswith("--- a/unit_humaneval_0.py\n+++ b/unit_humaneval_0.py\n")
    assert m.key == sha256_text(UNIT.src_hash + "\n" + m.diff)
    assert Mutant.from_dict(m.to_dict()) == m


def test_enumerate_yields_nothing_when_the_operator_matches_nothing():
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


def test_make_mutant_returns_none_when_the_result_does_not_compile():
    # Real case from HumanEval/76: replacing unary '-' with 'not' turns ``n == -1`` into
    # ``n == not 1``, which does not parse. A module that will not compile fails every
    # test for the wrong reason and would score as a kill the agent never earned.
    src = "def f(n):\n    if n == -1:\n        return 1\n    return 0\n"
    unit = Unit("HumanEval/76", "unit_humaneval_76", "f", src, "", "", sha256_text(src), 1, 0, ())
    spec = enumerate_specs(src, ["ReplaceUnaryOperator_USub_Not"])[0]
    assert apply_spec(src, spec) == "def f(n):\n    if n == not 1:\n        return 1\n    return 0\n"
    assert make_mutant(unit, spec) is None


def test_make_mutant_keeps_a_mutant_whose_source_only_warns():
    # ``x is 'a'`` compiles, but CPython emits a SyntaxWarning for it. Under ``-W error``
    # that warning becomes a SyntaxError, so an unguarded compile() would silently drop a
    # valid mutant -- and this one *is* the bug we want the agent to find.
    src = "def f(x):\n    return x == 'a'\n"
    unit = Unit("HumanEval/3", "unit_humaneval_3", "f", src, "", "", sha256_text(src), 1, 0, ())
    spec = enumerate_specs(src, ["ReplaceComparisonOperator_Eq_Is"])[0]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("error", SyntaxWarning)   # the hostile setting, made local
        m = make_mutant(unit, spec)
    assert m is not None and m.mutated_src == "def f(x):\n    return x is 'a'\n"
    assert caught == []                                  # and nothing leaked to the caller


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


def test_sample_specs_prefers_distinct_spans():
    # Six ARITH operators land on the *same* '+' token and one on the '-': spans are
    # [X, X, X, X, X, X, Y]. With per_family=2 a sampler that just took the first two of a
    # shuffled pool would pick two mutations of the same token 5 times in 7.
    same_span = ["ReplaceBinaryOperator_Add_Sub", "ReplaceBinaryOperator_Add_Mul", "ReplaceBinaryOperator_Add_Div",
                 "ReplaceBinaryOperator_Add_Mod", "ReplaceBinaryOperator_Add_Pow", "ReplaceBinaryOperator_Add_FloorDiv"]
    specs = enumerate_specs(SRC, [*same_span, "ReplaceBinaryOperator_Sub_Add"])
    assert len(specs) == 7 and len({s.span for s in specs}) == 2
    for seed in range(20):
        chosen = sample_specs(specs, per_family=2, rng=random.Random(seed))
        assert len(chosen) == 2
        assert len({s.span for s in chosen}) == 2, f"seed {seed} sampled the same span twice"


def test_component_round_trips():
    from crucible.stream.mutants import Component
    c = Component("Op", 3, ((2, 1), (2, 5)))
    assert Component.from_dict(c.to_dict()) == c
    assert c.to_dict()["span"] == [[2, 1], [2, 5]]


def test_mutant_components_round_trip_and_default():
    from crucible.stream.mutants import Component
    m = Mutant("u", "k", "Op", 0, "ARITH", ((1, 0), (1, 2)), "s", "d")   # positional, no components
    assert m.components == ()
    m2 = Mutant("u", "k", "Op", 0, "ARITH", ((1, 0), (1, 2)), "s", "d",
                components=(Component("Op", 0, ((1, 0), (1, 2))), Component("Op2", 1, ((2, 0), (2, 2)))))
    assert Mutant.from_dict(m2.to_dict()) == m2


def test_mutant_from_dict_accepts_pre_stack2_records():
    d = Mutant("u", "k", "Op", 0, "ARITH", ((1, 0), (1, 2)), "s", "d").to_dict()
    d.pop("components")                       # a record written before this field existed
    assert Mutant.from_dict(d).components == ()
