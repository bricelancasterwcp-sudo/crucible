import pytest
from crucible.sandbox.runner import run_tests
from crucible.stream.oracle import Expected
from crucible.stream.testgen import _round_trips, render_tests

MOD = "def add(a, b):\n    return a + b\n"
IDENT = "def ident(x):\n    return x\n"


def test_render_and_run_generated_tests():
    exp = [Expected(0, True, "3", None), Expected(1, False, None, "raised:TypeError"),
           Expected(2, True, "0.30000000000000004", None)]
    src, dropped = render_tests("unit_x", "add", [[1, 2], [None, 1], [0.1, 0.2]], exp, prefix="v", atol=0)
    assert dropped == [("v1", "raised:TypeError")]
    assert "def test_v0" in src and "def test_v2" in src and "def test_v1" not in src
    r = run_tests("unit_x", MOD, src)
    assert r.all_passed and set(r.passed) == {"test_v0", "test_v2"}
    r2 = run_tests("unit_x", MOD.replace("a + b", "a - b"), src)
    assert r2.killed


def test_render_is_deterministic_and_collectable_shape():
    # run_tests() collects the file standalone with the unit stubbed, so anything that
    # reads the unit at collection time (parametrize over unit data, fixtures, star
    # imports) would read as an infra error rather than a measurement.
    exp = [Expected(0, True, "3", None), Expected(1, True, "5", None)]
    inputs = [[1, 2], [2, 3]]
    first, _ = render_tests("unit_x", "add", inputs, exp, prefix="v", atol=0)
    second, _ = render_tests("unit_x", "add", inputs, exp, prefix="v", atol=0)
    assert first == second
    assert "parametrize" not in first and "fixture" not in first and "import *" not in first


def test_unrenderable_inputs_are_dropped_with_reason():
    # The arguments are rendered as a literal too, so an input the test file's namespace
    # cannot evaluate (inf, nan) would raise NameError there -- classified as a *failure*,
    # i.e. blamed on the unit. Drop it at the source instead (ruling R-T7-2).
    inputs = [[1.0], [float("inf")], [float("nan")]]
    exp = [Expected(0, True, "1.0", None), Expected(1, True, "1.0", None), Expected(2, True, "1.0", None)]
    src, dropped = render_tests("unit_x", "ident", inputs, exp, prefix="v", atol=0)
    assert dropped == [("v1", "no-roundtrip"), ("v2", "no-roundtrip")]
    assert "def test_v0" in src and "def test_v1" not in src and "def test_v2" not in src
    assert "inf" not in src and "nan" not in src
    r = run_tests("unit_x", IDENT, src)
    assert r.all_passed and set(r.passed) == {"test_v0"}


# --- tolerance (ruling R-T7-4): floats compare through the generated _eq, at any depth ---

NEAR = "def near(x):\n    return x + 1e-09\n"
NESTED = "def nested(x):\n    return [[x + 1e-12]] if x == 0.1 else {'a': (x + 1e-12,)}\n"
WRONG_NESTED = "def nested(x):\n    return [[x + 0.5]] if x == 0.1 else {'a': (x + 0.5,)}\n"
QUIRK = "def quirk(x):\n    return (1, 2) if x == 0 else 1.0000005\n"
NESTED_EXPECTED = [Expected(0, True, "[[0.1]]", None), Expected(1, True, "{'a': (0.2,)}", None)]
NESTED_INPUTS = [[0.1], [0.2]]


def test_scalar_float_within_atol_passes():
    src, _ = render_tests("unit_x", "near", [[0.3]], [Expected(0, True, "0.3", None)], prefix="h", atol=1e-6)
    assert "ATOL = 1e-06" in src
    assert run_tests("unit_x", NEAR, src).all_passed


def test_nested_floats_compare_with_tolerance():
    # A last-bit difference nested inside a list, tuple or dict is not a behavioural kill.
    src, _ = render_tests("unit_x", "nested", NESTED_INPUTS, NESTED_EXPECTED, prefix="h", atol=0)
    assert "ATOL = 1e-06" in src  # EvalPlus's default when the problem ships no atol
    r = run_tests("unit_x", NESTED, src)
    assert r.all_passed and set(r.passed) == {"test_h0", "test_h1"}


def test_wrong_nested_value_is_still_killed():
    src, _ = render_tests("unit_x", "nested", NESTED_INPUTS, NESTED_EXPECTED, prefix="h", atol=0)
    r = run_tests("unit_x", WRONG_NESTED, src)
    assert r.killed and set(r.failed) == {"test_h0", "test_h1"}


def test_sequence_type_and_bool_are_not_blurred():
    # A tuple is not a list even element-wise, and a bool is compared with ==, so a float
    # a hair away from 1 is not "True" just because it is within atol of it.
    exp = [Expected(0, True, "[1, 2]", None), Expected(1, True, "True", None)]
    src, _ = render_tests("unit_x", "quirk", [[0], [1]], exp, prefix="v", atol=0)
    r = run_tests("unit_x", QUIRK, src)
    assert r.killed and set(r.failed) == {"test_v0", "test_v1"}


# --- caller-side alignment (ruling R-T7-5): never render an unbacked expectation ---

def test_length_mismatch_is_a_caller_error():
    with pytest.raises(ValueError):
        render_tests("unit_x", "add", [[1], [2], [3]], [Expected(0, True, "1", None)], prefix="v", atol=0)


def test_out_of_order_expected_is_a_caller_error():
    exp = [Expected(1, True, "1", None), Expected(0, True, "2", None)]
    with pytest.raises(ValueError):
        render_tests("unit_x", "add", [[1], [2]], exp, prefix="v", atol=0)


def test_ok_without_value_repr_is_a_caller_error():
    with pytest.raises(ValueError):
        render_tests("unit_x", "add", [[1]], [Expected(0, True, None, None)], prefix="v", atol=0)


# --- literal-only evaluation (ruling R-T7-6) ---

def test_round_trips_rejects_non_literal_reprs():
    # repr(range(0, 3)) is a CALL expression: ast.literal_eval refuses it while bare eval
    # accepts it -- exactly the R-T7-3 difference. (A generated file could in fact evaluate
    # it, since builtins are in scope; literal-only is the deliberate conservative choice.)
    # Measured caveat: ast.literal_eval DOES accept "set()" -- CPython special-cases the
    # empty-set call -- so set() does not discriminate on 3.12.
    assert _round_trips([1, 2]) and _round_trips([set()])
    assert not _round_trips([range(0, 3)])
    exp = [Expected(0, True, "1", None), Expected(1, True, "1", None)]
    src, dropped = render_tests("unit_x", "ident", [[1], [range(0, 3)]], exp, prefix="v", atol=0)
    assert dropped == [("v1", "no-roundtrip")] and "def test_v1" not in src
