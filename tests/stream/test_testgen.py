from crucible.sandbox.runner import run_tests
from crucible.stream.oracle import Expected
from crucible.stream.testgen import render_tests

MOD = "def add(a, b):\n    return a + b\n"


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


def test_float_uses_approx_with_atol():
    exp = [Expected(0, True, "0.3", None)]
    src, _ = render_tests("unit_x", "add", [[0.1, 0.2]], exp, prefix="h", atol=1e-6)
    assert "pytest.approx" in src
    assert run_tests("unit_x", MOD, src).all_passed


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
