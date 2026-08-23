import time

import pytest
from crucible.sandbox.runner import _classify, run_tests

MOD = "def add(a, b):\n    return a + b\n"
TESTS = ("import pytest\nfrom unit_x import add as candidate\n"
         "def test_v0():\n    assert candidate(1, 2) == 3\n"
         "def test_v1():\n    assert candidate(2, 2) == 4\n")


def test_passing_module():
    r = run_tests("unit_x", MOD, TESTS)
    assert r.infra_error is None and set(r.passed) == {"test_v0", "test_v1"} and r.all_passed


def test_mutant_is_killed():
    r = run_tests("unit_x", MOD.replace("a + b", "a - b"), TESTS)
    assert r.killed and set(r.failed) == {"test_v0", "test_v1"} and not r.passed


def test_subset_runs_only_named_tests():
    r = run_tests("unit_x", MOD, TESTS, subset=["test_v1"])
    assert r.passed == ("test_v1",) and not r.failed


def test_hang_is_timed_out_not_infra():
    r = run_tests("unit_x", "def add(a, b):\n    while True: pass\n", TESTS,
                  per_test_timeout_s=1.0, wall_cap_s=20.0)
    assert r.infra_error is None and set(r.timed_out) == {"test_v0", "test_v1"} and r.killed


def test_syntax_error_module_is_collection_error_not_infra():
    r = run_tests("unit_x", "def add(a, b)\n    return a + b\n", TESTS)
    assert r.infra_error is None and r.errored == ("__collection__",) and r.killed


def test_broken_test_file_is_infra():
    r = run_tests("unit_x", MOD, "this is not python\n")
    assert r.infra_error is not None and not r.killed and r.wall_s > 0


def test_no_tests_collected_is_infra():
    r = run_tests("unit_x", MOD, "from unit_x import add\n")
    assert r.infra_error is not None and not r.killed


def test_suite_hang_past_wall_cap_is_suite_timeout_not_infra():
    # pytest never returns (the unit hangs at import), so no junit is written: the wall-cap
    # kill must still read as a failure, not as a missing-junit infra error.
    r = run_tests("unit_x", "import time\ntime.sleep(120)\n", TESTS,
                  per_test_timeout_s=60.0, wall_cap_s=5.0)
    assert r.infra_error is None and r.timed_out == ("__suite__",) and r.killed


BAD_PARAMETRIZE = ("import pytest\nfrom unit_x import add as candidate\n"
                   "@pytest.mark.parametrize('a,b,c', 1)\n"
                   "def test_v0(a, b, c):\n    assert candidate(a, b) == c\n")

INTERRUPTING = ("import pytest\nfrom unit_x import add as candidate\n"
                "def test_v0():\n    assert candidate(1, 2) == 3\n"
                "def test_v1():\n    pytest.exit('bail out')\n")


def test_test_file_that_fails_during_collection_is_infra():
    # Imports fine, breaks when pytest evaluates the parametrize argvalues: our bug,
    # so it must never be charged to the unit as a collection error.
    r = run_tests("unit_x", MOD, BAD_PARAMETRIZE)
    assert r.infra_error is not None and not r.killed and r.errored == ()


def test_interrupted_run_with_partial_results_is_infra():
    # pytest.exit() mid-run: rc=2 with a junit holding one passed test. A partial run is
    # not a measurement.
    r = run_tests("unit_x", MOD, INTERRUPTING)
    assert r.infra_error is not None and not r.killed and not r.passed


def test_unparseable_junit_is_infra_not_an_exception():
    r = _classify(1, False, '<?xml version="1.0"?><testsuites><testsu', "", 0.5)
    assert r.infra_error is not None and "unparseable" in r.infra_error and not r.killed


def test_wall_s_includes_the_probe():
    t0 = time.monotonic()
    r = run_tests("unit_x", MOD, TESTS)
    elapsed = time.monotonic() - t0
    assert r.infra_error is None
    assert r.wall_s >= 0.7 * elapsed, f"wall_s={r.wall_s} omits most of the {elapsed}s run"


def test_rejects_module_name_that_collides_with_the_test_file():
    with pytest.raises(ValueError):
        run_tests("test_unit", MOD, TESTS)


def test_rejects_module_name_that_is_not_an_identifier():
    with pytest.raises(ValueError):
        run_tests("../x", MOD, TESTS)
