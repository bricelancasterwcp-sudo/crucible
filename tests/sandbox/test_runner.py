from crucible.sandbox.runner import run_tests

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
    assert r.infra_error is not None and not r.killed


def test_no_tests_collected_is_infra():
    r = run_tests("unit_x", MOD, "from unit_x import add\n")
    assert r.infra_error is not None and not r.killed


def test_suite_hang_past_wall_cap_is_suite_timeout_not_infra():
    # pytest never returns (the unit hangs at import), so no junit is written: the wall-cap
    # kill must still read as a failure, not as a missing-junit infra error.
    r = run_tests("unit_x", "import time\ntime.sleep(120)\n", TESTS,
                  per_test_timeout_s=60.0, wall_cap_s=5.0)
    assert r.infra_error is None and r.timed_out == ("__suite__",) and r.killed
