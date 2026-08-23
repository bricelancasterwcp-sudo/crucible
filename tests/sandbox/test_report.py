from crucible.sandbox.report import TestReport, parse_junit

JUNIT = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest" tests="4">
<testcase classname="test_unit" name="test_v0" time="0.001"/>
<testcase classname="test_unit" name="test_v1" time="0.001"><failure message="assert 1 == 2">x</failure></testcase>
<testcase classname="test_unit" name="test_v2" time="5.0"><failure message="Failed: Timeout &gt;5.0s">x</failure></testcase>
<testcase classname="test_unit" name="test_v3" time="0.0"><error message="boom">x</error></testcase>
</testsuite></testsuites>"""


def test_parse_junit_buckets_by_outcome():
    p, f, t, e = parse_junit(JUNIT)
    assert p == ("test_v0",) and f == ("test_v1",) and t == ("test_v2",) and e == ("test_v3",)


def test_report_flags():
    ok = TestReport(("a",), (), (), (), 0.1, None)
    assert ok.all_passed and not ok.killed
    k = TestReport((), ("a",), (), (), 0.1, None)
    assert k.killed and not k.all_passed
    infra = TestReport((), (), (), (), 0.1, "server down")
    assert not infra.killed and not infra.all_passed
    empty = TestReport((), (), (), (), 0.1, None)
    assert not empty.all_passed, "zero tests passed must not count as all_passed"


def test_report_round_trip():
    r = TestReport(("a", "b"), ("c",), ("__suite__",), (), 1.5, None)
    assert TestReport.from_dict(r.to_dict()) == r
    assert set(r.to_dict()) == {"passed", "failed", "timed_out", "errored", "wall_s", "infra_error"}
