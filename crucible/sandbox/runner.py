"""Run one test file against one unit inside the sandbox and classify the outcome.

Classification is the whole point (ruling R7). Three things look alike from the outside
-- a hanging unit, a unit that will not import, and a test file we ourselves generated
badly -- and only the first two are measurements:

* wall-cap hang        -> ``timed_out=("__suite__",)``      (a failure)
* per-test hang        -> ``timed_out=(<test names>,)``     (a failure, via pytest-timeout)
* unit will not import -> ``errored=("__collection__",)``   (a failure)
* test file broken / no tests / no junit / pytest rc 3,4,5 -> ``infra_error`` (not scored)

Telling "the unit does not import" from "the test file is broken" is done by loading the
test file first, on its own, in a cheap sandbox run with the unit replaced by a stub: if
that import fails the fault is ours, so any collection error pytest reports afterwards can
only come from the real unit. Parsing alone is not enough -- ``this is not python`` parses
(it is a comparison of two names) and only fails when imported.
"""
from __future__ import annotations

import os
import shutil
import sys

from .exec import execute
from .report import TestReport, parse_junit

_TEST_MODULE = "test_unit"
TEST_FILE = f"{_TEST_MODULE}.py"
_JUNIT = "junit.xml"
_PROBE_SRC = f"import {_TEST_MODULE}"
# Stand-in for the unit while the test file is checked alone: a PEP 562 module-level
# __getattr__ satisfies any ``from <unit> import name`` the test file performs.
_UNIT_STUB_SRC = "def __getattr__(name):\n    return lambda *a, **k: None\n"
# Deliberately independent of the caller's wall cap: a caller with a tight budget must not
# be able to turn a slow-but-valid test file into a false infra error.
_PROBE_WALL_CAP_S = 10.0
_STDERR_TAIL = 400
_PROBE_STDERR_TAIL = 300


def run_tests(module_name: str, module_src: str, test_src: str, *, subset: list[str] | None = None,
              per_test_timeout_s: float = 5.0, wall_cap_s: float = 60.0, mem_limit_bytes: int = 4 << 30,
              python: str = sys.executable) -> TestReport:
    """Execute ``test_src`` against ``module_src`` and report what happened.

    ``subset`` selects tests by bare function name. Filenames inside the sandbox are
    chosen here (``{module_name}.py`` and ``test_unit.py``) and never taken from callers;
    ``module_name`` must therefore not be ``test_unit``.
    """
    probe = execute([python, "-c", _PROBE_SRC],
                    {TEST_FILE: test_src, f"{module_name}.py": _UNIT_STUB_SRC},
                    wall_cap_s=_PROBE_WALL_CAP_S, mem_limit_bytes=mem_limit_bytes)
    if probe.returncode != 0:
        return TestReport((), (), (), (), probe.wall_s,
                          f"test file does not load: {probe.stderr[-_PROBE_STDERR_TAIL:]}")
    targets = [f"{TEST_FILE}::{name}" for name in subset] if subset else [TEST_FILE]
    argv = [python, "-m", "pytest", *targets, "-q", "-p", "no:cacheprovider", "--tb=line",
            "-W", "ignore", f"--timeout={per_test_timeout_s}", "--timeout-method=signal",
            f"--junitxml={_JUNIT}", "-o", "junit_logging=no"]
    files = {f"{module_name}.py": module_src, TEST_FILE: test_src}
    res = execute(argv, files, wall_cap_s=wall_cap_s, mem_limit_bytes=mem_limit_bytes, keep=True)
    try:
        xml = _read_junit(res.workdir)
    finally:
        shutil.rmtree(res.workdir, ignore_errors=True)
    return _classify(res.returncode, res.timed_out, xml, res.stderr, res.wall_s)


def _read_junit(workdir: str) -> str | None:
    """The junit report pytest left behind, or None when it never wrote one."""
    path = os.path.join(workdir, _JUNIT)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _classify(rc: int | None, timed_out: bool, xml: str | None, stderr: str, wall_s: float) -> TestReport:
    """Turn a raw pytest exit into a verdict, keeping failures and infra errors apart."""
    if timed_out:
        return TestReport((), (), ("__suite__",), (), wall_s, None)
    if xml is None:
        return TestReport((), (), (), (), wall_s, f"no junit written (rc={rc}): {stderr[-_STDERR_TAIL:]}")
    passed, failed, t_out, errored = parse_junit(xml)
    if rc == 2 and not passed and not failed and not t_out:
        # pytest exit 2 = interrupted (collection errors): the module under test failed to
        # import. The probe already imported the test file standalone, so the fault is the unit's.
        return TestReport((), (), (), ("__collection__",), wall_s, None)
    if rc not in (0, 1, 2):
        return TestReport((), (), (), (), wall_s, f"pytest rc={rc}: {stderr[-_STDERR_TAIL:]}")
    if not (passed or failed or t_out or errored):
        return TestReport((), (), (), (), wall_s, "no tests collected")
    return TestReport(passed, failed, t_out, errored, wall_s, None)
