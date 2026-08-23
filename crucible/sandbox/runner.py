"""Run one test file against one unit inside the sandbox and classify the outcome.

Classification is the whole point (ruling R7). Three things look alike from the outside
-- a hanging unit, a unit that will not import, and a test file we ourselves generated
badly -- and only the first two are measurements:

* wall-cap hang        -> ``timed_out=("__suite__",)``      (a failure)
* per-test hang        -> ``timed_out=(<test names>,)``     (a failure, via pytest-timeout)
* unit will not import -> ``errored=("__collection__",)``   (a failure)
* test file will not collect / no junit / unparseable junit / interrupted run / pytest
  rc outside {0,1,2} -> ``infra_error`` (counted, never charged)

Telling "the unit is broken" from "our test file is broken" is done by collecting the test
file first, on its own, with the unit replaced by a stub (``_probe``): only if that clean
collection succeeds can a later failure be attributed to the unit. The probe runs pytest's
own ``--collect-only`` rather than a plain import because collection does more than import
-- it evaluates parametrize argvalues, collects classes and builds fixture definitions, and
a test file that dies in any of those is ours, not the unit's (ruling R-T3-3).
"""
from __future__ import annotations

import os
import shutil
import sys
import xml.etree.ElementTree as ET

from .exec import ExecResult, execute
from .report import TestReport, parse_junit

_TEST_MODULE = "test_unit"
TEST_FILE = f"{_TEST_MODULE}.py"
_JUNIT = "junit.xml"
# Stand-in for the unit while the test file is collected alone: a PEP 562 module-level
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

    ``subset`` selects tests by bare function name. Filenames inside the sandbox are chosen
    here (``{module_name}.py`` and ``test_unit.py``) and never taken from callers, so
    ``module_name`` is validated rather than trusted.

    ``wall_s`` covers both sandbox runs -- the collect probe and pytest -- and is therefore
    bounded by ``wall_cap_s + _PROBE_WALL_CAP_S`` (ruling R-T3-6).
    """
    _check_module_name(module_name)
    probe = _probe(python, module_name, test_src, mem_limit_bytes)
    if probe.timed_out or probe.returncode != 0:
        return TestReport((), (), (), (), probe.wall_s,
                          f"test file does not collect: {probe.stderr[-_PROBE_STDERR_TAIL:]}")
    res = _run_pytest(python, module_name, module_src, test_src, subset,
                      per_test_timeout_s, wall_cap_s, mem_limit_bytes)
    try:
        xml = _read_junit(res.workdir)
    finally:
        shutil.rmtree(res.workdir, ignore_errors=True)
    return _classify(res.returncode, res.timed_out, xml, res.stderr, probe.wall_s + res.wall_s)


def _check_module_name(module_name: str) -> None:
    """Reject a name that would collide with our own filenames or escape the workdir.

    This is a caller bug, not a property of the unit, so it raises instead of returning a
    report: a silent collision drops one of the two source files and fabricates a kill.
    """
    if not module_name.isidentifier() or module_name == _TEST_MODULE:
        raise ValueError(f"module_name must be a Python identifier other than "
                         f"{_TEST_MODULE!r}; got {module_name!r}")


def _probe(python: str, module_name: str, test_src: str, mem_limit_bytes: int) -> ExecResult:
    """Collect the test file alone, with the unit stubbed, to prove the file is sound."""
    argv = [python, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider",
            "-W", "ignore", TEST_FILE]
    files = {TEST_FILE: test_src, f"{module_name}.py": _UNIT_STUB_SRC}
    return execute(argv, files, wall_cap_s=_PROBE_WALL_CAP_S, mem_limit_bytes=mem_limit_bytes)


def _run_pytest(python: str, module_name: str, module_src: str, test_src: str,
                subset: list[str] | None, per_test_timeout_s: float, wall_cap_s: float,
                mem_limit_bytes: int) -> ExecResult:
    """The measurement itself: the real unit, the real test file, junit on disk."""
    targets = [f"{TEST_FILE}::{name}" for name in subset] if subset else [TEST_FILE]
    argv = [python, "-m", "pytest", *targets, "-q", "-p", "no:cacheprovider", "--tb=line",
            "-W", "ignore", f"--timeout={per_test_timeout_s}", "--timeout-method=signal",
            f"--junitxml={_JUNIT}", "-o", "junit_logging=no"]
    files = {f"{module_name}.py": module_src, TEST_FILE: test_src}
    return execute(argv, files, wall_cap_s=wall_cap_s, mem_limit_bytes=mem_limit_bytes, keep=True)


def _read_junit(workdir: str) -> str | None:
    """The junit report pytest left behind, or None when it never wrote one.

    Decoded with ``errors="replace"``: a run killed by RLIMIT_FSIZE or SIGKILL can leave a
    truncated file split mid-character, and the instrument must not raise on it.
    """
    path = os.path.join(workdir, _JUNIT)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def _classify(rc: int | None, timed_out: bool, xml: str | None, stderr: str, wall_s: float) -> TestReport:
    """Turn a raw pytest exit into a verdict, keeping failures and infra errors apart."""
    if timed_out:
        return TestReport((), (), ("__suite__",), (), wall_s, None)
    if xml is None:
        return TestReport((), (), (), (), wall_s, f"no junit written (rc={rc}): {stderr[-_STDERR_TAIL:]}")
    if rc not in (0, 1, 2):
        return TestReport((), (), (), (), wall_s, f"pytest rc={rc}: {stderr[-_STDERR_TAIL:]}")
    try:
        passed, failed, t_out, errored = parse_junit(xml)
    except ET.ParseError as exc:
        return TestReport((), (), (), (), wall_s, f"junit unparseable (rc={rc}): {exc}")
    if rc == 2:
        if passed or failed or t_out:
            # Interrupted part-way (e.g. pytest.exit from generated code): a partial run
            # is not a measurement, whatever it managed to record (ruling R-T3-4).
            return TestReport((), (), (), (), wall_s, "pytest interrupted (rc=2)")
        # Collection failed inside pytest. The probe already collected the test file
        # standalone, so the fault can only be the unit's.
        return TestReport((), (), (), ("__collection__",), wall_s, None)
    if not (passed or failed or t_out or errored):
        return TestReport((), (), (), (), wall_s, "no tests collected")
    return TestReport(passed, failed, t_out, errored, wall_s, None)
