"""The verdict of one test run, and the junit-xml parsing that produces it.

``TestReport`` separates *measurements* from *instrument failures*. A unit that fails,
hangs, or refuses to import is a measurement (``killed``); an instrument that could not
produce a verdict -- a test file that does not parse, a run with no tests, a missing
junit report -- sets ``infra_error`` and is never scored (ruling R7).

Unmeasured buckets stay empty and ``infra_error`` stays ``None``; nothing here defaults
to a value that would read as a measurement.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass

_NAME_FIELDS = ("passed", "failed", "timed_out", "errored")


@dataclass(frozen=True)
class TestReport:
    """Outcome of running a test file against a unit. Test ids are bare function names."""

    # Not a pytest test class despite the name; keeps importers' output free of
    # PytestCollectionWarning.
    __test__ = False

    passed: tuple[str, ...]
    failed: tuple[str, ...]
    timed_out: tuple[str, ...]
    errored: tuple[str, ...]
    wall_s: float
    infra_error: str | None

    @property
    def all_passed(self) -> bool:
        """True only when tests actually ran and every one of them passed."""
        return (self.infra_error is None and bool(self.passed)
                and not self.failed and not self.timed_out and not self.errored)

    @property
    def killed(self) -> bool:
        """True when the suite produced a real failure: a fail, a hang, or a collection error."""
        return self.infra_error is None and bool(self.failed or self.timed_out or self.errored)

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in _NAME_FIELDS:
            d[k] = list(d[k])
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TestReport":
        return cls(tuple(d["passed"]), tuple(d["failed"]), tuple(d["timed_out"]),
                   tuple(d["errored"]), float(d["wall_s"]), d["infra_error"])


def _is_timeout(msg: str) -> bool:
    """Recognise pytest-timeout's failure message (``Failed: Timeout >5.0s``)."""
    return "Timeout" in msg and (">" in msg or "timeout" in msg.lower())


def parse_junit(xml_text: str) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Bucket junit ``<testcase>`` names into (passed, failed, timed_out, errored)."""
    root = ET.fromstring(xml_text)
    passed: list[str] = []
    failed: list[str] = []
    timed_out: list[str] = []
    errored: list[str] = []
    for tc in root.iter("testcase"):
        name = tc.get("name", "")
        fail = tc.find("failure")
        err = tc.find("error")
        if err is not None:
            errored.append(name)
        elif fail is not None:
            (timed_out if _is_timeout(fail.get("message", "")) else failed).append(name)
        elif tc.find("skipped") is not None:
            continue
        else:
            passed.append(name)
    return tuple(passed), tuple(failed), tuple(timed_out), tuple(errored)
