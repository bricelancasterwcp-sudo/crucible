"""The verification-budget meter: K charged test executions per task (spec §4.6).

The budget's unit is an *execution that is a measurement*. An execution whose
``TestReport.infra_error`` is set produced no verdict, so it is counted (``infra``)
and reported, but never charged -- the same instrument-honesty rule that keeps such
runs out of the scores (ruling R7).

``check()`` is the gate to call *before* spending an execution; ``charge()`` records
the one that was spent. Counters start at 0 because nothing has been observed yet;
no field defaults to a value that would read as a measurement.
"""
from __future__ import annotations

from .report import TestReport


class BudgetExhausted(RuntimeError):
    """Raised by ``BudgetMeter.check`` when the task's charged executions are spent."""


class BudgetMeter:
    """Counts test executions that are measurements. Infra failures are counted separately, never charged."""

    def __init__(self, k: int = 8):
        self.k, self.charged, self.infra = k, 0, 0

    def check(self) -> None:
        if self.charged >= self.k:
            raise BudgetExhausted(f"verification budget exhausted: {self.charged}/{self.k}")

    def charge(self, report: TestReport) -> None:
        if report.infra_error is None:
            self.charged += 1
        else:
            self.infra += 1

    def remaining(self) -> int:
        return max(0, self.k - self.charged)

    def to_dict(self) -> dict:
        return {"k": self.k, "charged": self.charged, "infra": self.infra}
