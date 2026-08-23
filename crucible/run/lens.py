"""The lens: reduce a run's ``TaskRecord``s (one per arm-attempt) to the success rates the
endpoints E1/E2 read. The ceiling pilot (Task 14) takes ``build_lens(...).succ_overall``
as p0.

*Honest measurement is the whole point.* ``succ_overall`` is the mean of ``hidden_pass``
over ONLY the records that were actually scored -- the ones where ``hidden_pass is not
None``. An attempt whose infra died, or that was never measured, has ``hidden_pass=None``;
it is EXCLUDED from the success denominator, NEVER charged as a failure. Coercing that
``None`` to ``False`` here would bias every rate downward and turn "we don't know" into
"it failed" -- the exact dishonesty ``records.py`` keeps the ``None`` around to prevent.
``infra_rate`` reports those excluded attempts separately, so nothing is swept away.

The per-kind rates (``succ_phase1``/``second``/``novel``) apply the SAME measured-only
filter, restricted to ``kind == "first"/"second"/"novel"``. Every rate is 0.0 on an empty
denominator -- no ``ZeroDivisionError``. ``ArmLens`` is frozen; all its fields are
JSON-native scalars, so ``to_dict`` is a plain ``asdict`` and ``from_dict`` a plain
``cls(**d)``, exact inverses.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ArmLens:
    """One arm's success reduced by phase and exposure -- the input row E1/E2 read.

    ``succ_overall`` and the per-kind rates are means of ``hidden_pass`` over MEASURED
    attempts only; ``infra_rate`` is the fraction of attempts that were never scored (or
    hit infra), reported apart so excluded attempts stay visible instead of vanishing.
    """

    arm: str
    n: int
    succ_overall: float
    succ_phase1: float
    succ_second: float
    succ_novel: float
    landing_rate: float
    abstain_rate: float
    infra_rate: float

    def to_dict(self) -> dict:
        """JSON-ready form. Every field is a JSON-native scalar, so ``asdict`` is exact."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ArmLens":
        """Inverse of :meth:`to_dict`; no field needs coercion to restore equality."""
        return cls(**d)


def _rate(flags: list) -> float:
    """Fraction of truthy entries; 0.0 on an empty list (no ``ZeroDivisionError``)."""
    return sum(1 for f in flags if f) / len(flags) if flags else 0.0


def build_lens(task_recs) -> ArmLens:
    """Reduce one arm's ``TaskRecord``s to an :class:`ArmLens`.

    ``succ_overall`` = mean ``hidden_pass`` over records where ``hidden_pass is not None``
    (infra/not-measured excluded). ``succ_phase1/second/novel`` apply that same filter,
    restricted to ``kind == "first"/"second"/"novel"``. ``infra_rate`` = fraction with
    ``hidden_pass is None`` OR ``infra_error`` set. ``landing_rate`` = mean ``landed``,
    ``abstain_rate`` = fraction with ``status == "abstain"``. Empty input -> all rates 0.0.

    Raises ``ValueError`` if the records do not all share one arm.
    """
    recs = list(task_recs)

    arms = {r.arm for r in recs}
    if len(arms) > 1:
        raise ValueError(f"build_lens got mixed arms: {sorted(arms)}")
    arm = next(iter(arms)) if arms else ""

    measured = [r for r in recs if r.hidden_pass is not None]

    return ArmLens(
        arm=arm,
        n=len(recs),
        succ_overall=_rate([r.hidden_pass for r in measured]),
        succ_phase1=_rate([r.hidden_pass for r in measured if r.kind == "first"]),
        succ_second=_rate([r.hidden_pass for r in measured if r.kind == "second"]),
        succ_novel=_rate([r.hidden_pass for r in measured if r.kind == "novel"]),
        landing_rate=_rate([r.landed for r in recs]),
        abstain_rate=_rate([r.status == "abstain" for r in recs]),
        infra_rate=_rate([r.hidden_pass is None or r.infra_error is not None for r in recs]),
    )
