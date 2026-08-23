"""The ceiling pilot: run A_noMem over the phase-1 stream and read off p0 (spec §4.8.4).

Before the arms are measured, one question decides whether the stream is worth measuring on:
can the *frozen* proposer, with no memory and no help, already repair most of it? That rate
-- ``p0`` -- is the ceiling every downstream comparison sits under. If it is *too high* the
ceiling is undiscriminating: A_full vs A_noMem cannot separate on a stream the naive proposer
mostly solves, so the stream must be HARDENED before it locks. Spec §4.8.4 draws that line at
``p0 > 0.70`` -- STRICT: a p0 of exactly 0.70 is not too easy, it is the last "proceed".

Two honest-measurement choices are load-bearing.

*p0 is the measured-only success fraction, nothing else.* It is ``build_lens(...).succ_overall``
-- the mean of ``hidden_pass`` over the attempts that were actually scored -- NOT ``landing_rate``
(did the codec parse), not the visible reward, not the fraction of attempts that ran. Those
other rates can sit far from the real success rate (a submission that lands and passes every
visible test can still fail the hidden suite), so reading p0 off one of them would answer a
different, easier question than "did the repair actually work" and mis-gate the §4.8.4 lock.

*The threshold lives in one place, and it is strict.* ``_verdict`` owns the ``p0 >
TOO_EASY_THRESHOLD`` comparison; ``ceiling_pilot`` never re-derives it. Flipping that ``>`` to
``>=`` is the mutation the exact-0.70 boundary test is built to catch.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from crucible.run.arm import ARMS
from crucible.run.driver import run_arm, select_pilot_tasks
from crucible.run.lens import build_lens
from crucible.run.records import read_task_records

TOO_EASY_THRESHOLD = 0.70
"""Spec §4.8.4: the stream is too easy iff p0 is STRICTLY above this. Exactly 0.70 = proceed."""

PROCEED = "proceed"
"""The recommendation when the ceiling is discriminating enough to measure the arms on."""

FIRST_HARDENING_RUNG = "harden: stack two mutations per unit"
"""The first rung of the hardening ladder (spec §4.8.4) named when the stream is too easy."""


@dataclass(frozen=True)
class PilotVerdict:
    """The ceiling pilot's answer: the measured p0, its n, and what to do next.

    ``too_easy`` iff ``p0 > TOO_EASY_THRESHOLD``. ``recommendation`` is :data:`PROCEED` when
    the stream is discriminating, else the first hardening-ladder rung. Every field is a
    JSON-native scalar, so ``to_dict`` is a plain ``asdict`` the CLI can print verbatim.
    """

    p0: float
    n: int
    too_easy: bool
    recommendation: str

    def to_dict(self) -> dict:
        """JSON-ready form -- all fields are JSON-native scalars, so ``asdict`` is exact."""
        return asdict(self)


def _verdict(p0: float, n: int) -> PilotVerdict:
    """Turn a measured ``p0`` over ``n`` tasks into the pilot's verdict.

    ``too_easy`` iff ``p0 > TOO_EASY_THRESHOLD`` -- STRICT (spec §4.8.4 says ">0.70", so a p0
    of exactly 0.70 is NOT too easy). The threshold lives ONLY here, so the mutation that
    weakens ``>`` to ``>=`` shows up as an exact-0.70 boundary that flips to "too easy".
    """
    too_easy = p0 > TOO_EASY_THRESHOLD
    return PilotVerdict(p0=p0, n=n, too_easy=too_easy,
                        recommendation=FIRST_HARDENING_RUNG if too_easy else PROCEED)


def ceiling_pilot(stream_dir: Path, out_dir: Path, proposer, value, *,
                  n: int = 30, seed: int = 0, log=print) -> PilotVerdict:
    """Run ``A_noMem`` over ``n`` phase-1 tasks of ``stream_dir``; return the ceiling verdict.

    Draws the pilot's tasks with :func:`select_pilot_tasks` (seeded, phase-1 only), runs
    ``ARMS["A_noMem"]`` over them via :func:`run_arm` (resumable; records land under
    ``out_dir/A_noMem/``), then reduces the task records with :func:`build_lens` and reads
    ``p0 = lens.succ_overall`` -- the honest measured-only success rate. ``too_easy`` and
    ``recommendation`` follow :func:`_verdict` (STRICT ``p0 > 0.70``).
    """
    stream_dir, out_dir = Path(stream_dir), Path(out_dir)
    keys = select_pilot_tasks(stream_dir, n, seed=seed)
    out_path = run_arm(ARMS["A_noMem"], stream_dir, keys, proposer, value, out_dir, log=log)
    lens = build_lens(read_task_records(out_path))
    return _verdict(lens.succ_overall, lens.n)
