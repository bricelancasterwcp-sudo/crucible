"""Tests for the ported REx Thompson-sampling scheduler (pure math, unwrapped)."""

import random

from crucible.search.rex import RexScheduler


def test_selection_is_deterministic_under_a_seeded_rng():
    def run():
        s = RexScheduler(rng=random.Random("t"))
        for a in ("x", "y", "z"):
            s.add_arm(a)
        return [s.select() for _ in range(5)]

    assert run() == run()


def test_reward_shifts_selection_toward_the_rewarded_arm():
    s = RexScheduler(rng=random.Random("t"))
    for a in ("x", "y"):
        s.add_arm(a)
    for _ in range(30):
        s.update("x", 1.0)
        s.update("y", 0.0)
    picks = [s.select() for _ in range(200)]
    assert picks.count("x") > picks.count("y") * 3


def test_failures_on_one_arm_shift_selection_to_the_other():
    """Both arms earn the same successes; only 'y' then fails repeatedly.

    The failures must drive 'y' below 'x' via the ``beta += 1 - reward`` update.
    Drop that update and the two arms collapse to the SAME Beta(alpha, smoothing)
    posterior, so selection becomes a 50/50 coin flip and this assertion fails.
    That is what pins the beta half of ``update`` under mutation.
    """
    s = RexScheduler(rng=random.Random("t"))
    for a in ("x", "y"):
        s.add_arm(a)
    for _ in range(30):
        s.update("x", 1.0)
        s.update("y", 1.0)  # identical successes on both arms
    for _ in range(30):
        s.update("y", 0.0)  # y alone then accrues failures
    picks = [s.select() for _ in range(200)]
    assert picks.count("x") > picks.count("y") * 3
