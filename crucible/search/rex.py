"""REx Thompson-sampling scheduler (Beta bandit), ported from haotang1995/REx (MIT).

Ported from ``acr/scheduler/rex.py`` in github.com/haotang1995/REx. Each candidate
patch is an "arm" with a Beta(alpha, beta) posterior; :meth:`RexScheduler.select`
draws one sample per arm and returns the argmax (Thompson sampling), and
:meth:`RexScheduler.update` folds a reward in ``[0, 1]`` into that arm's posterior
(``alpha += reward``; ``beta += 1 - reward``).

The one adaptation for determinism: the original draws from a module-level numpy
``default_rng`` and uses ``rng.beta(...)``. Here a ``random.Random`` is injected via
the constructor and every draw goes through ``rng.betavariate(alpha, beta)`` instead.
Two schedulers seeded ``random.Random(same_seed)`` and driven with the same call
sequence therefore produce identical selections. Nothing else about the algorithm
changes.
"""

from __future__ import annotations

import random
from typing import Hashable


class _Arm:
    """One bandit arm's Beta(alpha, beta) posterior."""

    __slots__ = ("alpha", "beta")

    def __init__(self, alpha: float, beta: float) -> None:
        self.alpha = alpha
        self.beta = beta


class RexScheduler:
    """Thompson-sampling scheduler over arms with Beta(alpha, beta) posteriors.

    Args:
        smoothing: Beta prior added to both alpha and beta of every new arm.
        heuristic_weight: multiplier on an arm's optional heuristic reward.
        rng: injected ``random.Random`` used for every Beta draw (keyword-only).
    """

    def __init__(
        self,
        *,
        smoothing: float = 1.0,
        heuristic_weight: float = 1.0,
        rng: random.Random,
    ) -> None:
        self._smoothing = smoothing
        self._heuristic_weight = heuristic_weight
        self._rng = rng
        self._arms: dict[Hashable, _Arm] = {}

    def add_arm(self, arm_id: Hashable, heuristic_reward: float = 0.0) -> None:
        """Register an arm with prior Beta(smoothing + weight*heuristic, smoothing)."""
        alpha = self._smoothing + self._heuristic_weight * heuristic_reward
        beta = self._smoothing
        self._arms[arm_id] = _Arm(alpha, beta)

    def select(self) -> Hashable:
        """Return the arm with the largest Beta(alpha, beta) sample (one draw each)."""
        if not self._arms:
            raise ValueError("RexScheduler.select called with no arms")
        return max(self._arms, key=self._sample)

    def _sample(self, arm_id: Hashable) -> float:
        """Draw one Thompson sample from the arm's current posterior."""
        arm = self._arms[arm_id]
        return self._rng.betavariate(arm.alpha, arm.beta)

    def update(self, arm_id: Hashable, reward: float) -> None:
        """Fold a reward into the arm's posterior: alpha += reward, beta += 1 - reward."""
        arm = self._arms[arm_id]
        arm.alpha += reward
        arm.beta += 1 - reward
