"""``OnlineValue``: v1 trained scorer -- a hand-rolled logistic regression over node +
task-context features, satisfying the :class:`~crucible.value.model.Value` protocol.

v0 (:class:`~crucible.value.model.ConstantValue`) is an honest no-op baseline; v1 learns.
The model is deliberately minimal -- stdlib ``math`` only, plain SGD, no numpy -- because
the feature vector is small (12 floats) and the training signal (one outcome per scored
node) arrives one example at a time, so a batched/vectorised model would buy nothing and
would add a dependency this spike does not need.

**Determinism is non-negotiable** (pre-reg S3 Task 7): the search is reproducible from its
seed, so ``score`` must return the same value for the same node under the same task
context every time it is called, with no randomness anywhere. There is none here -- the
model is pure arithmetic over ``self.w`` and the node's own features.

**The feature vector** (spec section 6), in fixed order::

    [1.0 (bias), mean_logprob or 0.0, self_certainty or 0.0, float(depth),
     *family_onehot(7), retrieval_hit]

``mean_logprob``/``self_certainty`` are ``None`` when the serving path returned no
logprobs; absent is folded to ``0.0`` rather than treated as a sentinel the model has to
learn around. ``family`` and ``retrieval_hit`` are TASK-level, not node-level -- every node
scored within one task shares them, so they are not read off the node at all.
:meth:`begin_task` is how the A_full driver sets them once per task, before the search
loop runs; the search loop itself only ever calls ``score``/``update`` on ``Value``, so it
never has to know this context exists.

**Two update paths, two feature sources.** :meth:`score` computes features fresh from the
live node + current task context and caches them in ``self._seen`` keyed by
``node.node_id`` -- the cache exists because the outcome (hidden-suite pass/fail) is only
known later, once execution finishes, and by then the caller may only have the id, not the
node. :meth:`update` is given the node directly, so it recomputes fresh ("live") features
from the current context rather than trusting a possibly-stale cache entry.
:meth:`update_by_id` has no node, only an id, so it trains against exactly the features
that were cached at score time -- even if :meth:`begin_task` has since moved on to a new
task's context. This is intentional: the outcome being folded in belongs to the task the
node was scored under, not whatever task happens to be active when the outcome lands.
"""
from __future__ import annotations

import math

# Sorted; CONST is included for schema stability even though rung-1 streams carry none of
# it -- dropping it now would change the feature vector's width the day a stream does.
FAMILIES: tuple[str, ...] = ("ARITH", "BOOL", "CMP", "CONST", "FLOW", "SDL", "UNARY")

_LEARNING_RATE = 0.1
# bias, mean_logprob, self_certainty, depth, family one-hot, retrieval_hit
_FEATURE_DIM = 4 + len(FAMILIES) + 1


def _sigmoid(x: float) -> float:
    """Numerically stable logistic sigmoid -- avoids ``OverflowError`` on large ``|x|``."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _dot(w: list[float], x: list[float]) -> float:
    return sum(wi * xi for wi, xi in zip(w, x))


class OnlineValue:
    """Trained P(hidden pass | node) scorer: hand-rolled logistic regression, plain SGD.

    Weights init to all zeros (an untrained model with zero weights predicts 0.5 for any
    input, the same honest starting point as v0's constant). ``lr=0.1`` is fixed per spec,
    not tuned -- S3 measures whether *any* trained scorer beats the constant baseline, not
    which learning rate is optimal.
    """

    def __init__(self) -> None:
        self.w: list[float] = [0.0] * _FEATURE_DIM
        self._family: str | None = None
        self._retrieval_hit: bool = False
        self._seen: dict[str, list[float]] = {}
        self.n_scores: int = 0
        self.n_updates: int = 0

    def begin_task(self, family: str, retrieval_hit: bool) -> None:
        """Set the task-level context (family, retrieval hit) subsequent scores read.

        Called by the A_full driver once before each task; the search loop never calls
        this -- it only sees ``score``/``update``, both of which read whatever context was
        last set here (or the safe all-zero default, before the first call).
        """
        self._family = family
        self._retrieval_hit = retrieval_hit

    def _features(self, node) -> list[float]:
        """Build the fixed-order feature vector from ``node`` plus the current task context."""
        mean_logprob = node.candidate.mean_logprob
        self_certainty = node.candidate.self_certainty
        family_onehot = [1.0 if fam == self._family else 0.0 for fam in FAMILIES]
        return [
            1.0,
            mean_logprob if mean_logprob is not None else 0.0,
            self_certainty if self_certainty is not None else 0.0,
            float(node.depth),
            *family_onehot,
            1.0 if self._retrieval_hit else 0.0,
        ]

    def _sgd_step(self, features: list[float], outcome: bool) -> None:
        """One plain-SGD step: ``w += lr * (outcome - sigmoid(w.x)) * x``."""
        prediction = _sigmoid(_dot(self.w, features))
        error = float(outcome) - prediction
        self.w = [wi + _LEARNING_RATE * error * xi for wi, xi in zip(self.w, features)]

    def score(self, node) -> float:
        """P(hidden pass | node) under the current task context. Deterministic; caches features."""
        features = self._features(node)
        self._seen[node.node_id] = features
        self.n_scores += 1
        return _sigmoid(_dot(self.w, features))

    def update(self, node, outcome: bool) -> None:
        """Fold a realised outcome for ``node`` into the model using freshly computed features."""
        features = self._features(node)
        self._sgd_step(features, outcome)
        self.n_updates += 1

    def update_by_id(self, node_id: str, outcome: bool) -> bool:
        """Train against ``node_id``'s score-time cached features; ``False`` if never scored.

        Never raises on an unseen id -- the driver logs the miss and moves on, so a bug
        upstream (an outcome reported for a node that was never scored) degrades to a
        skipped training step, not a crashed run.
        """
        features = self._seen.get(node_id)
        if features is None:
            return False
        self._sgd_step(features, outcome)
        self.n_updates += 1
        return True

    def snapshot(self) -> dict:
        """Record-keeping snapshot: weights plus call counters, JSON-native."""
        return {
            "w": list(self.w),
            "n_scores": self.n_scores,
            "n_updates": self.n_updates,
        }

    def restore(self, snapshot: dict) -> None:
        """Inverse of :meth:`snapshot`; subsequent ``score`` calls match the snapshotted model."""
        self.w = list(snapshot["w"])
        self.n_scores = snapshot["n_scores"]
        self.n_updates = snapshot["n_updates"]
