"""Per-class isotonic calibration of value scores into P(hidden pass), plus abstention
(spec S3 Section 7).

**Why this exists.** The A_full arm's value scorer (``crucible.value.online.OnlineValue``)
is a plain logistic model -- a reasonable RANKING signal, but not a calibrated probability:
two nodes scored 0.9 do not necessarily have the same true P(hidden pass) if one came from
a retrieval-hit task in search phase 1 and the other from a no-hit task deep in phase 2.
``Calibrator`` fits a separate isotonic map from raw score to calibrated confidence PER
PROVENANCE CLASS (:data:`PROVENANCE_CLASSES`), so the abstention decision can condition on
"which regime is this score from," not just the raw number.

**crepes vs. stdlib PAVA (implementer decision, recorded here and in the Task 8 report).**
This module implements isotonic regression via a hand-rolled Pool Adjacent Violators
Algorithm (PAVA) over stdlib floats, NOT the ``crepes`` package the design doc's Section 7
mentions. Reasons:

1. ``crepes`` is a CONFORMAL REGRESSION library -- its Mondrian machinery buckets a
   CONTINUOUS regression target into prediction INTERVALS with guaranteed coverage. What
   this module needs is isotonic calibration of a BINARY outcome (hidden pass / fail)
   against a score -- a probability-calibration problem, not an interval-coverage problem.
   Forcing crepes' regression-interval API to answer "what is P(outcome=1 | score)" is
   exactly the awkward fit the brief's implementer-choice clause warns about.
2. The observations arrive one at a time, streamed through ``observe()``, and each class's
   isotonic fit needs to be recomputed from that class's own accumulating history (and,
   post-recalibrate, from a trailing window of it) -- a small, from-scratch PAVA over a
   Python list of floats does this in a few lines with no batching/vectorisation benefit to
   buy, mirroring this repo's established preference for hand-rolled online models over a
   heavier dependency (see ``crucible/value/online.py``'s module docstring: "a small
   feature vector ... arrives one example at a time, so a batched/vectorised model would
   buy nothing and would add a dependency this spike does not need").
3. No new dependency was added to ``pyproject.toml`` -- ``crepes`` was NOT installed.

**Honest cold start.** Before :data:`MIN_OBS` observations have landed in a class,
:meth:`Calibrator.confidence` returns the RAW score unchanged. A half-trained isotonic fit
on a handful of points is not more honest than the raw score; passthrough says so plainly
rather than dressing up noise as calibration.

**Isotonic regression cannot invert, only flatten.** By definition, an isotonic fit is
non-decreasing in score: ``fit(x1) <= fit(x2)`` whenever ``x1 <= x2``. Feeding it data
where high scores correlate with BAD outcomes does not produce a decreasing curve (that
would violate the fit's own monotonicity) -- PAVA's pooling step merges the conflicting
region into a single flat block at the local mean instead. This is a hard mathematical
property, not an implementation gap; see ``tests/uncertainty/test_conformal.py``'s module
docstring for how the ``recalibrate`` tests are built around it.

**Abstention composes with the reward rule, and REPLACES the raw-confidence compare (S3
wiring).** The search loop's abstain rule has two halves. Its REWARD half (``reward <
crucible.search.loop.ABSTAIN_THRESHOLD``, structurally ``0.5``) is unconditional and this
module never touches it -- an arm still cannot abstain on a submission that is passing most
of its visible suite. Its CONFIDENCE half is what changes: for an arm with no calibration
hook the loop compares a RAW value-model score against that same structural ``0.5``, and for
a calibrated arm (A_full, via ``crucible.run.full``'s per-task hook) the loop calls
:meth:`Calibrator.should_abstain` on a CALIBRATED probability at :data:`ABSTAIN_P` (``0.2``)
instead.

The two numbers are deliberately different rules, not one constant that drifted: ``0.5`` on
a raw ranking score means "the ranker does not believe this either", while ``0.2`` on a
calibrated P(hidden pass) is the pre-registered §6 gate. Comparing a calibrated probability
against ``0.5`` would abstain on most of a hard stream (at p0 ~ 0.27 the honest calibrated
probability is usually below a half), and comparing a raw score against ``0.2`` would
abstain almost never. Neither module re-derives the other's threshold, and there is
deliberately no cross-module equality pin between them.

**The fit is a pure function of the observation multiset, not of insertion order.**
:func:`_fit_isotonic` pools every pair sharing the same score into one weighted-mean point
BEFORE running PAVA (see its docstring) -- without that pre-pooling step, two exact-tied
scores with different outcomes could land in either order depending on which ``observe()``
call happened first, and PAVA's merge-on-monotonicity-violation check does not fire between
two blocks with EQUAL means, so a third, unrelated block could end up pooled in or not
depending on that arbitrary tie order. Pinned by
``test_fit_isotonic_pools_ties_before_running_pava`` and
``test_confidence_is_order_independent_at_tied_scores`` (both in
``tests/uncertainty/test_conformal.py``).

All state here is in-memory for one run; :meth:`Calibrator.snapshot`/:meth:`restore` exist
for record-keeping only (arm records already carry the observations), not persistence.
"""
from __future__ import annotations

PROVENANCE_CLASSES: tuple[str, ...] = ("hit-p1", "hit-p2", "nohit-p1", "nohit-p2")

# Below this many observations in a class, `confidence` is honest raw-score passthrough --
# an isotonic fit on fewer points is not a trustworthy calibration.
MIN_OBS = 10

# Inclusive gate, same convention as spec Section 4.7's `landing >= 0.95`: p == ABSTAIN_P
# exactly does NOT abstain. Composes with the search loop's REWARD half and replaces its
# raw-confidence compare for a calibrated arm -- see the module docstring.
ABSTAIN_P = 0.2


def provenance_class(retrieval_hit: bool, phase: int) -> str:
    """Map ``(retrieval_hit, phase)`` to one of :data:`PROVENANCE_CLASSES`.

    ``phase`` must be ``1`` or ``2`` (the two search phases); any other value raises
    ``ValueError`` -- fail loud on a caller bug rather than minting an unrecognised 5th
    class, matching ``OnlineValue.begin_task``'s exclusion-is-explicit idiom for family.
    """
    if phase not in (1, 2):
        raise ValueError(f"unknown phase {phase!r}; must be 1 or 2")
    hit = "hit" if retrieval_hit else "nohit"
    return f"{hit}-p{phase}"


def _fit_isotonic(pairs: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Pool Adjacent Violators: the monotone non-decreasing step-function fit of ``pairs``
    (score, outcome-as-0-or-1), by squared error.

    Returns ``(x_max, mean_y)`` anchors sorted ascending by ``x_max``, one per pooled
    block; :func:`_predict` walks them to find the block covering a query score.

    **Ties are pooled to a weighted mean BEFORE PAVA runs** (sklearn's ``_make_unique``
    approach): every pair sharing the same ``x`` is collapsed into one ``(x, sum_y,
    weight)`` point first, using plain dict accumulation, which is commutative and thus
    independent of ``pairs``' iteration order. Skipping this step made the fit a function
    of INSERTION ORDER among exact ties, not just the observation multiset -- the stack
    merge below only pools two blocks when their MEANS violate monotonicity, and two
    same-``x`` singleton blocks with equal means never trigger that check, so which of them
    ended up adjacent to a *third*, unrelated block (and therefore whether that unrelated
    block got pooled in) depended on which same-``x`` point was appended first. Fixed and
    pinned by ``test_fit_isotonic_pools_ties_before_running_pava`` (mutation-checked:
    removing this pooling step is caught by that test and by
    ``test_confidence_is_order_independent_at_tied_scores``).

    Once ties are pooled, each unique ``x`` appears exactly once, in ascending order, and
    the classic stack-based PAVA runs over those: whenever the newly-appended block's mean
    is LOWER than its predecessor's (a monotonicity violation), merge them into one block
    and keep checking backward -- a merge can cascade through several prior blocks in one
    step (see the module docstring for why a fully score-decreasing input pools into a
    single flat block).
    """
    pooled: dict[float, list[float]] = {}
    for x, y in pairs:
        bucket = pooled.setdefault(x, [0.0, 0.0])  # [sum_y, weight]
        bucket[0] += y
        bucket[1] += 1.0
    ordered = sorted(pooled.items())  # ascending by x; each x appears exactly once

    blocks: list[list[float]] = []  # each: [sum_y, weight, x_max]
    for x, (sum_y, weight) in ordered:
        blocks.append([sum_y, weight, x])
        while len(blocks) >= 2 and (blocks[-2][0] / blocks[-2][1]) > (blocks[-1][0] / blocks[-1][1]):
            block_sum_y, block_weight, _ = blocks.pop()
            prev_sum_y, prev_weight, _ = blocks.pop()
            blocks.append([prev_sum_y + block_sum_y, prev_weight + block_weight, x])
    return [(x_max, sum_y / weight) for sum_y, weight, x_max in blocks]


def _predict(anchors: list[tuple[float, float]], x: float) -> float:
    """Step-function isotonic prediction at ``x``: the pooled block's mean whose range
    covers ``x``, extrapolated flat beyond the observed range on either side.
    """
    result = anchors[0][1]
    for x_max, mean_y in anchors:
        if x < x_max:
            break
        result = mean_y
    return result


class Calibrator:
    """Per-class isotonic calibration of value scores, plus the abstention gate.

    ``observe`` records one (score, outcome) pair for a provenance class and re-fits that
    class's isotonic model against its FULL history so far -- each class's model and
    history live in their own dict entry, so training one class never touches another's
    (the per-class-independence invariant the tests mutation-check).

    ``recalibrate(window)`` is the post-accepted-sleep hook: sleep training breaks the
    exchangeability the running fit assumed, so each class's model is re-fit from only its
    OWN last ``window`` observations. This is a PERMANENT truncation, not a one-shot
    correction: ``recalibrate`` drops every pre-truncation pair from ``cls``'s history, not
    just from the model it fits right now, so every LATER plain ``observe()`` call in that
    class also builds on the truncated history (growing from ``window`` observations
    onward), never on the pre-sleep pairs again. An accepted sleep breaks exchangeability
    for good, not for one fit.
    """

    def __init__(self) -> None:
        self._history: dict[str, list[tuple[float, float]]] = {}
        self._models: dict[str, list[tuple[float, float]]] = {}

    def observe(self, score: float, cls: str, outcome: bool) -> None:
        """Record one realised (score, outcome) pair for ``cls`` and re-fit its model.

        Raises ``ValueError`` if ``cls`` is not one of :data:`PROVENANCE_CLASSES` -- the
        write boundary where a typo'd class would otherwise silently start its own bucket.
        """
        if cls not in PROVENANCE_CLASSES:
            raise ValueError(f"unknown provenance class {cls!r}; must be one of {PROVENANCE_CLASSES}")
        pairs = self._history.setdefault(cls, [])
        pairs.append((float(score), 1.0 if outcome else 0.0))
        self._models[cls] = _fit_isotonic(pairs)

    def confidence(self, score: float, cls: str) -> float:
        """Calibrated P(hidden pass) for ``score`` under provenance class ``cls``.

        Isotonic regression fit per class over its observed (score, outcome) pairs. Before
        :data:`MIN_OBS` observations have landed in ``cls`` (including a class never
        observed at all), this is honest passthrough: the raw ``score`` is returned
        unchanged rather than trusting a fit built on too little data.
        """
        pairs = self._history.get(cls, [])
        if len(pairs) < MIN_OBS:
            return score
        anchors = self._models.get(cls)
        if not anchors:
            return score
        return _predict(anchors, score)

    def should_abstain(self, p: float, cls: str) -> bool:
        """Whether calibrated confidence ``p`` (for provenance class ``cls``) is low enough
        to abstain: ``p < ABSTAIN_P``, an INCLUSIVE gate -- ``p == ABSTAIN_P`` exactly does
        NOT abstain, matching spec Section 4.7's ``landing >= 0.95`` convention.

        ``cls`` is accepted for signature symmetry with :meth:`confidence` (and for a
        future per-class threshold, if S4 needs one); :data:`ABSTAIN_P` is currently a
        single module-level constant applied the same way to every class.

        This is the CONFIDENCE half of ``crucible.search.loop._status``'s abstain rule for a
        calibrated arm; the reward half (``< ABSTAIN_THRESHOLD``) still has to hold too. See
        the module docstring for why ``0.2`` here and ``0.5`` there are two different rules.
        """
        del cls  # not read yet -- see the docstring
        return p < ABSTAIN_P

    def recalibrate(self, window: int) -> None:
        """PERMANENTLY truncate every observed class's history to its last ``window``
        observations and re-fit from what remains -- the post-accepted-sleep hook (accepted
        sleep training breaks the exchangeability the running fit assumed, so pre-sleep
        observations must stop influencing every later fit, not just the very next one).

        The truncation replaces ``self._history[cls]``, not just the model: a later plain
        ``observe()`` call re-fits from this truncated-then-grown history, never from the
        dropped pre-truncation pairs. If truncation leaves a class below :data:`MIN_OBS`,
        :meth:`confidence` correctly falls back to raw-score passthrough again until enough
        post-truncation observations land.

        Classes with fewer than ``window`` total observations keep everything they have (a
        no-op truncation). Classes never observed are untouched (nothing to recalibrate).
        Raises ``ValueError`` if ``window`` is not positive.
        """
        if window <= 0:
            raise ValueError(f"window must be positive, got {window}")
        for cls, pairs in self._history.items():
            pairs = self._history[cls] = pairs[-window:]
            self._models[cls] = _fit_isotonic(pairs)

    def snapshot(self) -> dict:
        """Record-keeping snapshot of every class's history and fitted model, JSON-native."""
        return {
            "history": {cls: [list(pair) for pair in pairs] for cls, pairs in self._history.items()},
            "models": {cls: [list(anchor) for anchor in anchors] for cls, anchors in self._models.items()},
        }

    def restore(self, snapshot: dict) -> None:
        """Inverse of :meth:`snapshot`; subsequent ``confidence`` calls match the
        snapshotted state.
        """
        self._history = {
            cls: [(float(x), float(y)) for x, y in pairs] for cls, pairs in snapshot["history"].items()
        }
        self._models = {
            cls: [(float(x), float(y)) for x, y in anchors] for cls, anchors in snapshot["models"].items()
        }
