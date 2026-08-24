"""Tests for per-class isotonic calibration + abstention (spec S3 Task 8 brief, section 7).

``Calibrator`` maps a raw value score into a calibrated P(hidden pass) per PROVENANCE
CLASS (retrieval-hit x search phase), so a low-confidence proposal late in a search can be
told apart from an equally-scored one early in a search even though the raw value score
alone cannot see that difference. Isotonic regression here is a hand-rolled PAVA (Pool
Adjacent Violators Algorithm) over stdlib floats -- no numpy, no new dependency -- per the
brief's implementer-choice clause; see the module docstring in
``crucible/uncertainty/conformal.py`` and the Task 8 report for the crepes-vs-PAVA
rationale (crepes targets conformal *regression* intervals, not binary-outcome isotonic
calibration, so adopting it here would be the awkward fit the brief warns about).

Four properties are mutation-checked (see the Task 8 report and its review-fix addendum
for the literal kill runs):

1. **Honest cold start.** Before ``MIN_OBS=10`` observations land in a class, ``confidence``
   returns the RAW score unchanged -- a mutant that fits early anyway is caught by
   :func:`test_confidence_is_passthrough_before_min_obs`.
2. **Per-class independence.** A mutant that shares one bucket/model across all provenance
   classes (ignoring the ``cls`` argument) is caught by
   :func:`test_classes_are_independent_training_one_leaves_another_at_passthrough`: an
   UNOBSERVED class must stay exact passthrough no matter how much another class has been
   trained.
3. **Inclusive abstention boundary.** ``should_abstain`` uses spec Section 4.7's inclusive-
   gate convention (``landing >= 0.95`` passes) -- ``p == ABSTAIN_P`` exactly does NOT
   abstain. A mutant that flips ``<`` to ``<=`` is caught by
   :func:`test_should_abstain_false_exactly_at_threshold`.
4. **The fit is a pure function of the observation multiset, not insertion order.** Review
   finding (fixed): ``_fit_isotonic`` used to run PAVA directly over the raw pair list, so
   two exact-tied scores with different outcomes could pool differently into a neighbouring
   block depending on which ``observe()`` call happened first (verified computationally:
   the multiset ``{(5, 0), (5, 1), (6, 0.5)}`` gave ``predict(5.0) == 0.0`` in one insertion
   order and ``0.5`` in the other). Fixed by pooling identical scores to a weighted mean
   BEFORE PAVA runs (sklearn's ``_make_unique`` approach). A mutant that removes that
   pre-pooling step is caught by :func:`test_fit_isotonic_pools_ties_before_running_pava`
   and :func:`test_confidence_is_order_independent_at_tied_scores`.

**On the recalibrate "flip" test:** isotonic regression is BY DEFINITION a monotone
non-decreasing function of score -- ``fit(x1) <= fit(x2)`` whenever ``x1 <= x2`` is the
whole point of "isotonic," and PAVA's pooling step enforces it unconditionally. That means
NO valid isotonic fit can ever produce ``confidence(0.9) < confidence(0.1)`` (that would
require the fitted function to decrease); feeding it a purely score-decreasing pattern
("high score, bad outcome") does not invert the curve, it POOLS the whole thing into one
flat block at the batch's mean. So the "ordering flips" property tested below is: a clear
stale separation (old data: high-good) is fully DESTROYED (not merely weakened -- pooled
to a single flat value) once ``recalibrate`` restricts the fit to a purely contradictory
recent window, and the high-probe's confidence visibly drops from its stale value. That is
the strongest assertion a correct isotonic fit can support for genuinely contradictory
recent data, and it is exactly "drops influence of old observations" (the brief's own
first clause) made precise.
"""
from __future__ import annotations

import json

import pytest

from crucible.uncertainty.conformal import (
    ABSTAIN_P,
    MIN_OBS,
    PROVENANCE_CLASSES,
    Calibrator,
    _fit_isotonic,
    _predict,
    provenance_class,
)


# --- provenance_class: the 4-way mapping ----------------------------------------


def test_provenance_class_hit_phase_1():
    assert provenance_class(retrieval_hit=True, phase=1) == "hit-p1"


def test_provenance_class_hit_phase_2():
    assert provenance_class(retrieval_hit=True, phase=2) == "hit-p2"


def test_provenance_class_nohit_phase_1():
    assert provenance_class(retrieval_hit=False, phase=1) == "nohit-p1"


def test_provenance_class_nohit_phase_2():
    assert provenance_class(retrieval_hit=False, phase=2) == "nohit-p2"


def test_provenance_classes_is_exactly_the_pinned_4_tuple():
    assert PROVENANCE_CLASSES == ("hit-p1", "hit-p2", "nohit-p1", "nohit-p2")


def test_provenance_class_raises_on_unknown_phase():
    # Fail loud on a caller bug rather than minting an unrecognised 5th class -- matches
    # OnlineValue.begin_task's exclusion-is-explicit idiom for an unrecognised family.
    with pytest.raises(ValueError):
        provenance_class(retrieval_hit=True, phase=3)


# --- cold start: honest passthrough before MIN_OBS -------------------------------


def test_confidence_is_passthrough_before_min_obs():
    cal = Calibrator()
    cls = "hit-p1"
    for _ in range(MIN_OBS - 1):  # one short of the threshold
        cal.observe(score=0.9, cls=cls, outcome=True)
    assert cal.confidence(0.42, cls) == 0.42


def test_confidence_is_passthrough_for_a_never_observed_class():
    cal = Calibrator()
    assert cal.confidence(0.73, "nohit-p2") == 0.73


def test_confidence_stops_being_passthrough_once_min_obs_is_reached():
    cal = Calibrator()
    cls = "hit-p1"
    for _ in range(MIN_OBS):  # exactly at the threshold
        cal.observe(score=0.9, cls=cls, outcome=True)
        cal.observe(score=0.1, cls=cls, outcome=False)
    # 2 * MIN_OBS observations now recorded -- must be using the fitted model, not raw.
    assert cal.confidence(0.9, cls) != 0.9


# --- discrimination: 20 informative pairs move confidence off the raw score ------


def test_confidence_discriminates_after_20_informative_pairs():
    cal = Calibrator()
    cls = "hit-p1"
    for _ in range(10):
        cal.observe(score=0.9, cls=cls, outcome=True)
        cal.observe(score=0.1, cls=cls, outcome=False)

    high = cal.confidence(0.9, cls)
    low = cal.confidence(0.1, cls)
    assert high > low
    assert high != 0.9
    assert low != 0.1


# --- tie pooling: the fit is a pure function of the multiset (review fix) --------


def test_fit_isotonic_pools_ties_before_running_pava():
    # THE mutation guard at the fix site: the reviewer's exact triple, computationally
    # verified to diverge (0.0 vs 0.5 at the tied score 5.0) before pre-pooling was added.
    # Both insertion orders of the SAME multiset must now fit identically.
    order_a = [(5.0, 0.0), (5.0, 1.0), (6.0, 0.5)]
    order_b = [(5.0, 1.0), (5.0, 0.0), (6.0, 0.5)]

    anchors_a = _fit_isotonic(order_a)
    anchors_b = _fit_isotonic(order_b)

    assert _predict(anchors_a, 5.0) == _predict(anchors_b, 5.0)
    assert anchors_a == anchors_b  # the whole fit matches, not just the one probe point


def test_confidence_is_order_independent_at_tied_scores():
    # End-to-end version of the same guard through the public API a real caller uses.
    # `outcome` is bool-only, so the reviewer's (6, 0.5) point is reproduced as two
    # observations at x=6 (one True, one False -- mean 0.5), added identically in both
    # variants. Filler observations (also identical in both variants) clear MIN_OBS
    # without touching the tied region -- only the insertion order of the (5, 0)/(5, 1)
    # pair differs between order_a and order_b.
    cls = "hit-p1"

    order_a = Calibrator()
    for _ in range(7):
        order_a.observe(score=0.0, cls=cls, outcome=False)
    order_a.observe(score=5.0, cls=cls, outcome=False)
    order_a.observe(score=5.0, cls=cls, outcome=True)
    order_a.observe(score=6.0, cls=cls, outcome=True)
    order_a.observe(score=6.0, cls=cls, outcome=False)

    order_b = Calibrator()
    for _ in range(7):
        order_b.observe(score=0.0, cls=cls, outcome=False)
    order_b.observe(score=5.0, cls=cls, outcome=True)
    order_b.observe(score=5.0, cls=cls, outcome=False)
    order_b.observe(score=6.0, cls=cls, outcome=True)
    order_b.observe(score=6.0, cls=cls, outcome=False)

    assert order_a.confidence(5.0, cls) == order_b.confidence(5.0, cls)


# --- class independence (mutation guard: a shared/global fit must be killed) -----


def test_classes_are_independent_training_one_leaves_another_at_passthrough():
    # THE primary mutation guard: a mutant that shares one history/model bucket across all
    # classes (ignoring `cls`) makes "nohit-p2" spuriously see hit-p1's >= MIN_OBS
    # observations and stop being passthrough. A correct per-class implementation cannot.
    cal = Calibrator()
    for _ in range(15):
        cal.observe(score=0.9, cls="hit-p1", outcome=True)
        cal.observe(score=0.1, cls="hit-p1", outcome=False)

    assert cal.confidence(0.55, "nohit-p2") == 0.55


def test_classes_are_independent_produce_different_calibrated_values():
    # Reinforces independence with two SIMULTANEOUSLY-trained classes on different data --
    # a shared-fit mutant would make these collide; per-class buckets keep them distinct.
    cal = Calibrator()
    for _ in range(12):
        cal.observe(score=0.9, cls="hit-p1", outcome=True)
        cal.observe(score=0.1, cls="hit-p1", outcome=False)
        cal.observe(score=0.7, cls="nohit-p2", outcome=True)
        cal.observe(score=0.3, cls="nohit-p2", outcome=False)

    assert cal.confidence(0.8, "hit-p1") != cal.confidence(0.8, "nohit-p2")


# --- should_abstain: inclusive boundary (mutation guard: < vs <=) ----------------


def test_should_abstain_true_strictly_below_threshold():
    assert Calibrator().should_abstain(ABSTAIN_P - 0.01, "hit-p1") is True


def test_should_abstain_false_exactly_at_threshold():
    # THE mutation guard: p == ABSTAIN_P exactly must NOT abstain -- the same inclusive
    # gate convention as spec Section 4.7's `landing >= 0.95`.
    assert Calibrator().should_abstain(ABSTAIN_P, "hit-p1") is False


def test_should_abstain_false_strictly_above_threshold():
    assert Calibrator().should_abstain(ABSTAIN_P + 0.01, "hit-p1") is False


# --- recalibrate(window): drops the influence of old observations ----------------


def test_recalibrate_drops_old_observations_and_destroys_stale_separation():
    cal = Calibrator()
    cls = "hit-p1"
    old_n = 12
    for _ in range(old_n):
        cal.observe(score=0.9, cls=cls, outcome=True)
        cal.observe(score=0.1, cls=cls, outcome=False)

    stale_high = cal.confidence(0.9, cls)
    stale_low = cal.confidence(0.1, cls)
    assert stale_high > stale_low  # old data says high-good

    recent_n = 12
    for _ in range(recent_n):
        cal.observe(score=0.9, cls=cls, outcome=False)
        cal.observe(score=0.1, cls=cls, outcome=True)
    cal.recalibrate(window=recent_n * 2)  # exactly the recent (high-bad) batch, no older

    fresh_high = cal.confidence(0.9, cls)
    fresh_low = cal.confidence(0.1, cls)
    assert fresh_high == fresh_low  # stale separation destroyed, pooled flat
    assert fresh_high < stale_high  # and visibly dropped from the stale belief


def test_recalibrate_only_touches_classes_it_has_history_for():
    cal = Calibrator()
    for _ in range(15):
        cal.observe(score=0.9, cls="hit-p1", outcome=True)
    cal.recalibrate(window=5)
    # "nohit-p2" was never observed -- recalibrate must not conjure a model for it.
    assert cal.confidence(0.5, "nohit-p2") == 0.5


def test_recalibrate_raises_on_non_positive_window():
    with pytest.raises(ValueError):
        Calibrator().recalibrate(window=0)


# --- snapshot/restore --------------------------------------------------------------


def test_snapshot_restore_round_trips_confidence():
    cal = Calibrator()
    cls = "hit-p1"
    for _ in range(15):
        cal.observe(score=0.9, cls=cls, outcome=True)
        cal.observe(score=0.1, cls=cls, outcome=False)
    snap = cal.snapshot()
    expected_high = cal.confidence(0.9, cls)
    expected_low = cal.confidence(0.1, cls)

    restored = Calibrator()
    restored.restore(snap)

    assert restored.confidence(0.9, cls) == expected_high
    assert restored.confidence(0.1, cls) == expected_low


def test_snapshot_restore_preserves_cold_start_for_untouched_classes():
    cal = Calibrator()
    for _ in range(15):
        cal.observe(score=0.9, cls="hit-p1", outcome=True)
    snap = cal.snapshot()

    restored = Calibrator()
    restored.restore(snap)
    assert restored.confidence(0.5, "nohit-p2") == 0.5


def test_snapshot_is_json_native():
    cal = Calibrator()
    cal.observe(score=0.9, cls="hit-p1", outcome=True)
    json.dumps(cal.snapshot())  # must not raise
