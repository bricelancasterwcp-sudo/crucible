"""RED/GREEN tests for the B-lite gate evaluator (prereg §5.4/§6):
paired DeLong AUROC, the static floor, calibration, and the ONE-READ rule.

This module is the most mutation-critical of the phase -- it computes the
pre-registered verdict. Every test below is a deliberate mutation pin, not
incidental coverage:

* `test_delong_paired_hand_computed_six_item_case` -- a fully hand-derived
  6-item case (see this file's inline derivation): a perfect separator
  (`auroc1 == 1.0` exactly) vs. a one-inversion scorer (`auroc2 == 8/9`
  exactly), plus the hand-derived `se_diff == sqrt(2)/9` and
  `z == 1/sqrt(2)`.
* `test_delong_paired_identical_scores_gives_zero_se_and_zero_diff_exactly`
  -- `s2 = s1` (a fresh equal-valued array, not the same object) must give
  `diff == 0.0` and `se_diff == 0.0` EXACTLY. This is the mutation pin
  called out in the brief: an UNPAIRED SE formula (independently combining
  each scorer's own variance) gives a nonzero SE here even though the two
  scorers are identical -- only the paired form (variance of the
  PER-SAMPLE difference) collapses to exactly zero.
* `test_delong_paired_tie_handling_matches_hand_value` -- a case with a
  cross-class tie, checked against a hand-derived AUROC.
* `test_ece_*` -- an exactly-zero perfectly-calibrated construction and a
  large-gap inverted one.
* `test_fit_static_floor_*` -- AUROC > 0.9 on a separable synthetic, and
  bit-for-bit-deterministic across two independent fits with the same seed.
* `test_evaluate_gate_*` -- the one-read pin (a second call against the
  same `out_path` raises before doing anything else), the alignment pin (a
  missing score key raises), and the comparison-direction pin (P1's verdict
  flips between a scenario where B-lite clearly beats control and one where
  it clearly does not).

No real corpus, no real proposer, no real jina/codeexecutor model anywhere
in this file -- synthetic corpora are written directly in Task 2/3's
on-disk `samples.jsonl` format (same pattern as `test_corpus.py` /
`test_train.py`), and score files are plain JSON dicts per this module's
own documented format.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from crucible.latent import corpus
from crucible.latent import eval as gate_eval
from crucible.latent.corpus import Sample
from crucible.latent.eval import (
    bootstrap_diff_ci,
    delong_paired,
    ece,
    evaluate_gate,
    fit_static_floor,
)

# -- shared helpers -------------------------------------------------------------


def _fn_id_for_split(split: str, seed: int = 0, prefix: str = "fn") -> str:
    """Search synthetic fn_ids until one hashes into `split` -- exercises
    the real `assign_split` instead of hardcoding a hash value (same
    pattern as `test_corpus.py` / `test_train.py`)."""
    for i in range(10_000):
        candidate = f"{prefix}-{i}"
        if corpus.assign_split(candidate, seed) == split:
            return candidate
    raise AssertionError(f"no synthetic fn_id landed in split={split!r} within 10000 tries")


def _write_samples_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _sample_row(fn_id: str, *, outcome: str, pad: str, args: str = "()") -> dict:
    return {
        "fn_id": fn_id,
        "function_src": f"def f():\n    return 1  # {pad}\n",
        "args": args,
        "outcome": outcome,
        "return_repr": "1" if outcome == "return" else None,
        "snapshots": [],
    }


def _make_sample(fn_id: str, *, outcome: str, args: str = "()", src_len: int = 20) -> Sample:
    return Sample(
        fn_id=fn_id,
        function_src=f"def f():\n    return 1  # {'x' * src_len}\n",
        args=args,
        outcome=outcome,
        return_repr="1" if outcome == "return" else None,
        snapshots=(),
    )


# =============================================================================
# delong_paired
# =============================================================================


def test_delong_paired_hand_computed_six_item_case():
    """y = [1,1,1,0,0,0]; s1 perfectly separates -> auroc1 == 1.0 exactly.
    s2 has exactly one inversion (the lowest positive loses to one
    negative) -> auroc2 == 8/9 exactly (7 winning pairs count as full wins,
    1 pair inverted, out of 3*3 = 9 total pairs).

    Hand-derived structural components (see task report / self-review for
    the full derivation):
      V10_1 = [1, 1, 1]         V01_1 = [1, 1, 1]
      V10_2 = [1, 1, 2/3]       V01_2 = [1, 1, 2/3]
      d10 = d01 = [0, 0, 1/3]   var(d10, ddof=1) = var(d01, ddof=1) = 1/27
      var(diff) = (1/27)/3 + (1/27)/3 = 2/81
      se_diff = sqrt(2)/9 ; z = (1/9) / (sqrt(2)/9) = 1/sqrt(2)
    """
    y = np.array([1, 1, 1, 0, 0, 0])
    s1 = np.array([10.0, 9.0, 8.0, 3.0, 2.0, 1.0])  # positives all > negatives
    s2 = np.array([10.0, 9.0, 8.0, 3.0, 2.0, 8.5])  # last negative beats the weakest positive

    result = delong_paired(y, s1, s2)

    assert result["auroc1"] == 1.0
    assert result["auroc2"] == pytest.approx(8.0 / 9.0)
    assert result["diff"] == pytest.approx(1.0 / 9.0)
    assert result["se_diff"] > 0.0
    assert result["se_diff"] == pytest.approx(math.sqrt(2) / 9)
    assert result["z"] == pytest.approx(1.0 / math.sqrt(2))


def test_delong_paired_identical_scores_gives_zero_se_and_zero_diff_exactly():
    """Mutation pin (brief): an UNPAIRED SE formula gives a nonzero SE even
    when the two scorers are identical -- only the PAIRED form (variance of
    the per-sample structural-component DIFFERENCE) collapses to exactly
    zero. `s2` is a freshly-built array equal in VALUE to `s1`, not the
    same object, so this cannot pass by some `is`-identity shortcut."""
    y = np.array([1, 1, 1, 0, 0, 0, 1, 0])
    s1 = np.array([0.9, 0.3, 0.7, 0.1, 0.5, 0.2, 0.6, 0.4])
    s2 = np.array(s1, copy=True)

    result = delong_paired(y, s1, s2)

    assert result["diff"] == 0.0
    assert result["se_diff"] == 0.0
    assert result["z"] == 0.0
    assert result["auroc1"] == result["auroc2"]


def test_delong_paired_tie_handling_matches_hand_value():
    """y = [1,1,0,0], s = [2,1,1,0]: positive-2 (score 1) ties negative-1
    (score 1). Pairs: (2,1) win, (2,0) win, (1,1) tie=0.5, (1,0) win ->
    (1+1+0.5+1)/4 = 3.5/4 = 0.875. Calling `delong_paired(y, s, s)` reuses
    the real implementation for both arguments and additionally re-confirms
    the identical-scores zero-SE property on a tied array specifically."""
    y = np.array([1, 1, 0, 0])
    s = np.array([2.0, 1.0, 1.0, 0.0])

    result = delong_paired(y, s, s)

    assert result["auroc1"] == pytest.approx(0.875)
    assert result["auroc2"] == pytest.approx(0.875)
    assert result["se_diff"] == 0.0


def test_delong_paired_raises_when_a_class_is_absent():
    y = np.array([1, 1, 1])
    s = np.array([0.1, 0.2, 0.3])
    with pytest.raises(ValueError):
        delong_paired(y, s, s)


# =============================================================================
# bootstrap_diff_ci
# =============================================================================


def test_bootstrap_diff_ci_contains_the_point_estimate_diff():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 0])
    s1 = np.array([0.9, 0.8, 0.7, 0.6, 0.2, 0.1, 0.3, 0.4, 0.85, 0.15])
    s2 = np.array([0.5, 0.55, 0.45, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

    point = delong_paired(y, s1, s2)
    lo, hi = bootstrap_diff_ci(y, s1, s2, n=500, seed=0)

    assert lo <= point["diff"] <= hi


def test_bootstrap_diff_ci_is_deterministic_given_same_seed():
    y = np.array([1, 1, 1, 0, 0, 0, 1, 0])
    s1 = np.array([0.9, 0.3, 0.7, 0.1, 0.5, 0.2, 0.6, 0.4])
    s2 = np.array([0.2, 0.6, 0.4, 0.5, 0.3, 0.7, 0.1, 0.9])

    ci_a = bootstrap_diff_ci(y, s1, s2, n=300, seed=7)
    ci_b = bootstrap_diff_ci(y, s1, s2, n=300, seed=7)

    assert ci_a == ci_b


# =============================================================================
# ece
# =============================================================================


def test_ece_perfectly_calibrated_synthetic_is_exactly_zero():
    """10 bins, 20 samples per bin, all sharing the bin's own probability
    `p = (2b+1)/20` with EXACTLY `2b+1` positives among the 20 -- accuracy
    equals confidence in every bin by construction, so ECE is ~0 up to the
    float64 rounding of `p` itself (e.g. `0.05` has no exact binary
    representation) -- not literally bit-exact `0.0`, but far tighter than
    the "large" gap the inverted case pins below."""
    ys: list[int] = []
    ps: list[float] = []
    for b in range(10):
        p = (2 * b + 1) / 20.0
        positives = 2 * b + 1
        ys.extend([1] * positives + [0] * (20 - positives))
        ps.extend([p] * 20)

    assert ece(np.array(ys), np.array(ps), bins=10) == pytest.approx(0.0, abs=1e-9)


def test_ece_inverted_is_large():
    y = np.zeros(50)
    p = np.full(50, 0.9)

    assert ece(y, p) == pytest.approx(0.9)


def test_ece_empty_input_is_zero():
    assert ece(np.array([]), np.array([])) == 0.0


# =============================================================================
# fit_static_floor
# =============================================================================


def _separable_synthetic_samples(n_per_class: int = 25) -> list[Sample]:
    samples = []
    for i in range(n_per_class):
        samples.append(_make_sample(f"pos-{i}", outcome="return", src_len=200 + i))
        samples.append(_make_sample(f"neg-{i}", outcome="exception:ValueError", src_len=5 + (i % 3)))
    return samples


def test_fit_static_floor_separates_synthetic_above_0_9_auroc():
    samples = _separable_synthetic_samples()
    predict = fit_static_floor(samples, seed=0)
    probs = predict(samples)

    y = np.array([1 if s.outcome == "return" else 0 for s in samples])
    auroc = delong_paired(y, probs, probs)["auroc1"]

    assert auroc > 0.9


def test_fit_static_floor_is_deterministic_across_two_fits():
    train_samples = _separable_synthetic_samples()
    test_samples = _separable_synthetic_samples(n_per_class=5)

    predict_a = fit_static_floor(train_samples, seed=0)
    predict_b = fit_static_floor(train_samples, seed=0)

    probs_a = predict_a(test_samples)
    probs_b = predict_b(test_samples)

    np.testing.assert_allclose(probs_a, probs_b, rtol=0, atol=0)


def test_fit_static_floor_predict_returns_ndarray_of_right_shape():
    train_samples = _separable_synthetic_samples(n_per_class=3)
    predict = fit_static_floor(train_samples, seed=0)
    probs = predict(train_samples)

    assert isinstance(probs, np.ndarray)
    assert probs.shape == (len(train_samples),)
    assert np.all((probs >= 0.0) & (probs <= 1.0))


# =============================================================================
# evaluate_gate
# =============================================================================


def _build_gate_corpus(corpus_dir: Path, *, n_train_each: int = 8, n_test_each: int = 8) -> tuple[list, list]:
    """A synthetic corpus with `n_train_each` positive + negative TRAIN
    functions and `n_test_each` positive + negative TEST functions (never
    "val" -- irrelevant to this module). TRAIN and TEST function_src bodies
    both correlate a padding comment's length with the binary label, so
    `fit_static_floor` has real (if crude) signal to fit on. Returns
    `(test_pos_ids, test_neg_ids)`.
    """
    rows = []
    for i in range(n_train_each):
        fn_id = _fn_id_for_split("train", 0, prefix=f"tr1-{i}")
        rows.append(_sample_row(fn_id, outcome="return", pad="X" * 200))
    for i in range(n_train_each):
        fn_id = _fn_id_for_split("train", 0, prefix=f"tr0-{i}")
        rows.append(_sample_row(fn_id, outcome="exception:ValueError", pad="X" * 5))

    test_pos_ids, test_neg_ids = [], []
    for i in range(n_test_each):
        fn_id = _fn_id_for_split("test", 0, prefix=f"te1-{i}")
        rows.append(_sample_row(fn_id, outcome="return", pad="X" * 200))
        test_pos_ids.append(fn_id)
    for i in range(n_test_each):
        fn_id = _fn_id_for_split("test", 0, prefix=f"te0-{i}")
        rows.append(_sample_row(fn_id, outcome="exception:ValueError", pad="X" * 5))
        test_neg_ids.append(fn_id)

    _write_samples_jsonl(corpus_dir / "samples.jsonl", rows)
    return test_pos_ids, test_neg_ids


def _scores_for(ids: list[str], prob: float) -> dict:
    return {f"{fn_id}:()": prob for fn_id in ids}


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data))


def test_evaluate_gate_verdict_flips_with_comparison_direction(tmp_path):
    """Scenario A: B-lite scores a perfect hard separator (1.0/0.0),
    control scores a constant 0.5 (chance, all ties) -> diff = 0.5,
    se_diff == 0.0 exactly (every structural component is constant within
    each arm) -> P1 must PASS (0.5 >= 0).

    Scenario B: the SAME two score sets, with B-lite and control SWAPPED
    -> diff = -0.5, se_diff == 0.0 -> P1 must FAIL (-0.5 >= 0 is false).

    A mutant that flips the `>=` to `<=`, or that swaps which arm's AUROC
    is `auroc1` vs `auroc2` inside `evaluate_gate`, flips at least one of
    these two verdicts relative to the correct answer -- this is the
    comparison-direction pin."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    test_pos_ids, test_neg_ids = _build_gate_corpus(corpus_dir)

    good = {**_scores_for(test_pos_ids, 1.0), **_scores_for(test_neg_ids, 0.0)}
    chance = {**_scores_for(test_pos_ids, 0.5), **_scores_for(test_neg_ids, 0.5)}

    blite_good_path = tmp_path / "blite_good.json"
    ctrl_chance_path = tmp_path / "ctrl_chance.json"
    _write_json(blite_good_path, good)
    _write_json(ctrl_chance_path, chance)

    report_a = evaluate_gate(corpus_dir, blite_good_path, ctrl_chance_path, tmp_path / "report_a.json")
    assert report_a["p1"]["verdict"] == "PASS"
    assert report_a["p1"]["diff"] == pytest.approx(0.5)
    assert report_a["p1"]["se_diff"] == 0.0

    # Scenario B: same two score files, arms swapped (blite <- chance, ctrl <- good).
    report_b = evaluate_gate(corpus_dir, ctrl_chance_path, blite_good_path, tmp_path / "report_b.json")
    assert report_b["p1"]["verdict"] == "FAIL"
    assert report_b["p1"]["diff"] == pytest.approx(-0.5)
    assert report_b["p1"]["se_diff"] == 0.0


def test_evaluate_gate_report_has_expected_shape_and_writes_file(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    test_pos_ids, test_neg_ids = _build_gate_corpus(corpus_dir)

    good = {**_scores_for(test_pos_ids, 0.95), **_scores_for(test_neg_ids, 0.05)}
    bad = {**_scores_for(test_pos_ids, 0.55), **_scores_for(test_neg_ids, 0.45)}
    blite_path = tmp_path / "blite.json"
    ctrl_path = tmp_path / "ctrl.json"
    _write_json(blite_path, good)
    _write_json(ctrl_path, bad)

    out_path = tmp_path / "gate_report.json"
    report = evaluate_gate(corpus_dir, blite_path, ctrl_path, out_path)

    assert out_path.exists()
    on_disk = json.loads(out_path.read_text())
    assert on_disk == report

    assert report["n_test"] == len(test_pos_ids) + len(test_neg_ids)
    assert set(report["p1"]) == {"blite_auroc", "ctrl_auroc", "diff", "se_diff", "z", "bootstrap_ci", "verdict"}
    assert set(report["p2"]) == {"blite_vs_floor", "ctrl_vs_floor"}
    assert set(report["p3"]) == {"multiclass_breakdown", "ece"}
    assert report["majority_floor"]["auroc"] == pytest.approx(0.5)


def test_evaluate_gate_raises_on_second_call_same_out_path(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    test_pos_ids, test_neg_ids = _build_gate_corpus(corpus_dir)
    scores = {**_scores_for(test_pos_ids, 0.9), **_scores_for(test_neg_ids, 0.1)}
    blite_path = tmp_path / "blite.json"
    ctrl_path = tmp_path / "ctrl.json"
    _write_json(blite_path, scores)
    _write_json(ctrl_path, scores)

    out_path = tmp_path / "gate_report.json"
    evaluate_gate(corpus_dir, blite_path, ctrl_path, out_path)

    with pytest.raises(FileExistsError):
        evaluate_gate(corpus_dir, blite_path, ctrl_path, out_path)


def test_evaluate_gate_second_call_never_reaches_load_split(tmp_path, monkeypatch):
    """Stronger form of the one-read pin: the second call must be refused
    BEFORE `crucible.latent.corpus.load_split` is ever invoked again --
    not merely fail some other way after re-reading test. Monkeypatches
    `load_split` to explode if called a second time."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    test_pos_ids, test_neg_ids = _build_gate_corpus(corpus_dir)
    scores = {**_scores_for(test_pos_ids, 0.9), **_scores_for(test_neg_ids, 0.1)}
    blite_path = tmp_path / "blite.json"
    ctrl_path = tmp_path / "ctrl.json"
    _write_json(blite_path, scores)
    _write_json(ctrl_path, scores)

    out_path = tmp_path / "gate_report.json"
    evaluate_gate(corpus_dir, blite_path, ctrl_path, out_path)

    def _explode(*args, **kwargs):
        raise AssertionError("load_split must not be called on a second evaluate_gate call")

    monkeypatch.setattr(gate_eval._corpus, "load_split", _explode)

    with pytest.raises(FileExistsError):
        evaluate_gate(corpus_dir, blite_path, ctrl_path, out_path)


def test_evaluate_gate_raises_on_missing_score_key(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    test_pos_ids, test_neg_ids = _build_gate_corpus(corpus_dir)

    scores = {**_scores_for(test_pos_ids, 0.9), **_scores_for(test_neg_ids, 0.1)}
    del scores[f"{test_pos_ids[0]}:()"]  # drop one required key

    blite_path = tmp_path / "blite.json"
    ctrl_path = tmp_path / "ctrl.json"
    full_scores = {**_scores_for(test_pos_ids, 0.9), **_scores_for(test_neg_ids, 0.1)}
    _write_json(blite_path, scores)
    _write_json(ctrl_path, full_scores)

    with pytest.raises(ValueError):
        evaluate_gate(corpus_dir, blite_path, ctrl_path, tmp_path / "gate_report.json")


def test_evaluate_gate_raises_on_extra_score_key(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    test_pos_ids, test_neg_ids = _build_gate_corpus(corpus_dir)

    scores = {**_scores_for(test_pos_ids, 0.9), **_scores_for(test_neg_ids, 0.1)}
    scores["not-a-real-key:()"] = 0.5

    blite_path = tmp_path / "blite.json"
    ctrl_path = tmp_path / "ctrl.json"
    full_scores = {**_scores_for(test_pos_ids, 0.9), **_scores_for(test_neg_ids, 0.1)}
    _write_json(blite_path, scores)
    _write_json(ctrl_path, full_scores)

    with pytest.raises(ValueError):
        evaluate_gate(corpus_dir, blite_path, ctrl_path, tmp_path / "gate_report.json")
