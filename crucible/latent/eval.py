"""The B-lite gate evaluator: paired DeLong AUROC, floors, calibration, and
the ONE-READ rule on the test split (prereg §5.4/§6).

`evaluate_gate` is the ONLY function in this whole codebase allowed to pass
`"test"` to `crucible.latent.corpus.load_split`, and it does so EXACTLY
ONCE per call -- `crucible.latent.train._load` (used by both the B-lite and
control training harnesses) structurally forbids `"test"` at its own call
site, so the test split is unreachable from anywhere except here. On top of
that, `evaluate_gate` REFUSES to run a second time against the same
`out_path` (checked as the very FIRST statement in the function body,
before `load_split` is ever called) -- the pre-registered "one read" is not
a convention, it is a function that cannot execute past its first line a
second time.

Two independent statistical primitives, both pure numpy (no scipy, no
scikit-learn):

* `delong_paired`: DeLong (1988)'s paired AUROC comparison, computed via
  Sun & Xu (2014)'s fast midrank structural-components method -- O(N log N)
  instead of the naive O(m*n) pairwise comparison, and, critically, PAIRED:
  because both scorers are evaluated against the same ordered `y`, their
  per-sample structural components share sample identity, so subtracting
  them element-wise before taking a variance is what captures the
  correlation between two scorers evaluated on the same data. An unpaired
  combination (independently combining each scorer's own SE) discards that
  correlation and gets the wrong answer even in the degenerate case where
  the two scorers are IDENTICAL (see `test_delong_paired_identical_scores_
  gives_zero_se_and_zero_diff_exactly` in the test file -- only the paired
  form gives exactly zero there).
* `ece`: the standard equal-width-bin expected calibration error.

`fit_static_floor` is a SEPARATE, deliberately tiny torch logistic
regression on three cheap structural features of a (function, input) pair
-- "the meaningful floor" B-lite and the control arm must beat by more than
surface code statistics (prereg §5.4/§6), as distinguished from the
TRIVIAL majority-class floor `evaluate_gate` also reports (a constant
score, whose AUROC is 0.5 by construction -- reported for transparency,
never gated on).

Score-file format (this is the interface Tasks 10/11 must write to, DEFINED
here precisely since nothing else in this codebase specifies it): a JSON
object at `blite_scores_path` / `ctrl_scores_path`, mapping a per-sample key
`f"{fn_id}:{args}"` (both fields exactly as stored on `corpus.Sample`) to a
single float -- the arm's predicted probability that the sample's outcome
is a clean return (`crucible.latent.gen.binary_label`'s `1`). Every key in
the test split must appear in the score file, and every key in the score
file must be a real test-split sample -- `evaluate_gate` RAISES on either a
missing or an extra key (`_align_scores`) rather than silently scoring the
inner join, which would fabricate the test-set denominator the whole gate's
credibility rests on.

References:
  DeLong, DeLong & Clarke-Pearson (1988), "Comparing the Areas under Two or
  More Correlated Receiver Operating Characteristic Curves: A Nonparametric
  Approach", Biometrics 44(3), 837-845.
  Sun & Xu (2014), "Fast Implementation of DeLong's Algorithm for Comparing
  the Areas Under Correlated Receiver Operating Characteristic Curves",
  IEEE Signal Processing Letters 21(11), 1389-1393 -- the midrank
  structural-component computation this module's `_structural_components`
  implements.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import torch

from crucible.latent import corpus as _corpus
from crucible.latent.gen import binary_label

# -- midranks + DeLong structural components ----------------------------------


def _midrank(x: np.ndarray) -> np.ndarray:
    """1-indexed MIDRANKS of `x`: a run of `k` tied values, spanning ranks
    `[i+1, i+k]` under an arbitrary tie-break, all get the AVERAGE rank
    over that span instead -- e.g. two tied lowest values both get rank
    1.5, never 1 and 2 in whatever order they happen to appear. Standard
    tie handling for a rank-based AUROC estimator; the property this relies
    on elsewhere in this module is that `sum(midranks) == N*(N+1)/2`
    EXACTLY regardless of how many ties there are (redistributing a
    contiguous rank block's sum equally among its members never changes
    the block's total)."""
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    n = len(x)
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0  # 1-indexed average over the tied run [i, j]
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    return ranks


def _structural_components(y: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """`(V10, V01, auroc)` for one scorer `s` against binary labels `y`,
    DeLong (1988)'s own notation: `V10[i]` is indexed over the `m` POSITIVE
    samples (`i = 1..m`) -- the fraction of the `n` negatives that positive
    `i` beats, ties counted as half a win: `(1/n) * sum_j Psi(X_i, Y_j)`.
    `V01[j]` is indexed over the `n` NEGATIVE samples, the symmetric
    quantity: `(1/m) * sum_i Psi(X_i, Y_j)`. `mean(V10) == mean(V01) ==
    auroc` (the Mann-Whitney U form of AUROC) -- this function returns
    `mean(V10)` as `auroc`, a single source of truth shared with the
    variance computation in `delong_paired`, rather than a separately
    re-derived point estimate that could drift out of sync with it.

    Computed via Sun & Xu (2014)'s three midrank passes (never the naive
    O(m*n) pairwise comparison):
      - `tz`: midrank of every sample within the FULL combined set.
      - `tx`: midrank of positive samples within the positives ONLY.
      - `ty`: midrank of negative samples within the negatives ONLY.
      V10 = (tz_pos - tx) / n
      V01 = 1 - (tz_neg - ty) / m

    Raises `ValueError` if either class is absent from `y` -- AUROC is
    undefined without both classes present, and this is the gate's own
    statistic, not a routine per-batch training probe that must tolerate a
    degenerate split composition (contrast `crucible.latent.train.
    _rank_auroc`, which returns 0.5 for exactly that reason in a different
    context).
    """
    y = np.asarray(y)
    s = np.asarray(s, dtype=np.float64)
    pos_mask = y == 1
    neg_mask = y == 0
    m = int(pos_mask.sum())
    n = int(neg_mask.sum())
    if m == 0 or n == 0:
        raise ValueError("both classes must be present in y to compute an AUROC")

    tz = _midrank(s)
    tx = _midrank(s[pos_mask])
    ty = _midrank(s[neg_mask])

    v10 = (tz[pos_mask] - tx) / n
    v01 = 1.0 - (tz[neg_mask] - ty) / m

    auroc = float(v10.mean())
    return v10, v01, auroc


def _sample_var(x: np.ndarray) -> float:
    """Sample variance, ddof=1 -- `0.0` for `len(x) < 2` (no degrees of
    freedom to estimate a spread from a single observation) instead of the
    NaN-with-a-divide-by-zero-warning `np.var(x, ddof=1)` would otherwise
    produce."""
    if len(x) < 2:
        return 0.0
    return float(np.var(x, ddof=1))


def delong_paired(y: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> dict:
    """DeLong (1988) paired comparison of two scorers' AUROC on the same
    `y`, via `_structural_components`' fast midrank method (Sun & Xu 2014).
    See this module's docstring for full references.

        var(diff) = var(V10_1 - V10_2, ddof=1) / m + var(V01_1 - V01_2, ddof=1) / n

    -- the PAIRED form. `s1` and `s2` score the SAME `y`-ordered samples,
    so `V10_1[i]` and `V10_2[i]` both refer to positive sample `i` (and
    likewise `V01_1[j]`/`V01_2[j]` to negative sample `j`); subtracting
    element-wise before the variance is exactly what lets a correlation
    between the two scorers (e.g. `s1 is s2`) collapse the estimated
    variance toward zero. An UNPAIRED combination -- independently
    computing each scorer's own variance and summing
    (`var(V10_1)/m + var(V01_1)/n + var(V10_2)/m + var(V01_2)/n`) -- throws
    that correlation away and gives the wrong (generally nonzero) SE even
    when `s1` and `s2` are bit-identical.

    Returns `{"auroc1", "auroc2", "diff", "se_diff", "z"}`:
    `diff = auroc1 - auroc2`; `se_diff = sqrt(var(diff))`;
    `z = diff / se_diff`, defined as exactly `0.0` when `se_diff == 0.0`
    (a genuinely tied comparison -- not a division error to propagate as
    NaN/inf).
    """
    y = np.asarray(y)
    v10_1, v01_1, auroc1 = _structural_components(y, s1)
    v10_2, v01_2, auroc2 = _structural_components(y, s2)

    m = v10_1.shape[0]
    n = v01_1.shape[0]

    d10 = v10_1 - v10_2
    d01 = v01_1 - v01_2

    var_diff = _sample_var(d10) / m + _sample_var(d01) / n
    se_diff = float(np.sqrt(var_diff))

    diff = auroc1 - auroc2
    z = diff / se_diff if se_diff != 0.0 else 0.0

    return {"auroc1": auroc1, "auroc2": auroc2, "diff": diff, "se_diff": se_diff, "z": z}


# -- paired bootstrap CI on the AUROC diff -------------------------------------


def bootstrap_diff_ci(y, s1, s2, n: int = 10000, seed: int = 0) -> tuple[float, float]:
    """Paired percentile bootstrap (2.5/97.5) on `auroc1 - auroc2`, via a
    seeded `numpy.random.default_rng(seed)`. PAIRED: every draw resamples
    ONE shared index vector and applies it to `y`, `s1`, AND `s2` together,
    so a sample's true label always stays matched to both arms' scores for
    that draw -- never resampled independently per array.

    A draw that happens to lose one binary class entirely (AUROC
    undefined) is skipped and does not count toward `n`, up to a generous
    attempt ceiling -- raises `ValueError` if not even one valid draw was
    produced (e.g. `y` itself has fewer than 2 of some class).
    """
    y = np.asarray(y)
    s1 = np.asarray(s1, dtype=np.float64)
    s2 = np.asarray(s2, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n_samples = len(y)

    diffs: list[float] = []
    attempts = 0
    max_attempts = max(n * 20, 100)
    while len(diffs) < n and attempts < max_attempts:
        attempts += 1
        idx = rng.integers(0, n_samples, size=n_samples)
        yb = y[idx]
        if (yb == 1).sum() == 0 or (yb == 0).sum() == 0:
            continue
        _, _, auroc1 = _structural_components(yb, s1[idx])
        _, _, auroc2 = _structural_components(yb, s2[idx])
        diffs.append(auroc1 - auroc2)

    if not diffs:
        raise ValueError("bootstrap_diff_ci: no resample produced both classes present")

    lo, hi = np.percentile(np.asarray(diffs), [2.5, 97.5])
    return float(lo), float(hi)


# -- calibration ----------------------------------------------------------------


def ece(y, p, bins: int = 10) -> float:
    """Standard expected calibration error: `bins` equal-width bins over
    `p` in `[0, 1]` (the last bin's upper edge is closed, so `p == 1.0`
    lands in it rather than nowhere); for each non-empty bin,
    `|mean(y) - mean(p)|` weighted by the bin's share of `N`, summed.
    Empty bins contribute nothing. `N == 0` -> `0.0`."""
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    total_n = len(y)
    if total_n == 0:
        return 0.0

    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        if i == bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        count = int(mask.sum())
        if count == 0:
            continue
        accuracy = float(y[mask].mean())
        confidence = float(p[mask].mean())
        total += (count / total_n) * abs(accuracy - confidence)
    return float(total)


# -- static floor: tiny seeded torch logistic on 3 structural features --------

_STATIC_FLOOR_STEPS = 300
_STATIC_FLOOR_LR = 0.05
_STATIC_FLOOR_SEED = 0


def _ast_node_count(function_src: str) -> int:
    """AST node count of `function_src` (`ast.walk` over the parsed tree),
    `0` on a parse failure. A defensive guard, not a real code path: every
    `function_src` reaching this function came from a `Sample` that already
    passed `crucible.latent.gen.validate` at corpus-generation time, which
    itself requires a successful `ast.parse` -- this branch should be
    unreachable for a validated corpus. Kept anyway per the brief: a static
    floor that raises on one malformed row would take the whole gate down
    with it."""
    try:
        tree = ast.parse(function_src)
    except SyntaxError:
        return 0
    return sum(1 for _ in ast.walk(tree))


def _n_args(args: str) -> int:
    """`len(ast.literal_eval(args))`; `0` on any parse failure or on a
    literal that is not a sized container. `args` is always a `repr()` of
    an argument tuple in this corpus (`crucible.latent.gen.
    parse_candidate`) -- the `TypeError` branch is defensive, not a real
    code path."""
    try:
        value = ast.literal_eval(args)
    except (SyntaxError, ValueError):
        return 0
    try:
        return len(value)
    except TypeError:
        return 0


def _static_floor_features(samples) -> np.ndarray:
    """`(N, 3)` feature matrix: `[len(function_src), ast node count,
    n_args]` per sample -- the fixed, tiny structural feature set the
    static floor logistic regresses on (prereg §5.4/§6)."""
    return np.array(
        [[len(s.function_src), _ast_node_count(s.function_src), _n_args(s.args)] for s in samples],
        dtype=np.float64,
    )


def fit_static_floor(train_samples, *, seed: int = 0):
    """Fit a tiny seeded torch logistic regression on
    `_static_floor_features(train_samples)` against each sample's
    `crucible.latent.gen.binary_label`, and return
    `predict(samples) -> np.ndarray` of predicted probabilities. `predict`
    RE-EXTRACTS features from whatever `samples` it is called with -- it
    never memoizes `train_samples`' own features, so calling it on the test
    split scores the test split's actual function/args pairs.

    "The meaningful floor" (prereg §5.4/§6), distinct from the trivial
    majority-class floor `evaluate_gate` also reports: three cheap
    structural features of a function/input pair, fit with a linear
    decision boundary -- a bar B-lite and the control arm must clear by
    more than surface code statistics, not a claim this floor is a strong
    classifier on its own.

    Determinism: `torch.manual_seed(seed)` is called ONCE, before any
    tensor or `nn.Module` is constructed, and is the ONLY randomness source
    this function touches -- feature extraction is pure numpy/`ast`, and
    the fit loop is FULL-BATCH gradient descent (no shuffling, no dropout,
    no other stochastic op). Two calls with the same `seed` and
    `train_samples` therefore produce bit-identical fitted weights, and so
    bit-identical `predict` output for the same input samples.

    Features are standardized using `train_samples`' own mean/std, computed
    ONCE and closed over by `predict` -- test-time features never
    contribute to the standardization. A zero-variance feature (constant
    across `train_samples`) gets `std := 1.0` instead of a divide-by-zero.
    """
    torch.manual_seed(seed)

    features = _static_floor_features(train_samples)
    labels = np.array([binary_label(s.outcome) for s in train_samples], dtype=np.float32)

    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std = np.where(std == 0, 1.0, std)

    x_train = torch.tensor((features - mean) / std, dtype=torch.float32)
    y_train = torch.tensor(labels, dtype=torch.float32)

    linear = torch.nn.Linear(3, 1)
    optimizer = torch.optim.Adam(linear.parameters(), lr=_STATIC_FLOOR_LR)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    linear.train()
    for _ in range(_STATIC_FLOOR_STEPS):
        optimizer.zero_grad()
        logits = linear(x_train).squeeze(-1)
        loss = loss_fn(logits, y_train)
        loss.backward()
        optimizer.step()
    linear.eval()

    def predict(samples) -> np.ndarray:
        features_s = _static_floor_features(samples)
        x_s = torch.tensor((features_s - mean) / std, dtype=torch.float32)
        with torch.no_grad():
            logits = linear(x_s).squeeze(-1)
            probs = torch.sigmoid(logits).numpy()
        return probs.astype(np.float64)

    return predict


# -- score-file loading + alignment --------------------------------------------


def _score_key(sample) -> str:
    """The score-file key for one `Sample`: `f"{fn_id}:{args}"` -- see this
    module's docstring for the score-file format this must match."""
    return f"{sample.fn_id}:{sample.args}"


def _load_scores(path) -> dict:
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"score file {path} must be a JSON object mapping sample keys to probabilities")
    return {str(k): float(v) for k, v in data.items()}


def _align_scores(samples, scores: dict, arm_name: str) -> np.ndarray:
    """`scores` reordered to match `samples`, or `ValueError` on ANY
    mismatch between the two key sets -- missing OR extra. A silent inner
    join (scoring only the overlap) would fabricate the test-set
    denominator: the gate's whole credibility rests on accounting for
    EVERY test sample, not whichever subset happened to have a score."""
    sample_keys = [_score_key(s) for s in samples]
    sample_key_set = set(sample_keys)
    score_key_set = set(scores.keys())
    missing = sample_key_set - score_key_set
    extra = score_key_set - sample_key_set
    if missing or extra:
        raise ValueError(
            f"{arm_name} score file misaligned with the test split: "
            f"{len(missing)} missing key(s), {len(extra)} extra key(s) -- "
            "refusing a silent inner join (a fabricated denominator)"
        )
    return np.array([scores[k] for k in sample_keys], dtype=np.float64)


# -- multiclass breakdown (P3) --------------------------------------------------


def _outcome_bucket(outcome: str) -> str:
    """3-way multiclass bucket name for `outcome` (prereg §4's
    {pass-return, exception, timeout} split -- the same three buckets as
    `config.N_OUTCOME_CLASSES`, but named here, not indexed, and computed
    by an INDEPENDENT implementation from `crucible.latent.train
    ._outcome_class` -- that function is private to train.py and returns
    an int index; this one exists purely for P3's human-readable report
    keys and shares no code with it."""
    if outcome == "return":
        return "return"
    if outcome == "timeout":
        return "timeout"
    return "exception"


def _threshold_rates(y: np.ndarray, scores: np.ndarray, outcomes: list) -> dict:
    """Per-`_outcome_bucket` accuracy of `scores >= 0.5` against `y`, plus
    each bucket's sample count -- P3's multiclass breakdown."""
    preds = (scores >= 0.5).astype(int)
    correct = preds == y
    buckets: dict[str, list[bool]] = {}
    for outcome, ok in zip(outcomes, correct):
        buckets.setdefault(_outcome_bucket(outcome), []).append(bool(ok))
    return {
        bucket: {"n": len(flags), "rate": float(np.mean(flags))}
        for bucket, flags in buckets.items()
    }


# -- the gate itself ------------------------------------------------------------


def evaluate_gate(corpus_dir, blite_scores_path, ctrl_scores_path, out_path) -> dict:
    """Compute the pre-registered B-lite gate verdict (prereg §5.4/§6) and
    write it to `out_path`. Returns the same dict.

    Order of operations, load-bearing for the ONE-READ discipline:

    1. `out_path` existence is checked FIRST, before anything else runs --
       a second call against the same `out_path` raises `FileExistsError`
       immediately, before `crucible.latent.corpus.load_split` is ever
       called. This is what makes "reads the test split ONCE" true of the
       function as a whole, not just of one call in isolation: there is no
       way to reach a second `load_split(corpus_dir, "test")` through this
       function, ever, for a given `out_path`.
    2. `load_split(corpus_dir, "test")` -- the ONE call, assigned to
       `test_samples` and reused for every downstream computation (P1, P2,
       the static floor's `predict`, P3). No other `load_split(..., "test")`
       call site exists anywhere in this function.
    3. Score files are loaded and aligned to `test_samples` by
       `f"{fn_id}:{args}"` key (see module docstring for the exact format);
       ANY missing or extra key raises (`_align_scores`).
    4. `fit_static_floor` is fit on `load_split(corpus_dir, "train")` --
       train, never test; its `predict` is then applied to `test_samples`
       to score the floor on the same held-out set the two arms are scored
       on.
    5. P1: `delong_paired(y, blite_scores, ctrl_scores)`; verdict `"PASS"`
       iff `diff >= 2 * se_diff` (B-lite must beat control by at least two
       standard errors), plus a bootstrap CI on that diff.
    6. P2: each arm vs. the static floor, same `>= 2*se_diff` rule,
       independently for B-lite and control.
    7. P3: per-outcome-bucket accuracy-at-0.5 for both arms, ECE for both
       arms. Also reports the TRIVIAL majority-class floor (a constant
       score at train's class-1 prevalence; AUROC 0.5 BY CONSTRUCTION,
       since every comparison against a constant score is a tie) --
       informational only, never gated on.

    Writes the full report as JSON to `out_path` and returns it.
    """
    out_path = Path(out_path)
    if out_path.exists():
        raise FileExistsError(
            f"{out_path} already exists -- evaluate_gate refuses to run twice "
            "(the one-read rule: this function reads the test split exactly "
            "once per out_path, and a rerun would read it again)"
        )

    corpus_dir = Path(corpus_dir)

    # The ONE call site of load_split(..., "test") in this function -- see
    # this function's own docstring for why that makes the whole module's
    # one-read discipline true.
    test_samples = _corpus.load_split(corpus_dir, "test")
    if not test_samples:
        raise ValueError("empty test split -- nothing to gate on")

    y = np.array([binary_label(s.outcome) for s in test_samples], dtype=np.int64)
    outcomes = [s.outcome for s in test_samples]

    blite_scores_raw = _load_scores(blite_scores_path)
    ctrl_scores_raw = _load_scores(ctrl_scores_path)
    s_blite = _align_scores(test_samples, blite_scores_raw, "blite")
    s_ctrl = _align_scores(test_samples, ctrl_scores_raw, "ctrl")

    train_samples = _corpus.load_split(corpus_dir, "train")
    if not train_samples:
        raise ValueError("empty train split -- cannot fit the static floor")
    floor_predict = fit_static_floor(train_samples, seed=_STATIC_FLOOR_SEED)
    s_floor = floor_predict(test_samples)

    train_y = np.array([binary_label(s.outcome) for s in train_samples], dtype=np.int64)
    majority_p = float(train_y.mean())
    s_majority = np.full(len(test_samples), majority_p, dtype=np.float64)
    _, _, majority_auroc = _structural_components(y, s_majority)

    # -- P1: blite vs ctrl -----------------------------------------------
    p1_delong = delong_paired(y, s_blite, s_ctrl)
    p1_pass = p1_delong["diff"] >= 2 * p1_delong["se_diff"]
    p1_ci = bootstrap_diff_ci(y, s_blite, s_ctrl)

    # -- P2: each arm vs the static floor ---------------------------------
    p2_blite = delong_paired(y, s_blite, s_floor)
    p2_ctrl = delong_paired(y, s_ctrl, s_floor)
    p2_blite_pass = p2_blite["diff"] >= 2 * p2_blite["se_diff"]
    p2_ctrl_pass = p2_ctrl["diff"] >= 2 * p2_ctrl["se_diff"]

    # -- P3: multiclass breakdown + calibration ----------------------------
    p3 = {
        "multiclass_breakdown": {
            "blite": _threshold_rates(y, s_blite, outcomes),
            "ctrl": _threshold_rates(y, s_ctrl, outcomes),
        },
        "ece": {
            "blite": ece(y, s_blite),
            "ctrl": ece(y, s_ctrl),
        },
    }

    report = {
        "n_test": len(test_samples),
        "n_train": len(train_samples),
        "p1": {
            "blite_auroc": p1_delong["auroc1"],
            "ctrl_auroc": p1_delong["auroc2"],
            "diff": p1_delong["diff"],
            "se_diff": p1_delong["se_diff"],
            "z": p1_delong["z"],
            "bootstrap_ci": list(p1_ci),
            "verdict": "PASS" if p1_pass else "FAIL",
        },
        "p2": {
            "blite_vs_floor": {
                "arm_auroc": p2_blite["auroc1"],
                "floor_auroc": p2_blite["auroc2"],
                "diff": p2_blite["diff"],
                "se_diff": p2_blite["se_diff"],
                "verdict": "PASS" if p2_blite_pass else "FAIL",
            },
            "ctrl_vs_floor": {
                "arm_auroc": p2_ctrl["auroc1"],
                "floor_auroc": p2_ctrl["auroc2"],
                "diff": p2_ctrl["diff"],
                "se_diff": p2_ctrl["se_diff"],
                "verdict": "PASS" if p2_ctrl_pass else "FAIL",
            },
        },
        "p3": p3,
        "majority_floor": {
            "p": majority_p,
            "auroc": majority_auroc,
        },
    }

    out_path.write_text(json.dumps(report, indent=2))
    return report
