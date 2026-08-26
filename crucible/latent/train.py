"""B-lite training harness: collapse probes + val-only early stop (prereg §5.2/§5.5).

`train_blite` is the ONLY place the B-lite arm's training loop lives. Two
disclosure disciplines run through this whole file, both non-negotiable per
the pre-registration:

* **The test split is structurally unreachable from here.** `_load` is the
  single call site of `crucible.latent.corpus.load_split` in this module,
  and it raises before ever calling `load_split("test", ...)` -- not a
  convention, a literal guard. `train_blite` itself only ever calls
  `_load(corpus_dir, "train")` and `_load(corpus_dir, "val")` with hardcoded
  string literals, never a variable that could be swapped for `"test"` at a
  distance. The gate (`crucible.latent.eval`, Task 8) is the only code path
  that reads test, and it reads it exactly once.
* **Collapse probes are disclosures, not decoration** (prereg §5.5): every
  `EVAL_EVERY` steps this module writes `{step, val_auroc, latent_std_mean,
  effective_rank}` to `out_dir/probes.jsonl` UNCONDITIONALLY -- including the
  very first eval, even if the run has already collapsed by then. A model
  that collapses from step 1 must still be visible in the probe log, not
  quietly absent from it.

`code_embedder` is INJECTED (`(list[str]) -> torch.Tensor (B, d_model)`) --
this module never imports `transformers` and never knows whether the frozen
encoder behind it is the real `jinaai/jina-embeddings-v2-base-code` (wired
by ops) or a test's seeded random-projection stub. Its output is always
`.detach()`-ed here regardless, as a second, structural guarantee (on top of
whatever the injected callable itself does) that no gradient can flow back
through the "frozen" code encoder via this training loop.

AUROC is computed here with a small, dependency-free rank-based (Mann-
Whitney) estimator -- no scikit-learn. Ties are handled via MIDRANKS (the
average rank across every tied entry), never an arbitrary tie-breaking
order. `crucible.latent.eval`'s paired DeLong comparison (Task 8) is a
separate, independent implementation for the gate itself; this module does
not import it and does not depend on it.
"""
from __future__ import annotations

import contextlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from crucible.latent import config
from crucible.latent import corpus as _corpus
from crucible.latent.config import MAX_SNAPSHOTS
from crucible.latent.corpus import Sample
from crucible.latent.gen import binary_label
from crucible.latent.model import BLite, blite_loss
from crucible.latent.state import PAD, VOCAB_SIZE, encode_input, encode_state_sequence

# -- config resolution --------------------------------------------------------

# Every config.py knob train_blite() will read (and config_overrides may
# override). An explicit allowlist -- not "whatever getattr(config, key)
# happens to resolve" -- so an unknown override key is a loud ValueError at
# the call boundary, not a silently-ignored typo.
_OVERRIDABLE_KEYS = (
    "LR", "BATCH", "MAX_STEPS", "EVAL_EVERY", "PATIENCE", "TRAIN_SEED",
    "D_MODEL", "STATE_ENC_D", "STATE_ENC_LAYERS", "STATE_ENC_HEADS",
    "PRED_LAYERS", "PRED_HEADS", "LAMBDA_ISO", "N_OUTCOME_CLASSES",
)


def _resolve_config(overrides: dict | None) -> dict:
    """`config.py`'s current values for `_OVERRIDABLE_KEYS`, with `overrides`
    (if any) applied on top. Raises `ValueError` on an unknown override key
    rather than silently ignoring it -- a typo'd override key should fail
    loud, not train with the un-overridden default and look like it worked.
    """
    cfg = {key: getattr(config, key) for key in _OVERRIDABLE_KEYS}
    if overrides:
        unknown = set(overrides) - set(_OVERRIDABLE_KEYS)
        if unknown:
            raise ValueError(f"unknown config_overrides key(s): {sorted(unknown)}")
        cfg.update(overrides)
    return cfg


# -- outcome -> 3-way class index ---------------------------------------------


def _outcome_class(outcome: str) -> int:
    """`outcome` string -> `GroundedHead`'s 3-way class index (config.py's
    own `N_OUTCOME_CLASSES` docstring ordering): 0 = pass-return, 1 =
    exception, 2 = timeout. Consistent with `binary_label` (`crucible.
    latent.gen`): "return" -> binary 1 / class 0, everything else -> binary
    0, split further into class 1 (exception) or class 2 (timeout)."""
    if outcome == "return":
        return 0
    if outcome == "timeout":
        return 2
    return 1  # exception:<Type>


# -- the test-split literal guard ---------------------------------------------


def _load(corpus_dir: Path, split: str) -> list[Sample]:
    """One split of the corpus, via `crucible.latent.corpus.load_split` --
    except `split == "test"`, which raises `ValueError` UNCONDITIONALLY,
    before `load_split` is ever called. The gate (`crucible.latent.eval`,
    Task 8) reads the test split exactly once; this training code path must
    be structurally unable to reach it at all, not merely discouraged from
    doing so by convention. This is the only call site of `load_split` in
    this module."""
    if split == "test":
        raise ValueError(
            "crucible.latent.train never reads the test split -- "
            "the gate (crucible.latent.eval) reads it exactly once."
        )
    return _corpus.load_split(Path(corpus_dir), split)


# -- batching ------------------------------------------------------------------


def _pad_sequences(seqs: list[list[int]], pad_id: int) -> tuple[Tensor, Tensor]:
    """`seqs`: B token-id lists of possibly different lengths -> `(ids
    (B, L), pad_mask (B, L))`; `pad_mask` is True at PAD positions, padded to
    the BATCH's own max length (floored at 1 so no dimension collapses to
    zero on an all-empty batch)."""
    max_len = max((len(s) for s in seqs), default=0)
    max_len = max(max_len, 1)
    ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    pad_mask = torch.ones((len(seqs), max_len), dtype=torch.bool)
    for i, s in enumerate(seqs):
        n = len(s)
        if n:
            ids[i, :n] = torch.tensor(s, dtype=torch.long)
            pad_mask[i, :n] = False
    return ids, pad_mask


def _pad_state_batch(per_sample: list[list[list[int]]], pad_id: int) -> tuple[Tensor, Tensor, Tensor]:
    """`per_sample[i]`: sample i's own list of per-snapshot token-id lists
    (already capped to `MAX_SNAPSHOTS` by harvest + `encode_state_sequence`).
    Pads to a common `T` (the BATCH's max snapshot count) and `L_state` (the
    batch's max per-snapshot token length) -> `(state_ids (B, T, L_state),
    state_pad_mask (B, T, L_state), seq_mask (B, T + 2))`.

    `seq_mask` positions 0/1 (code, input) are always True; state position
    `2 + t` is True for `t < len(per_sample[i])` -- a CONTIGUOUS True prefix
    per sample, the invariant `LatentPredictor`'s own `last_idx` gather
    relies on (model.py's review note). A batch-padding snapshot SLOT
    (beyond one sample's own real snapshot count) is left entirely PAD --
    `BLite`'s `_no_fully_masked_rows` guard exists exactly for that all-pad
    row.
    """
    b = len(per_sample)
    t_max = max((len(seqs) for seqs in per_sample), default=0)
    t_max = max(t_max, 1)
    l_state = 1
    for seqs in per_sample:
        for tok in seqs:
            l_state = max(l_state, len(tok))

    state_ids = torch.full((b, t_max, l_state), pad_id, dtype=torch.long)
    state_pad_mask = torch.ones((b, t_max, l_state), dtype=torch.bool)
    seq_mask = torch.zeros((b, t_max + 2), dtype=torch.bool)
    seq_mask[:, :2] = True

    for i, seqs in enumerate(per_sample):
        for t, tok in enumerate(seqs):
            n = len(tok)
            if n:
                state_ids[i, t, :n] = torch.tensor(tok, dtype=torch.long)
                state_pad_mask[i, t, :n] = False
            seq_mask[i, 2 + t] = True
    return state_ids, state_pad_mask, seq_mask


def _flatten_valid(x: Tensor, valid_mask: Tensor) -> Tensor:
    """`(B, T, d)` + `(B, T)` bool -> `(N_valid, d)`: only the real
    (non-batch-padding) positions, via boolean advanced indexing."""
    return x[valid_mask]


def _forward_batch(model: BLite, code_embedder, samples: list[Sample], device: torch.device):
    """One batch of `Sample`s through `model`. Returns the same six-tuple
    `BLite.forward` returns, plus `(binary_y, class_y)` -- the grounded
    targets, built here from each sample's `outcome` string.

    Honest test-time contract (final review CRITICAL): `pred_states` /
    `target_states` / `valid_mask` still come from `model.forward(...)`,
    teacher-forced on the RECORDED state sequence -- `prediction_loss` and
    `isotropy_loss` are unchanged, both still measured on the model's
    ability to predict the ACTUAL observed next state. `binary_logit` /
    `class_logits`, the grounded outcome targets the gate is graded on, are
    computed SEPARATELY via `model.unroll(...)` + `model.head(...)` --
    never from `forward`'s own (state-conditioned) `final_hidden`, whose
    `binary_logit`/`class_logits` this function discards unused. This is
    the honest test-time contract: the grounded head is trained on exactly
    the representation it will be scored on at eval/gate time (`unroll`
    alone), never on one that got to peek at the recorded trace.
    """
    code_texts = [s.function_src for s in samples]
    # `.detach()` regardless of what the injected callable itself does --
    # a structural guarantee, on top of whatever "frozen" discipline the
    # callable already observes, that no gradient reaches the code encoder
    # through this training loop.
    code_embed = code_embedder(code_texts).to(device).detach()

    input_seqs = [encode_input(s.args) for s in samples]
    input_ids, input_pad_mask = _pad_sequences(input_seqs, PAD)
    input_ids, input_pad_mask = input_ids.to(device), input_pad_mask.to(device)

    state_seqs = [encode_state_sequence(s.snapshots, MAX_SNAPSHOTS) for s in samples]
    state_ids, state_pad_mask, seq_mask = _pad_state_batch(state_seqs, PAD)
    state_ids, state_pad_mask = state_ids.to(device), state_pad_mask.to(device)
    seq_mask = seq_mask.to(device)

    binary_y = torch.tensor([binary_label(s.outcome) for s in samples], dtype=torch.float32, device=device)
    class_y = torch.tensor([_outcome_class(s.outcome) for s in samples], dtype=torch.long, device=device)

    pred_states, target_states, valid_mask, _leaky_binary_logit, _leaky_class_logits = model(
        code_embed, input_ids, input_pad_mask, state_ids, state_pad_mask, seq_mask
    )

    unrolled_final_hidden = model.unroll(code_embed, input_ids, input_pad_mask, n_steps=config.N_UNROLL_STEPS)
    binary_logit, class_logits = model.head(unrolled_final_hidden)

    return pred_states, target_states, valid_mask, binary_logit, class_logits, binary_y, class_y


def _batch_iterator(samples: list[Sample], batch_size: int):
    """Yields shuffled batches of `samples` forever, reshuffling (via the
    GLOBAL `random` module -- seeded once by the caller from `TRAIN_SEED`)
    at the start of every pass, so the exact sequence of batches is a pure
    function of that seed and `len(samples)`."""
    order = list(range(len(samples)))
    while True:
        random.shuffle(order)
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            yield [samples[j] for j in idx]


# -- collapse probes: rank-based AUROC + effective rank -----------------------


def _midrank(x: np.ndarray) -> np.ndarray:
    """1-indexed MIDRANKS of `x`: tied values get the AVERAGE of the ranks
    they would span under an arbitrary tie-break, e.g. two tied lowest
    values get rank 1.5 each (not 1 and 2 in whatever order they happen to
    appear) -- the standard tie handling for a rank-based AUROC estimator."""
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


def _rank_auroc(y: np.ndarray, scores: np.ndarray) -> float:
    """AUROC via the Mann-Whitney U / rank-sum relation, pure numpy, no
    scikit-learn:

        AUROC = (sum_of_ranks(positives) - n_pos*(n_pos+1)/2) / (n_pos*n_neg)

    using `_midrank` for ties (see its docstring). `y`: `(N,)` binary
    `{0, 1}` labels; `scores`: `(N,)` predicted scores (higher => more
    likely class 1). A degenerate single-class input (`n_pos == 0` or
    `n_neg == 0`) returns 0.5 (chance) rather than raising -- AUROC is
    undefined without both classes present, and this is a per-eval probe
    that must not crash a training run over a val-split composition
    accident."""
    y = np.asarray(y)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = _midrank(scores)
    sum_ranks_pos = ranks[y == 1].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _effective_rank(z: np.ndarray) -> float:
    """Roy & Vetterli (2007) effective rank of the `(N, d)` latent batch
    `z`: `exp(entropy(p))`, where `p` is `z`'s singular-value spectrum
    normalized to a probability distribution (`p_i = s_i / sum(s_i)`). A
    batch collapsed onto a single direction has one dominant singular value
    -> entropy ~0 -> effective rank ~1 (total collapse); a batch spread
    evenly across `k` directions -> effective rank ~k. Exactly-zero singular
    values are dropped before the log (the standard `0 log 0 := 0` entropy
    convention, without ever evaluating `log(0)`). `N == 0` (no valid
    latents in this eval pass at all) -> `0.0`, a value no genuine spectrum
    can produce, so it reads unambiguously as "nothing was measured" rather
    than as a real collapse."""
    if z.shape[0] == 0:
        return 0.0
    s = np.linalg.svd(z, compute_uv=False)
    s = s[s > 1e-12]
    if s.size == 0:
        return 0.0
    p = s / s.sum()
    entropy = float(-np.sum(p * np.log(p)))
    return float(np.exp(entropy))


def _unroll_batch(model: BLite, code_embedder, samples: list[Sample], device: torch.device):
    """One batch of `Sample`s through `model.unroll` ONLY -- NEVER calls
    `encode_state_sequence` and never builds `state_ids`/`state_pad_mask`/
    `seq_mask` at all. This is the val-path counterpart to `_forward_batch`
    (final review CRITICAL): early stopping now selects on exactly the
    representation the gate reads at test time, so this function has no
    code path through which a recorded snapshot sequence could reach it.
    Returns `(binary_logit, unrolled_final_hidden, binary_y)`.
    """
    code_texts = [s.function_src for s in samples]
    code_embed = code_embedder(code_texts).to(device).detach()

    input_seqs = [encode_input(s.args) for s in samples]
    input_ids, input_pad_mask = _pad_sequences(input_seqs, PAD)
    input_ids, input_pad_mask = input_ids.to(device), input_pad_mask.to(device)

    binary_y = torch.tensor([binary_label(s.outcome) for s in samples], dtype=torch.float32, device=device)

    unrolled_final_hidden = model.unroll(code_embed, input_ids, input_pad_mask, n_steps=config.N_UNROLL_STEPS)
    binary_logit, _class_logits = model.head(unrolled_final_hidden)
    return binary_logit, unrolled_final_hidden, binary_y


@torch.no_grad()
def _evaluate(model: BLite, code_embedder, val_samples: list[Sample], device: torch.device,
              batch_size: int, d_model: int) -> dict:
    """One full pass over `val_samples`: grounded binary AUROC (via
    `_rank_auroc`) plus the two collapse probes (`latent_std_mean`,
    `effective_rank`) -- ALL THREE computed from `model.unroll`'s output
    ONLY (`_unroll_batch`, final review CRITICAL). This function never
    calls `encode_state_sequence` and never touches a sample's recorded
    trace: early stopping now selects on exactly the honest test-time
    regime the gate (`crucible.latent.eval`) reads, not on a
    state-conditioned representation that would never be reachable at
    actual gate time. The collapse probes are measured on `unroll`'s own
    self-predicted final hidden (the model's "dreamed" state, the same
    branch every grounded decision is now read from), replacing the
    pre-fix probes' use of the teacher-forced `pred_states` -- consistent
    with "the gate's regime" this early-stop signal is meant to track.
    """
    model.eval()
    score_parts, y_parts, latent_parts = [], [], []
    for start in range(0, len(val_samples), batch_size):
        batch = val_samples[start:start + batch_size]
        binary_logit, unrolled_final_hidden, binary_y = _unroll_batch(model, code_embedder, batch, device)
        score_parts.append(torch.sigmoid(binary_logit).detach().cpu().numpy())
        y_parts.append(binary_y.detach().cpu().numpy())
        latent_parts.append(unrolled_final_hidden.detach().float().cpu().numpy())
    model.train()

    scores = np.concatenate(score_parts) if score_parts else np.zeros(0)
    ys = np.concatenate(y_parts) if y_parts else np.zeros(0)
    val_auroc = _rank_auroc(ys, scores) if scores.size else 0.5

    z_all = np.concatenate(latent_parts, axis=0) if latent_parts else np.zeros((0, d_model))
    latent_std_mean = float(z_all.std(axis=0).mean()) if z_all.shape[0] else 0.0
    effective_rank = _effective_rank(z_all)

    return {
        "val_auroc": val_auroc,
        "latent_std_mean": latent_std_mean,
        "effective_rank": effective_rank,
    }


def _save_checkpoint(model: BLite, cfg: dict, path: Path) -> None:
    """State dict + the resolved config snapshot (training hyperparams AND
    the model dims used to build `model`, so a later loader can reconstruct
    the exact same architecture) -- both required, per the brief."""
    torch.save({"state_dict": model.state_dict(), "config": cfg}, path)


def _use_bf16(device: str) -> bool:
    """"bf16 iff cuda" (prereg §5.2), decided by the device's TYPE
    (`torch.device(device).type`), never by string-comparing the raw
    `device` argument -- `device="cuda:0"` (a real, common caller value,
    not a hypothetical one) is cuda just as much as bare `"cuda"` is, and a
    literal `device == "cuda"` compare would silently fall back to fp32 for
    it."""
    return torch.device(device).type == "cuda"


# -- the training loop ----------------------------------------------------------


def train_blite(
    corpus_dir,
    out_dir,
    *,
    code_embedder,
    device: str,
    config_overrides: dict | None = None,
) -> dict:
    """Train B-lite on `corpus_dir`'s train split, val-only early-stop on its
    val split, write collapse-probe disclosures + the best checkpoint +
    a summary to `out_dir`, and return that summary dict.

    Data path: `_load(corpus_dir, "train")` / `_load(corpus_dir, "val")`
    ONLY -- both literal strings, never a variable. See `_load`'s own
    docstring for why the test split is structurally unreachable here.

    Precision: bf16 autocast iff `device`'s TYPE is cuda (see `_use_bf16` --
    `device="cuda:0"` counts, not just bare `"cuda"`), else fp32
    unconditionally (no autocast at all on cpu) -- both training and
    evaluation forward passes share this discipline.

    NaN/inf total loss raises `RuntimeError` IMMEDIATELY -- per prereg §6,
    this is an infra failure (CONFOUNDED), never a training result to
    continue past.

    Early stopping: tracks the best val AUROC seen across evals with
    `PATIENCE` (config); the comparison is STRICTLY greater (`>`), so a
    perfectly flat val AUROC (a collapsed-from-the-start run, say) still
    exhausts patience and stops -- it is never treated as "still improving"
    by tying the previous best. Every eval writes a probe line first,
    including the very first one, so a collapse visible from step 1 is
    still logged before any early-stop decision is made on it.

    Writes `out_dir/probes.jsonl` (one JSON line per eval: `{step,
    val_auroc, latent_std_mean, effective_rank}`), `out_dir/best.pt`
    (`{state_dict, config}` at the best val AUROC seen, or a final fallback
    save if `MAX_STEPS` never reached a single eval), and
    `out_dir/train_summary.json` (`{steps_run, best_val_auroc,
    stopped_reason: "early_stop"|"max_steps", wall_s}`), returned verbatim.
    """
    corpus_dir = Path(corpus_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = _resolve_config(config_overrides)

    seed = cfg["TRAIN_SEED"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_samples = _load(corpus_dir, "train")
    val_samples = _load(corpus_dir, "val")
    if not train_samples:
        raise ValueError("empty train split -- nothing to train on")

    torch_device = torch.device(device)
    model = BLite(
        VOCAB_SIZE,
        d_model=cfg["D_MODEL"],
        d_state=cfg["STATE_ENC_D"],
        state_enc_layers=cfg["STATE_ENC_LAYERS"],
        state_enc_heads=cfg["STATE_ENC_HEADS"],
        pred_layers=cfg["PRED_LAYERS"],
        pred_heads=cfg["PRED_HEADS"],
        n_classes=cfg["N_OUTCOME_CLASSES"],
    ).to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["LR"])

    use_bf16 = _use_bf16(device)
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16
        else contextlib.nullcontext()
    )

    probes_path = out_dir / "probes.jsonl"
    best_ckpt_path = out_dir / "best.pt"
    probes_path.write_text("")  # fresh log for this run

    best_val_auroc = -math.inf
    evals_without_improvement = 0
    stopped_reason = "max_steps"
    wall_start = time.monotonic()
    batches = _batch_iterator(train_samples, cfg["BATCH"])

    step = 0
    while step < cfg["MAX_STEPS"]:
        batch = next(batches)
        model.train()
        optimizer.zero_grad()
        with autocast_ctx:
            (pred_states, target_states, valid_mask, binary_logit,
             class_logits, binary_y, class_y) = _forward_batch(model, code_embedder, batch, torch_device)
            z_for_iso = _flatten_valid(pred_states, valid_mask)
            total, _parts = blite_loss(
                pred_states, target_states, valid_mask, z_for_iso,
                binary_logit, binary_y, class_logits, class_y,
                lambda_iso=cfg["LAMBDA_ISO"],
            )
        if not torch.isfinite(total):
            raise RuntimeError(
                f"non-finite total loss ({total.item()!r}) at step {step + 1} -- "
                "infra failure per prereg §6 (CONFOUNDED), not a training result"
            )
        total.backward()
        optimizer.step()
        step += 1

        if step % cfg["EVAL_EVERY"] == 0:
            probe = _evaluate(model, code_embedder, val_samples, torch_device, cfg["BATCH"], cfg["D_MODEL"])
            with probes_path.open("a") as f:
                f.write(json.dumps({"step": step, **probe}) + "\n")

            if probe["val_auroc"] > best_val_auroc:
                best_val_auroc = probe["val_auroc"]
                evals_without_improvement = 0
                _save_checkpoint(model, cfg, best_ckpt_path)
            else:
                evals_without_improvement += 1
                if evals_without_improvement >= cfg["PATIENCE"]:
                    stopped_reason = "early_stop"
                    break

    if not best_ckpt_path.exists():
        # MAX_STEPS was reached before a single EVAL_EVERY boundary --
        # best.pt must still exist (the module's contract), so save the
        # current (only) model state as the fallback.
        _save_checkpoint(model, cfg, best_ckpt_path)

    summary = {
        "steps_run": step,
        "best_val_auroc": best_val_auroc if math.isfinite(best_val_auroc) else None,
        "stopped_reason": stopped_reason,
        "wall_s": time.monotonic() - wall_start,
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


# -- honest test-time scoring: unroll-only, no snapshots (final review CRITICAL) --


def score_split(
    checkpoint_path,
    corpus_dir,
    split: str,
    out_path,
    *,
    code_embedder,
    device: str,
    allow_test: bool = False,
) -> None:
    """Score every sample in `split` with a trained B-lite checkpoint,
    using ONLY `BLite.unroll` (never the recorded trace) -- the same
    honest test-time contract `_evaluate`'s val-AUROC probe uses. Writes
    the score-file format `crucible.latent.eval` documents: a JSON object
    mapping `f"{fn_id}:{args}"` -> the predicted P(clean return) (sigmoid
    of the unrolled binary logit).

    Calls the SAME `_load` this module's training loop calls for
    `split in ("train", "val")` -- `_load` itself has NO `allow_test`
    escape hatch and never will: its whole job is to make "test"
    structurally unreachable from the training code path, unconditionally.

    `split == "test"` is refused HERE too, unless `allow_test=True` is
    passed explicitly. When it is, this function bypasses `_load` entirely
    (it would always raise) and calls `crucible.latent.corpus.load_split(
    ..., "test")` directly. This does NOT reopen the one-read hole
    `crucible.latent.eval.evaluate_gate` closes: `evaluate_gate` remains
    the only code path that reads test AND gates a verdict on it, and it
    still refuses a second run against the same `out_path` (its own lock
    sentinel). `score_split`'s `allow_test=True` branch is a narrow,
    explicit exception whose docstring is also its contract: it may be
    called EXACTLY ONCE per arm, by the ops gate procedure, AFTER the
    prereg lock, to produce the very score file `evaluate_gate` reads --
    never from a retry loop, never speculatively, and never more than once
    per arm's test scoring.

    Loads `checkpoint_path` (a `train_blite`/`_save_checkpoint`-format
    `{state_dict, config}` file) and reconstructs the exact `BLite`
    architecture from its `config` snapshot -- including
    `STATE_ENC_HEADS`, which `train_blite` now includes in every
    checkpoint it writes for exactly this reason.
    """
    if split == "test" and not allow_test:
        raise ValueError(
            'crucible.latent.train.score_split refuses split="test" unless '
            "allow_test=True is passed explicitly -- see this function's own "
            "docstring for the one legitimate caller (the ops gate procedure, "
            "exactly once per arm, after the prereg lock)"
        )

    corpus_dir = Path(corpus_dir)
    samples = (
        _corpus.load_split(corpus_dir, "test") if split == "test" else _load(corpus_dir, split)
    )

    checkpoint = torch.load(Path(checkpoint_path), weights_only=False)
    cfg = checkpoint["config"]
    torch_device = torch.device(device)
    model = BLite(
        VOCAB_SIZE,
        d_model=cfg["D_MODEL"],
        d_state=cfg["STATE_ENC_D"],
        state_enc_layers=cfg["STATE_ENC_LAYERS"],
        state_enc_heads=cfg["STATE_ENC_HEADS"],
        pred_layers=cfg["PRED_LAYERS"],
        pred_heads=cfg["PRED_HEADS"],
        n_classes=cfg["N_OUTCOME_CLASSES"],
    ).to(torch_device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    scores: dict[str, float] = {}
    batch_size = cfg["BATCH"]
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch = samples[start:start + batch_size]
            code_texts = [s.function_src for s in batch]
            code_embed = code_embedder(code_texts).to(torch_device).detach()

            input_seqs = [encode_input(s.args) for s in batch]
            input_ids, input_pad_mask = _pad_sequences(input_seqs, PAD)
            input_ids, input_pad_mask = input_ids.to(torch_device), input_pad_mask.to(torch_device)

            unrolled_final_hidden = model.unroll(code_embed, input_ids, input_pad_mask, n_steps=config.N_UNROLL_STEPS)
            binary_logit, _class_logits = model.head(unrolled_final_hidden)
            probs = torch.sigmoid(binary_logit).cpu().numpy()

            for sample, prob in zip(batch, probs):
                scores[f"{sample.fn_id}:{sample.args}"] = float(prob)

    Path(out_path).write_text(json.dumps(scores))
