"""Token-space CONTROL arm fine-tune harness (prereg §3).

Pre-registered as the FAIR control for B-lite: "the control must share the
treatment's data path and early-stop protocol exactly -- an unfair control
makes the gate meaningless." Every protocol element that could make the
comparison unfair is therefore IDENTICAL IN KIND to `crucible.latent.
train`'s B-lite loop, not merely similar -- see `train_control`'s own
docstring for the full accounting of what is shared, what is reused by
IMPORT (not reimplemented), and the one deliberate structural difference
(per-epoch vs per-step) and why it does not touch the shared protocol
elements.

The control model sees ONLY `render_control_input(function_src, args)` --
raw source text plus the call's argument literal, run through an injected
tokenizer. It never sees execution-state snapshots (B-lite's exclusive
input, `crucible.latent.state`) -- that asymmetry is the entire point of
this arm: does giving B-lite the executed-state trace actually help, versus
a token-space model with the same code and the same labels but no trace?

This module never imports `transformers` and never knows whether the model
behind `model_factory` is the real `microsoft/codeexecutor` (wired by ops)
or a test's tiny stub -- see `train_control`'s docstring for the exact
model-call contract.
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from crucible.latent import config
from crucible.latent.corpus import Sample
from crucible.latent.gen import binary_label
from crucible.latent.train import _load, _rank_auroc

# `_load` is REUSED, not reimplemented: `crucible.latent.train._load` is
# the single call site of `crucible.latent.corpus.load_split` in train.py,
# and it raises unconditionally before ever calling `load_split("test",
# ...)`. Importing the function object directly (rather than writing a
# second copy of the same guard) makes it structurally impossible for the
# two arms' test-split guards to drift apart -- there is only one guard,
# shared by both.
#
# `_rank_auroc` is REUSED for the same reason: it is the shared val metric
# (midrank/Mann-Whitney AUROC, no scikit-learn) both arms must be graded on
# identically. Both names are private to train.py; imported here anyway,
# by explicit brief instruction, rather than duplicated.

# -- config resolution --------------------------------------------------------

# Every config.py knob train_control() reads. Unlike train_blite's
# `_resolve_config`, this takes no override dict -- train_control has no
# `config_overrides` parameter (see its own docstring for why: it never
# constructs the model, so there is no model-dimension config to resolve or
# override). Each key is read fresh via `getattr` at call time, so a test
# can still adjust one with `monkeypatch.setattr(config, "PATIENCE", ...)`
# and have it take effect exactly as if it had been an explicit override.
_CONFIG_KEYS = ("CTRL_LR", "CTRL_MAX_EPOCHS", "CTRL_MAXLEN", "PATIENCE", "TRAIN_SEED", "BATCH")


def _resolve_config() -> dict:
    """`config.py`'s current values for `_CONFIG_KEYS`. `PATIENCE`,
    `TRAIN_SEED`, and `BATCH` are the exact same shared knobs train.py
    reads (not CTRL_-prefixed duplicates) -- reused, not duplicated, so the
    control arm's early-stop patience, seed, and batch size can never drift
    from the treatment arm's."""
    return {key: getattr(config, key) for key in _CONFIG_KEYS}


# -- render_control_input -------------------------------------------------------


def render_control_input(function_src: str, args_literal: str) -> str:
    """`function_src + "\\nINPUT: " + args_literal` -- pure string
    concatenation, deterministic (same inputs -> same output, always), and
    NEVER truncates. Truncation to `config.CTRL_MAXLEN` TOKENS happens
    later, at tokenization time, in this module's own batching code
    (`_tokenize_truncated`) -- this function has no notion of tokens at all
    and must not attempt a character-level truncation, which could not know
    where an arbitrary injected tokenizer's token boundaries fall.
    """
    return function_src + "\nINPUT: " + args_literal


# -- tokenization + batching ---------------------------------------------------


def _tokenize_truncated(tokenizer: Callable[[str], list[int]], text: str, max_len: int) -> list[int]:
    """`text` -> `list(tokenizer(text))[:max_len]`.

    TRUNCATION HAPPENS HERE (the loader), not in `render_control_input` and
    not assumed of `tokenizer` itself: `tokenizer` is contracted to return
    the FULL, untruncated token-id sequence for `text`, and this function
    slices it to the first `max_len` ids -- a token-boundary-respecting cut
    (never mid-token) because it slices the tokenizer's own already-tokenized
    output, not the raw text.
    """
    return list(tokenizer(text))[:max_len]


def _tokenize_batch(
    tokenizer: Callable[[str], list[int]], samples: list[Sample], max_len: int
) -> list[list[int]]:
    """Every sample in `samples` -> `render_control_input(...)` -> tokenized
    and truncated to `max_len` tokens (see `_tokenize_truncated`). One
    token-id list per sample, in the same order as `samples`."""
    return [
        _tokenize_truncated(tokenizer, render_control_input(s.function_src, s.args), max_len)
        for s in samples
    ]


def _pad_token_batch(token_lists: list[list[int]]) -> tuple[Tensor, Tensor]:
    """`token_lists`: one already-truncated token-id list per sample ->
    `(input_ids (B, L), attention_mask (B, L))`, padded to the BATCH's own
    max length (floored at 1 so no dimension collapses to zero on an
    all-empty batch).

    Pad id is 0; `attention_mask` is 1 at REAL token positions and 0 at PAD
    positions -- the OPPOSITE polarity from train.py's own `pad_mask`
    convention (True means IGNORE, matching `nn.TransformerEncoder`'s
    `src_key_padding_mask`). This is deliberate, not an inconsistency: this
    mask is handed to an injected HuggingFace-style model, whose
    `attention_mask` convention is 1-means-attend. The literal PAD id value
    (0) does not otherwise matter -- it is always masked out.
    """
    max_len = max((len(t) for t in token_lists), default=0)
    max_len = max(max_len, 1)
    b = len(token_lists)
    input_ids = torch.zeros((b, max_len), dtype=torch.long)
    attention_mask = torch.zeros((b, max_len), dtype=torch.long)
    for i, ids in enumerate(token_lists):
        n = len(ids)
        if n:
            input_ids[i, :n] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, :n] = 1
    return input_ids, attention_mask


def _extract_logits(output) -> Tensor:
    """`output` (the return of `model(input_ids, attention_mask)`, per
    `train_control`'s model contract) -> the per-sample logits, squeezed to
    `(B,)`.

    Accepts EITHER an object exposing a `.logits` attribute
    (HuggingFace-`ModelOutput`-style, `return_dict=True`) OR a plain
    tuple/list whose FIRST element is the logits tensor
    (`return_dict=False`-style) -- ops's real `microsoft/codeexecutor`-plus-
    head wrapper may return either depending on how it is configured, and
    this module's own tests exercise both shapes.
    """
    logits = output.logits if hasattr(output, "logits") else output[0]
    if logits.dim() == 2 and logits.shape[-1] == 1:
        logits = logits.squeeze(-1)
    return logits


def _forward_batch(
    model, tokenizer: Callable[[str], list[int]], samples: list[Sample],
    device: torch.device, max_len: int,
) -> tuple[Tensor, Tensor]:
    """One batch of `Sample`s through `model` -> `(logits (B,), binary_y
    (B,))`. `binary_y` is built here from each sample's `outcome` string via
    `crucible.latent.gen.binary_label` -- the same binary reduction train.py
    and corpus.py both use (1 = clean return, 0 = exception or timeout)."""
    token_lists = _tokenize_batch(tokenizer, samples, max_len)
    input_ids, attention_mask = _pad_token_batch(token_lists)
    input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)

    binary_y = torch.tensor([binary_label(s.outcome) for s in samples], dtype=torch.float32, device=device)

    output = model(input_ids, attention_mask)
    logits = _extract_logits(output)
    return logits, binary_y


# -- val AUROC probe (the control arm's only probe) -----------------------------


@torch.no_grad()
def _evaluate(
    model, tokenizer: Callable[[str], list[int]], val_samples: list[Sample],
    device: torch.device, batch_size: int, max_len: int,
) -> dict:
    """One full pass over `val_samples` -> `{"val_auroc": float}` -- the
    ONLY probe this arm discloses (unlike B-lite's collapse probes, which
    are specific to B-lite's own latent representation and have no
    token-space analogue here; prereg §3's control job is a fair
    apples-to-apples comparison on the shared val-AUROC metric).

    AUROC is computed by `crucible.latent.train._rank_auroc` -- the SAME
    midrank/Mann-Whitney estimator train.py uses for its own val AUROC
    probe, imported (not reimplemented), so both arms are graded on
    byte-for-byte the same metric. A degenerate empty val split returns
    0.5 (chance), same convention as `_rank_auroc` itself for a
    single-class input.
    """
    model.eval()
    score_parts, y_parts = [], []
    for start in range(0, len(val_samples), batch_size):
        batch = val_samples[start:start + batch_size]
        logits, binary_y = _forward_batch(model, tokenizer, batch, device, max_len)
        score_parts.append(torch.sigmoid(logits).detach().cpu().numpy())
        y_parts.append(binary_y.detach().cpu().numpy())
    model.train()

    scores = np.concatenate(score_parts) if score_parts else np.zeros(0)
    ys = np.concatenate(y_parts) if y_parts else np.zeros(0)
    val_auroc = _rank_auroc(ys, scores) if scores.size else 0.5
    return {"val_auroc": val_auroc}


def _save_checkpoint(model, cfg: dict, path: Path) -> None:
    """State dict + the resolved TRAINING-hyperparam config snapshot.
    Unlike train.py's B-lite `_save_checkpoint`, this does NOT also snapshot
    model-architecture dims -- this module never constructs the model
    (`model_factory` does, externally), so it has no architecture
    information to snapshot in the first place; reconstructing the exact
    model from this checkpoint alone is the caller's own responsibility."""
    torch.save({"state_dict": model.state_dict(), "config": cfg}, path)


# -- the training loop ----------------------------------------------------------


def train_control(
    corpus_dir,
    out_dir,
    *,
    model_factory: Callable[[], "torch.nn.Module"],
    tokenizer: Callable[[str], list[int]],
    device: str,
) -> dict:
    """Fine-tune the token-space CONTROL arm on `corpus_dir`'s train split,
    val-only early-stop on its val split, write a val-AUROC probe log + the
    best checkpoint + a summary to `out_dir`, and return that summary dict.

    Fairness accounting (prereg §3 -- "the control must share the
    treatment's data path and early-stop protocol exactly"): every element
    below is IDENTICAL IN KIND to train.py's B-lite loop, not merely
    similar --

    * **Data path.** `_load(corpus_dir, "train")` / `_load(corpus_dir,
      "val")` ONLY, both hardcoded string literals, never a variable.
      `_load` is `crucible.latent.train._load` itself, imported (not
      reimplemented) -- see this module's own top docstring and `_load`'s
      docstring in train.py for why "test" raises unconditionally, before
      `load_split` is ever called. The gate (Task 8) is the only code path
      that reads test, for either arm.
    * **Val metric.** `_evaluate` (this module) computes val AUROC via
      `crucible.latent.train._rank_auroc`, imported (not reimplemented) --
      both arms are graded on the identical midrank/Mann-Whitney estimator.
    * **Early-stop rule.** Tracks the best val AUROC seen with
      `config.PATIENCE` (the SAME shared knob train.py reads, never a
      CTRL_-prefixed duplicate); the comparison is STRICTLY greater (`>`),
      so a perfectly flat val AUROC still exhausts patience and stops --
      never treated as "still improving" by tying the previous best. This
      is the exact same rule train.py enforces, applied at a different
      GRAIN (see below).
    * **Seed.** `config.TRAIN_SEED` -- the same shared seed train.py reads,
      mixed into `random`/`numpy`/`torch`'s global RNG state the same way
      (`random.seed`, `np.random.seed`, `torch.manual_seed`), at the start
      of this function, before any data shuffling or model construction.
    * **NaN/inf loss.** Raises `RuntimeError` IMMEDIATELY, BEFORE
      `.backward()`/`optimizer.step()` are ever called on it -- an infra
      failure, never a training result to continue past, exactly as in
      train.py.

    The ONE deliberate structural difference from train.py: this loop is
    per-EPOCH (`config.CTRL_MAX_EPOCHS` caps it, not `config.MAX_STEPS`),
    and one "eval" here is one full pass over the train split followed by
    one val-AUROC pass, not an `EVAL_EVERY`-step boundary -- so
    `config.PATIENCE` counts consecutive non-improving EPOCHS here, rather
    than consecutive non-improving `EVAL_EVERY`-spaced step-evals. This
    reflects the arms' different natural training grain (B-lite: a
    from-scratch model trained step-wise over a large step budget; control:
    fine-tuning an already-pretrained language model over a small number of
    full passes) -- it does NOT touch the early-stop RULE itself (still
    strict `>`, still the same shared `PATIENCE` value) or any of the other
    fairness elements above.

    `model_factory` is a zero-arg callable returning a fresh
    `torch.nn.Module` model (ops wires `microsoft/codeexecutor` plus a
    linear classification head; tests pass a tiny 2-layer stub with the
    same call contract). The model MUST support the standard
    `torch.nn.Module` protocol (`.to(device)`, `.train()`, `.eval()`,
    `.parameters()`) plus:

        model(input_ids, attention_mask) -> output

    where `input_ids`/`attention_mask` are `(B, L)` long tensors
    (`attention_mask`: 1 at real-token positions, 0 at pad -- see
    `_pad_token_batch`'s docstring for why this is the opposite polarity
    from train.py's own mask convention) and `output` is EITHER an object
    exposing a `.logits` attribute OR a plain tuple/list whose first
    element is the logits tensor -- `logits`: `(B,)` or `(B, 1)` (both
    accepted, squeezed to `(B,)`), ONE real-valued score per sample,
    POSITIVE meaning more likely a clean return
    (`crucible.latent.gen.binary_label(outcome) == 1`). See
    `_extract_logits`. This loop feeds it straight to
    `F.binary_cross_entropy_with_logits` against that target during
    training and to `torch.sigmoid` for the val AUROC probe.

    `tokenizer` is a callable `(text: str) -> list[int]` returning the
    FULL, UNTRUNCATED token-id sequence for `text` over the model's own
    vocabulary. Truncation to `config.CTRL_MAXLEN` tokens happens HERE, in
    this module's own batching code (`_tokenize_truncated`) -- never inside
    `render_control_input` (pure string concatenation, no notion of tokens)
    and never assumed of `tokenizer` itself.

    Unlike `train_blite`, this function has NO `config_overrides`
    parameter: it never constructs the model (that is entirely
    `model_factory`'s job), so there is no model-dimension config for it to
    resolve or override. Every OTHER knob it reads (`CTRL_LR`,
    `CTRL_MAX_EPOCHS`, `CTRL_MAXLEN`, `PATIENCE`, `TRAIN_SEED`, `BATCH`) is
    read fresh from `crucible.latent.config` at call time (`_resolve_config`),
    so a test can still adjust one via `monkeypatch.setattr(config, ...)`.

    Writes `out_dir/probes.jsonl` (one JSON line per epoch: `{epoch,
    val_auroc}` -- this arm's only probe, see `_evaluate`), `out_dir/best.pt`
    (`{state_dict, config}` at the best val AUROC seen, or a final fallback
    save if the run somehow never completes a single epoch), and
    `out_dir/train_summary.json` (`{epochs_run, best_val_auroc,
    stopped_reason: "early_stop"|"max_epochs", wall_s}` -- train.py's own
    summary schema with `steps_run` renamed `epochs_run` and `"max_steps"`
    renamed `"max_epochs"`), returned verbatim.
    """
    corpus_dir = Path(corpus_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = _resolve_config()

    seed = cfg["TRAIN_SEED"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_samples = _load(corpus_dir, "train")
    val_samples = _load(corpus_dir, "val")
    if not train_samples:
        raise ValueError("empty train split -- nothing to train on")

    torch_device = torch.device(device)
    model = model_factory().to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["CTRL_LR"])

    probes_path = out_dir / "probes.jsonl"
    best_ckpt_path = out_dir / "best.pt"
    probes_path.write_text("")  # fresh log for this run

    best_val_auroc = -math.inf
    evals_without_improvement = 0
    stopped_reason = "max_epochs"
    wall_start = time.monotonic()

    epoch = 0
    while epoch < cfg["CTRL_MAX_EPOCHS"]:
        epoch += 1
        model.train()
        order = list(range(len(train_samples)))
        random.shuffle(order)  # GLOBAL random module, seeded from TRAIN_SEED above
        for start in range(0, len(order), cfg["BATCH"]):
            idx = order[start:start + cfg["BATCH"]]
            batch = [train_samples[j] for j in idx]

            optimizer.zero_grad()
            logits, binary_y = _forward_batch(model, tokenizer, batch, torch_device, cfg["CTRL_MAXLEN"])
            loss = F.binary_cross_entropy_with_logits(logits, binary_y)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite loss ({loss.item()!r}) at epoch {epoch} -- "
                    "infra failure, not a training result to continue past"
                )
            loss.backward()
            optimizer.step()

        probe = _evaluate(model, tokenizer, val_samples, torch_device, cfg["BATCH"], cfg["CTRL_MAXLEN"])
        with probes_path.open("a") as f:
            f.write(json.dumps({"epoch": epoch, **probe}) + "\n")

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
        # CTRL_MAX_EPOCHS somehow never reached a single epoch boundary --
        # best.pt must still exist (the module's contract), so save the
        # current (only) model state as the fallback.
        _save_checkpoint(model, cfg, best_ckpt_path)

    summary = {
        "epochs_run": epoch,
        "best_val_auroc": best_val_auroc if math.isfinite(best_val_auroc) else None,
        "stopped_reason": stopped_reason,
        "wall_s": time.monotonic() - wall_start,
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
