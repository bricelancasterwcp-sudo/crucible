"""B-lite: the treatment-arm model + LeWM two-term objective (prereg §3/§5.2).

Pure ``torch.nn`` -- NO ``transformers`` import here. The frozen code encoder
(``jinaai/jina-embeddings-v2-base-code``, revision-pinned, ``requires_grad_(False)``)
lives OUTSIDE this module, in ops/training code; ``BLite`` takes its output,
``code_embed``, as a plain tensor input. That split is deliberate: this file
should be loadable, testable, and mutation-tested with nothing heavier than
torch on the box, and it should be impossible for a change in here to
accidentally start downloading or fine-tuning the frozen encoder.

Three trained modules (prereg §3, "B-lite" arm):

* ``StateEncoder``    -- one execution snapshot's fixed-vocab token ids
                          (crucible.latent.state) -> a D_MODEL-d embedding.
                          Reused for BOTH the call's input encoding
                          (``encode_input``) and each state snapshot
                          (``encode_snapshot``) -- both are token-id
                          sequences over the same fixed 326-token vocabulary.
* ``LatentPredictor``  -- EB-JEPA-shaped causal transformer over
                          ``[z_code, z_input, z_s1..z_sT]`` that predicts each
                          next latent state from everything causally before it.
* ``GroundedHead``     -- final predictor hidden -> the grounded outcome
                          targets (binary clean-return-or-not, plus the
                          descriptive 3-way {pass-return, exception, timeout}).

``BLite`` composes the three. The loss functions (``prediction_loss``,
``isotropy_loss``, ``blite_loss``) implement the LeWM (MIT) two-term
objective -- reimplemented from the paper/its MIT-licensed identifiability
repo, NEVER copied from the CC-BY-NC-licensed reference implementation:
prediction (1 - cosine, masked over valid steps) plus an isotropic-Gaussian
regularizer on the latent batch (SIGReg-style: penalize the batch covariance's
deviation from a scaled identity). Per prereg §3: **no EMA, no stop-grad** --
both branches of every cosine-similarity pair stay fully differentiable.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from crucible.latent.config import (
    D_MODEL,
    LAMBDA_ISO,
    N_OUTCOME_CLASSES,
    PRED_HEADS,
    PRED_LAYERS,
    STATE_ENC_D,
    STATE_ENC_LAYERS,
)


def _no_fully_masked_rows(pad_mask: Tensor) -> Tensor:
    """Guard against an all-True (fully padded) row in a key-padding mask.

    ``nn.MultiheadAttention`` has no valid key to attend to when an entire
    row of ``src_key_padding_mask`` is True, which is a real NaN-producing
    edge case (not hypothetical -- some PyTorch attention backends return
    NaN, not just "undefined", for a fully-masked softmax row). Every real
    snapshot this repo ever emits carries at least one non-PAD token (the
    LINE token, unconditionally first -- see state.py), so a fully-padded
    row should never occur on real data; this guard exists purely so a
    degenerate all-pad input (e.g. a padding row assembled by a caller for
    batch-shape reasons) fails soft -- one unmasked dummy position -- instead
    of propagating NaN through the whole batch.
    """
    all_masked = pad_mask.all(dim=1)
    if not bool(all_masked.any()):
        return pad_mask
    safe = pad_mask.clone()
    safe[all_masked, 0] = False
    return safe


class StateEncoder(nn.Module):
    """Token ids for one snapshot (or the call's input literal) -> D_MODEL-d.

    embedding -> TransformerEncoder (padding-masked) -> masked mean-pool over
    the sequence dim -> Linear(d_state -> d_model). ~20M params at the
    prereg-locked defaults (d_state=512, d_model=768, 4 layers).
    """

    def __init__(
        self,
        vocab_size: int,
        d_state: int = STATE_ENC_D,
        d_model: int = D_MODEL,
        n_layers: int = STATE_ENC_LAYERS,
        n_heads: int = 8,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_state)
        layer = nn.TransformerEncoderLayer(
            d_model=d_state,
            nhead=n_heads,
            dim_feedforward=4 * d_state,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.proj = nn.Linear(d_state, d_model)

    def forward(self, token_ids: Tensor, pad_mask: Tensor) -> Tensor:
        """``(B, L)`` token ids + ``(B, L)`` pad mask (True at PAD positions,
        i.e. positions to IGNORE -- ``nn.TransformerEncoder``'s own
        ``src_key_padding_mask`` convention) -> ``(B, d_model)``.
        """
        safe_pad_mask = _no_fully_masked_rows(pad_mask)
        x = self.embed(token_ids)  # (B, L, d_state)
        h = self.encoder(x, src_key_padding_mask=safe_pad_mask)  # (B, L, d_state)

        valid = (~safe_pad_mask).unsqueeze(-1).to(h.dtype)  # (B, L, 1)
        summed = (h * valid).sum(dim=1)  # (B, d_state)
        counts = valid.sum(dim=1).clamp(min=1.0)  # (B, 1) -- never truly 0, see guard above
        pooled = summed / counts
        return self.proj(pooled)  # (B, d_model)


class LatentPredictor(nn.Module):
    """Causal transformer over ``[z_code, z_input, z_s1..z_sT]``, ``(B, S, d)``.

    Exposes the per-position hidden state -- under the causal mask, hidden
    position ``i`` is exactly the model's predicted embedding for position
    ``i + 1`` (its next-state prediction), computed from positions ``0..i``
    only -- plus ``final_hidden``, the hidden at the last VALID (non-pad)
    position per sample, which is what ``GroundedHead`` grounds against.
    ~100M params at the prereg-locked defaults (d_model=768, 12 layers, 12
    heads -- EB-JEPA shape, prereg §3).
    """

    def __init__(
        self,
        d_model: int = D_MODEL,
        n_layers: int = PRED_LAYERS,
        n_heads: int = PRED_HEADS,
    ) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)

    def forward(self, seq: Tensor, seq_mask: Tensor) -> tuple[Tensor, Tensor]:
        """``seq``: ``(B, S, d_model)``. ``seq_mask``: ``(B, S)`` bool, True at
        VALID (non-pad) positions -- positions 0 (``z_code``) and 1
        (``z_input``) are always valid; state positions are valid up to that
        sample's real snapshot count and False (padding) past it.

        Returns ``(hidden, final_hidden)``:
          * ``hidden``: ``(B, S, d_model)`` -- per-position causal hidden
            state. ``hidden[:, i]`` depends only on ``seq[:, :i+1]``.
          * ``final_hidden``: ``(B, d_model)`` -- ``hidden`` gathered at each
            sample's own last True index in ``seq_mask``.
        """
        B, S, _ = seq.shape
        # Boolean causal mask (True = blocked/"future", matching
        # src_key_padding_mask's own True-means-ignore convention below) --
        # NOT a float additive (-inf) mask. This torch version emits a
        # "Support for mismatched src_key_padding_mask and mask is
        # deprecated" UserWarning when the two masks have different dtypes;
        # keeping both boolean is the documented, warning-free combination.
        # Correctness of the causal masking itself (not just the absence of
        # a warning) is pinned separately by the causality tests below.
        causal_mask = torch.triu(
            torch.ones(S, S, dtype=torch.bool, device=seq.device), diagonal=1
        )
        pad_mask = ~seq_mask
        safe_pad_mask = _no_fully_masked_rows(pad_mask)

        hidden = self.encoder(
            seq, mask=causal_mask, src_key_padding_mask=safe_pad_mask
        )  # (B, S, d_model)

        valid = ~safe_pad_mask  # (B, S)
        # seq_mask always has >=1 True per row (positions 0/1 are never
        # padded), so this index is always in range -- the clamp is a pure
        # defensive floor, not something real inputs should ever hit.
        last_idx = valid.sum(dim=1).clamp(min=1) - 1  # (B,)
        final_hidden = hidden[torch.arange(B, device=seq.device), last_idx]  # (B, d_model)
        return hidden, final_hidden


class GroundedHead(nn.Module):
    """Final predictor hidden -> (binary clean-return logit, 3-way class logits)."""

    def __init__(self, d_model: int = D_MODEL, n_classes: int = N_OUTCOME_CLASSES) -> None:
        super().__init__()
        self.binary = nn.Linear(d_model, 1)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, final_hidden: Tensor) -> tuple[Tensor, Tensor]:
        """``(B, d_model)`` -> ``(binary_logit (B,), class_logits (B, n_classes))``."""
        binary_logit = self.binary(final_hidden).squeeze(-1)
        class_logits = self.classifier(final_hidden)
        return binary_logit, class_logits


class BLite(nn.Module):
    """Bundles ``StateEncoder`` + ``LatentPredictor`` + ``GroundedHead``.

    ``code_embed`` is passed IN -- the frozen encoder that produces it lives
    outside this module entirely (ops/training code); ``BLite`` never
    computes it and never imports anything that could.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = D_MODEL,
        d_state: int = STATE_ENC_D,
        state_enc_layers: int = STATE_ENC_LAYERS,
        state_enc_heads: int = 8,
        pred_layers: int = PRED_LAYERS,
        pred_heads: int = PRED_HEADS,
        n_classes: int = N_OUTCOME_CLASSES,
    ) -> None:
        super().__init__()
        self.state_encoder = StateEncoder(
            vocab_size,
            d_state=d_state,
            d_model=d_model,
            n_layers=state_enc_layers,
            n_heads=state_enc_heads,
        )
        self.predictor = LatentPredictor(d_model=d_model, n_layers=pred_layers, n_heads=pred_heads)
        self.head = GroundedHead(d_model=d_model, n_classes=n_classes)

    def forward(
        self,
        code_embed: Tensor,
        input_ids: Tensor,
        input_pad_mask: Tensor,
        state_ids: Tensor,
        state_pad_mask: Tensor,
        seq_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Args:
            code_embed: ``(B, d_model)`` -- frozen encoder output, computed
                and (per prereg §3) ``requires_grad_(False)``-frozen OUTSIDE
                this module; used here exactly as given.
            input_ids, input_pad_mask: ``(B, L_in)`` -- ``encode_input``'s
                token ids + PAD mask (True = PAD).
            state_ids, state_pad_mask: ``(B, T, L_state)`` -- ``T`` snapshots
                per sample (``encode_state_sequence``, padded to a common
                ``T`` across the batch), each ``(token ids, PAD mask)``.
            seq_mask: ``(B, T + 2)`` bool, True at VALID positions of the
                predictor's own ``[z_code, z_input, z_s1..z_sT]`` sequence --
                positions 0/1 always True; state positions True up to that
                sample's real snapshot count.

        Returns:
            pred_states: ``(B, T, d_model)`` -- predicted embeddings for
                ``z_s1..z_sT``, one per snapshot position.
            target_states: ``(B, T, d_model)`` -- the actual
                (StateEncoder-encoded) ``z_s1..z_sT``, aligned 1:1 with
                ``pred_states``. Per prereg §3 (no EMA, no stop-grad) this
                stays fully differentiable -- it is NOT detached.
            valid_mask: ``(B, T)`` bool -- True where that snapshot position
                is real (not batch padding); feed straight to
                ``prediction_loss``.
            binary_logit: ``(B,)``.
            class_logits: ``(B, n_classes)``.
        """
        B, T, L_state = state_ids.shape

        z_input = self.state_encoder(input_ids, input_pad_mask)  # (B, d_model)

        flat_ids = state_ids.reshape(B * T, L_state)
        flat_pad = state_pad_mask.reshape(B * T, L_state)
        z_states = self.state_encoder(flat_ids, flat_pad).reshape(B, T, -1)  # (B, T, d_model)

        seq = torch.cat([code_embed.unsqueeze(1), z_input.unsqueeze(1), z_states], dim=1)  # (B, T+2, d)
        hidden, final_hidden = self.predictor(seq, seq_mask)

        pred_states = hidden[:, 1:-1, :]  # (B, T, d) -- hidden[1..T] predicts seq[2..T+1]
        target_states = seq[:, 2:, :]  # (B, T, d) -- the actual z_s1..z_sT
        valid_mask = seq_mask[:, 2:]  # (B, T)

        binary_logit, class_logits = self.head(final_hidden)
        return pred_states, target_states, valid_mask, binary_logit, class_logits


def prediction_loss(pred: Tensor, target: Tensor, valid_mask: Tensor) -> Tensor:
    """LeWM prediction term: ``1 - cosine_similarity`` per step, masked mean.

    ``pred``, ``target``: ``(B, T, d)``. ``valid_mask``: ``(B, T)`` bool,
    True where that step counts. Identical vectors -> 0 per step; opposite
    (antiparallel) vectors -> 2 per step (cosine = -1) -- the sign convention
    a ``1 + cosine`` mutant would silently invert.

    Zero valid steps overall -> ``tensor(0.)``, never NaN: the masked mean's
    denominator is clamped to >= 1 and the whole result is then multiplied by
    a 0/1 "any valid steps at all" indicator, rather than branching on
    ``count == 0`` and dividing by it directly.
    """
    cos = F.cosine_similarity(pred, target, dim=-1)  # (B, T), in [-1, 1]
    per_step = 1.0 - cos
    mask = valid_mask.to(per_step.dtype)
    total_valid = mask.sum()
    denom = total_valid.clamp(min=1.0)
    masked_mean = (per_step * mask).sum() / denom
    return masked_mean * (total_valid > 0).to(masked_mean.dtype)


def isotropy_loss(z: Tensor) -> Tensor:
    """LeWM/SIGReg-style collapse-prevention term (prereg §3).

    ``z``: ``(B, d)``. Mean-center over the batch, form the (unbiased)
    sample covariance ``C``, and penalize its squared Frobenius deviation
    from the nearest scaled identity ``(tr(C) / d) * I`` -- exactly zero for
    an isotropic (equal-variance-in-every-direction) batch, and large for a
    collapsed (low-rank) one, since a low-rank ``C`` cannot be close to any
    full-rank scaled identity.

    ``B < 2`` -> ``tensor(0.)`` (an unbiased covariance is undefined at
    ``B <= 1``; a single centered point is identically the zero vector
    regardless, so there is nothing to measure).
    """
    B, d = z.shape
    if B < 2:
        return torch.zeros((), dtype=z.dtype, device=z.device)
    z_centered = z - z.mean(dim=0, keepdim=True)
    cov = (z_centered.T @ z_centered) / (B - 1)  # (d, d)
    scale = cov.diagonal().sum() / d  # tr(C) / d
    eye = torch.eye(d, dtype=z.dtype, device=z.device)
    diff = cov - scale * eye
    return (diff * diff).sum() / d


def blite_loss(
    pred: Tensor,
    target: Tensor,
    valid_mask: Tensor,
    z_for_iso: Tensor,
    binary_logit: Tensor,
    binary_y: Tensor,
    class_logits: Tensor,
    class_y: Tensor,
    lambda_iso: float = LAMBDA_ISO,
) -> tuple[Tensor, dict[str, float]]:
    """LeWM two-term latent objective + grounded outcome supervision.

    ``total = prediction_loss + lambda_iso * isotropy_loss``
              ``+ BCE(binary_logit, binary_y)`` (primary grounded target)
              ``+ 0.5 * CE(class_logits, class_y)`` (descriptive/aux target,
              half-weighted -- prereg §4: the binary clean-return-or-not
              split is what the head is graded on; the 3-way outcome is
              reported descriptively, not gated).

    Returns ``(total, parts)`` where ``parts`` holds each term (and the
    total) as a plain detached float, for cheap logging without holding
    graph references.
    """
    pred_l = prediction_loss(pred, target, valid_mask)
    iso_l = isotropy_loss(z_for_iso)
    binary_l = F.binary_cross_entropy_with_logits(binary_logit, binary_y.to(binary_logit.dtype))
    class_l = F.cross_entropy(class_logits, class_y.long())

    total = pred_l + lambda_iso * iso_l + binary_l + 0.5 * class_l

    parts = {
        "prediction": float(pred_l.detach()),
        "isotropy": float(iso_l.detach()),
        "binary": float(binary_l.detach()),
        "class": float(class_l.detach()),
        "total": float(total.detach()),
    }
    return total, parts
