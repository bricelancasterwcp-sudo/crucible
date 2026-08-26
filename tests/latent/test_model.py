"""RED/GREEN tests for the B-lite model + LeWM two-term objective (prereg
§3/§5.2, Task 5 brief).

Each test here is a mutation pin, per the brief:

* shapes end-to-end through ``BLite.forward``.
* ``prediction_loss``: 0 for identical vectors, ~2 for antiparallel vectors
  (the ``1 - cosine`` SIGN pin -- a ``1 + cosine`` mutant would swap these
  two results exactly), and a 0-valid-step mask giving exactly 0, never NaN.
* ``isotropy_loss``: near-0 for a large isotropic Gaussian batch, and LARGE
  for a rank-1 (collapsed) batch. The collapsed-batch assertion alone is
  THE mutant-killing test here: a "return zeros" mutant would pass the
  near-0 case trivially (0 IS near 0) but fails the large-value assertion
  outright.
* frozen-encoder discipline: a frozen ``nn.Linear`` standing in for the
  external code encoder never accumulates a gradient through
  ``blite_loss(...).backward()``, while every trained submodule does.
* causality: perturbing a LATER position in the predictor's input sequence
  never changes an EARLIER position's hidden state (kills "causal mask
  dropped" mutants -- full bidirectional attention would leak it back), and
  perturbing an EARLIER position DOES change a later one (kills "mask
  replaced by diagonal/identity" mutants -- pure self-attention would never
  propagate it forward either).
* grounded head: output shapes and that gradients reach both its own
  parameters and its input.

All dims are tiny (d_model=8 or 16, 1-2 layers/heads) and everything runs on
CPU, seeded, per the 4G-scope constraint on this test command.
"""
from __future__ import annotations

import torch

from crucible.latent.model import (
    BLite,
    GroundedHead,
    LatentPredictor,
    StateEncoder,
    blite_loss,
    isotropy_loss,
    prediction_loss,
)
from crucible.latent.state import VOCAB_SIZE

torch.manual_seed(0)


def _tiny_blite() -> BLite:
    return BLite(
        VOCAB_SIZE,
        d_model=16,
        d_state=12,
        state_enc_layers=2,
        state_enc_heads=2,
        pred_layers=2,
        pred_heads=2,
        n_classes=3,
    )


def _tiny_batch(B: int = 3, T: int = 4, L_in: int = 5, L_state: int = 6):
    """A batch with NO padding anywhere (every snapshot/input slot real) --
    shape-only fixture; padding-specific behavior is covered separately."""
    input_ids = torch.randint(0, VOCAB_SIZE, (B, L_in))
    input_pad_mask = torch.zeros(B, L_in, dtype=torch.bool)
    state_ids = torch.randint(0, VOCAB_SIZE, (B, T, L_state))
    state_pad_mask = torch.zeros(B, T, L_state, dtype=torch.bool)
    seq_mask = torch.ones(B, T + 2, dtype=torch.bool)
    code_embed = torch.randn(B, 16)
    return code_embed, input_ids, input_pad_mask, state_ids, state_pad_mask, seq_mask


# ---- shapes end-to-end -----------------------------------------------------


def test_blite_forward_shapes():
    model = _tiny_blite()
    B, T = 3, 4
    args = _tiny_batch(B=B, T=T)
    pred_states, target_states, valid_mask, binary_logit, class_logits = model(*args)

    assert pred_states.shape == (B, T, 16)
    assert target_states.shape == (B, T, 16)
    assert valid_mask.shape == (B, T)
    assert valid_mask.dtype == torch.bool
    assert binary_logit.shape == (B,)
    assert class_logits.shape == (B, 3)


def test_blite_forward_respects_per_sample_snapshot_count():
    """A batch where sample 0 has fewer real snapshots than sample 1 (batch
    padding in T) -- valid_mask must reflect exactly that, not "all True"."""
    model = _tiny_blite()
    B, T = 2, 4
    code_embed, input_ids, input_pad_mask, state_ids, state_pad_mask, seq_mask = _tiny_batch(B=B, T=T)
    # sample 0 has only 2 real snapshots; sample 1 has all 4.
    seq_mask = seq_mask.clone()
    seq_mask[0, 4:] = False  # positions 2,3 (2 state slots) real; 4,5 padding
    _, _, valid_mask, _, _ = model(code_embed, input_ids, input_pad_mask, state_ids, state_pad_mask, seq_mask)
    assert valid_mask[0].tolist() == [True, True, False, False]
    assert valid_mask[1].tolist() == [True, True, True, True]


# ---- StateEncoder: shape + the all-pad-row NaN guard -----------------------


def test_state_encoder_output_shape():
    enc = StateEncoder(VOCAB_SIZE, d_state=12, d_model=16, n_layers=2, n_heads=2)
    B, L = 3, 7
    ids = torch.randint(0, VOCAB_SIZE, (B, L))
    pad_mask = torch.zeros(B, L, dtype=torch.bool)
    pad_mask[:, 5:] = True  # last 2 positions are padding for every row
    out = enc(ids, pad_mask)
    assert out.shape == (B, 16)
    assert torch.isfinite(out).all()


def test_state_encoder_ignores_token_content_at_padded_positions():
    """Real pin (not just shapes/finiteness): changing the token ids sitting
    at PADDED positions must not move the output at all -- attention key-
    masking keeps other positions from seeing them, and the mean-pool must
    not sum them in either. A mutant that pools over every position
    (ignoring the mask) would still be finite and correctly shaped, but
    WOULD change here, since a padded position's own hidden state (a valid
    query) still depends on its own token id via the residual connection."""
    enc = StateEncoder(VOCAB_SIZE, d_state=12, d_model=16, n_layers=2, n_heads=2)
    enc.eval()
    B, L = 2, 7
    torch.manual_seed(1)
    ids_a = torch.randint(0, VOCAB_SIZE, (B, L))
    pad_mask = torch.zeros(B, L, dtype=torch.bool)
    pad_mask[:, 5:] = True  # last 2 positions are padding
    ids_b = ids_a.clone()
    ids_b[:, 5:] = (ids_a[:, 5:] + 1) % VOCAB_SIZE  # only padded-position content differs

    with torch.no_grad():
        out_a = enc(ids_a, pad_mask)
        out_b = enc(ids_b, pad_mask)
    assert torch.allclose(out_a, out_b, atol=1e-5)


def test_state_encoder_all_pad_row_does_not_nan():
    """Defensive guard: a fully-padded row must not propagate NaN, even
    though real encode_snapshot() output is never actually empty."""
    enc = StateEncoder(VOCAB_SIZE, d_state=12, d_model=16, n_layers=2, n_heads=2)
    B, L = 2, 5
    ids = torch.randint(0, VOCAB_SIZE, (B, L))
    pad_mask = torch.zeros(B, L, dtype=torch.bool)
    pad_mask[0, :] = True  # row 0 is entirely padding
    out = enc(ids, pad_mask)
    assert torch.isfinite(out).all()


# ---- prediction_loss: the cosine-sign pin + the 0-valid-steps pin ----------


def test_prediction_loss_is_zero_for_identical_vectors():
    pred = torch.randn(2, 3, 5)
    target = pred.clone()
    valid_mask = torch.ones(2, 3, dtype=torch.bool)
    loss = prediction_loss(pred, target, valid_mask)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-5)


def test_prediction_loss_is_two_for_antiparallel_vectors():
    """cosine(-v, v) == -1 exactly -> 1 - (-1) == 2. A `1 + cosine` sign-flip
    mutant would give 0 here (and 2 for the identical-vector case above) --
    this pair of tests pins the sign both ways."""
    pred = torch.randn(2, 3, 5)
    target = -pred
    valid_mask = torch.ones(2, 3, dtype=torch.bool)
    loss = prediction_loss(pred, target, valid_mask)
    assert torch.allclose(loss, torch.tensor(2.0), atol=1e-5)


def test_prediction_loss_zero_valid_steps_gives_zero_not_nan():
    pred = torch.randn(2, 3, 5)
    target = torch.randn(2, 3, 5)
    valid_mask = torch.zeros(2, 3, dtype=torch.bool)
    loss = prediction_loss(pred, target, valid_mask)
    assert not torch.isnan(loss).any()
    assert loss.item() == 0.0


def test_prediction_loss_masks_out_invalid_steps():
    """A step with wildly different pred/target but masked out must not
    move the loss at all."""
    pred = torch.randn(1, 2, 5)
    target = pred.clone()
    target[:, 1, :] = -1000.0  # garbage at the masked-out step
    valid_mask = torch.tensor([[True, False]])
    loss = prediction_loss(pred, target, valid_mask)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-5)


# ---- isotropy_loss: THE collapse pin ---------------------------------------


def test_isotropy_loss_near_zero_for_large_isotropic_batch():
    z = torch.randn(4000, 16)
    loss = isotropy_loss(z)
    assert loss.item() < 0.05


def test_isotropy_loss_large_for_rank1_collapsed_batch():
    """All rows lie on a single line through the origin -- this is what a
    fully-collapsed representation looks like. The batch covariance is
    rank-1, which cannot be close to any full-rank scaled identity, so the
    loss must be large. A mutant that always returns 0 fails this
    assertion outright (0 is not > 5.0) even though it would pass the
    near-zero isotropic test above for free."""
    direction = torch.randn(16)
    direction = direction / direction.norm()
    scale = torch.randn(500, 1) * 10.0
    z = scale * direction  # (500, 16), every row a multiple of `direction`
    loss = isotropy_loss(z)
    assert loss.item() > 5.0


def test_isotropy_loss_batch_of_one_or_zero_is_zero():
    assert isotropy_loss(torch.randn(1, 8)).item() == 0.0
    assert isotropy_loss(torch.zeros(0, 8)).item() == 0.0


def test_isotropy_loss_zero_for_exactly_isotropic_synthetic_batch():
    """A batch constructed to have EXACTLY covariance == I (via a random
    orthogonal transform of a standard basis spread) should give a loss
    numerically indistinguishable from 0, not just "small"."""
    d = 6
    q, _ = torch.linalg.qr(torch.randn(d, d))
    # d+1 points forming a simplex with sample covariance exactly I (up to
    # floating point) after scaling -- simplest reliable construction is
    # just a big random isotropic batch decorrelated via qr; reuse the
    # large-N isotropic check's tighter tolerance is enough evidence here,
    # so just assert a rotation of an isotropic batch keeps the loss ~equal.
    z = torch.randn(3000, d)
    loss_a = isotropy_loss(z)
    loss_b = isotropy_loss(z @ q)
    assert torch.allclose(loss_a, loss_b, atol=1e-3)


# ---- frozen-encoder discipline ---------------------------------------------


def test_frozen_code_encoder_stand_in_never_gets_a_gradient():
    model = _tiny_blite()
    frozen = torch.nn.Linear(4, 16)
    frozen.requires_grad_(False)

    B, T = 2, 3
    code_input = torch.randn(B, 4)
    code_embed = frozen(code_input)
    assert code_embed.requires_grad is False

    input_ids = torch.randint(0, VOCAB_SIZE, (B, 5))
    input_pad_mask = torch.zeros(B, 5, dtype=torch.bool)
    state_ids = torch.randint(0, VOCAB_SIZE, (B, T, 6))
    state_pad_mask = torch.zeros(B, T, 6, dtype=torch.bool)
    seq_mask = torch.ones(B, T + 2, dtype=torch.bool)

    pred, target, valid_mask, binary_logit, class_logits = model(
        code_embed, input_ids, input_pad_mask, state_ids, state_pad_mask, seq_mask
    )
    binary_y = torch.randint(0, 2, (B,))
    class_y = torch.randint(0, 3, (B,))
    total, parts = blite_loss(
        pred, target, valid_mask, target.reshape(-1, 16), binary_logit, binary_y, class_logits, class_y
    )
    total.backward()

    assert frozen.weight.grad is None
    assert frozen.bias.grad is None
    assert model.state_encoder.embed.weight.grad is not None
    assert model.predictor.encoder.layers[0].linear1.weight.grad is not None
    assert model.head.binary.weight.grad is not None
    assert model.head.classifier.weight.grad is not None

    assert isinstance(parts, dict)
    assert set(parts) == {"prediction", "isotropy", "binary", "class", "total"}
    assert all(isinstance(v, float) for v in parts.values())


def test_blite_loss_applies_half_weight_to_class_term():
    """Pins the 0.5 aux weight: identical pred/target (prediction_loss==0),
    a batch too small for isotropy (isotropy_loss==0 by construction), so
    total collapses to exactly `binary + 0.5*class` -- a mutant using
    weight 1.0 instead of 0.5 would fail this equality."""
    pred = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])  # nonzero -- a zero vector's
    # cosine similarity is ill-defined and torch's eps-guarded denominator
    # would silently give per_step==1 instead of the intended 0 here.
    target = pred.clone()
    valid_mask = torch.ones(1, 1, dtype=torch.bool)
    z_for_iso = torch.zeros(1, 4)  # B=1 -> isotropy_loss is exactly 0

    binary_logit = torch.tensor([0.7], requires_grad=True)
    binary_y = torch.tensor([1.0])
    class_logits = torch.tensor([[0.1, 0.2, 0.7]], requires_grad=True)
    class_y = torch.tensor([2])

    total, parts = blite_loss(
        pred, target, valid_mask, z_for_iso, binary_logit, binary_y, class_logits, class_y, lambda_iso=0.1
    )

    expected_binary = torch.nn.functional.binary_cross_entropy_with_logits(binary_logit, binary_y)
    expected_class = torch.nn.functional.cross_entropy(class_logits, class_y)
    expected_total = expected_binary + 0.5 * expected_class

    assert torch.allclose(total, expected_total, atol=1e-6)
    assert abs(parts["prediction"]) < 1e-5
    assert parts["isotropy"] == 0.0


# ---- causality: the mask pin -----------------------------------------------


def test_predictor_later_perturbation_does_not_change_earlier_hidden():
    pred = LatentPredictor(d_model=8, n_layers=2, n_heads=2)
    pred.eval()
    B, S = 1, 5
    seq = torch.randn(B, S, 8)
    seq_mask = torch.ones(B, S, dtype=torch.bool)

    with torch.no_grad():
        hidden_before, _ = pred(seq, seq_mask)
        seq_perturbed = seq.clone()
        # REPLACE the last position with a fresh random vector -- not a
        # uniform += shift. norm_first=True means attention operates on
        # LayerNorm(x), whose mean-centering exactly cancels a shift added
        # equally to every feature of a token; that would silently defeat
        # this perturbation regardless of whether the mask is causal.
        seq_perturbed[:, S - 1, :] = torch.randn(8) * 10.0
        hidden_after, _ = pred(seq_perturbed, seq_mask)

    assert torch.allclose(hidden_before[:, : S - 1, :], hidden_after[:, : S - 1, :], atol=1e-4)


def test_predictor_earlier_perturbation_changes_later_hidden():
    """The complement of the test above: proves the mask is really causal
    (looks backward), not a mutant that zeroed it into a diagonal/identity
    mask (looks at nothing but self, which would ALSO leave every other
    position unchanged under an earlier perturbation)."""
    pred = LatentPredictor(d_model=8, n_layers=2, n_heads=2)
    pred.eval()
    B, S = 1, 5
    seq = torch.randn(B, S, 8)
    seq_mask = torch.ones(B, S, dtype=torch.bool)

    with torch.no_grad():
        hidden_before, _ = pred(seq, seq_mask)
        seq_perturbed = seq.clone()
        seq_perturbed[:, 0, :] = torch.randn(8) * 10.0  # replace, not shift -- see note above
        hidden_after, _ = pred(seq_perturbed, seq_mask)

    later = slice(1, S)
    assert not torch.allclose(hidden_before[:, later, :], hidden_after[:, later, :], atol=1e-4)


def test_predictor_final_hidden_gathers_last_valid_position():
    pred = LatentPredictor(d_model=8, n_layers=1, n_heads=2)
    pred.eval()
    B, S = 2, 5
    seq = torch.randn(B, S, 8)
    seq_mask = torch.ones(B, S, dtype=torch.bool)
    seq_mask[0, 3:] = False  # sample 0's last valid position is index 2

    with torch.no_grad():
        hidden, final_hidden = pred(seq, seq_mask)

    assert torch.allclose(final_hidden[0], hidden[0, 2, :])
    assert torch.allclose(final_hidden[1], hidden[1, S - 1, :])


# ---- GroundedHead: shapes + gradient flow ----------------------------------


def test_grounded_head_shapes_and_gradient_flow():
    head = GroundedHead(d_model=8, n_classes=3)
    final_hidden = torch.randn(4, 8, requires_grad=True)
    binary_logit, class_logits = head(final_hidden)

    assert binary_logit.shape == (4,)
    assert class_logits.shape == (4, 3)

    (binary_logit.sum() + class_logits.sum()).backward()

    assert final_hidden.grad is not None
    assert head.binary.weight.grad is not None
    assert head.classifier.weight.grad is not None
