# Crucible Pillar-1 pre-registration — B-lite: a latent execution predictor vs a token-space control

**Date authored:** 2026-08-25. **Status:** DRAFT until tagged `prereg-lock-blite`; after
the tag, §6–§8 freeze and only the §12 amendment protocol may touch them.

## 1. Provenance — what licenses this spike

This is the program's founding pillar, and the one part it never tested. The Phase-A
prereg named Pillar 1 (a non-token latent "intuition" predictor) as the pre-committed
follow-up and gated it behind a GO that never came; Phases A–C tested the memory pillar
instead and closed per GATE-C §6. Brice ruled (2026-08-25): the non-LLM intention was
the point — write up and run the skipped pillar now, as its own pre-registered spike
outside the closed A–C gate chain. Design basis: `docs/research/05-latent-predictors-
world-models.md` (survey 2026-08-23, every license verified by command) and the Phase-A
prereg §13 pre-commitment, including its mandatory control and its kill rule.

## 2. Question

Every 2025–26 execution-reasoning system narrates program behavior **as tokens**; the
survey found zero published latent-space execution predictors for code. Can a small
(~100–150M) **non-token** predictor — frozen code encoder, trained state encoder,
action-conditioned latent dynamics, grounded outcome head — predict the *observable
outcome* of running code better than the best same-scale token-space model? If a latent
"intuition" cannot beat a 125M token model at outcome prediction, it has not earned a
place beside any proposer, and Pillar 1 stops (§13's own rule).

## 3. Arms

| Arm | What it is | Params (trained) | License basis |
|---|---|---|---|
| **B-lite** (treatment) | frozen `jinaai/jina-embeddings-v2-base-code` (137M, apache-2.0) + trained state encoder (~20M) + action-conditioned latent predictor (~100M, d≈768, ~12 layers, EB-JEPA shape) + grounded outcome head | ~120M | LeWM (MIT) two-term objective — prediction + isotropic-Gaussian regularizer, **no EMA, no stop-grad**; SIGReg reimplemented from arXiv:2511.08544 or lifted from `klindtlab/lejepa-identifiability` (MIT) — **never** from `lejepa` (CC-BY-NC) |
| **Control** (mandatory, §13) | `microsoft/codeexecutor` (125M, MIT weights) with a classification head, fine-tuned on the SAME training split for the SAME outcome targets | ~125M | MIT |
| Floors (non-gating disclosure) | majority-class predictor; logistic regression on static features (code length, AST node count) | — | — |

Both arms train on identical data, with identical early-stopping protocol (validation
split), and are evaluated **once** on the same held-out test outcomes. Same-scale is the
point: the control is the strongest permissively-licensed token-space model at ~125M.

## 4. Corpus (built pre-lock, floors are kill criteria)

CRUXEval-style generation (MIT pipeline blueprint): the Phase-A frozen 1.5B proposer
generates short self-contained functions + inputs → execute → filter (short, low-memory,
**deterministic** — screened by `sensorium refocus` MATCH verdicts) → record traces with
**sensorium** (PEP 669; local project) — captures per-line locals for focused code,
marks-and-counts truncation instead of silently dropping, and `sensorium diff` provides
edit-effect labels. Outcome label per (function, input): {pass-return, exception-type,
timeout} reduced to the binary the head predicts (clean-return vs not) plus the
multiclass reported descriptively.

- **Targets (chosen, sanity-checked, not derived):** 5,000 accepted functions × ≥3
  inputs ≈ 15,000 (code, input, outcome) samples. **Floor: 3,000 functions** — below it
  after the budgeted generation window, STOP pre-lock (kill criterion), report.
- **Split by FUNCTION, never by input** (leakage guard): 80/10/10 train/val/test. Test
  is read exactly once, at the gate.
- Class balance is measured and disclosed at lock; if the binary outcome is more skewed
  than 80/20, generation continues with rejection sampling on the majority class until
  balance or the window closes (recorded either way).

## 5. Instrument (built pre-lock, mutation-tested like all crucible instruments)

1. Harvest driver + exporter: sensorium runs → `(code_span, input, state_snapshots,
   outcome)` tensors + a manifest (per-run SQLite → one corpus manifest; truncated or
   REFUSED-determinism samples excluded and counted).
2. B-lite model + training harness (bf16, batch ≈64, ~6 GB peak — fits the 16 GB box
   with the encoder frozen).
3. Control fine-tune harness (same data loader, same early-stop rule).
4. Evaluation: **paired DeLong** AUROC comparison on the shared test outcomes (paired —
   the arms score the same items), plus bootstrap CI as a cross-check.
5. Collapse probes, logged every eval interval: latent per-dimension std, effective
   rank, and val grounded-head AUROC. These are *disclosures and diagnostics*; smooth
   collapse is a RESULT (it fails P1 honestly), not a confound.

## 6. Endpoints (frozen at lock)

- **P1 (gating, §13 verbatim):** AUROC_B-lite − AUROC_control ≥ **2·SE_diff** (paired
  DeLong SE) on the held-out test outcomes. Pass → GO_P1. Fail → **NO-GO: Pillar 1
  stops**, per the pre-commitment's own words.
- **P2 (non-gating):** both arms vs the floors (each must beat majority + static
  logistic by 2·SE to count as having learned *anything*; a control that fails this
  makes P1 vacuous — reported plainly, verdict CONFOUNDED-by-weak-control and the gate
  is not exercisable).
- **P3 (non-gating):** multiclass outcome breakdown; calibration (ECE); collapse-probe
  trajectories; wall-clock + VRAM for both arms.
- **CONFOUNDED** (instrument only): NaN/divergence in training (fix infra, clean rerun —
  R-S4-1 discipline), data-pipeline defect discovered post-lock, test-split
  contamination, control fails P2.

## 7. Verdict rule (frozen at lock)

GO_P1 → the latent path earned a Phase-2 design (integration with search — LightZero
MCTS per the survey — as a NEW prereg). NO-GO → Pillar 1 closes; the program's full
write-up records that the founding hypothesis was tested and failed at this scale, with
the honest risk register (§9) as the prior. **The point estimate decides. No extension,
re-run, re-split, threshold change, or added arm after any test-set number is read.
Findings ship either way.**

## 8. Kill criteria

- Corpus floor (§4) missed → stop pre-lock.
- Determinism screening rejects > 40% of generated samples → the corpus recipe is
  broken; stop pre-lock, redesign.
- Control fine-tune cannot beat the floors on VAL (before any test read) → the control
  harness is defective; fix before lock, else stop.
- Two consecutive infra-killed training runs after fixes → stop, report.

## 9. Honest prior (from the survey, recorded so the verdict is read against it)

No published latent execution predictor exists to inherit; program state is discrete
and adversarial to smooth latents (the off-by-one problem); CWM needed 32B params for
token-space trace competence and a 150M latent model is not a small CWM; collapse is
quiet and the default. The control is strong and cheap. **The pre-registered expectation
is that this is hard and P1 may well fail** — the spike's value is a clean, licensed,
reproducible first measurement either way.

## 10. Ops

All crucible run discipline applies (detached runs + marker-or-death monitor for
anything > 30 min, fresh dirs, never resume a gating training run — archive and rerun,
secret-scan before push, sync verify after). GPU: corpus generation ~2–4 h (1.5B served,
vLLM n-batched); B-lite training ≤ ~4 h at 6 GB; control fine-tune ≤ ~1 h. Everything
local to the 16 GB box.

## 11. Lock record (`prereg-lock-blite`)

Tag lands on the commit recording: this document's final text; corpus manifest hash,
counts, class balance, determinism-rejection rate; split assignment hash; encoder/control
revisions + weight digests; the exact training configs (steps, batch, lr, early-stop
rule) for both arms; SE method (DeLong, paired); library versions.

## 12. Amendments

Pre-lock: edit with date + old value, mark the line. Post-lock: §6–§8 immutable.

*(none yet)*
