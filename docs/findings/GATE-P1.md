# Pillar-1 B-lite gating result — pre-registered verdict: **NO-GO** (Pillar 1 stops)

**Date:** 2026-08-27. **Lock:** `prereg-lock-blite` (446d4b8). P1 computed exactly as
locked, in ONE `evaluate_gate` read of the 6,063-outcome test split (sentinel + report
now refuse any second read). Records: `runs/blite-gate/` + `runs/blite-corpus/`
(git-ignored); corpus and configs as recorded in LOCK-BLITE.

## 1. The gate

| | AUROC (test, n=6,063) | |
|---|---|---|
| **B-lite** (latent unroll from (code, input) only) | **0.99981** | 120M trained params, 66 min on the pod |
| **Control** (microsoft/codeexecutor + head, token space) | **0.99998** | 125M, 22 min |
| **P1: diff ≥ 2·SE?** | diff = **−1.67×10⁻⁴**, paired DeLong SE = 4.1×10⁻⁵, **z = −4.07**; bootstrap CI [−2.6×10⁻⁴, −9.4×10⁻⁵] | **FAIL → NO-GO** |

The control is not merely non-inferior — it is statistically significantly better. Per
the §13 pre-commitment carried into §6: **B stops.**

## 2. Supporting endpoints

- **P2 (floors):** both arms beat the static floor (0.5184) by ~+0.48 with SE ≈ 0.008 —
  both models genuinely learned the task; the gate was exercisable. §8 control-vs-floor
  VAL check: 0.9998 vs 0.5144 (**PASS**; run post-lock due to a sequencing slip,
  pre-any-test-read — disclosed).
- **P3 (where B-lite loses):** on exception samples the arms are near-tied (0.9976 vs
  0.9983); on **clean-return samples the control is PERFECT (1.0000) and B-lite is
  0.9631** — the latent rollout occasionally cannot tell a working function from a
  crasher, while the token model reads it off the text. ECE: control 0.0012, B-lite
  0.0084 (both well-calibrated).
- Training: B-lite early-stopped at 11,500/20,000 steps (best val 0.99947); control ran
  its 5 epochs (val 0.99980). Collapse probes stayed healthy throughout — the LeWM
  two-term objective held without EMA or stop-grad at 120M params on real data.

## 3. Honest reading (against the §9 prior)

The founding hypothesis was **not refuted as impossible — it was outcompeted.** A 120M
non-token predictor, unrolling its own latent dynamics with no access to any trace,
reached AUROC 0.9998 on held-out functions: latent execution intuition *works*. But the
same-scale token-space model is near-oracle on this corpus, because the battery-balanced
outcome distribution (TypeError-dominated, per LOCK-BLITE's disclosure) is largely
readable from surface text — exactly the control's home turf, and exactly what §9
predicted ("the cheap baseline is very strong"). The pre-registered bar was "beat the
token model or stop"; the point estimate decides; B stops. A future revival (new prereg
only) would need a corpus whose outcomes are NOT surface-readable — semantic bugs, not
type crashes — which is where a latent path could in principle separate.

## 4. Environment & cost

Both arms trained AND scored on one RunPod pod (`egz3gwbzkhvygd`, RTX 3090 24 GB,
SECURE, `runpod/pytorch:1.1.0-cu1290-torch291-ubuntu2404`, torch 2.9.1+cu129,
**transformers pinned 4.49.0** — jina's pinned remote code is incompatible with tf 5.x;
a first control run under tf 5.16.1 was discarded and retrained under 4.49.0 so both
arms share one environment, per the lock's fairness rule). Offline HF mode with
snapshot-shipped pinned revisions. Checkpoint sha256/12: B-lite `f91b7afb5d01`,
control `d2115f6a6cf5`. **Pod cost $2.97** (cap $8). Local GPU: ~7 h of 1.5B serving for
corpus generation. Sample provenance chain (replay-corrupt archive → reharvest →
battery) as per LOCK-BLITE.
