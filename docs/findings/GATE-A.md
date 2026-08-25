# Phase-A gating runs — pre-registered verdict: **PARTIAL** (Brice rules)

**Date:** 2026-08-25. **Lock:** `prereg-lock-a` (6bc3da0). All §6 endpoints computed
exactly as written after all arms completed; no number was read before its arm finished,
nothing was re-run after a number was read. Records under `runs/gate-*/` (git-ignored;
`lens.json` written per arm per §12).

## 1. Runs

| Arm | Result | Wall | Notes |
|---|---|---|---|
| A_noMem | 450/450, **0 infra** | 79 min | first, per §4.8 ordering |
| A_full (attempt 1) | **infra kill** at task 184 | 39 min | HTTP 400: memory-augmented refinement prompt exceeded the 8192-token serve window (probe-confirmed). Serve window → 16384 (native 32k; sampling-neutral, KV pool peaked 1.4%). R-S4-1: discarded, clean rerun. Archived `runs/gate-a-full-infra1/` |
| A_full (rerun) | 450/450, **0 infra** | 97 min | 14 sleeps, **14/14 accepted**, all hot-loaded live |
| B_search | 450/450, **0 infra** | 37 min | after a 10-task §8 preflight (5.8 s/task) |
| B_naive | 450/450, **0 infra** | 29 min | non-gating (E6) |

A_noMem's measurement stands across the window change: it completed 450/450 with zero
infra at 8192, so no request it made was affected; the window is pure KV capacity.

## 2. Endpoints (Δ_min = 0.0884, derived at lock)

| ID | Value | Bar | Verdict |
|---|---|---|---|
| **E1** memory works | Δ_A = succ(A_full, second) − succ(A_full, phase-1) = 0.5750 − 0.4350 = **+0.1400**; control Δ = 0.4200 − 0.4250 = **−0.0050** | Δ_A ≥ 0.0884 ∧ control ∈ ±0.0884 | **PASS** |
| **E2a** beats big on repeats | A_full second 0.5750 vs B_search second 0.7850 | ≥ | **FAIL** |
| **E2b** competitive on new | A_full p1∪novel 0.4360 vs B_search 0.7720 − 0.0884 = 0.6836 | ≥ | **FAIL** |
| **E3a** uncertainty informative | AUROC **0.5880** (n+ = 224, n− = 226); bar 0.5 + 2·SE_HM = 0.5535 | ≥ | **PASS** |
| **E3b** abstention honest | abstained-task failure rate **0.9787** (46/47); bar = 2 × overall failure 0.5022 = **1.0044** | ≥ | **FAIL** (bar > 1.0 — see §4) |
| E4 (non-gating) | QuixBugs held-out | — | **not run** |
| E5 (non-gating, report) | A_full novel − A_noMem novel = 0.4400 − 0.5000 = **−0.0600** | — | reported |
| E6 (non-gating, report) | B_search 0.7778 vs B_naive 0.7222 overall (+5.6 pp for search) | — | reported |

**§7 verdict: PARTIAL** — E1 passes with the control clean, E2a/E2b/E3b fail.
Per the pre-registration: *Brice rules; I do not.* Nothing was extended, re-run, or
re-thresholded after these numbers were read.

## 3. What the point estimates say

- **The thesis held where it was tested: explicit memory + sleep moved second-exposure
  success by +14.0 pp** against an 8.84 pp bar, with the no-memory control flat (−0.5 pp).
  The gain is consistent, not drift-compensated: A_full ≥ A_noMem in every family with
  n ≥ 5 (ARITH +2.5, BOOL +4.6, CMP +7.3, SDL +9.7, UNARY +20 pp).
- **Small+memory does not reach big+search at this scale:** the 14B beats A_full by
  ~21 pp on repeats and ~34 pp on first exposures. E2's premise (that memory could close
  a ~10× parameter gap on this stream) is answered: not at 1.5B vs 14B.
- **E5 is mildly negative (−6.0 pp on novel, n=50):** consistent with the exemplar
  file-familiarity caveat — memory helps re-exposures, not novel units.
- A_full also ran **faster** than A_noMem (3019 s vs 4723 s of task wall) — retrieval
  shortens search — while spending 1139 s of measured sleep GPU time (§3 asymmetry:
  A_full total ≈ 69 min GPU vs A_noMem 79 min).

## 4. Instrument honesty notes (lenses, named)

- **E3a lens:** computed over SUBMITTED candidates (task-level, n=450), not the §5
  "every executed candidate" node level: the hidden suite only ever runs on the submitted
  candidate, so per-node hidden outcomes were never measured by the built instrument.
  This is the only endpoint whose lens narrows the pre-reg text; recorded here, not
  silently.
- **E3b's bar saturated:** 2 × overall failure (0.5022) = 1.0044 > 1.0 — the pass
  condition as written is unsatisfiable whenever overall failure ≥ 50%. The point
  estimate decides, so E3b is reported FAIL as written; the un-thresholded observation
  (recorded, not substituted): abstained tasks failed at 97.9% vs 50.2% overall, a 1.95×
  ratio, 46 of 47 abstains correct. A future pre-registration should bar the RATIO
  against min(1, 2×overall) or cap the bar below 1.
- Sampler, seeds, stream, and thresholds identical across arms; every arm ran the same
  450 tasks in the same order from the same manifest, one server process per arm-run,
  identity asserted at start and after every hot-load.

## 5. GPU accounting (§3 declared asymmetry)

| Arm | Task wall | Sleep GPU (measured `gpu_s`) |
|---|---|---|
| A_full | 3019 s | **1139 s** (14 sleeps, 14 accepted) |
| A_noMem | 4723 s | — |
| B_search | 2237 s | — |
| B_naive | 1716 s | — |

## 6. Not run (labelled per §7)

Exploratory arms (A_mem−sleep, A_sleep−mem) and E4 (QuixBugs held-out) were not run in
this pass; they remain available GPU-time-permitting and never promote to gating.
