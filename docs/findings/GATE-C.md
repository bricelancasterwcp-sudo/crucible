# Phase-C gating result — pre-registered verdict: **NO-GO**

**Date:** 2026-08-25. **Lock:** `prereg-lock-c` (ada99a9). Endpoints computed exactly as
locked (spec §5–§6) after both arms completed; no number read before both finished;
nothing re-run. Records under `runs/gate-c-symmem/`, `runs/abl-symmem-15b/` (git-ignored;
`lens.json` + pool rates per arm).

## 1. Runs

| Arm | Result | Wall | Probes |
|---|---|---|---|
| B_symmem (gating) | 450/450, **0 infra** | 40 min | `symptom_probes.txt` = 450 ✓ |
| A_symmem (exploratory) | 450/450, **0 infra** | 78 min | 450 ✓ |

## 2. Gating endpoint (Δ_min = 0.0764; bar 0.8364; τ = 0.8051, locked)

Gate hygiene in locked order: saturation clause not triggered; CONFOUNDED checks clean
(infra 0.0, landing 1.0, identity asserted); **repeat guard holds** (|0.9200 − 0.9050| =
0.0150 ≤ 0.0586 — the exact-class path reproduced B_mem's repeats).

| ID | Value | Bar | Verdict |
|---|---|---|---|
| **C1** symptom transfer moves the non-repeat pool | pool(B_symmem) = 193/250 = **0.7720** vs frozen B_mem 0.7600 (+1.2 pp) | ≥ 0.8364 | **FAIL → NO-GO** |

## 3. Non-gating endpoints (as pre-declared)

| ID | Value | Frozen comparator | Read |
|---|---|---|---|
| C2 novel | 0.6800 | 0.6800 | the third *exactly-equal* novel reading on the 14B (B_search, B_mem, B_symmem all 0.68) — novel units are impervious at every retrieval grain tried |
| C3 silence rate | 291/450 non-exact tasks silent (64.7%) | — | τ = 0.8051 is a high bar and truly-transferable lessons are scarce; the matcher spoke on 159 tasks |
| C4 matched vs silent (descriptive, selection-confounded as pre-stated) | matched **158/159 = 99.4%** vs silent 219/291 = 75.3% | — | when the matcher speaks it is almost never wrong — precision is not the problem; *reach* is |
| C5 wall | 2026 s | B_mem 2459 s | faster despite +450 probe executions |
| — phase-1 | 0.7950 (+1.5 pp vs B_mem 0.7800) | — | the whole pool gain sits in phase-1 (159/200 vs 156/200); novel contributed zero |

## 4. Exploratory: A_symmem vs frozen comparators

| | pool | second | novel | abstain |
|---|---|---|---|---|
| A_mem_nosleep *(frozen)* | 0.4200 | 0.5600 | 0.4000 | 0.087 |
| A_symmem | 0.4240 | 0.5650 | 0.4400 | 0.091 |

Same story at 1.5B: pool +0.4 pp (nothing), repeats preserved, novel 0.44 — equal to
A_mem_exactonly's silence arm. Symptom-matching neither helps nor hurts where
family-noise and silence already tied.

## 5. Verdict and what §6 prescribes

**NO-GO.** Cross-unit symptom-conditioned retrieval at the lexical grain, with an honest
false-positive-controlled threshold, does not move the non-repeat pool (+1.2 pp against
a +7.6 pp bar) on either model. The instrument behaved exactly as designed — the C4
precision (99.4%) says the scorer finds real matches; the C3 silence rate says such
matches barely exist in a 179-unit corpus. Transfer fails for want of *transferable
material*, not for want of matching. Per the locked §6: **the three-phase findings arc
ships with the store's repeat result as headline; the program closes.** Any revival
(bigger corpus, richer match grain) is a new pre-registration.

## 6. GPU accounting

B_symmem 2382 s + A_symmem 4696 s + smoke 126 s + server loads ≈ **2.1 h**; τ
calibration CPU-only. Phase-C total build: 11 commits, 4 task-level fix rounds, 6/6
mutation sweep, final review CLEAN.
