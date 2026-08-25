# Phase-B gating result — pre-registered verdict: **GO_B**

**Date:** 2026-08-25. **Lock:** `prereg-lock-b` (dd2d7d6). Endpoints computed exactly as
locked (spec §5–§6), after both arms completed; no number was read before its arm
finished; nothing was re-run. Records under `runs/gate-b2-mem/`, `runs/abl-mem-exactonly/`
(git-ignored; `lens.json` per arm).

## 1. Runs

| Arm | Result | Wall | Notes |
|---|---|---|---|
| B_mem (gating) | 450/450, **0 infra** | 41 min | 14B + store-only MemHooks; every second exposure carried memory; `adapter_id` None throughout |
| A_mem_exactonly (exploratory) | 450/450, **0 infra** | 83 min | 1.5B, exact-class retrieval; **0/50 novel tasks received a block** (strangers got silence, by construction); no sleep file |

Pre-lock smoke: 16 tasks, window 16384 verdict "stands" (0 overflows, 0 KV preemptions).

## 2. Gating endpoint (Δ_min = 0.0807, derived at lock)

| ID | Value | Bar | Verdict |
|---|---|---|---|
| **B1** store stacks on 14B | Δ_B = succ_second − succ_phase1 = 0.9050 − 0.7800 = **+0.1250**; frozen control Δ = −0.0100 ∈ ±0.0807 ✓ | ≥ 0.0807 | **PASS → GO_B** |

Gate hygiene, in order: saturation clause not triggered (0.780 ≤ 0.9193); CONFOUNDED
checks clean (infra 0.0, landing 1.0, served identity asserted at construction).

## 3. Non-gating endpoints (as pre-declared)

| ID | Value | Frozen comparator | Read |
|---|---|---|---|
| B2 novel | 0.6800 | 0.6800 | exactly equal — family memory neither helps nor hurts the 14B's strangers (n=50, ±2SE ≈ 0.13) |
| B3 cross-arm second | 0.9050 | 0.7850 | +12.0 pp, consistent with B1's within-arm form |
| B4 task wall | 2459 s | 2237 s | +10% — retrieval slightly *slows* the 14B (contrast: the A-arm speedup came from adapters, ABLATIONS-A §5) |
| — phase-1 | 0.7800 | 0.7950 | −1.5 pp, well inside ±Δ_min: no first-exposure harm |

## 4. Exploratory: A_mem_exactonly vs frozen A_mem_nosleep (reading guide pre-stated in spec §5)

| | phase-1 | second | Δ second | novel | abstain |
|---|---|---|---|---|---|
| A_mem_nosleep *(frozen)* | 0.4250 | 0.5600 | +0.1350 | 0.4000 | 0.087 |
| A_mem_exactonly | 0.4200 | 0.5700 | **+0.1500** | 0.4400 | 0.067 |

Both pre-stated reads answered: (1) Δ_second survives fully without the family-wide
fallback — **exact-class content drives the repeat gain**. (2) Novel moved 0.40 → 0.44
with strangers receiving silence — *partial* recovery toward A_noMem's 0.50, directionally
consistent with family-wide lessons being mildly harmful to strangers, but n=50
(±2SE ≈ 0.14) — descriptive, not conclusive.

## 5. Verdict and what it licenses

**GO_B** per §6: the explicit store is a **model-agnostic upgrade** on re-exposures —
+13.5 pp on a 1.5B (ABLATIONS-A), +12.5 pp on a 14B ten times its size, in both cases as
pure prompt-side retrieval with no training. Per the locked §6, Phase-C is retrieval
*policy* (symptom-conditioned retrieval, learned silence), to be pre-registered before
any Phase-C number is read. The novel-unit gap remains the open frontier: memory still
does nothing for strangers on either model (B2 exactly 0.68 = control; E5's −6 pp at
1.5B), and the exact-only probe hints the right first move is teaching retrieval when
NOT to speak.

## 6. GPU accounting

B_mem 2465 s + A_mem_exactonly 4987 s + smoke 97 s + two server loads ≈ **2.2 h** total,
all local, zero training.
