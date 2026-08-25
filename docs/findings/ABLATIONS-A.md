# Phase-A exploratory ablations — A_mem−sleep / A_sleep−mem (protocol)

**Status: EXPLORATORY, NON-GATING.** Pre-registered as such (prereg §"Exploratory",
line: *"A_mem−sleep (explicit memory, no LoRA) and A_sleep−mem (LoRA, no explicit
store) — separates the store from the weights"*). The §7 gate verdict — **PARTIAL**,
recorded in [GATE-A.md](GATE-A.md) — is final and is not affected by anything below.
Per the lock: *"Exploratory arms are labelled exploratory in every table and never
promote to gating."*

**Protocol authored 2026-08-25, BEFORE the ablation switch existed in the code and
before any ablation task ran.** Brice authorized the runs ("do it") after the gate.

## 1. Question

E1 passed on a compound treatment: A_full = A_noMem + {explicit memory retrieval,
value v1 + calibrated abstention, sleep LoRA consolidation}. Which mechanism carries
the +14.0 pp second-exposure gain?

## 2. Arms

Each ablation is **A_full minus exactly one mechanism** (never "A_noMem plus one" —
the value model and calibrator stay in both, as in A_full):

| Arm name in records | Prereg name | Retrieval (prompt) | Sleep (LoRA) | Value v1 + calibrated abstain |
|---|---|---|---|---|
| `A_mem_nosleep` | A_mem−sleep | ON | OFF | ON |
| `A_sleep_nomem` | A_sleep−mem | OFF | ON | ON |

Serving identity is byte-identical to A_full/A_noMem: same 1.5B checkpoint, chat-served,
same locked serve flags (util 0.45, max-model-len 16384), same seed 0, same search budget.
Same locked stream (hash `1158e92f…`), `--tasks all` (450). Episodes and lessons are
written identically in both arms (in A_sleep_nomem they are inert on the prompt path;
sleep's SFT set reads verified episodes, which exist in both).

## 3. Pre-declared computations (descriptive; no new thresholds)

For each arm, from its `lens.json` (the same `build_lens` the gate used):
1. succ_phase1, succ_second, succ_novel, succ_overall, abstain_rate, infra_rate,
   landing_rate; **Δ_second = succ_second − succ_phase1**.
2. Comparison against the frozen GATE-A rows: A_full Δ = +0.1400 (0.4350 → 0.5750),
   A_noMem Δ = −0.0050 (0.4250 → 0.4200).
3. Task wall-time per arm (A_full 3019 s vs A_noMem 4723 s — does retrieval alone
   reproduce the speedup?); for A_sleep_nomem, sleep count / accept rate / gpu_s.

Reading guide, stated in advance: Δ(A_mem_nosleep) ≈ +0.14 → the store carries the
gain; Δ(A_sleep_nomem) ≈ +0.14 → the weights carry it; both ≪ +0.14 → the gain needs
the interaction (or the value/abstain column). These are descriptive reads on n=200
per exposure class (SE of a single rate ≈ 3.5 pp) — no verdict is attached.

## 4. Run discipline

Identical to the gate runs even though non-gating: fresh out dirs via
`scripts/run_arm_detached.sh` (refuses non-empty dirs), §8 VRAM floor before serving,
served-identity assert, an infra-killed run is archived and cleanly rerun from zero
(R-S4-1), never resumed, never spliced. One run per arm; the numbers reported are the
numbers the first completed run produced.

## 5. Results

*(appended after both runs complete — nothing above this line changes after this
commit)*
