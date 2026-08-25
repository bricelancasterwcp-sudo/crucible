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

**Run 2026-08-25, both arms 450/450, zero infra, landing 1.0, one run each, never
resumed.** A_mem_nosleep: EXIT=0, wall 4995 s. A_sleep_nomem: EXIT=0, wall 6445 s
(3858 s task wall + sleep overhead). Structural invariants held: A_mem_nosleep wrote
**no** sleep-records file and stamped no adapter on any record; A_sleep_nomem stamped
`retrieved_ids=()` on all 450 records while its organ minted lessons normally.
Records + `lens.json` under `runs/abl-mem-nosleep/`, `runs/abl-sleep-nomem/`
(git-ignored, same as the gate runs). Gate rows are the frozen GATE-A lenses.

| Arm | phase-1 | second | **Δ second** | novel | overall | abstain | task wall | sleeps (acc) | sleep gpu_s |
|---|---|---|---|---|---|---|---|---|---|
| A_noMem *(gate)* | 0.4250 | 0.4200 | −0.0050 | 0.5000 | 0.4311 | 0.000 | 4723 s | — | — |
| A_full *(gate)* | 0.4350 | 0.5750 | **+0.1400** | 0.4400 | 0.4978 | 0.104 | 3019 s | 14 (14) | 1139 |
| A_mem_nosleep | 0.4250 | 0.5600 | **+0.1350** | 0.4000 | 0.4822 | 0.087 | 4990 s | 0 (by construction) | 0 |
| A_sleep_nomem | 0.4350 | 0.4450 | **+0.0100** | 0.4400 | 0.4400 | 0.067 | 3858 s | 12 (12) | 636 |

Readings, in the §3 pre-declared frame (descriptive; single-rate SE ≈ 3.5 pp at
n=200, ≈ 7 pp for novel at n=50):

1. **The store carries the E1 gain.** Retrieval without any weight consolidation
   reproduces +13.5 pp of A_full's +14.0 pp second-exposure gain — ~96% of the
   effect at zero sleep GPU.
2. **Weights alone do not move repeats.** A_sleep_nomem's Δ = +1.0 pp, within
   control noise (A_noMem −0.5 pp), despite 12/12 sleeps accepted and every
   post-sleep attempt generating from the latest adapter. LoRA consolidation on
   ~200 verified fixes does not act as usable episodic memory here.
3. **Inside A_full, sleep's marginal contribution was speed, not accuracy.**
   Retrieval alone does not reproduce the wall-time win (4990 s ≈ A_noMem's
   4723 s); adapters alone cut it to 3858 s; the combination to 3019 s. Sleep
   bought ~0.5 pp of repeat accuracy and ~40% of wall time relative to
   A_mem_nosleep.
4. **No memory variant helps novel units** (E5's caveat holds mechanism-by-
   mechanism): A_noMem's 0.50 is the best novel rate; every memory arm sits at
   0.40–0.44.
5. Calibrated abstention engaged in all full-family arms (6.7–10.4%).

**Phase-B implication (for Brice's ruling, not a verdict):** the payoff lever for
*repeat* performance is the explicit store; a Phase-B built around consolidation
would be building on the weakest component measured here. The E5/novel-unit gap is
untouched by every mechanism tried.

*Ops note (instrument honesty):* an initial hand-rolled count in the session used
`status == "abstained"` and read abstain = 0.0 for both arms; the recorded status
string is `abstain`. The lens (`build_lens`) — the instrument §3 committed to —
was used for every number above. The named bug class strikes again; the protocol's
"name the instrument" rule is what caught it.
