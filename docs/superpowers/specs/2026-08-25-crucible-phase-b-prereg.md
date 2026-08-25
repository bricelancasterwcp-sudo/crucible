# Crucible Phase-B pre-registration — does the store stack on the strong model?

**Date authored:** 2026-08-25. **Status:** DRAFT until tagged `prereg-lock-b`; after the
tag, §5–§7 are frozen and only the §11 amendment protocol may touch them.

## 1. Provenance — what licenses this phase

Phase-A's pre-registered verdict was **PARTIAL** ([GATE-A.md](../../findings/GATE-A.md)):
E1 (memory works, +14.0 pp second-exposure) and E3a passed; E2a/b (beat the 14B) and E3b
(saturated bar) failed. Under §7 of the Phase-A prereg, PARTIAL puts the next step to
Brice. The §13 pre-commitment (B-lite latent predictor) was conditioned on GO and does
not bind. The exploratory ablations ([ABLATIONS-A.md](../../findings/ABLATIONS-A.md))
showed the **explicit store carries ~96% of the E1 gain** (retrieval-only +13.5 pp, zero
sleep GPU) while weight consolidation alone moves repeats +1.0 pp (control noise).

Brice ruled (2026-08-25): Phase-B = memory transfer, primary question = **is the explicit
store a model-agnostic upgrade** — does retrieval-only memory lift the strong 14B the way
it lifted the 1.5B? Design approach approved: one gating arm, one mechanism, frozen
Phase-A control, one exploratory ride-along.

## 2. Question

Phase-A left the store confounded with model weakness: retrieval pays +13.5 pp on a model
that fails 57% of first exposures. The 14B fails only 20.5% — and its own frozen
second-exposure delta is **−1.0 pp** (0.795 → 0.785): the strong model does not
self-improve on repeats. If the store is a general mechanism, bolting it (and nothing
else) onto the 14B should produce a within-arm second-exposure gain the bare 14B lacks.
If the gain was a small-model crutch, it won't.

## 3. Arms

| Arm | Model / serving | Hooks | Role |
|---|---|---|---|
| **B_mem** | Qwen2.5-Coder-14B-Instruct (AWQ-served under this name), chat, seed 0, k=8, width=4 | **MemHooks**: retrieval ON (full policy), sleep OFF, **no value v1, no calibrator** | **gating** |
| B_search *(frozen)* | identical serving identity | none | control — **never re-run** |
| A_mem_exactonly | Qwen2.5-Coder-1.5B-Instruct, chat, seed 0 | FullHooks: retrieval **exact-class only** (no family-wide lesson fallback), sleep OFF, value v1 + calibrator ON | exploratory |

- **B_mem differs from the frozen control by the store alone.** `MemHooks.task_confidence()`
  returns `None`, so the search's status/abstention path is byte-identical to B_search's
  structural rule; the value model is `ConstantValue`, as B_search's was. Episodes are
  written for every attempt and lessons distilled from verified ones (same gates as
  FullHooks); the store starts empty and fills during the run.
- **A_mem_exactonly differs from frozen A_mem_nosleep by retrieval policy alone** (same
  value v1 + calibrator + no-sleep configuration). True strangers — classes with no
  exact-class content — receive **silence**, not family-wide lessons. The exemplar path is
  already class-exact and is untouched.
- The control is the GATE-A B_search lens (n=450, phase-1 0.7950, second 0.7850, novel
  0.6800, landing 1.0, infra 0). Re-running a completed measurement is prohibited
  (Phase-A discipline); its `lens.json` sha256 is recorded at lock. Comparability rests on:
  same locked stream, same task order, same serve table entry, same box, same seed.

## 4. Instrument changes (pre-lock, each mutation-tested)

1. `ARMS` gains `B_mem` (14B serving identity == B_search) and `A_mem_exactonly` (1.5B
   identity == A_full); pinned by test like the Phase-A ablation arms.
2. `MemHooks` in `crucible/run/full.py` implementing the ArmHooks protocol as above; CLI
   gates it by arm name. Nothing on the B_mem path constructs a value model beyond
   `ConstantValue` or a calibrator.
3. `FULL_FAMILY` values become `(retrieval_mode: "full"|"exact"|"off", sleep: bool)`;
   `retrieve()` gains `exact_only` (skip the family-wide lesson pool; exemplar unchanged).
4. **14B serve window 8192 → 16384** (`SERVE` table). Same class of fix as the 1.5B's
   A_full overflow (GATE-A run ledger): memory-augmented refinement prompts exceed
   8192−2048. Pure KV capacity; sampling-neutral. AWQ weights ~10 GiB under util 0.90
   leave ~4 GiB KV; the §8 smoke must show no KV stalls, else fall back to 12288 (still
   clears the measured 6145-token worst case + 2048 generation) — a fallback taken
   pre-lock is recorded here as an amendment, post-lock it is an infra fix.
5. No other serving, search, prompt, or stream change of any kind.

## 5. Endpoints (frozen at lock)

**The instrument is `build_lens` — measured-only denominators (`hidden_pass is not
None`), per-kind rates, exactly as Phase-A.** All rates below are lens fields.

**Δ_min is derived, not chosen:** Δ_min = 2·√(2·p̄(1−p̄)/C) with p̄ = the frozen control's
phase-1 rate 0.7950 and C = 200 tasks per exposure class → **0.0807** (recomputed and
recorded to 4 dp in the lock record).

- **B1 (gating, the only gating endpoint):**
  Δ_B = succ_second(B_mem) − succ_phase1(B_mem) **≥ Δ_min**, AND the frozen control's own
  within-arm delta (−0.0100) lies in ±Δ_min (it does; frozen — recorded, not re-measured).
  *Saturation clause (the E3b lesson):* if succ_phase1(B_mem) > 1 − Δ_min the bar is
  unsatisfiable; B1 is then **not exercisable** and is reported unthresholded — it does not
  silently pass or fail.
- **B2 (non-gating):** succ_novel(B_mem) vs frozen 0.6800 — does family-wide memory hurt
  the strong model's strangers (n=50; SE ≈ 7 pp; descriptive, ±2·SE quoted).
- **B3 (non-gating):** cross-arm succ_second(B_mem) vs frozen 0.7850 — reported with the
  drift caveat (different run days); the within-arm B1 is the protected form.
- **B4 (non-gating):** task wall-time B_mem vs the frozen control's task wall of 2237 s
  (sum of wall_s over its 450 records) — does retrieval speed the strong model's search.
- **Exploratory (A_mem_exactonly, reading guide pre-stated):** vs frozen A_mem_nosleep
  (phase-1 0.4250, second 0.5600, novel 0.4000, abstain 0.087). If Δ_second stays ≈ +13
  pp → exact-class content drives the repeat gain. If succ_novel recovers toward
  A_noMem's 0.5000 → the novel-unit harm is family-lesson-driven. Both reads are
  descriptive; neither gates.

No abstention endpoint: B_mem has no calibrator, deliberately.

## 6. Verdict rule (frozen at lock)

- **GO_B** = B1 passes. → Phase-C is retrieval *policy* (symptom-conditioned retrieval,
  learned silence), pre-registered then.
- **NO-GO** = B1 fails with a clean instrument. → Ship the full Phase-A+B findings arc;
  the spike closes. A NO-GO with B_mem phase-1 ≥ Δ_min *below* the control's 0.7950 is
  additionally reported as "memory harms first exposures at 14B" — a finding, not a
  confound.
- **CONFOUNDED** = instrument failure only: infra_rate(B_mem) > 0.02, landing_rate <
  0.98, served-identity mismatch, or the §5 saturation clause. → fix, archive, clean
  rerun from zero. A phase-1 *shift* alone is never CONFOUNDED (family-wide lessons can
  legitimately touch first exposures; the within-arm form absorbs it conservatively).
- **The point estimate decides.** No extension, re-run, threshold change, stream change,
  or added arm after any B_mem number is read. Exploratory results never promote.

## 7. Kill criteria

- Pre-lock smoke (§8) fails twice on KV/window grounds after the 12288 fallback → stop,
  redesign serving before any lock.
- Any evidence the control's serving conditions cannot be reproduced (SERVE table drift,
  driver diff touching the request path since the gate) → stop; Phase-B needs a fresh
  paired control run instead, and this document is amended pre-lock to say so.

## 8. Run discipline (Phase-A rules, unchanged)

Locked stream `1158e92f…` (hash in LOCK-A), `--tasks all` (450), seed 0. One run per arm,
fresh out dirs (`runs/gate-b2-mem/`, `runs/abl-mem-exactonly/`), `run_arm_detached.sh`
(non-empty-dir refusal), OS-detach + marker-or-death monitor, free-VRAM ≥ 13 GiB before
serving, served-identity assert, R-S4-1: infra kill = archive + clean rerun, never
resume, never splice. Order: **smoke (non-gating dir, ~15 tasks, store pre-warmed to
force memory-augmented refinement prompts) → lock tag → B_mem → A_mem_exactonly**.
`lens.json` written per arm; teardown = kill EngineCore by pid, wait port-down.

## 9. Lock record (`prereg-lock-b`)

The tag lands on the commit recording: this document's final pre-lock text; Δ_min to 4 dp
with its derivation inputs; sha256 of the frozen control `lens.json` and of the frozen
A_mem_nosleep `lens.json`; the stream hash; the 14B AWQ revision + shard digests (as in
LOCK-A) and the 1.5B revision; the SERVE table entries verbatim (post-window-change);
vLLM version; the exact arm names and task set (`all`).

## 10. Cost

Smoke ~15 min; B_mem ~60–80 min (B_search ran 37 min memoryless; retrieval lengthens
prompts); A_mem_exactonly ~85 min; total ≤ 3 h GPU, all local, no training.

## 11. Amendments

Pre-lock amendments edit this section with date + old value, and mark the amended line.
Post-lock, §5–§7 are immutable; anything else follows the Phase-A amendment footnote
protocol.

*(none yet)*
