# S2 — ceiling pilot: BLOCKED at the §4.7 codec-landing gate

**Date:** 2026-08-23
**Slice:** S2 (search + arms + ceiling pilot), operational run (plan Task 16)
**Status:** **The ceiling pilot did not run.** The pre-registered §4.7 codec-landing
gate fails for the pre-registered small-arm proposer *and* for its §2 alternative, at
the pinned sampler. Clearing the gate requires amending pinned/frozen pre-registration
(the §3 `max_new_tokens` pin and/or the §4.4 codec) — an operator decision, recorded
below rather than made unilaterally to pass a gate.

The S2 *build* (plan Tasks 1–15 + docstring fix) is complete and reviewed clean; it is
unaffected by this. What is blocked is the *operational* ceiling-pilot run that S2's
exit criterion asks for.

---

## 1. What was run

Per spec **§4.7** ("Codec landing pre-check"): before any arm runs, the full-module-rewrite
codec must produce **≥ 95% parseable submissions on 30 smoke tasks** against each served
model. Failure ⇒ **baseline fallback (§2) or codec fix**, before any arm runs.

The ceiling pilot (**§4.8.4**) runs **A_noMem**, whose proposer for all small arms is
**Qwen3.5-2B** (LoRA-attach succeeded in S1, so the §2 LoRA-failure swap to the 1.5B never
fired — see `S1-serving.md §7`). So the gate that stands between us and the pilot is the
**2B proposer's** landing rate.

The landing probe was run against the **real 450-task stream** (`streams/full/dd5912cddedc`,
which passed every §4.8.1–3 structural pre-check) at the **pinned sampler** (§3: temperature
0.7, top_p 0.95, `max_new_tokens 1024`, thinking OFF), each pytest touching the sandbox
wrapped in the mandatory `systemd-run --user --scope -p MemoryMax=… -p MemorySwapMax=0`
memory cap.

## 2. Result — both pre-registered models fail the gate

| Proposer | role | `max_new_tokens` | landing (frozen codec) | §4.7 (≥0.95) |
|---|---|---|---|---|
| `Qwen/Qwen3.5-2B` | pre-registered small-arm proposer | **1024** (pinned) | **0.767** (23/30) | **FAIL** |
| `Qwen/Qwen2.5-Coder-1.5B-Instruct` | §2 small-proposer alternative | **1024** (pinned) | **0.80** (24/30) | **FAIL** |

Neither the pre-registered proposer nor its pre-registered alternative clears the gate at
the pinned sampler. Per §4.7 the "baseline fallback" remedy is therefore exhausted for the
small arm; the remaining sanctioned remedy is a **codec fix**.

## 3. Diagnosis — the failures are decoding artifacts, not model incapability

Sampling the actual completions shows the sub-95% rate is dominated by **truncation and
degenerate decoding**, not by the model being unable to write the corrected module:

- **Truncation (dominant).** The frozen codec (§4.4, `crucible/proposer/prompt.py`) asks the
  model to re-emit the **entire module *and* reproduce the whole visible test harness**
  inside one ```` ```python ```` block. On these tasks that routinely exceeds 1024 tokens, so
  the completion is cut off **mid-file with the fence still open** → `extract_module` reports
  `no-fence` → rejected. Example (1.5B): a valid `import math` / `from unit_… import … as
  candidate` module header that then degenerates into `1, 1, 1, 1, …` repetition and hits the
  1024 cap before closing the fence.
- **Empty completions (~6%).** The 1.5B occasionally returns an empty string (immediate EOS).
- **Fragments / test-echo (2B only).** The 2B (a Qwen3-VL base, not code-instruction-tuned)
  sometimes emits a bare code fragment (e.g. a lone `if` block, no `def`) or regurgitates the
  **visible test file** instead of the module. These *parse* (so the raw 0.767 is if anything
  optimistic) but are not full-module rewrites.

### Diagnostic: the codec fix is concrete and scoped

Re-running the 1.5B coder at a larger token budget and with an unclosed-fence salvage
(throwaway measurement, n=50, real stream):

| Proposer | `max_new_tokens` | frozen codec | + unclosed-fence salvage |
|---|---|---|---|
| `Qwen2.5-Coder-1.5B-Instruct` | 1024 | 0.80 | — |
| `Qwen2.5-Coder-1.5B-Instruct` | **2048** | **0.92** (46/50) | **~0.94** (residual = 3 empty completions) |

Raising the token budget from 1024→2048 alone lifts landing **0.80 → 0.92** — confirming
truncation is the dominant cause. Salvaging a truncated-but-valid module (recover the code
before the missing closing fence) rescues the residual `no-fence`; the last ~6% is the
empty-completion rate, which an in-search resample (the arm already draws *k* candidates per
node) absorbs but a single-shot pre-check counts as a miss.

## 4. Why this stops here (the pre-registration crux)

The two levers that clear the gate are both **frozen, pre-registered** knobs:

1. **`max_new_tokens` is pinned at 1024 in §3.** The measured root cause is that this pin is
   too small for a codec that re-emits the module *plus* the whole test harness. Raising it is
   a pin amendment.
2. **`extract_module` (§4.4 codec) rejects unclosed fences.** Adding salvage — or, better,
   trimming the codec so the model no longer has to reproduce the visible test file it can
   already see — is a change to frozen codec code.

Amending a pinned value or the codec **to make a pre-registered gate pass** is exactly the
decision this project holds for the operator — the whole point of the binding spec and the
honest-measurement rules is that the machinery does not quietly re-tune itself to clear its
own gate. So the pilot stops at the gate. **p0 is unmeasured; the "too-easy" verdict is N/A
until the gate is cleared under an approved configuration.**

## 5. Recommended remedy (for operator decision)

All three are within §4.7's sanctioned "codec fix"; ordered by impact and by how small an
amendment they are:

- **(A) Raise the pinned `max_new_tokens` to 2048** (§3 amendment). Biggest single lift
  (0.80→0.92 on the 1.5B). Cheap, low-risk, arguably corrects a pin that was mis-set for this
  codec from the start. **Recommended as the minimum.**
- **(B) Add unclosed-fence salvage to `extract_module`** (§4.4 codec code). Recovers truncated
  modules; pushes landing to ~0.94 on the 1.5B; still one mutation-tested function.
- **(C) Trim the codec so the model emits only the module, not a reproduction of the visible
  test harness** (§4.4 codec redesign). Removes the token pressure at its source and kills the
  test-echo failure mode. Largest change; needs its own review pass.
- **Model choice, once the codec is fixed:** re-run the §4.7 probe for the **2B** (pre-registered
  proposer) under the amended config. If the 2B clears 0.95, keep it (LoRA-attach already works
  on it — the thesis arm A_full needs that). If only the 1.5B clears, the small-arm proposer
  becomes the 1.5B and A_full's LoRA path must be re-verified on it. (A + B raised the *1.5B*
  to ~0.94 in the diagnostic; the 2B was only measured at the pinned 1024 = 0.767 and should be
  re-probed at 2048 before it is ruled out — its raw failures were more test-echo than pure
  truncation, so it may need C, not just A.)

The separate **baseline "big" proposer** (§2: `Qwen3.5-9B`, fallback `Qwen2.5-Coder-14B-Instruct
Q4_K_M`) was **not** landing-probed here — it gates the B_naive/big arm, not the pilot, and is
S3/S4 work. It must clear §4.7 too before those arms run.

## 6. Exit-criterion status

Spec §10 S2 exit: *"A_noMem pilot number recorded; rung fixed; B arms smoke-tested."*

- **A_noMem pilot number:** **NOT recorded** — blocked at the §4.7 gate (this document is the
  record of why).
- **Hardening rung:** not fixed (the pilot that would inform it did not run).
- **B arms smoke-tested:** the arm *machinery* (search, REx, driver, records, lens, pilot,
  landing-check) is built and unit-reviewed clean (plan Tasks 1–15); the *operational* smoke is
  gated behind the same landing fix.

The build slice is done and merge-ready; the operational slice resumes once the operator picks
a §4.7 remedy (§5) and approves the corresponding pin/codec amendment.
