# Crucible Phase-C pre-registration — symptom-conditioned retrieval + learned silence

**Date authored:** 2026-08-25. **Status:** DRAFT until tagged `prereg-lock-c`; after the
tag, §5–§7 are frozen and only the §11 amendment protocol may touch them.

## 1. Provenance — what licenses this phase

Phase-B returned **GO_B** ([GATE-B.md](../../findings/GATE-B.md)): the explicit store is
model-agnostic on repeats (+12.5 pp on the 14B, +13.5 pp on the 1.5B). Its locked §6
names Phase-C: retrieval *policy* — symptom-conditioned retrieval and learned silence —
pre-registered before any number is read. The open frontier is uniform across every
result of this program: **memory does nothing for units it has never seen** (B2 novel
exactly = control at 14B; E5 −6 pp at 1.5B), and the exact-only probe showed silence
directionally beats family-wide noise. Brice approved this design 2026-08-25
("14B non-repeat pool" gate; mechanism and arms as below).

## 2. Question

Class-keyed retrieval transfers nothing across units. Can retrieval keyed on the
**failure itself** — the symptom text and the shape of prior fixes — carry repair
knowledge to units the store has never seen, and can a pre-set silence threshold keep it
from spraying noise where it has nothing? The gated population is exactly where
class-exact memory is empty: **phase-1 ∪ novel tasks (n = 250)**, on the strong model.

## 3. Arms

| Arm | Model / serving | Hooks | Retrieval mode | Role |
|---|---|---|---|---|
| **B_symmem** | 14B (AWQ-served), chat, seed 0, k=8, w=4 | MemHooks (no value model, no calibrator, no sleep) | `"symptom"` | **gating** |
| B_mem *(frozen, GATE-B)* | identical serving identity | MemHooks | `"full"` | control — **never re-run** |
| A_symmem | 1.5B, chat, seed 0 | FullHooks (value v1 + calibrator ON, sleep OFF) | `"symptom"` | exploratory |
| A_mem_nosleep *(frozen, ABLATIONS-A)* | identical serving identity | FullHooks | `"full"` | exploratory comparator |

Each new arm differs from its frozen comparator by **retrieval policy alone** — same
hooks class, same value/calibrator configuration, same serving identity (pinned by test,
as in Phases A/B).

Frozen comparator values (recomputed from records at lock; lens digests recorded):
B_mem pool (phase-1∪novel, measured-only) = (156+34)/250 = **0.7600**; B_mem second
0.9050; B_mem novel 0.6800. A_mem_nosleep pool = (85+20)/250 = **0.4200**.

## 4. Instrument changes (pre-lock, each mutation-tested)

1. `ARMS` gains `B_symmem` (== B_search/B_mem serving identity) and `A_symmem`
   (== A_full identity); pinned by test.
2. **`crucible/memory/symmatch.py`** (new): a deterministic lexical relevance scorer and
   the v2 retrieval policy. Policy, in order:
   a. **Exact-class fast path unchanged**: a live exact-class lesson (and the class-exact
      exemplar) retrieves exactly as mode `"full"` does today — the proven repeat
      machinery is byte-untouched.
   b. Otherwise: score **all live lessons cross-unit** by lexical overlap between the
      query (current unit source + rendered symptom text) and the lesson (its
      `landed_diff` + the cited episode's symptom as rendered in its stored ``root_prompt``,
      extracted at runtime from the episodic store). Family is a scoring *feature*, never a hard filter. Unit-local
      test NAMES are excluded from both sides (they are meaningless across units).
      Top-2 lessons with score ≥ **τ**; the class-exact-only exemplar rule is unchanged
      (strangers get no exemplar).
   c. Below τ: **silence** — `RetrievedBlock(None, ())`, never `""`.
3. **Retrieval-mode vocabulary grows `"symptom"`** (shared by FullHooks and MemHooks;
   `FULL_FAMILY` gains `A_symmem: ("symptom", False)`, the MemHooks CLI gate maps
   `B_symmem → "symptom"`). In symptom mode `before_task` runs ONE extra **uncharged**
   driver-side symptom execution (`run(unit, unit.module_src, None)` — deterministic,
   byte-identical to the free symptom run the search itself performs; never in
   `executions_charged`; counted in a disclosed hooks counter reported in the findings).
   No record-schema change.
4. **τ is fixed pre-lock by a pre-registered rule**: over the Phase-A/B memory databases
   (`runs/gate-a-full`, `runs/abl-mem-nosleep`, `runs/abl-mem-exactonly`,
   `runs/gate-b2-mem` — disclosed calibration data, none of it Phase-C data),
   τ = **P95 of the unrelated-pair score distribution**. Pairs are formed WITHIN each
   database (a run's own episodes × its own live lessons) and pooled across the four:
   *unrelated* = episode and lesson from a different unit AND different family;
   *related* = same class. A false-positive control: speak only when the
   match beats ~all cross-family noise. The script, the pair counts, the distributions'
   summary statistics, and the resulting τ are recorded in LOCK-C.
5. No serving, search, prompt-template, or stream change of any kind. (The serve window
   stays 16384, as locked for Phase-B.)

## 5. Endpoints (frozen at lock)

**The instrument is `build_lens` plus one derived pool rate** — pool = mean
`hidden_pass` over measured phase-1∪novel records (the same measured-only discipline;
recomputed from records, not from the two lens fields, so mixed-infra cases stay honest).

Δ_min values derived at lock:
- **Δ_min(pool)** = 2·√(2·p̄(1−p̄)/250), p̄ = 0.7600 → **0.0764**.
- Δ_min(repeat guard) = 2·√(2·0.905·0.095/200) = **0.0586**.

- **C1 (gating, the only gating endpoint):**
  pool(B_symmem) ≥ pool(B_mem frozen) + Δ_min(pool) = 0.7600 + 0.0764 = **0.8364**.
  *Saturation clause (E3b lesson):* if 0.7600 + Δ_min > 1 the bar is unsatisfiable and
  C1 is not exercisable — cannot occur at these values, stated anyway.
- **Repeat guard (confound detector, NOT an endpoint):** |succ_second(B_symmem) −
  0.9050| ≤ 0.0586. Attribution stated honestly: seconds whose class was SOLVED at
  phase 1 take the byte-unchanged exact-class path (≈78% of seconds, per B_mem's
  phase-1 rate); seconds whose first attempt failed take the new symptom path where
  B_mem used family fallback. A violated guard therefore triggers **CONFOUNDED plus a
  mandatory diagnosis** — split the seconds by solved-at-phase-1 (derivable from the
  run's own records) and report both subset rates; a drift on the exact subset is
  instrument/serving, a drift confined to the failed-first subset is a policy effect
  and is reported as such alongside the CONFOUNDED verdict.
- **CONFOUNDED** additionally: infra_rate > 0.02, landing_rate < 0.98, served-identity
  mismatch.
- **Non-gating, pre-declared:** C2 novel-only vs frozen 0.6800 (n=50, ±2SE quoted);
  C3 silence rate = fraction of non-exact-class tasks with no block; C4 succ among
  symptom-matched vs silent non-exact tasks (descriptive; selection-confounded and said
  so); C5 wall vs B_mem's 2459 s; the uncharged-symptom-run counter (must equal task
  count in symptom mode).
- **Exploratory (A_symmem vs frozen A_mem_nosleep, reading guide pre-stated):** pool
  0.4200 comparator; novel 0.4000/0.4400/0.5000 (nosleep / exactonly / noMem) — does
  symptom-matching beat both family-noise and silence? Descriptive only, never gates.

## 6. Verdict rule (frozen at lock)

- **GO_C** = C1 passes with a clean repeat guard. → Symptom-grain transfer works; the
  program's write-up gains its third leg; any Phase-D (corpus expansion / scaling) is a
  new pre-registration.
- **NO-GO** = C1 fails with a clean instrument. → Transfer fails at this grain; the
  three-phase findings arc ships with the store's repeat result as headline.
  **Findings ship either way.**
- **CONFOUNDED** = the §5 instrument checks only (incl. the repeat guard) → fix,
  archive, clean rerun from zero.
- **The point estimate decides.** No extension, re-run, threshold change (τ included),
  stream change, or added arm after any B_symmem number is read. Exploratory results
  never promote.

## 7. Kill criteria

- **Ranker-sanity gate (pre-lock, on calibration data):** if the median related-pair
  score (episode × lesson from the same class) does not exceed τ, the ranker cannot
  separate signal from cross-family noise — STOP before locking; redesign or abandon
  Phase-C with a report. Recorded in LOCK-C either way.
- Smoke fails twice on symptom-plumbing or window grounds → stop, redesign before lock.
- Serving conditions for the frozen comparators cannot be reproduced → stop; amend
  pre-lock to require a fresh paired control.

## 8. Run discipline (unchanged)

Locked stream `1158e92f…`, `--tasks all` (450), seed 0, one run per arm, fresh dirs
(`runs/gate-c-symmem/`, `runs/abl-symmem-15b/`), detached launcher + marker-or-death
monitor, ≥13 GiB free before serving, identity assert, R-S4-1 never-resume. Order:
**τ calibration + ranker-sanity gate → smoke (≥15 tasks incl. second exposures AND
novel/stranger tasks, symptom plumbing + silence exercised) → lock tag → B_symmem →
A_symmem → GATE-C.md**. `lens.json` + pool rate per arm; teardown by EngineCore pid.

## 9. Lock record (`prereg-lock-c`)

The tag lands on the commit recording: this document's final text; both Δ_min values
with derivation inputs; **τ with its rule, pair counts, and distribution summaries**;
frozen comparator lens sha256s (B_mem, A_mem_nosleep) and their recomputed pool rates;
stream hash; model revisions/digests (unchanged from LOCK-B); vLLM version; serve
entries verbatim; arm names and task set; ranker-sanity gate outcome.

## 10. Cost

τ calibration: CPU-only, minutes. Smoke ~15 min; B_symmem ~45–60 min (symptom run adds
~1 sandbox execution/task); A_symmem ~90 min; total ≤ 3 h GPU. Build: the largest of the
memory line (~S3-sized).

## 11. Amendments

Pre-lock amendments edit this section with date + old value and mark the amended line.
Post-lock, §5–§7 are immutable.

*(none yet)*
