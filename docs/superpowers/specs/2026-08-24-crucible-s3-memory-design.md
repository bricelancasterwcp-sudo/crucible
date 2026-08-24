# S3 — memory organ + value + uncertainty + sleep → A_full

**Date:** 2026-08-24
**Status:** design approved by Brice (chat, 2026-08-24); this document is the written spec.
**Parent:** pre-reg `2026-08-23-crucible-phase-a-prereg.md` §2 (arms), §4.4 (what A_full
sees), §8 (instrument honesty), §9 (module interfaces — binding), §10 S3 (slice exit).
**Context:** the stream is hardened (rung `stack2`, `streams/1158e92f40ad`, A_noMem
p0 = 0.267); the proposer is the A2 1.5B (chat-served; LoRA-attach verified 2026-08-24,
loss 1.25, 3.3 GiB peak); the §2 big-arm fallback (14B-AWQ) is active. S3 builds the
system under test: A_full = search + explicit memory + sleep.

## 1. Rulings (Brice, 2026-08-24, AskUserQuestion)

- **R-S3-1 — memory scope = episodic + semantic.** Both populated and retrieved in S3.
  The procedural store exists per the §9 schema but stays unpopulated (skills defer to a
  later slice; they mostly pay off on E5, not the E1 gate).
- **R-S3-2 — semantic writer = mechanical templates.** Lessons are extracted
  programmatically from verified episodes and rendered through one fixed template. No
  LLM distillation surface inside the instrument (a later exploratory variant may add
  proposer self-distillation behind a flag — not built now).
- **R-S3-3 — sleep cadence = episode threshold.** Sleep fires between tasks when ≥ N=16
  NEW verified episodes have accumulated since the last ACCEPTED sleep. N is a config
  knob pinned at `prereg-lock-a`. The trigger is a deterministic function of the run
  record, so a replay sleeps at identical points.

## 2. Memory organ (`crucible/memory/`)

- **Storage:** one SQLite file per arm run (a fresh organ per run — arms never share
  memory; the file path is part of the run record). Three typed stores; schema fields per
  pre-reg §9: MemOS `TextualMemoryMetadata` (confidence, source locators, status,
  version, history) + Graphiti temporal (`valid_at/invalid_at/expired_at/episodes[]`) +
  MIRIX credibility/evidence/lineage + ours (`last_verified_at`, `falsified_by`,
  `verification_method`). Content-addressed ids (hash of identity fields — the cognee
  idea); keys are identities, not descriptions.
- **Episodic record** — written mechanically by the driver for EVERY (task, arm) attempt
  set, success or not: task_key, class_id, phase, kind, prompt hash, per-candidate
  (patch hash, sandbox report summary, node status), landed module hash if any, budget
  spent, hidden verdict. `verified` ⇔ the pre-reg success definition (hidden∪visible
  pass, module-only, untampered). Only verified episodes feed distillation and sleep;
  all episodes are queryable.
- **Semantic lesson** — one per verified episode, per R-S3-2. Fields: unit_id, family,
  class_id, mutated spans, the landed unified diff (vs the mutated source), visible
  tests that flipped fail→pass, killing-test names, cited episode id. Rendered through
  ONE fixed template (template text is part of the spec-locked prompt surface).
- **Interfaces (per §9, binding):** `retrieve(unit, family, symptom) -> [Item with
  provenance]`; `write(episode)`; `refalsify(item) -> bool`.

## 3. Retrieval (the §4.4 A_full-only prompt block)

- Policy: exact (unit_id, family) class match first; else family-wide fallback — a NOVEL
  task can only ever get family-level lessons (this is where negative transfer is
  measurable, pre-reg §4.3).
- Ranking: not-falsified first, then `last_verified_at` recency, then confidence.
- **Context budget (hard):** ≤ 2 semantic lessons + ≤ 1 episodic exemplar, ≤ 4800 rendered
  characters (~1200 tokens — a character budget, since the client has no tokenizer), so
  prompt + 2048 generation fits the 8192 serving window. Overflow
  drops the exemplar first, then the second lesson — never truncates mid-item.
- Every task record carries the retrieved item ids (empty list ⇔ no-hit — None-vs-zero
  discipline). The A_noMem prompt is byte-identical to S2's (arms differ by exactly the
  pre-registered columns).

## 4. Falsification scheduler

- `refalsify(item)`: re-run the lesson's cited flipped visible test against its stored
  landed module in the sandbox. Pass ⇒ bump `last_verified_at`. Fail ⇒ set
  `falsified_by` (execution record id) — the item leaves retrieval permanently.
- Cadence: at each sleep event, batch: every item cited by episodes entering SFT + every
  retrieval-eligible item not verified since the previous sleep.
- Accounting: falsification executions are sleep-internal — never charged to any task's
  K=8 — and counted in the SleepRecord.

## 5. Sleep (`crucible/sleep/`)

- Trigger: R-S3-3 (≥16 new verified episodes since last accepted sleep), checked between
  tasks only.
- SFT set: ALL verified episodes to date, formatted as (the task's real prompt → the
  landed module) pairs. **Cumulative retrain from BASE each sleep** — not
  adapter-on-adapter — so there is no compounding drift and the adapter is a
  deterministic function of the episode set; rehearsal is inherent (old episodes stay in
  the set). PEFT LoRA rank 16 (the S1-verified config), TRL SFT, seeded.
- Regression slice: a seeded sample of min(12, solved-so-far) previously-solved tasks, re-run
  at K=1 greedy under the candidate adapter. **Accept iff solved count drops by ≤ 1.**
  Slice executions are sleep-internal (uncharged), recorded.
- On accept: hot-swap via vLLM's runtime LoRA API (the 1.5B SERVE entry carries
  `--enable-lora --max-lora-rank 32`; attach verified 2026-08-24). On reject: keep the
  prior adapter; the episode counter does NOT reset (next check fires next task).
- `SleepRecord` per pre-reg §8.6: episodes selected, adapter id, slice result, accepted,
  refalsification tally, GPU seconds. Adapter registry maps adapter_id → episode-set
  hash + parent base digest (lens `proposer.adapter_id` comes from here).

## 6. Value v1 (`crucible/value/`)

- Features per node: visible-pass fraction, executions used, mean_logprob,
  self_certainty, depth, family (one-hot over the 7 real families), retrieval-hit flag.
- Model: tiny online logistic regression (hand-rolled or sklearn SGD — no new heavy
  deps); `score(node) -> float` = P(hidden pass); `update(node, outcome)` fires once per
  task after the hidden verdict lands (closes the S2 "update() not wired" note).
- A_full's REx ordering uses v1; **A_noMem keeps ConstantValue** (its pilot/record
  already ran; arms differ only by pre-registered columns).

## 7. Uncertainty (`crucible/uncertainty/`)

- Isotonic calibration over value scores; crepes Mondrian conformal with provenance
  classes = {retrieval-hit, no-hit} × {phase 1, phase 2}.
- `confidence(node) -> (p, class)`; `should_abstain(p, cls)` implements the pre-reg §6
  abstention rule (2× multiplier lives in scoring, not here).
- `recalibrate(window)` fires after every ACCEPTED sleep (exchangeability is broken by
  design; recalibration is the pre-registered answer).

## 8. A_full assembly (`crucible/run/`)

- `ArmConfig` gains memory/sleep/value columns; A_full = the A_noMem driver + retrieval
  block + value v1 + abstention + the sleep loop between tasks. Exploratory sub-arm
  flags (`A_mem-sleep`, `A_sleep-mem`) exist in config but are not run in S3.
- Fresh memory DB per run; DB path, adapter lineage, and retrieval stats land in the
  ArmRecord and the lens.

## 9. S3 exit (per pre-reg §10)

- A_full end-to-end **smoke** on a 30-task slice of `1158e92f40ad` (NOT the gating run;
  the gating run is S4, after lock), run with the sleep threshold overridden to N=4 —
  at p0≈0.27 a 30-task smoke yields ~8 verified episodes, which never reaches N=16;
  the override is a smoke-only config value, and the N pinned at lock is untouched:
  completes within budget rules, ≥1 sleep event
  fires and is honestly accepted or rejected, all record types schema-complete +
  round-trip, refalsification observed at least once, lens carries adapter_id.
- All §9 interfaces implemented and tested; mutation-checked pins on load-bearing lines
  (budget non-charging of sleep/falsification executions, verified-only selection,
  accept threshold, retrieval ranking, context-budget drop order, counter non-reset).
- GPU steps (SFT, hot-swap) get a CPU-mockable seam; one live GPU smoke each.

## 10. Non-goals (S3)

Procedural skills; LLM self-distillation; the gating A_full run; E4 QuixBugs transfer;
any B-arm work; any pre-reg threshold change. THIRD_PARTY.md updated at merge for
newly-vendored schema ports (fields only — no code vendored from MemOS/Graphiti/MIRIX).
