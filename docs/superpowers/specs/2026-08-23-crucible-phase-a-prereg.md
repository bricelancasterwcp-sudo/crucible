# crucible — Phase A pre-registration and design

**Status:** DRAFT → becomes LOCKED at git tag `prereg-lock-a` (see §12). Written 2026-08-23 after the five open-component surveys (`docs/research/01..05`). Approved in chat by Brice 2026-08-23 (arms, endpoints, verdict rule, the three chosen numbers, the four survey amendments).

**Working name:** crucible. Rename freely; nothing here depends on it.

---

## 0. The question

Current LLMs learn once, freeze, and carry everything in weights. crucible inverts that: a **small frozen proposer** wrapped by **structured memory written continuously**, **reasoning as tree search whose nodes are scored by executing tests**, a **value function trained on real outcomes**, and **uncertainty derived from provenance + verification status**.

Phase A asks one falsifiable question:

> Does small-proposer + continuous structured memory + verify-by-execution search (a) get better at a task *class* on second exposure, and (b) match or beat a bigger frozen model given the same verification budget?

Pillar 1 (a non-token latent "intuition" predictor) is **not** tested in Phase A. It is the pre-committed follow-up if Phase A returns GO (§13).

## 1. Decisions of record (Brice, 2026-08-23)

| Decision | Ruling |
|---|---|
| Compute | This box (RTX 5080 16 GB, 29 GB RAM) + short cloud rentals only for training that won't fit. Phase A is expected to need zero rental. |
| Domain | Code repair of mutation-injected bugs in Python — verification is free and exact. |
| Approach | **Option A**: loop first, borrowed intuition. Frozen small open LLM as proposer; build pillars 2+3+5 around it; bigger frozen model as baseline. |
| Licensing | Apache-2.0 / MIT / BSD adoptable. Copyleft (GPL/AGPL/LGPL) never enters the tree. CC-BY-NC / research-only / custom / no-license = reference-only. Every adopted artifact's license is verified by command and recorded in `THIRD_PARTY.md`. |
| Chosen numbers | M = 50 (sleep interval, tasks) [amended pre-lock — see A2], K = 8 (test executions per task), abstention enrichment ≥ 2×. All three are **chosen, not derived** — they are fixed here so they cannot drift, and they are not tuned after any number is seen. |

## 2. Arms

All arms see the **identical task sequence** (same seeds, same order), the same sampler settings, and the same verification budget (§4.6). Only the rows below differ.

| Arm | Proposer | Explicit memory | Sleep (LoRA consolidation) | Search/verify | Role |
|---|---|---|---|---|---|
| **A_full** | small (Qwen3.5-2B) | yes | yes | yes | the system under test |
| **A_noMem** | small | no | no | yes | isolates memory from search; **structural control** (§4.8) |
| **B_search** | big (Qwen3.5-9B) | no | no | yes | fair "small+memory vs big" comparison |
| **B_naive** | big | no | no | best-of-K, no tree | "how LLMs operate today"; secondary, cheap |

**Exploratory (non-gating), run only if GPU-time allows after the gating arms:** A_mem−sleep (explicit memory, no LoRA) and A_sleep−mem (LoRA, no explicit store) — separates the store from the weights.

**Declared asymmetry:** A_full spends extra GPU time in sleep (LoRA training between tasks). This is the thesis (learning in the loop), not a confound; it is **reported** as GPU-minutes per arm, and it never counts against the per-task verification budget.

**Proposer fallback (pre-registered):** if PEFT LoRA cannot be attached to Qwen3.5-2B's Gated-DeltaNet layers (verified in Slice 1), the proposer for **all small arms** becomes Qwen2.5-Coder-1.5B-Instruct (Apache-2.0, plain `Qwen2ForCausalLM`). This is an infrastructure swap made before lock, not a post-hoc model change. Qwen2.5-Coder-**3B** is never used (qwen-research license).

**Baseline fallback:** Qwen2.5-Coder-14B-Instruct Q4_K_M (Apache-2.0) if Qwen3.5-9B fails the codec-landing check (§4.7).

## 3. Sampler and serving (pinned)

- Temperature 0.7, top_p 0.95, max_new_tokens **2048** [amended pre-lock — see A1], **thinking mode OFF** for all arms. Seed = hash(run_id, task_key, node_id, k) — deterministic and recorded.
- One inference server process per arm-run; served model identity (path + digest) asserted by the driver before the first task and re-asserted after every sleep reload. Mismatch = infrastructure failure, run aborts.
- vLLM is the intended server (n-best, logprobs, `/v1/load_lora_adapter` hot-swap). **llama.cpp `llama-server` is the committed fallback** (already on this box; `POST /lora-adapters` hot-swap; `n_probs`). Which one is used is recorded in the lens (§7). The sm_120 install is timeboxed to one working day in Slice 1; on expiry, fall back.

**Amendments (pre-lock; §12 protocol — date + old value):**
- **A1 (2026-08-23): `max_new_tokens` 1024 → 2048.** The S4.7 landing pre-check (run on the real 450-task stream before any arm) found the 1024 cap was the *dominant* codec-landing failure: the full-module-rewrite codec (§4.4) asks the model to re-emit the module **and** reproduce the whole visible test harness in one block, which routinely exceeds 1024 tokens, truncating the completion with its fence still open (rejected as `no-fence`). At the pinned 1024, Qwen3.5-2B landed 0.767 and Qwen2.5-Coder-1.5B-Instruct 0.80 — both below the 0.95 gate; raising the 1.5B to 2048 lifted landing to 0.92. This is the §4.7-sanctioned "codec fix, before any arm runs." Old value 1024. Evidence: `docs/findings/S2-ceiling-pilot.md`. Single source: `crucible.proposer.client.MAX_NEW_TOKENS`.
- Sleep is an explicit **stop-serve → train → evaluate adapter → reload** cycle (serving and training cannot co-reside on 14.5 GB usable VRAM) [amended pre-lock — see A3].
- **A2 (2026-08-24): sleep cadence — task-interval M = 50 → verified-episode threshold N = 16.** The approved S3 spec (R-S3-3, `docs/superpowers/specs/2026-08-24-crucible-s3-memory-design.md`) triggers sleep when 16 NEW verified episodes have accumulated since the last accepted adapter, checked between tasks — training-set growth, not task count, is what makes retraining worthwhile; a fixed task interval would fire empty or near-empty sleeps on hard stretches. Training is cumulative from the BASE model over ALL verified episodes each sleep (full-rehearsal equivalent; no separate rehearsal buffer). Old value: M = 50 tasks. N = 16 is chosen-not-derived, fixed here, not tuned after any number is seen (the ops smoke used threshold 4 purely to exercise loop mechanics — `docs/findings/S3-smoke.md`). Single source: `crucible/sleep/loop.py` `SLEEP_THRESHOLD_DEFAULT`.
- **A3 (2026-08-24): sleep serving cycle — stop-serve → train → reload becomes live co-residency + runtime hot-swap.** The stop-serve assumption was measured false: the 1.5B served at gpu-memory-utilization 0.45 (≈8.1 GiB actual hold) plus the trainer at batch 1 + gradient-accumulation 8 + checkpointing (≈5.2 GiB peak) ran four consecutive sleep cycles that trained and hot-loaded (`/v1/load_lora_adapter`) beside the live server (`docs/findings/S3-smoke.md` §2). Old text: explicit stop-serve → train → evaluate → reload cycle. Served identity is still re-asserted after every hot-load (mismatch = infrastructure failure), per this section's identity rule.
- **A4 (2026-08-24): adapter accept gate — fixed 20-unit regression slice from *excluded* units with Δ_min → min(12, |solved|) holdout slice of the arm's own already-solved stream tasks with ACCEPT_MAX_DROP = 1.** The gate's job is "did consolidation break what the arm could already do", which excluded-unit generality does not measure. Accept iff slice-solved-after ≥ slice-solved-before − 1. Old values: 20 excluded units, Δ_min pass-rate drop. Caveat (recorded in CARRIED-DEBT S3): the slice is graded on hidden suites of solved tasks — acceptable for a weights-gate, never for a score. Single source: `crucible/sleep/loop.py` `SLICE_SIZE`, `ACCEPT_MAX_DROP`.
- **A5 (2026-08-24): uncertainty instrument — crepes Mondrian conformal (abstain on empty prediction set at α = 0.1, categories by verification state) → stdlib PAVA isotonic calibration per provenance class {hit,nohit}×{p1,p2}, abstain iff calibrated p < 0.2 (inclusive at the threshold: p = 0.2 does not abstain).** crepes is a conformal-regression-interval library — the wrong instrument for calibrating a binary P(hidden pass); per-class PAVA keeps the Mondrian idea (conditional coverage by class) without the dependency (crepes stays a design reference in `THIRD_PARTY.md`, no code). Below MIN_OBS = 10 observations a class is honest passthrough (raw score, no fit). Post-sleep recalibration truncates each class's history to the trailing window, permanently — post-sleep fits never mix pre-sleep pairs (final-review I-1 ruling). Old values: crepes BSD-3 dependency, α = 0.1, verification-state categories {unexecuted, visible_partial, visible_passed}. Single source: `crucible/uncertainty/conformal.py`.

## 4. Task stream

### 4.1 Seed units
- **Sources (train stream):** EvalPlus HumanEval+ (164, Apache-2.0) and MBPP+ (378, Apache-2.0). HumanEvalPack (MIT) supplies the same 164 HumanEval problems plus `bug_type` / `failure_symptoms` fields used only for the realism check (§4.9). **542 unique functions.**
- **Held-out transfer set:** QuixBugs (40 programs, MIT, pre-LLM, human-authored one-line defects). Used for E4 only; never in the train stream.
- **Unit** = `<name>.py` (the canonical solution, **docstring stripped**, signature kept) + `test_<name>.py` (pytest, generated) + hidden tests. Each unit must pass its own full suite on the canonical solution, or it is dropped (named in `dropped`).
- **Visible vs hidden tests:** visible = asserts generated from EvalPlus `base_input`; hidden = asserts from `plus_input`. The agent may run visible tests (budgeted). **Success is judged on visible ∪ hidden.** This is what makes "verified-by-agent" distinct from "actually correct" and gives E3 its signal.

### 4.2 Mutants
- Engine: cosmic-ray (MIT) `mutate_code(src, operator, occurrence)` + a ported MutPy `StatementDeletion` operator (Apache-2.0, rewritten for `ast.Constant`). No CLI, no work-DB.
- **Operator families (pre-registered; frozen name→family map in `crucible/stream/families.py` at lock):**

  | Family | cosmic-ray operators |
  |---|---|
  | ARITH | `ReplaceBinaryOperator_*` (Add/Sub/Mul/Div/FloorDiv/Mod/Pow/bitwise/shift) |
  | CMP | `ReplaceComparisonOperator_*` |
  | BOOL | `AddNot`, `ReplaceTrueWithFalse`, `ReplaceFalseWithTrue`, `ReplaceAndWithOr`, `ReplaceOrWithAnd` |
  | UNARY | `ReplaceUnaryOperator_*` |
  | CONST | `NumberReplacer` |
  | FLOW | `ReplaceBreakWithContinue`, `ReplaceContinueWithBreak`, `ZeroIterationForLoop` |
  | EXC | `ExceptionReplacer`, `RemoveDecorator` |
  | VAR | `VariableReplacer`, `VariableInserter` |
  | SDL | ported statement deletion |

- **Valid mutant** := applies cleanly ∧ compiles ∧ differs from the original ∧ is **killed by the visible suite** (≥1 visible test fails — so the agent always has a symptom) within the sandbox limits (§4.5). A mutant that hangs is *killed-by-timeout* and is valid (that is a real bug class) but flagged `kills_by_timeout=true`. Mutants killed only by hidden tests, or not killed at all, are excluded (no symptom / equivalent — no signal); both exclusion reasons are counted in the stream manifest.
- **Mutant key** = sha256(unit_src_hash ‖ unified_diff). Keys are identities, not descriptions.

### 4.3 Classes, exposures, novelty
- **Class** = (unit, family). A class is **eligible** only if it has ≥ 2 valid mutants at **distinct source sites** (different `(line, col)` span).
- **Phase 1 (first exposure):** C eligible classes, one mutant each (`m1`). Random order, seed 0.
- **Phase 2 (second exposure + novel):** for each phase-1 class, a *different* mutant `m2` at a different site; plus N_nov tasks from **novel units** (units absent from phase 1 entirely), one mutant each. Interleaved, random order, seed 1.
- **Targets:** C = 200, N_nov = 50. Actual counts recorded at lock; Δ_min is computed from them (§6).
- Novel tasks sit after memory has accumulated so negative transfer is measurable.

### 4.4 What the agent sees and emits
- Input: the mutated module source, the visible test file, the output of **one free execution** of the visible suite (the symptom), and — A_full only — retrieved memory (§8.3). The agent does not see the hidden tests or the original source.
- Output: a **full-module rewrite** (the unit is one function; this codec lands ~100% and removes codec-landing as a variable — §4.7), plus `status ∈ {verified_visible, believed, abstain}` and `confidence ∈ [0,1]`.
- Test files are hash-locked; a submission that alters them, or touches anything but the module, scores as failure with reason `tampered`.

### 4.5 Sandbox
- Each execution runs in a fresh temp dir, `PYTHONDONTWRITEBYTECODE=1`, `__pycache__` purged before every run, no network, `RLIMIT_AS` 1 GiB, **per-test timeout 5 s** (pytest-timeout), **per-execution wall cap 60 s**, subprocess-isolated (mini-swe-agent `LocalEnvironment` pattern; Docker only for a future tier-2 corpus).
- `TestReport = {passed: [id], failed: [id], timed_out: [id], errored: [id], wall_s, infra_error: str|None}`. `infra_error` set ⇒ the execution is **not** a measurement and is not charged to the budget; it is counted and reported.

### 4.6 Verification budget (the fair unit)
- **K = 8** executions of the visible suite (any subset) per task, across the whole search tree, after the one free symptom run. Cached results are free. Hidden tests are never executable by the agent.
- Wall-clock safety cap 900 s per task — a valve, not a budget. If > 2% of tasks in any arm hit it, that is reported as a confound on that arm.
- Token budgets differ by model size by design; executions are the comparable unit. Tokens, wall time, and GPU-minutes are recorded per task and per arm.

### 4.7 Codec landing pre-check
Before lock, run `assay`'s landing probe (already on this box) against each served model with the full-module-rewrite codec. Required: ≥ 95% parseable submissions on 30 smoke tasks. Failure ⇒ baseline fallback (§2) or codec fix, before any arm runs.

### 4.8 Structural pre-checks (run before any arm sees a task)
Computable in seconds; all must pass or the stream is rebuilt:
1. Phase-1 vs phase-2 `m1`/`m2` sets: identical family distribution (by construction — assert it), and the distributions of (a) number of killing tests, (b) unit source length, (c) `kills_by_timeout` rate lie within 2·SE of each other.
2. Novel units are disjoint from phase-1 units (assert).
3. Every class has `m1.site ≠ m2.site` (assert). *[Clarified 2026-08-23, per §11, with the rung-1 build (see `2026-08-23-crucible-s2.5-stack2-design.md`): at rung ≥ 1 a task carries a site-**set** (two sites), and this assertion is **disjointness of m1's and m2's site-sets**. Original wording kept above; no threshold changes.]*
4. **Ceiling / contamination pilot:** run **A_noMem** on a 30-task pilot drawn from phase 1. If success > 0.70, the stream is too easy for E1 to be detectable; apply the **hardening ladder in this order, one rung at a time, re-running the pilot**: (i) two-site mutants; (ii) MBPP+-only seeds; (iii) α-rename identifiers (semantics-preserving); (iv) tier-2 real repos via SWE-smith procedural bugs. The rung reached is recorded and fixed at lock.
5. **Live control:** A_noMem runs **first and completely**. Its second-minus-first delta must lie within ±Δ_min of zero. If not, the stream is confounded: fix, rebuild, rerun A_noMem from zero. A_full never runs until this passes. (This is why it is a preflight and not a re-roll — no A_full number has been read.)

### 4.9 Realism check (reported, not gating)
Compare the family distribution of *valid* mutants against HumanEvalPack's human-authored `bug_type` distribution. A large mismatch is a stated limitation of every transfer claim, not a reason to alter the stream after lock.

## 5. Memory, search, value, uncertainty in A_full — behaviour contracts

These are contracts the implementation must satisfy; the design detail is in §8.

- **Memory is written after every task** (episodic always; semantic claims derived and stamped with provenance; procedural skills scored), never only on success.
- **Retrieval is structured first**: keyed on (unit, family, failing-test signature); any embedding similarity is a secondary tiebreak only and is logged as such. Retrieved items carry their provenance and verification status into the prompt.
- **Falsification is re-execution**: a semantic claim used in a task is re-verified by running the test it cites when its `last_verified_at` is older than the unit's last change; a failed re-check marks it `falsified_by=<run>` and it is never retrieved as a fact again (it remains in the ledger).
- **Sleep every M = 50 tasks** [amended pre-lock — see A2, A3 in §3]: select **verified-only** episodes (hidden-pass ∧ not `tampered`), train a LoRA (PEFT r=16, bf16, gradient checkpointing) with a rehearsal buffer, **evaluate** the adapter on a fixed 20-unit regression slice from the stream's *excluded* units; accept only if the slice's pass rate does not drop by more than Δ_min; reload [slice and gate amended pre-lock — see A4 in §3]. Rejected adapters are kept on disk and logged.
- **Search** = REx Thompson-sampling scheduler over candidate "arms" (each candidate patch is an arm; children are refinements), rewards in [0,1] = fraction of visible tests passing; budget-aware: it must never exceed K executions.
- **Value function** = predicts P(hidden pass | node features) **before** execution; trained online from outcomes; used to order which candidates get the scarce executions. v0 = logistic/MLP over (visible tests passed/failed so far, failing-test signature class, diff size, proposer mean logprob / self-certainty, retrieved-memory verification status); recorded per node.
- **Uncertainty** = isotonic-calibrated value score wrapped in Mondrian conformal prediction (crepes, BSD-3). **Mondrian category = the candidate's verification state** ∈ {unexecuted, visible_partial (executed, some visible tests fail), visible_passed}, so coverage is guaranteed conditional on how much the candidate has actually been checked. The per-node `confidence` is recorded **before** each execution. The submission `status` (§4.4) derives from it: `verified_visible` if the final candidate passed the visible suite; `believed` if submitted without a visible pass; **`abstain` if the best candidate's prediction set is empty at α = 0.1**. **Recalibrated after every sleep** on a rolling recent window (exchangeability is violated by design; this is a first-class risk). [Library, α, categories, and abstain rule amended pre-lock — see A5 in §3.]

## 6. Endpoints

Let `succ(arm, S)` be the hidden-suite success rate of an arm over task set S. Let `p0` be the A_noMem pilot success rate (§4.8.4).

**Δ_min (derived, computed at lock):** Δ_min = 2 · sqrt(2 · p0(1−p0) / C), using the actual C and p0 recorded at lock. Sizing target C = 200 so that Δ_min ≤ ~10 pp (at p0 = 0.3: 9.2 pp). Sanity mark: Δ_min must land in [5 pp, 12 pp]; outside that range, stop and re-derive in the open before lock.

| ID | Endpoint | Pass condition | Gating? |
|---|---|---|---|
| **E1** | memory works | Δ_A = succ(A_full, phase-2 second-exposure) − succ(A_full, phase-1) ≥ Δ_min, **and** the same delta for A_noMem lies in ±Δ_min (control). Point estimate decides; paired per-class differences and their CI are reported alongside. | yes |
| **E2a** | beats big on repeats | succ(A_full, second-exposure) ≥ succ(B_search, second-exposure) | yes |
| **E2b** | competitive on new | succ(A_full, phase-1 ∪ novel) ≥ succ(B_search, phase-1 ∪ novel) − Δ_min | yes |
| **E3a** | uncertainty is informative | AUROC of A_full's **node-level calibrated confidence, recorded before each execution** (§5), against that candidate's hidden pass, over all executed candidates, ≥ 0.5 + 2·SE_AUROC (Hanley–McNeil, from the lock-time n estimate) | yes |
| **E3b** | abstention is honest | among tasks where A_full abstains, the failure rate of its best candidate ≥ 2× the overall A_full failure rate. If A_full never abstains, E3b is **None / not exercised**, reported as such, and does not block. | yes (unless not exercised) |
| E4 | tactic transfer (held-out) | succ(A_full, QuixBugs) − succ(A_noMem, QuixBugs), both run after phase 2, reported with CI. Interpretation: > Δ_min ⇒ memory transfers tactics, not just unit answers. | no |
| E5 | negative transfer | succ(A_full, novel) − succ(A_noMem, novel), reported. | no |
| E6 | B_naive vs B_search | reported; quantifies what search alone buys the big model. | no |

## 7. Verdict rule (written now, obeyed later)

- **GO** = E1 ∧ E2a ∧ E2b ∧ E3a ∧ (E3b ∨ E3b not exercised). → Full system design + Phase-B pre-registration (§13).
- **NO-GO** = E1 fails with controls clean. The memory/consolidation story is wrong at this scale. Ship as findings; stop.
- **PARTIAL** = E1 passes, any of E2a/E2b/E3 fails. Memory helps but small+memory does not reach big+search, or uncertainty is not informative. **Brice rules**; I do not.
- **CONFOUNDED** = a §4.8 pre-check fails. No verdict. Fix the stream, rebuild, rerun from zero — permitted only because no A_full number has been read (A_noMem runs first).
- **The point estimate decides.** No extension, re-run, corpus change, threshold change, or additional arm after A_full's numbers are seen. Exploratory arms are labelled exploratory in every table and never promote to gating.
- **Infrastructure kills** (server crash, disk full, power) with no A_full numbers read may be cleanly rerun from zero. Partial data is never spliced; deterministic seeds make resumption byte-identical or the run restarts.

## 8. Instrument honesty (non-negotiable)

1. **None-vs-zero.** Every per-task field is `None` until measured; a `dropped: [(task_key, reason)]` list is written per arm. `rate: None` never passes an endpoint.
2. **Infra ≠ subject.** `infra_error` executions are not charged, not scored, counted and reported. A protocol-breaking server reply (HTTP 200 with missing usage/logprobs) raises as infrastructure (assay's "stats-free-200" class).
3. **Name the lens.** Every reported rate carries `lens = {success_def: "hidden∪visible pass, module-only patch, test files unmodified", budget: K=8 + 900s valve, sampler: {T, top_p, max_tokens, thinking:off, seed_rule}, proposer: {name, digest, adapter_id}, server: {kind, version}, stream: {hash, rung}}`.
4. **Per-test wall-clock kill** (5 s) and **per-execution cap** (60 s) are part of the success definition; a hang is `timed_out`, which is a failure, not an infra error.
5. **`__pycache__` purge + `PYTHONDONTWRITEBYTECODE=1`** on every execution, and in every mutation-test script.
6. **Records:** one JSONL line per (task, arm) keyed by `task_key`; per-node JSONL for search; per-sleep JSONL (episodes selected, adapter id, regression-slice result, accepted?). Schema-completeness test + dataclass round-trip test for every record type, so a new field cannot be silently dropped.
7. **Aggregates hide drift:** report success per family and per phase for every arm, not only whole-stream.
8. **Runs > 2 h are OS-detached** (`setsid nohup`), write a pid file and a `.DONE` marker, and a watcher distinguishes silence from success. GPU hygiene: unload Ollama and other daemons, assert free VRAM ≥ 13 GB before starting an arm.
9. **Mutation-test every test that pins a load-bearing behaviour** (budget meter, timeout handling, tamper detection, key hashing, record completeness, Δ_min derivation): break the line, the test must fail, restore.
10. **Dry run before every long run** checks *artifacts* (is the value in the file the next step reads?), not just exit codes.

## 9. Architecture (interface level)

Python package `crucible/` (uv-managed venv, Python 3.12 — torch/vLLM wheel coverage; sensorium also needs 3.12+). Organised by feature; files ≤ 400 lines typical.

| Module | Responsibility | Key interface / invariant |
|---|---|---|
| `crucible/stream/` | unit builder (EvalPlus → module + visible/hidden pytest files), mutant generator (cosmic-ray + SDL), validator (sandbox-killed ⇒ valid), class/phase composer, structural pre-checks, stream manifest | `build_stream(seed, C, N_nov) -> StreamManifest` (content-hashed; deterministic); `precheck(manifest) -> Report` with every §4.8 assertion named |
| `crucible/sandbox/` | isolated pytest execution, budget meter | `run(unit, patch, subset) -> TestReport`; `BudgetMeter.charge(report)` raises `BudgetExhausted` before the 9th execution; infra errors never charged |
| `crucible/proposer/` | `Proposer` protocol + vLLM / llama.cpp adapters, identity assertion, logprob capture | `generate(prompt, n, seed) -> [Candidate{text, mean_logprob, self_certainty}]`; `assert_identity(expected_digest)` |
| `crucible/search/` | REx scheduler (ported verbatim, 4 files), `Node{patch, parent, report, citations, status}`, value-fn ordering, budget-awareness | search never calls `sandbox.run` beyond the meter; every node is recorded |
| `crucible/memory/` | SQLite typed stores: episodic, semantic, procedural; record schema ported from MemOS `TextualMemoryMetadata` (confidence, source locators, status, version, history) + Graphiti temporal fields (`valid_at/invalid_at/expired_at/episodes[]`) + MIRIX `skill_experience` (credibility, evidence, lineage) + our `last_verified_at`, `falsified_by`, `verification_method`; structured retrieval; falsification scheduler; writer | content-addressed ids (cognee `identity_fields` idea); `retrieve(unit, family, symptom) -> [Item with provenance]`; `write(episode)`; `refalsify(item) -> bool` re-runs the cited test |
| `crucible/sleep/` | SEAL-skeleton consolidation: select verified episodes → PEFT/TRL LoRA SFT with rehearsal → regression-slice eval → accept/reject → hot-swap via server API; adapter registry | `sleep(store, regression_slice) -> SleepRecord{adapter_id, accepted, slice_delta}` |
| `crucible/value/` | online value model v0 (features → P(hidden pass)); training from outcomes; feature schema | `score(node) -> float`; `update(node, outcome)` |
| `crucible/uncertainty/` | isotonic calibration + crepes Mondrian conformal; abstention rule; recalibration hook on sleep | `confidence(node) -> (p, provenance_class)`; `should_abstain(p, cls) -> bool`; `recalibrate(window)` |
| `crucible/run/` | arm configs (A_full, A_noMem, B_search, B_naive), driver, OS-detach wrapper, `.DONE` markers, record writers, `lens` builder, analysis (endpoints, Δ_min, AUROC, per-family tables), verdict printer | `run_arm(arm, manifest) -> ArmRecord`; `analyze(records) -> Verdict` with every endpoint's inputs named |
| `THIRD_PARTY.md` | every adopted/ported artifact: name, URL, license, verification command, what was taken | maintained at merge; CI check that every `vendor/` dir is listed |

**Adopted / ported (verified licenses):** REx (MIT) scheduler files verbatim; mini-swe-agent (MIT) `LocalEnvironment` pattern; cosmic-ray (MIT) operators; MutPy (Apache-2.0) SDL operator rewritten; moatless-tools (MIT) `pytest_parser.py`; MemOS (Apache-2.0) / Graphiti (Apache-2.0) / MIRIX (Apache-2.0) schemas only; cls-ledger (MIT) selection policy as design; SEAL (MIT) loop skeleton as design; PEFT/TRL (Apache-2.0), bitsandbytes (MIT), crepes (BSD-3), LM-Polygraph (MIT) as libraries. Reference-only and never vendored: pyactr (GPL), mergekit/TorchCP (LGPL), AlphaCodium (AGPL), swe-rl (CC-BY-NC), AutoCodeRover (source-available), BugsInPy / ReST-MCTS / Psearch / CodePRM / PURE (no license).

## 10. Slices (for the implementation plan)

Each slice ends with its acceptance tests green, mutation-tested, and a dry run that checks artifacts.

1. **S1 — environment + stream + sandbox + pre-checks.** uv venv (3.12), torch for sm_120 (timeboxed), vLLM-or-llama.cpp serving Qwen3.5-2B with an identity assertion, LoRA-attach smoke test on Qwen3.5-2B (decides the proposer fallback), unit builder, mutant generator + validator, stream composer, §4.8.1–3 pre-checks, sandbox with budget meter, codec-landing probe via assay. **Exit:** `build_stream` deterministic across two runs (same hash); 30-task smoke through the sandbox; pre-checks 1–3 green.
2. **S2 — search + A_noMem + B arms + ceiling pilot.** Proposer adapters, REx search, value-fn v0 scaffold (untrained: constant), driver, records, lens, analysis for E1/E2 inputs. Run the **ceiling pilot** (§4.8.4) and apply the hardening ladder if needed. **Exit:** A_noMem pilot number recorded; rung fixed; B arms smoke-tested.
3. **S3 — memory organ + value + uncertainty + sleep → A_full.** Stores, retrieval, writer, falsification, value training, calibration/conformal/abstention, sleep cycle with hot-swap and regression slice. **Exit:** A_full smoke on 10 tasks writes every record type; a forced falsification demo; one sleep cycle accepted or rejected with reasons.
4. **S4 — lock + run + analysis + verdict.** Compute Δ_min from pilot p0 and actual C; tag `prereg-lock-a`; run A_noMem fully (control check §4.8.5); then A_full, B_search, B_naive; E4 QuixBugs pass; analysis; verdict report with every lens; CARRIED-DEBT.md; withdrawn-claims log if any headline fails to hold.

Realism check (§4.9) and exploratory arms slot in after S4's gating arms if time allows.

## 11. Threats to validity (named now)

- **Contamination / memorization.** Small models have seen HumanEval. Mitigations: docstring stripping, hidden tests, the ceiling pilot + hardening ladder, novel-unit tasks (E5), QuixBugs transfer (E4). A GO on E1 alone could still be "it memorized unit answers via LoRA" — E4/E5 and the exploratory split arms are how that is distinguished; the report will say which it was.
- **Mutant realism.** §4.9 comparison is reported; no transfer claim beyond the mutant distribution is made without it.
- **Codec landing.** Removed as a variable by full-module rewrite + §4.7 pre-check.
- **Ceiling.** §4.8.4.
- **Exchangeability.** Recalibration after every sleep; E3 is reported per phase as well as overall.
- **Blackwell toolchain.** Timeboxed; llama.cpp fallback pre-committed; fallback recorded in the lens.
- **GPU time-sharing.** Sleep's stop/train/reload cycle is wall-clock accounted and reported; it never touches the verification budget.
- **Operator family taxonomy is a choice.** Frozen at lock; results are also reported per family.

## 12. Lock procedure and records

- **Lock** = git tag `prereg-lock-a` on the commit that records: stream hash, C, N_nov, rung, p0, Δ_min, proposer/baseline digests, server kind, families map. Nothing in §§2–7 changes after the tag; amendments before the tag are footnoted with date and old value.
- `docs/CARRIED-DEBT.md` appended at every slice merge: settled → deferred with rulings → process lessons; resolved items struck through, never deleted.
- `docs/WITHDRAWN-CLAIMS.md` records any headline that fails to replicate, with the arithmetic.
- All runs under `runs/<arm>/<run_id>/` (git-ignored), each with `lens.json`, `records.jsonl`, `nodes.jsonl`, `sleeps.jsonl`, `dropped.json`, `.DONE`.

## 13. Pre-commitment for Phase B (only on GO)

B-lite per `docs/research/05`: frozen Apache-2.0 code encoder (jina-embeddings-v2-base-code or Qwen3-Embedding-0.6B) + ~100–150M action-conditioned latent predictor (EB-JEPA shape, LeWorldModel two-term objective, SIGReg reimplemented from the paper or lifted from `klindtlab/lejepa-identifiability` — never from the CC-BY-NC `lejepa`) + a grounded head predicting the observable outcome; data from sensorium traces over a CRUXEval-style generated corpus; **mandatory control = `microsoft/codeexecutor` (MIT, 125M, token-space)**; ≈ 6 GB peak VRAM. Kill criterion to be pre-registered then: if B-lite's outcome-prediction AUROC does not exceed the codeexecutor control's by 2·SE on the same held-out traces, B stops.
