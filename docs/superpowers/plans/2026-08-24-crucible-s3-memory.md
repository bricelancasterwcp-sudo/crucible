# S3 Memory Organ + Value + Uncertainty + Sleep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build A_full — search + explicit memory + sleep — per the S3 spec, exiting on a 30-task smoke where at least one sleep fires and every record is schema-complete.

**Architecture:** Four new packages (`crucible/memory/`, `crucible/sleep/`, plus `crucible/value/online.py` and `crucible/uncertainty/`) behind the pre-reg §9 interfaces, threaded into the existing S2 machinery at four seams: `build_prompt` gains a `memory` block, `search`/`attempt_task` thread it, `run_arm` gains an optional hooks object, and the record/lens schemas gain trailing fields. A_noMem's behavior stays byte-identical (guard-tested).

**Tech Stack:** Python 3.12, SQLite (stdlib `sqlite3`), crepes (BSD-3, new dep), PEFT/TRL (already in `[serve]` extra, GPU-only paths seam-mocked).

**Spec:** `docs/superpowers/specs/2026-08-24-crucible-s3-memory-design.md` (rulings R-S3-1..3). Pre-reg §8 (instrument honesty) and §9 (interfaces) bind everything.

## Global Constraints

- **R-T2-6:** every pytest run touching `tests/sandbox` (incl. full-suite runs) goes under `systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0`. `PYTHONDONTWRITEBYTECODE=1` + `__pycache__` purge in every mutation check.
- **All randomness seeded and purpose-scoped** (`random.Random(f"{seed}:<purpose>")`); value scoring MUST be deterministic (pre-reg: "a value that varied between calls would fabricate the experiment").
- **Frozen dataclass field order is API:** new fields go at the END with defaults; `from_dict` stays backward-compatible via `d.get`/`pop`-with-default for every new field.
- **Schema-completeness + round-trip tests for EVERY record type** (pre-reg §8.6) — new types (EpisodicRecord, SemanticItem, SleepRecord) and extended ones (TaskRecord).
- **Sleep-internal and falsification executions are NEVER charged to any task's K=8** — mutation-pin this.
- **A_noMem is untouchable:** its prompt, search behavior, and records must be byte-identical to S2.5's (guard tests in Tasks 6 and 11). Arms differ only by pre-registered columns.
- **Memory organ is per-run:** fresh SQLite file per arm run; arms never share memory.
- **GPU-touching code (LoRA SFT, adapter hot-swap) sits behind seams** with fake implementations for unit tests; live GPU verification is ops (Task 12), not pytest.
- Verified ⇔ the pre-reg success definition: `hidden_pass is True` (never merely truthy; `None` = not measured).
- Commit convention `<type>(scope): summary`, no attribution trailers. Run tests with `.venv/bin/python -m pytest <path> --color=no` piped through `tr '\r' '\n'` (this box prints no `-q` summary line; use `-v` or exit codes for evidence).

---

### Task 1: Memory schema (`crucible/memory/schema.py`)

**Files:**
- Create: `crucible/memory/__init__.py` (empty), `crucible/memory/schema.py`
- Test: `tests/memory/__init__.py` (empty), `tests/memory/test_schema.py`

**Interfaces:**
- Produces: frozen dataclasses `EpisodicRecord` and `SemanticItem`, each with `to_dict`/`from_dict` (exact inverse, JSON-native values only) and a module function `content_id(kind: str, identity_fields: dict) -> str` = sha256 of the canonical JSON of `{"kind": kind, **identity_fields}` (sorted keys). Every record's `item_id` is derived by `content_id`, never hand-assigned.
- `EpisodicRecord` fields (in this order): `item_id: str`, `task_key: str`, `arm: str`, `unit_id: str`, `family: str`, `class_id: str`, `phase: int`, `kind: str`, `root_prompt: str`, `landed_module: str | None`, `visible_reward: float`, `executions_charged: int`, `hidden_pass: bool | None`, `verified: bool`, `memory_item_ids: tuple[str, ...]`, `created_at: str` (ISO-8601, caller-supplied — no clock reads inside the dataclass), `confidence: float`, `status: str`, `version: int`, `source_locator: str`, `valid_at: str`, `invalid_at: str | None`, `expired_at: str | None`, `last_verified_at: str | None`, `falsified_by: str | None`, `verification_method: str`. Identity fields for `content_id`: `{"task_key", "arm"}` (one episode per task per arm).
- `SemanticItem` fields: `item_id`, `unit_id`, `family`, `class_id`, `cited_episode_id: str`, `mutated_spans: tuple` (nested list JSON form like `TaskSpec.span`), `landed_diff: str`, `flipped_tests: tuple[str, ...]`, `killing_tests: tuple[str, ...]`, `created_at`, `confidence`, `status`, `version`, `source_locator`, `valid_at`, `invalid_at`, `expired_at`, `last_verified_at`, `falsified_by`, `verification_method`. Identity fields: `{"cited_episode_id"}` (one lesson per verified episode).
- `verified` on an episode is DERIVED at construction helper `episode_verified(hidden_pass, tampered) -> bool` = `hidden_pass is True and not tampered` — pin with a test that `hidden_pass=None` is NOT verified.

- [ ] **Step 1: Write failing tests** — round-trip equality for both types (`from_dict(to_dict(x)) == x` through `json.dumps`), schema-completeness (`set(to_dict()) == {f.name for f in fields(cls)}`), `content_id` stability (same identity fields ⇒ same id; different ⇒ different; insertion order of the dict irrelevant), `episode_verified(None, False) is False`, `episode_verified(True, True) is False`, `episode_verified(True, False) is True`. Write the fixtures concretely — no factories, literal values.
- [ ] **Step 2: RED.** `.venv/bin/python -m pytest tests/memory/test_schema.py -v` → ImportError.
- [ ] **Step 3: Implement.** Follow the repo's frozen-dataclass idiom exactly (`compose.py` `TaskSpec` is the template: `asdict`-based `to_dict` with explicit tuple↔list conversions, `from_dict` restoring tuple shapes). Module docstring states the MemOS/Graphiti/MIRIX field provenance (schema ported, no code vendored) and the identity-not-description key rule.
- [ ] **Step 4: GREEN**, then **Step 5: Commit** `feat(s3): memory schema — episodic + semantic records, content-addressed ids`.

---

### Task 2: SQLite store (`crucible/memory/store.py`)

**Files:**
- Create: `crucible/memory/store.py`
- Test: `tests/memory/test_store.py`

**Interfaces:**
- Consumes: Task 1's dataclasses + `content_id`.
- Produces: `class MemoryStore` with `__init__(self, db_path: Path)` (creates file + tables idempotently; three tables `episodic`, `semantic`, `procedural` — procedural created with the same column layout as semantic but never written in S3, per R-S3-1); `write_episode(rec: EpisodicRecord) -> None` (INSERT OR REPLACE by item_id); `write_semantic(item: SemanticItem) -> None`; `episodes(verified_only: bool = False) -> list[EpisodicRecord]` (insertion order); `semantic_for(unit_id: str, family: str) -> list[SemanticItem]` (exact class match); `semantic_family(family: str) -> list[SemanticItem]` (family-wide); `mark_falsified(item_id: str, falsified_by: str) -> None`; `mark_verified(item_id: str, at: str) -> None`; `count_verified_since(marker: int) -> int` and `verified_count() -> int` (for the sleep trigger); `close()`. Rows are stored as one JSON TEXT column `payload` plus indexed columns `item_id PRIMARY KEY, unit_id, family, verified INT` — retrieval filters on indexed columns, payload round-trips the full dataclass (no field can be silently dropped: reading is `from_dict(json.loads(payload))`).

- [ ] **Step 1: Failing tests** — create in `tmp_path`; write/read round-trip both types with dataclass equality; `episodes(verified_only=True)` returns only `verified` episodes; `semantic_for` exact-match vs `semantic_family` superset; `mark_falsified` then `semantic_for` still RETURNS the item but with `falsified_by` set (filtering is the retriever's job — the store is honest storage); idempotent re-open of an existing db; `verified_count` arithmetic; procedural table exists (`sqlite_master` query) and a `write_procedural` method does NOT exist (`not hasattr`).
- [ ] **Step 2: RED. Step 3: Implement** (stdlib `sqlite3`, `check_same_thread=False` NOT needed — single-threaded by design, say so in the docstring). **Step 4: GREEN.**
- [ ] **Step 5: Mutation check** — break `verified_only` filtering (drop the WHERE): the verified-only test must FAIL; restore, purge pycache, re-run. Commit `feat(s3): memory store — sqlite typed stores, honest payload round-trip`.

---

### Task 3: Mechanical distiller (`crucible/memory/distill.py`)

**Files:**
- Create: `crucible/memory/distill.py`
- Test: `tests/memory/test_distill.py`

**Interfaces:**
- Consumes: Task 1 types; `crucible.stream.mutants` diff helper style (build the landed diff with `difflib.unified_diff` against the MUTATED source, fixed headers `a/<module>.py -> b/<module>.py`, no dates — mirror `mutants._unified`).
- Produces: `distill(episode: EpisodicRecord, *, mutated_src: str, spans: tuple, flipped_tests: tuple[str, ...], killing_tests: tuple[str, ...], now: str) -> SemanticItem` — REFUSES (raises `ValueError`) a non-verified episode or one with `landed_module is None`; and `render_lesson(item: SemanticItem) -> str` — the ONE fixed template (R-S3-2), exact text:

      ### Prior verified fix in this code (family {family})
      The altered region was at spans {spans}. The repair that passed re-execution:
      ```diff
      {landed_diff}
      ```
      Visible tests that flipped from failing to passing: {flipped_tests}.

  (Template shown indented; the constant contains it UNindented. Rendered with
  `", ".join` for test lists and spans in JSON list form.)

  The template string is a module constant `LESSON_TEMPLATE` — spec-locked prompt surface, say so in its docstring.

- [ ] **Step 1: Failing tests** — distill on a verified episode produces a SemanticItem whose `landed_diff` contains the `-`/`+` lines of a known one-line change and whose `cited_episode_id` is the episode's id; `ValueError` on `verified=False` and on `landed_module=None`; `render_lesson` output contains the family, the diff fence, and the flipped test names, and is byte-stable for identical items (call twice, compare).
- [ ] **Step 2: RED. Step 3: Implement. Step 4: GREEN. Step 5:** Mutation check — make `distill` accept non-verified episodes (drop the guard): the ValueError test must FAIL. Commit `feat(s3): mechanical distiller — verified episodes only, one fixed lesson template`.

---

### Task 4: Retrieval (`crucible/memory/retrieve.py`)

**Files:**
- Create: `crucible/memory/retrieve.py`
- Test: `tests/memory/test_retrieve.py`

**Interfaces:**
- Consumes: `MemoryStore`, `render_lesson`.
- Produces: `CONTEXT_BUDGET_CHARS = 4800`; `retrieve(store: MemoryStore, unit_id: str, family: str) -> RetrievedBlock` where `RetrievedBlock` is a frozen dataclass `{block: str | None, item_ids: tuple[str, ...]}` (`block=None` ⇔ nothing retrieved — None-vs-zero: an empty organ yields `None`, never `""`). Policy per spec §3: class-exact `semantic_for` first; if empty, `semantic_family` (novel-task path); filter OUT falsified items (`falsified_by is not None`); rank by (`last_verified_at` DESC with `None` last, then `confidence` DESC, then `item_id` for determinism); take ≤2 lessons + ≤1 episodic exemplar (the exemplar: the landed module of the most recent verified episode for the same (unit, family) class, rendered under a `### A prior working version of this module` header — omit entirely on the family-fallback path, exemplars are class-specific); enforce the char budget by dropping the exemplar first, then the second lesson — never truncate mid-item; if the FIRST lesson alone exceeds the budget, return `block=None` (and no ids).
- Header line of a non-None block: `## Prior experience with this code` followed by the rendered items.

- [ ] **Step 1: Failing tests** — exact-class beats family-wide (seed a store with both; retrieved ids prove it); falsified items never appear; ranking (three items with distinct `last_verified_at`, assert order); budget drop order (construct oversized items — exemplar dropped first, then lesson 2; a single oversized lesson ⇒ `block=None`); determinism (two calls, equal results); empty store ⇒ `RetrievedBlock(None, ())`.
- [ ] **Step 2: RED. Step 3: Implement. Step 4: GREEN. Step 5:** Mutation checks — (a) remove the falsified filter → killed; (b) invert the drop order → killed. Commit `feat(s3): retrieval — class-then-family, ranked, hard char budget`.

---

### Task 5: Falsification (`crucible/memory/falsify.py`)

**Files:**
- Create: `crucible/memory/falsify.py`
- Test: `tests/memory/test_falsify.py` (sandbox-touching: cap any full-file run)

**Interfaces:**
- Consumes: `crucible.sandbox.task_run.run` (same call the arms use: `run(unit, patch, subset)`), `MemoryStore`.
- Produces: `refalsify(store, item: SemanticItem, unit: Unit, *, now: str) -> bool` — re-runs the item's `flipped_tests` subset (`run(unit_with_landed_module, landed_module, subset=flipped_tests)`) where the module under test is the cited episode's `landed_module`; pass (all flipped tests pass) ⇒ `mark_verified(item_id, now)` + return True; fail ⇒ `mark_falsified(item_id, <exec description string>)` + return False; an `infra_error` result changes NOTHING (not a measurement — item stays as it was, return True with a docstring note) — pin this. Also `falsify_batch(store, items_with_units: list[tuple[SemanticItem, Unit]], *, now) -> FalsifyTally` (frozen dataclass `{checked: int, passed: int, falsified: int, infra: int}`).
- These executions are NEVER charged to any task budget — there is no BudgetMeter in this module at all; state that in the module docstring and pin it by asserting the module source does not import `budget` (a one-line test: `"budget" not in Path(falsify.__file__).read_text()` — crude but honest).

- [ ] **Step 1: Failing tests** — use a tiny real unit (follow `tests/stream/test_stack.py::_unit`'s idiom): a lesson whose landed module genuinely passes its flipped test ⇒ True + `last_verified_at` bumped; a lesson whose stored module now fails (construct by storing a broken module) ⇒ False + `falsified_by` set; infra path (monkeypatch `run` to return a report with `infra_error="boom"`) ⇒ item untouched; tally arithmetic.
- [ ] **Step 2: RED. Step 3: Implement. Step 4: GREEN** (single-file run needs no cap unless you run the whole suite). **Step 5:** Mutation check — flip the infra branch to falsify on infra: the infra test must FAIL. Commit `feat(s3): falsification — re-execution of cited tests, infra never falsifies`.

---

### Task 6: Prompt + search threading (`memory` block, `Node.depth`, `SearchResult.root_prompt`)

**Files:**
- Modify: `crucible/proposer/prompt.py`, `crucible/search/node.py`, `crucible/search/loop.py`, `crucible/run/arm.py` (`_naive_attempt` + `attempt_task` signatures only)
- Test: `tests/proposer/test_prompt.py`, `tests/search/` existing files (append)

**Interfaces:**
- Produces: `build_prompt(unit, symptom, *, feedback=None, memory: str | None = None)` — when `memory` is not None its text is inserted as its own section BETWEEN the Symptom section and `_INSTRUCTION` (the block already carries its own `## Prior experience with this code` header); `memory=None` yields BYTE-IDENTICAL output to today's (guard test: compare against a literal golden call with the old signature semantics).
- `Node.for_candidate(cls, candidate, parent_id=None, depth: int = 0)` storing `self.depth`; `_add_candidate` passes its existing `node_depth` through (the `ctx.depth` dict stays — it is the loop's bookkeeping; `node.depth` is the value-feature surface).
- `search(unit, proposer, value, *, seed, k, width, memory: str | None = None, ...)` threading `memory` into `_Ctx` and into EVERY `build_prompt` call (root seeding AND refinement); `SearchResult` gains trailing `root_prompt: str = ""` set in `_finalize` (the root prompt WITH the memory block — it is what the episode stores and sleep trains on); `_naive_attempt` sets it too; `attempt_task(cfg, unit, taskspec, proposer, value, *, memory: str | None = None)` passes it down.
- A_noMem byte-identity guards: (1) `build_prompt(u, s)` equals `build_prompt(u, s, memory=None)`; (2) a seeded fixture `search(...)` run with and without `memory=None` explicit produces identical `SearchResult` minus `root_prompt` (compare `to_dict()` with the field popped); (3) `SearchResult.from_dict` on an S2-era dict (no `root_prompt` key) loads with `""`.

- [ ] **Step 1: Failing tests** for all three guard clauses above plus: memory block position (appears after "## Symptom" content and before the instruction text — assert index ordering in the string); refinement prompts carry the block (monkeypatch proposer to capture prompts; assert the block is in every captured prompt when memory is set, in none when None); `node.depth` equals the ctx depth for a refined node (existing search fixtures).
- [ ] **Step 2: RED. Step 3: Implement. Step 4: GREEN** — then run ALL of `tests/search tests/proposer tests/run` (no cap needed) and confirm zero regressions.
- [ ] **Step 5:** Mutation check — drop `memory` from the refinement prompt call only: the every-prompt test must FAIL. Commit `feat(s3): memory block threaded through prompt/search; A_noMem byte-identity pinned`.

---

### Task 7: Online value v1 (`crucible/value/online.py`)

**Files:**
- Create: `crucible/value/online.py`
- Test: `tests/value/test_online.py`

**Interfaces:**
- Consumes: `Node` (with `.depth`, `.candidate.mean_logprob`, `.candidate.self_certainty`), `Value` protocol (UNCHANGED — score/update signatures are pre-reg §9).
- Produces: `class OnlineValue` satisfying `Value`. Features (spec §6): `[1.0 (bias), mean_logprob or 0.0, self_certainty or 0.0, float(depth), *family_onehot(7), retrieval_hit]` — family and retrieval-hit are TASK-level: `begin_task(family: str, retrieval_hit: bool) -> None` sets them for subsequent scores (docstring: called by the A_full driver hook before each task; the search loop never calls it). `FAMILIES = ("ARITH", "BOOL", "CMP", "CONST", "FLOW", "SDL", "UNARY")` (sorted; CONST present for schema stability even though rung-1 streams carry none). Model: hand-rolled logistic regression, weights init 0.0, plain SGD `lr=0.1`, update rule `w += lr * (outcome - sigmoid(w·x)) * x`. `score(node)` = sigmoid(w·x) — deterministic, caches `(node_id -> features)` in `self._seen`; `update(node, outcome)` uses live features; `update_by_id(node_id: str, outcome: bool) -> bool` uses the score-time cache (returns False when the id was never scored — the driver logs, never raises). `snapshot() -> dict` / `restore(d)` for record-keeping (weights list + counters).
- No numpy needed beyond stdlib math; determinism test mandatory.

- [ ] **Step 1: Failing tests** — score in [0,1] and deterministic (two calls equal); update moves the score toward the outcome (score before < score after for outcome=True on same node); `begin_task` changes features (same node scores differently under different family/hit context — construct two contexts, assert different scores after some training); `update_by_id` on unseen id returns False; snapshot/restore round-trip equality of subsequent scores; protocol conformance `isinstance(OnlineValue(), Value)`.
- [ ] **Step 2: RED. Step 3: Implement. Step 4: GREEN. Step 5:** Mutation check — freeze the update (no-op body): the moves-toward-outcome test must FAIL. Commit `feat(s3): online value v1 — logistic P(hidden pass), task-context features`.

---

### Task 8: Uncertainty (`crucible/uncertainty/`)

**Files:**
- Create: `crucible/uncertainty/__init__.py`, `crucible/uncertainty/conformal.py`
- Modify: `pyproject.toml` (add `"crepes>=0.6"` to `[project] dependencies`; run `uv pip install --python .venv/bin/python crepes` and record the installed version in the report)
- Test: `tests/uncertainty/test_conformal.py`

**Interfaces:**
- Produces: `PROVENANCE_CLASSES = tuple of 4 strings` `"hit-p1" | "hit-p2" | "nohit-p1" | "nohit-p2"`; `provenance_class(retrieval_hit: bool, phase: int) -> str`; `class Calibrator` with `observe(score: float, cls: str, outcome: bool)`, `confidence(score: float, cls: str) -> float` (isotonic regression fit per class over observed (score, outcome) pairs; before `MIN_OBS=10` observations in a class, fall back to the raw score — say so in the docstring, it is the honest cold-start), `should_abstain(p: float, cls: str) -> bool` (threshold `ABSTAIN_P = 0.2`, a module constant with a docstring noting it composes with the search loop's existing ABSTAIN rule, not replaces it), `recalibrate(window: int) -> None` (refit isotonic per class over the last `window` observations — the post-sleep hook), `snapshot()/restore()`. Use crepes' Mondrian machinery if it fits cleanly; if crepes' API forces awkwardness for streaming isotonic, implement isotonic via stdlib (PAVA is ~20 lines) and use crepes only where it genuinely helps — the implementer decides and RECORDS the choice + rationale in the report; the interface above is what later tasks consume either way.
- All state in-memory within a run; observations recorded via the arm records anyway (no separate persistence).

- [ ] **Step 1: Failing tests** — `provenance_class` mapping (4 cases); cold-start passthrough (`confidence == score` before MIN_OBS); after observing 20 pairs where high scores always succeed and low always fail, `confidence(0.9, cls) > confidence(0.1, cls)` and both differ from raw; classes are independent (training one class leaves another at passthrough); `should_abstain` threshold boundary (p=0.2 exactly does NOT abstain — inclusive gate like §4.7's, document); `recalibrate(window)` drops influence of old observations (construct a flip: old data says high-good, recent window says high-bad; after recalibrate, ordering flips); snapshot/restore.
- [ ] **Step 2: RED. Step 3: Implement. Step 4: GREEN. Step 5:** Commit `feat(s3): uncertainty — per-class isotonic confidence + abstention + recalibrate hook` (include the pyproject change; note crepes-vs-PAVA decision in the commit body).

---

### Task 9: Sleep training seam (`crucible/sleep/select.py`, `train.py`, `registry.py`)

**Files:**
- Create: `crucible/sleep/__init__.py`, `crucible/sleep/select.py`, `crucible/sleep/train.py`, `crucible/sleep/registry.py`
- Test: `tests/sleep/__init__.py`, `tests/sleep/test_select.py`, `tests/sleep/test_registry.py`

**Interfaces:**
- `select.py`: `sft_pairs(store: MemoryStore) -> list[tuple[str, str]]` — ALL verified episodes (cumulative, R-S3-3 rationale in spec §5), pairs = `(episode.root_prompt, episode.landed_module)`; refuses (raises) an episode with `landed_module=None` reaching it (cannot happen for verified episodes — defensive, pinned); deterministic order (by `created_at`, then item_id); `episode_set_hash(pairs) -> str` (sha256 of the canonical JSON).
- `train.py`: `class Trainer(Protocol)` with `train(pairs: list[tuple[str, str]], seed: int, out_dir: Path) -> Path` (returns the adapter dir); `class FakeTrainer` (writes `out_dir/adapter_config.json` with `{"pairs": len(pairs), "seed": seed}` and returns it — the unit-test double); `class LoraTrainer` — the real one: PEFT LoRA rank 16 on the A2 model, TRL `SFTTrainer`, ALL imports inside `train()` (torch must not load at module import — the test env imports this module without CUDA), seeded (`transformers.set_seed`), trains from BASE every call (cumulative retrain, spec §5), completion-only loss on the module part if TRL's API makes that a one-liner, else full-sequence loss with the choice recorded. `LoraTrainer` gets NO unit test (GPU) — its live check is Task 12; `FakeTrainer` is what Tasks 10-11 wire in tests.
- `registry.py`: `class AdapterRegistry` (JSONL at a path): `record(adapter_id: str, episode_set_hash: str, base_digest: str, accepted: bool, created_at: str)`, `latest_accepted() -> str | None`; `adapter_id` = first 16 hex of `episode_set_hash` prefixed `ad-` (deterministic from the data, identity-not-description).

- [ ] **Step 1: Failing tests** — `sft_pairs` returns only verified episodes' pairs in deterministic order; the defensive raise; `episode_set_hash` stable/sensitive; `FakeTrainer` writes and returns; registry round-trip + `latest_accepted` skips rejected rows; `import crucible.sleep.train` succeeds with torch absent (simulate: run the import in a subprocess with `PYTHONPATH` set and a stub that raises on `import torch` — or simpler and acceptable: assert `"import torch" not in` the module's top-level (parse with `ast`, check no torch/peft/trl in module-level imports)).
- [ ] **Step 2: RED. Step 3: Implement. Step 4: GREEN. Step 5:** Commit `feat(s3): sleep selection + trainer seam + adapter registry`.

---

### Task 10: Sleep loop (`crucible/sleep/loop.py`) — trigger, gate, record

**Files:**
- Create: `crucible/sleep/loop.py`
- Test: `tests/sleep/test_loop.py`

**Interfaces:**
- Consumes: Tasks 2, 5, 9; a `ServerAdapter` seam and a `SliceRunner` seam (both defined here): `class ServerAdapter(Protocol): def load(self, adapter_dir: Path, adapter_id: str) -> None` (real impl `VllmAdapterLoader(base_url)` POSTs `/v1/load_lora_adapter` — mirror how S1's smoke did it, live-checked in Task 12; `FakeServerAdapter` records calls); `SliceRunner(Protocol): def solved(self, task_keys: list[str], adapter_id: str | None) -> int` (real impl re-runs tasks at K=1 greedy via the existing driver pieces; `FakeSliceRunner` returns scripted counts).
- Produces: `SLEEP_THRESHOLD_DEFAULT = 16`; `SLICE_SIZE = 12`; `ACCEPT_MAX_DROP = 1`; frozen `SleepRecord` dataclass (`sleep_index: int, adapter_id: str, episode_set_hash: str, episodes_selected: int, slice_task_keys: tuple[str, ...], slice_before: int, slice_after: int, accepted: bool, refalsify: dict` (the FalsifyTally as dict), `gpu_s: float | None, created_at: str`) with round-trip + completeness tests; `class SleepController` with `__init__(store, trainer, server, slice_runner, registry, *, threshold=SLEEP_THRESHOLD_DEFAULT, seed)`, `maybe_sleep(solved_task_keys: list[str], units_by_item: ..., now: str) -> SleepRecord | None`:
  1. trigger: `store.verified_count() - self._last_accepted_count >= threshold` else return None;
  2. refalsify batch (Task 5) over items cited by the SFT episodes + stale items; tally into the record;
  3. `pairs = sft_pairs(store)`; train via the seam; adapter_id from registry rule;
  4. regression slice: seeded sample (`random.Random(f"{seed}:slice:{sleep_index}")`) of `min(SLICE_SIZE, len(solved_task_keys))` from solved tasks; `before = slice_runner.solved(keys, current_adapter)`, `after = slice_runner.solved(keys, candidate)`; accept iff `before - after <= ACCEPT_MAX_DROP`;
  5. on accept: `server.load(...)`, registry record accepted, `self._last_accepted_count = store.verified_count()`; on reject: registry record rejected, counter NOT reset (pin!), server NOT called (pin!);
  6. return the SleepRecord either way.
- No BudgetMeter anywhere in `crucible/sleep/` (same crude source-scan pin as Task 5).

- [ ] **Step 1: Failing tests** (all with fakes) — below-threshold returns None and calls nothing; at threshold: full pipeline fires, record fields all set, accept path calls `server.load` exactly once and resets the counter (next call below threshold returns None); reject path (`FakeSliceRunner` scripted to drop 2): server NOT called, counter NOT reset (a subsequent call with 0 new episodes still fires — pin by asserting the second `maybe_sleep` trains again); slice determinism (same seed+index ⇒ same slice keys); `min(SLICE_SIZE, ...)` with 5 solved tasks ⇒ slice of 5; SleepRecord round-trip + completeness.
- [ ] **Step 2: RED. Step 3: Implement. Step 4: GREEN. Step 5:** Mutation checks — (a) reset counter on reject → the trains-again test must FAIL; (b) call server on reject → killed; (c) accept rule off-by-one (`< ACCEPT_MAX_DROP`) → add/verify a boundary test (drop of exactly 1 accepts). Commit `feat(s3): sleep loop — threshold trigger, regression gate, honest SleepRecord`.

---

### Task 11: A_full wiring (`crucible/run/`) — hooks, records, lens, CLI

**Files:**
- Modify: `crucible/run/arm.py` (ARMS + attempt_task already threaded in Task 6), `crucible/run/driver.py`, `crucible/run/records.py`, `crucible/run/lens.py`, `crucible/cli.py`
- Create: `crucible/run/full.py` (the A_full hook object)
- Test: `tests/run/test_full.py`, appends to `tests/run/test_records.py`, `tests/run/test_lens.py`, `tests/test_cli.py`

**Interfaces:**
- `records.py`: `TaskRecord` gains trailing `retrieved_ids: tuple[str, ...] = ()` and `adapter_id: str | None = None` (to_dict/from_dict backward-compatible via `d.get`; completeness tests updated — they will catch these automatically if written per §8.6).
- `arm.py`: `ARMS["A_full"] = ArmConfig("A_full", "Qwen/Qwen2.5-Coder-1.5B-Instruct", True, chat=True)` — same proposer as A_noMem by construction (arms differ by memory/sleep/value, which live in the HOOKS, not ArmConfig — document this: ArmConfig stays the serving identity; a comment points at `full.py`).
- `driver.py`: `run_arm(cfg, stream_dir, task_keys, proposer, value, out_dir, *, log=print, hooks: "ArmHooks | None" = None)` where `ArmHooks` is a Protocol defined in `full.py`: `before_task(unit, taskspec) -> str | None` (the memory block), `after_task(taskspec, record, result_root_prompt: str, landed_module: str | None, now: str) -> tuple[tuple[str, ...], str | None]` (returns (retrieved_ids, adapter_id) to stamp on the record — the driver rebuilds the TaskRecord with them via `dataclasses.replace`), `between_tasks(solved_task_keys, now) -> None` (sleep check). `hooks=None` ⇒ byte-identical behavior (guard: a fixture run_arm with hooks=None produces records equal to a pre-change golden — reuse the existing driver tests' fixtures and simply assert they still pass unchanged; plus one explicit `replace`-not-applied assertion).
- `full.py`: `class FullHooks` implementing the protocol. `before_task`: `retrieve(...)` (Task 4) + `value.begin_task(family, retrieval_hit)`. `after_task(taskspec, record, result, now)` (takes the whole `SearchResult`): build the EpisodicRecord from taskspec + record + `result.root_prompt` + `result.best_patch`; `store.write_episode`; if verified, `distill` + `store.write_semantic` with `flipped_tests = result.symptom_failed` (the visible tests failing pre-fix — a verified fix passed the whole visible suite, so all of them flipped; the docstring states that killing-test NAMES come from the symptom while the count comes from validation); `value.update_by_id(result.best_node_id, hidden_pass)` only when `hidden_pass is not None`; `calibrator.observe(...)`; returns `(retrieved_ids, adapter_id)` for the driver to stamp via `dataclasses.replace`. `between_tasks`: `sleep_controller.maybe_sleep(...)`; `calibrator.recalibrate` after an ACCEPTED sleep. HARD requirements: no extra sandbox executions, no extra generate calls, verified-only distillation, value update only on measured outcomes. `SearchResult` needs a trailing `symptom_failed: tuple[str, ...] = ()` (set from the ctx symptom's failed+timed_out+errored) — Task 6 adds it if not already present; otherwise this task adds it with its own round-trip guard.
- `lens.py`: `ArmLens` gains trailing `adapter_ids: tuple[str, ...] = ()` (distinct adapter ids that stamped records, in first-seen order); `build_lens` fills it.
- `cli.py`: `arm run --arm A_full` grows `--memory-db PATH` (default `runs/<arm>/memory.sqlite3`) and `--sleep-threshold INT` (default 16; the smoke's N=4 override, spec §9); wiring constructs FullHooks with the REAL trainer/server/slice-runner for A_full and passes hooks=None for every other arm (pin: `--arm A_noMem` never constructs a store — monkeypatch test).

- [ ] **Step 1: Failing tests** — hooks=None equivalence; FullHooks with an in-memory store + FakeTrainer/FakeServerAdapter/FakeSliceRunner over a 3-task fixture drive: episode written per task (verified and not), semantic written ONLY for verified, record stamped with retrieved_ids on a task where the store had a lesson, value.update called exactly once per measured task (spy), sleep fires when threshold crossed between tasks; records/lens round-trip + completeness with the new fields; CLI flags parse and `A_noMem` builds no store.
- [ ] **Step 2: RED. Step 3: Implement** (including the small Task-6 follow-through: `SearchResult.symptom_failed` trailing field + `_naive_attempt` setting it — if Task 6 already landed without it, add it here with its own round-trip guard). **Step 4: GREEN**, then the capped FULL suite. **Step 5:** Mutation checks — (a) distill on unverified episodes → killed by the semantic-only-verified test; (b) stamp retrieved_ids always-empty → killed; (c) hooks invoked for A_noMem → killed by the no-store test. Commit `feat(s3): A_full — hooks wiring, stamped records, lens adapters, CLI`.

---

### Task 12: OPS — live seams + the A_full 30-task smoke (spec §9 exit)

Operational, GPU. Controller-run or dispatched with care; every long step OS-detached with `.DONE` markers; builds/tests under the R-T2-6 cap; GPU hygiene (stop ollama; kill stray `VLLM::EngineCore` by pid — `pkill` on the launcher is not enough).

- [ ] **Step 1: Live LoRA trainer smoke** — `LoraTrainer.train` on 4 synthetic pairs (seeded): produces an adapter dir loadable by PEFT; record wall time + peak VRAM (expect ≈3.3 GiB per the attach smoke).
- [ ] **Step 2: Live hot-swap smoke** — serve the 1.5B (`PATH=.venv/bin:$PATH bash scripts/serve_model.sh Qwen/Qwen2.5-Coder-1.5B-Instruct`), `VllmAdapterLoader.load` the Step-1 adapter, then one `generate` against the adapter id; assert HTTP 200 paths and identity. (vLLM needs `VLLM_ALLOW_RUNTIME_LORA_UPDATING=true` for `/v1/load_lora_adapter` — if the load 404s/403s, set that env in `serve_model.sh` guarded by a comment, mirroring the S1 smoke's approach; record what was needed.)
- [ ] **Step 3: The smoke** — `crucible arm run --arm A_full <streams/1158e92f40ad> --base-url http://127.0.0.1:8010 --tasks phase1:30 --sleep-threshold 4 --out runs/s3-smoke` (exact task-selection flag per the CLI's existing `--tasks` semantics — read it before running). Watch: completes; ≥1 sleep event with an honest accept/reject; refalsification tally ≥1 checked; every record type parses (`read_task_records` + SleepRecord reads); lens carries adapter ids; NO task charged >K=8.
- [ ] **Step 4: Record** — `docs/findings/S3-smoke.md`: what ran, sleep events (accepted?), value update counts, retrieval hit counts, wall/GPU time, gotchas; CARRIED-DEBT S3 section; teardown (kill EngineCore by pid, verify `nvidia-smi`, restart ollama). Commit docs.

---

### Task 13: THIRD_PARTY + docs closeout

**Files:**
- Modify: `THIRD_PARTY.md`, `docs/CARRIED-DEBT.md` (S3 settled/deferred), pre-reg spec (NO changes expected — verify none needed; if any S3 reality contradicted the pre-reg text, STOP and surface to the controller rather than editing).

- [ ] **Step 1:** THIRD_PARTY entries: crepes (BSD-3, version, "conformal/calibration library"; if the implementer went PAVA-stdlib in Task 8, record "schema/design reference only"); MemOS/Graphiti/MIRIX already listed from S1 — verify, extend the "what was taken" note with the S3 field usage. **Step 2:** CARRIED-DEBT: settled (organ, value v1, uncertainty, sleep, A_full smoke), deferred (procedural store unpopulated R-S3-1; LLM distillation variant; exploratory sub-arms unrun; N pinned at lock). **Step 3:** Commit `docs(s3): third-party + carried-debt closeout`.

---

## Self-review (run after writing, fixed inline)

- Spec §2→T1-2, §3→T4+T6, §4→T5, §5→T9-10, §6→T7, §7→T8, §8→T11, §9 exit→T12, §10 non-goals→respected (no procedural writes, no LLM distillation, no gating run).
- Known judgment points delegated WITH requirements stated: Task 8 crepes-vs-PAVA; Task 11 hook signature (requirements list is binding); Task 12 `--tasks` flag semantics (read before running). These are deliberate — the file-local idioms decide, the requirements do not move.
- Type consistency: `RetrievedBlock{block, item_ids}` (T4) consumed in T11; `FalsifyTally` (T5) in T10's SleepRecord; `Trainer/ServerAdapter/SliceRunner` protocols (T9/T10) faked in T10-11; `SearchResult.root_prompt` (T6) + `symptom_failed` (T6-or-11) consumed in T11; `update_by_id` (T7) called in T11.
