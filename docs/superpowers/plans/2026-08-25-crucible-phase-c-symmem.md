# Phase-C: Symptom-Conditioned Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run the pre-registered Phase-C experiment: symptom-conditioned cross-unit retrieval with a τ-silence rule, gated on the 14B non-repeat pool (B_symmem vs frozen B_mem), with the A_symmem 1.5B exploratory mirror.

**Architecture:** A pure lexical scorer (`crucible/memory/symmatch.py`) + an additive policy function `retrieve_symptom` in `crucible/memory/retrieve.py` reusing its assembly internals; a `"symptom"` retrieval mode threaded through both hooks classes, whose `before_task` runs one uncharged, disclosed symptom probe; τ fixed pre-lock by `scripts/calibrate_tau.py` over the four Phase-A/B memory databases.

**Tech Stack:** Python 3.12 (`.venv/bin/python`), pytest, sqlite (existing store), vLLM 0.27.1 on port 8010. Zero new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-25-crucible-phase-c-prereg.md` — read §3–§9 first. §5–§7 freeze at `prereg-lock-c` (Task 10); nothing in the endpoint/verdict path changes after that.

## Global Constraints

- PUBLIC repo: secret-scan every staged diff before ANY push; verify origin sync after ANY push (`git fetch -q && git rev-parse master origin/master | uniq -c` → count 2).
- Full pytest ALWAYS under `systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q` (no tail summary on this box — check `$?`).
- Mutation checks: purge `__pycache__`, `PYTHONDONTWRITEBYTECODE=1`, restore from `cp` backup — NEVER `git checkout`.
- `runs/` and `streams/` gitignored. Frozen comparator lenses/records (`runs/gate-b2-mem`, `runs/abl-mem-nosleep`, plus the Phase-A DBs) are READ-ONLY evidence.
- Determinism is load-bearing everywhere: no wall-clock, no randomness, no dict-order dependence in scorer or policy; every ranking tie-broken by `item_id`.
- The exact-class fast path must stay byte-identical to today's `"full"`-mode behavior when a live exact-class lesson exists — the repeat guard (spec §5) depends on it.
- τ is `None` until the lock: `retrieve_symptom` must REFUSE `tau=None` loudly, so no run can happen with an uncalibrated instrument.
- Box gotchas: `.venv/bin` on PATH for `vllm`; kill vLLM by EngineCore pid from `nvidia-smi --query-compute-apps`; exit 144 = harness kill.

---

### Task 1: store + prompt seams (batch: three small additions)

**Files:**
- Modify: `crucible/memory/store.py`, `crucible/proposer/prompt.py`
- Test: `tests/memory/test_store.py`, `tests/proposer/test_prompt.py` (find the actual prompt-test file with `ls tests/proposer/`)

**Interfaces (produced):**
- `MemoryStore.semantic_all(self) -> list[SemanticItem]` — every semantic item, all units/families, ordered by `item_id` (SQL `ORDER BY item_id` for determinism). Mirrors `semantic_family`'s row-decoding exactly.
- `MemoryStore.episode_by_id(self, item_id: str) -> EpisodicRecord | None` — point query; `None` when absent.
- `crucible/proposer/prompt.py`: public alias `render_symptom = _render_symptom` with a one-line docstring ("public seam for symptom-mode retrieval, Phase-C prereg §4.2 — same renderer the prompt itself uses, so query text and prompt text can never drift").

- [ ] **Step 1: Failing tests.** In the store test file (mirror its existing fixture style):

```python
def test_semantic_all_returns_every_item_ordered_by_item_id(...):
    # seed two items in different units/families via the file's existing helpers
    items = store.semantic_all()
    assert [i.item_id for i in items] == sorted(i.item_id for i in items)
    assert len(items) == 2

def test_episode_by_id_point_query(...):
    # seed one episode; assert episode_by_id(ep.item_id) == ep and episode_by_id("absent") is None
```

In the prompt test file: `from crucible.proposer.prompt import render_symptom, _render_symptom; assert render_symptom is _render_symptom`.
- [ ] **Step 2: RED** (AttributeError/ImportError), **Step 3: implement**, **Step 4: GREEN** (both files), **Step 5: commit** `feat: store + prompt seams for symptom retrieval (Phase-C prereg §4.2)`.

### Task 2: the lexical scorer (`crucible/memory/symmatch.py`)

**Files:**
- Create: `crucible/memory/symmatch.py`
- Test: Create `tests/memory/test_symmatch.py`

**Interfaces (produced):**
```python
def tokenize(text: str) -> frozenset[str]
    # lowercase; split on non-alphanumeric; drop tokens of len < 2; drop any token
    # matching r"test\w*" (unit-local test NAMES are noise across units, spec §4.2b).
def score(query_tokens: frozenset[str], lesson_tokens: frozenset[str]) -> float
    # binary cosine: |Q ∩ L| / sqrt(|Q| * |L|); 0.0 when either side is empty.
def lesson_text(item: SemanticItem, episode_symptom: str) -> str
    # item.landed_diff + "\n" + episode_symptom + "\n" + item.family
    # (family rides as a TOKEN on both sides — a natural same-family boost, no magic weight)
def query_text(module_src: str, symptom_text: str, family: str) -> str
    # module_src + "\n" + symptom_text + "\n" + family
def rank(query_tokens, candidates: list[tuple[SemanticItem, frozenset[str]]]) -> list[tuple[float, SemanticItem]]
    # sorted by (-score, item.item_id) — the deterministic tie-break
def symptom_section(root_prompt: str) -> str
    # the text between "\n## Symptom\n" and the next "\n## " (or end); "" if absent.
```
Module docstring: what the scorer is (deterministic lexical, no deps), why test-name tokens are dropped, why family is a token not a weight, and that τ lives with the LOCK (module constant `TAU: float | None = None` with a comment citing spec §4.4 / LOCK-C; set at the lock commit, never before).

- [ ] **Step 1: Failing tests** (each is a future mutation pin — write them to kill the named mutant):

```python
def test_tokenize_drops_unit_local_test_names_but_keeps_code_tokens():
    toks = tokenize("failed: test_v0, test_h1\nreturn a + b")
    assert "test_v0" not in toks and "test_h1" not in toks and "return" in toks
    # MUTANT KILLED: removing the test\w* filter

def test_score_is_binary_cosine_and_zero_on_empty():
    q, l = tokenize("alpha beta gamma"), tokenize("beta gamma delta")
    assert abs(score(q, l) - 2/3) < 1e-9          # 2 shared / sqrt(3*3)
    assert score(frozenset(), l) == 0.0
    # MUTANT KILLED: swapping intersection for union, or dropping the sqrt

def test_family_token_boosts_same_family_pairs():
    q = tokenize(query_text("return x", "boom", "ARITH"))
    same = tokenize(lesson_text_stub(diff="return y", symptom="boom", family="ARITH"))
    other = tokenize(lesson_text_stub(diff="return y", symptom="boom", family="SDL"))
    assert score(q, same) > score(q, other)
    # (build lesson_text_stub inline from a minimal SemanticItem or call lesson_text directly)

def test_rank_breaks_ties_by_item_id():
    # two candidates with identical token sets -> equal scores; assert item_id order
    # MUTANT KILLED: dropping the tie-break

def test_symptom_section_extracts_between_headers():
    rp = "## Module under repair\nX\n\n## Symptom\nfailed: t\nboom\n\n## Instruction\nY"
    assert symptom_section(rp) == "failed: t\nboom"
    assert symptom_section("no symptom here") == ""
```
- [ ] **Step 2: RED**, **Step 3: implement exactly the interfaces above**, **Step 4: GREEN** (`tests/memory/test_symmatch.py`), **Step 5: commit** `feat: deterministic lexical symptom scorer (Phase-C prereg §4.2)`.

### Task 3: the v2 policy — `retrieve_symptom`

**Files:**
- Modify: `crucible/memory/retrieve.py` (additive function; NO change to `retrieve`)
- Test: `tests/memory/test_retrieve.py`

**Interfaces:**
- Consumes: Task 1's `semantic_all`/`episode_by_id`, Task 2's scorer functions.
- Produces:
```python
def retrieve_symptom(store: MemoryStore, unit_src: str, unit_id: str, family: str,
                     symptom_text: str, *, tau: float | None) -> RetrievedBlock
```
Behavior, in order (docstring states each as an invariant):
1. `tau is None` → `raise ValueError("tau is not locked — Phase-C runs before prereg-lock-c are forbidden")`.
2. **Exact fast path:** `live_exact` (as `retrieve` computes it) non-empty → lessons = `_rank_lessons(live_exact)[:2]` — the same items, same order, as mode `"full"` on that class.
3. Else: candidates = every live item from `semantic_all()`, excluding items whose `unit_id == unit_id` and family... NO — excluding nothing but falsified items (cross-unit IS the point; same-unit-other-family items are legitimate). For each, `lesson_text(item, symptom_section(store.episode_by_id(item.cited_episode_id).root_prompt))` (a missing episode → treat symptom as `""`, never crash). Rank with `rank(...)`; keep top 2 with `score >= tau`; below-τ or empty → lessons = [].
4. Exemplar: `_pick_exemplar(store, unit_id, family)` — unchanged, class-exact.
5. Assembly: the same `lesson_parts`/`exemplar_part`/`_budget_candidates`/`_assemble`/`CONTEXT_BUDGET_CHARS` path as `retrieve` (reuse the module privates in place; if that means extracting the last ~10 lines of `retrieve` into a shared private helper `_assemble_block(lesson_items, exemplar) -> RetrievedBlock`, do that refactor here and have BOTH functions call it — existing retrieve tests are the net).

- [ ] **Step 1: Failing tests** (reuse the file's fixtures):

```python
def test_symptom_fast_path_equals_full_policy_when_exact_lesson_exists(...):
    # seed exact-class lesson; assert retrieve_symptom(..., tau=0.99) == retrieve(store, unit, family)
    # MUTANT KILLED: fast path routed through the scorer

def test_symptom_match_carries_a_lesson_across_units(...):
    # seed a lesson in unit A whose landed_diff/symptom share distinctive tokens with the
    # query; call retrieve_symptom for stranger unit B with tau low (e.g. 0.05);
    # assert block is not None and the lesson's item_id is in item_ids
    # MUTANT KILLED: candidates filtered to same unit/family

def test_below_tau_is_silence_none_never_empty(...):
    # same seeding, tau=0.99 -> RetrievedBlock(None, ()) (assuming no exact content/exemplar)

def test_tau_none_raises(...):
    with pytest.raises(ValueError, match="not locked"):
        retrieve_symptom(store, "src", "X/0", "ARITH", "boom", tau=None)

def test_exemplar_rule_unchanged_under_symptom_mode(...):
    # class-exact verified episode, no lessons anywhere: block contains the exemplar
    # regardless of tau; a stranger unit gets no exemplar
```
- [ ] **Step 2: RED**, **Step 3: implement**, **Step 4: GREEN** (`tests/memory/test_retrieve.py` — ALL existing tests must still pass, they are the net for the assembly refactor), **Step 5: commit** `feat: retrieve_symptom — cross-unit lessons with tau-silence (Phase-C prereg §4.2)`.

### Task 4: hooks symptom mode + the uncharged probe

**Files:**
- Modify: `crucible/run/full.py`
- Test: `tests/run/test_full.py`

**Interfaces:**
- Consumes: Task 3's `retrieve_symptom`, Task 1's `render_symptom`, `crucible.sandbox.task_run.run`.
- Produces: retrieval-mode vocabulary grows `"symptom"` for BOTH hooks classes; `FULL_FAMILY` gains `"A_symmem": ("symptom", False)`; `MEM_ARMS` becomes `{"B_mem": "full", "B_symmem": "symptom"}` (a dict — update its docstring); `MemHooks.__init__(store, *, retrieval: str = "full", log=print)`; both classes get a public counter `uncharged_symptom_runs: int` starting 0.

`before_task` dispatch (both classes, same shape — MemHooks shown; FullHooks mirrors inside its existing mode dispatch):
```python
if self._retrieval == "symptom":
    symptom = run(unit, unit.module_src, None)      # UNCHARGED driver-side probe:
    self.uncharged_symptom_runs += 1                # deterministic, byte-identical to the
    block = retrieve_symptom(                       # search's own free symptom run; never
        self._store, unit.module_src,               # in executions_charged (spec §4.3)
        taskspec.unit_id, taskspec.family,
        render_symptom(symptom), tau=symmatch.TAU)
else:
    ...existing "full"/"exact"/"off" paths unchanged...
```
**Probe-count persistence (C5 needs a post-hoc artifact — the in-process counter dies
with the detached run):** both hooks accept `probe_log_path: Path | None = None`; in
symptom mode, after incrementing, overwrite that file with `str(self.uncharged_symptom_runs)`
(tiny atomic write per task). `build_mem_hooks` and `build_full_hooks` pass
`arm_dir / "symptom_probes.txt"`. Non-symptom modes never create the file. Add to the
Task-4 test: after two `before_task` calls the file reads "2"; with mode "full" it does
not exist.
**Module-docstring amendment (required, honesty):** full.py's invariant "*No extra sandbox executions.* This module never calls run/run_hidden on the task path" must be amended, not silently broken: append "— with ONE disclosed exception (Phase-C, spec §4.3): symptom-mode `before_task` runs a single uncharged symptom probe per task, identical in inputs and outcome to the free symptom run the search itself performs; it is counted in `uncharged_symptom_runs` and never appears in `executions_charged`."

- [ ] **Step 1: Failing tests:**

```python
def test_symptom_mode_probes_uncharged_and_hands_the_block_to_the_search(tmp_path, monkeypatch):
    """The probe must not touch executions_charged (it happens outside attempt_task), the
    counter must count it, and the block handed back must be retrieve_symptom's."""
    # monkeypatch crucible.run.full.retrieve_symptom with a spy returning RetrievedBlock("BLOCK", ("id1",))
    # and crucible.run.full.run with a spy returning a minimal TestReport
    # rig = _Rig(tmp_path, retrieval="symptom")   (extend _Rig's forwarding)
    # block = rig.hooks.before_task(U, SPEC)
    # assert block == "BLOCK" and rig.hooks.uncharged_symptom_runs == 1
    # assert the spy's tau argument is symmatch.TAU (is None today -> spy bypasses the raise)
    # MUTANT KILLED: probe charged / counter dropped / mode falls through to "full"

def test_full_family_and_mem_arms_map_the_phase_c_arms():
    assert FULL_FAMILY["A_symmem"] == ("symptom", False)
    assert MEM_ARMS == {"B_mem": "full", "B_symmem": "symptom"}
    # plus: every pre-existing FULL_FAMILY entry unchanged (assert full dict literal)
```
Also update the existing FULL_FAMILY pin test's expected literal (now 5 entries).
- [ ] **Step 2: RED**, **Step 3: implement**, **Step 4: GREEN** (`tests/run/test_full.py`), **Step 5: commit** `feat: symptom retrieval mode in both hooks + uncharged probe (Phase-C prereg §4.3)`.

### Task 5: ARMS + CLI wiring

**Files:**
- Modify: `crucible/run/arm.py`, `crucible/cli.py`
- Test: `tests/run/test_arm.py`, `tests/test_cli.py`

**Interfaces:** `ARMS["B_symmem"] == ArmConfig("B_symmem", ARMS["B_search"].model, True, chat=True)`; `ARMS["A_symmem"] == ArmConfig("A_symmem", ARMS["A_full"].model, True, chat=True)` (comment cites Phase-C prereg §3). CLI: the MemHooks branch becomes `elif cfg.name in MEM_ARMS: hooks = build_mem_hooks(cfg, a.stream_dir, a.out, memory_db=a.memory_db, retrieval=MEM_ARMS[cfg.name])` — `build_mem_hooks` gains the `retrieval: str = "full"` passthrough. A_symmem flows through the existing FULL_FAMILY branch untouched.

- [ ] **Step 1: Failing tests:** arm pin test mirroring `test_phase_b_arms_carry_their_controls_serving_identity` for the two new arms; CLI: extend the FULL_FAMILY parametrize with `("A_symmem", True, False)` + mode assert `"symptom"`; add:

```python
def test_cli_arm_run_b_symmem_wires_symptom_memhooks(_stream, tmp_path, monkeypatch):
    # as test_cli_arm_run_b_mem_wires_store_only_hooks, but --arm B_symmem and
    # assert seen["hooks"].retrieval == "symptom" (expose a read-only property on MemHooks
    # in Task 4: `retrieval`) plus ConstantValue + unwrapped proposer as before
```
(If Task 4 didn't add the `retrieval` property to MemHooks, add it here — read-only, like FullHooks' `retrieval_mode`.)
- [ ] **Step 2: RED**, **Step 3: implement**, **Step 4: GREEN** (both files), **Step 5: commit** `feat: Phase-C arms B_symmem + A_symmem wired (prereg §3)`.

### Task 6: `scripts/calibrate_tau.py`

**Files:**
- Create: `scripts/calibrate_tau.py`
- Test: Create `tests/memory/test_calibrate_tau.py` (import the script's functions; add a `main()` guard so import is side-effect-free)

**Interfaces (produced):** a script that takes DB paths + a stream dir and prints ONE json object:
```
usage: .venv/bin/python scripts/calibrate_tau.py --stream streams/1158e92f40ad \
    runs/gate-a-full/A_full/memory.sqlite3 runs/abl-mem-nosleep/A_mem_nosleep/memory.sqlite3 \
    runs/abl-mem-exactonly/A_mem_exactonly/memory.sqlite3 runs/gate-b2-mem/B_mem/memory.sqlite3
```
Logic (spec §4.4 + §7, exactly):
- Per DB: episodes = `store.episodes()`; lessons = live items from `semantic_all()`. For every (episode, lesson) pair IN THE SAME DB, excluding self-citation pairs (`lesson.cited_episode_id == episode.item_id`): query = `tokenize(query_text(unit_src, symptom_section(episode.root_prompt), episode.family))` with `unit_src` loaded from the stream (`stream_store.read_unit(stream_dir, episode.unit_id).module_src`); lesson tokens via `lesson_text(...)` with the lesson's OWN cited episode's symptom.
- *unrelated* pair: `lesson.unit_id != episode.unit_id and lesson.family != episode.family`. *related* pair: `lesson.class_id == episode.class_id`. Pairs that are neither are ignored.
- Output json: `{"n_unrelated": ..., "n_related": ..., "tau_p95_unrelated": ..., "median_related": ..., "unrelated_summary": {"p50":..,"p90":..,"p99":..}, "related_summary": {...}, "sanity": "PASS"|"FAIL"}` where sanity PASS iff `median_related > tau_p95_unrelated`. Percentiles: `statistics.quantiles(vals, n=100)` — document the exact method so the number is reproducible.
- Determinism: no randomness; iterate DBs in argv order, episodes/lessons in store order (already id-ordered).

- [ ] **Step 1: Failing test:** build a tmp store with 2 units × 2 families (reuse memory-test fixtures), mint 2 verified episodes + 2 lessons with engineered token overlap so related > unrelated; call the script's `calibrate(dbs, stream_loader)` function with a stub loader; assert sanity == "PASS", n_related/n_unrelated counts exact, and that a self-citation pair was EXCLUDED (seed one; count proves it).
- [ ] **Step 2: RED**, **Step 3: implement**, **Step 4: GREEN**, **Step 5: commit** `feat: tau calibration script with ranker-sanity gate (Phase-C prereg §4.4/§7)`.

### Task 7: mutation checks + full suite

Verification only (no commits unless a SURVIVED reveals a test defect — then STOP and report). Use the cp-backup harness from the Phase-B plan (never `git checkout`). Mutations, every one must print `killed:`:

```bash
mut2 crucible/memory/symmatch.py 's/re.compile(r"test\\w\*")/re.compile(r"zzzz\\w*")/' tests/memory/test_symmatch.py "test-name filter disabled"   # adapt pattern to the real code
mut2 crucible/memory/symmatch.py 's|math.sqrt(len(q) \* len(l))|max(len(q), len(l))|' tests/memory/test_symmatch.py "cosine denominator swapped"   # adapt to real expr
mut2 crucible/memory/retrieve.py 's/if live_exact:/if False:/2' tests/memory/test_retrieve.py::test_symptom_fast_path_equals_full_policy_when_exact_lesson_exists "fast path disabled"   # verify occurrence with git diff first
mut2 crucible/memory/retrieve.py 's/score >= tau/True/' tests/memory/test_retrieve.py::test_below_tau_is_silence_none_never_empty "tau gate dropped"   # adapt to real expr
mut2 crucible/run/full.py 's/self.uncharged_symptom_runs += 1/pass/' tests/run/test_full.py::test_symptom_mode_probes_uncharged_and_hands_the_block_to_the_search "probe counter dropped"
mut2 crucible/run/full.py 's/"B_symmem": "symptom"/"B_symmem": "full"/' tests/run/test_full.py::test_full_family_and_mem_arms_map_the_phase_c_arms "MEM_ARMS mode flip"
```
(Every sed is a TEMPLATE: check the real source line first, adapt, verify non-noop with `cmp`, verify the intended hunk with `git diff`, restore from backup.) Then the full suite under the 4G scope → exit 0; `git status --porcelain` clean. Controller pushes after review.

### Task 8 (ops, controller): τ calibration + lock prep

- [ ] Run `calibrate_tau.py` on the four real DBs + locked stream. If sanity FAIL → STOP per spec §7, report to Brice (kill criterion; do NOT lock).
- [ ] On PASS: set `symmatch.TAU = <tau_p95_unrelated>` (rounded to 4 dp, comment citing LOCK-C) + a pin test asserting the literal; append the pre-lock amendment to spec §11 (self-citation exclusion made explicit in §4.4's rule — date + wording); commit.
- [ ] Smoke file: 8 firsts + their seconds (as Phase-B) PLUS the first 4 `kind == "novel"` tasks from the manifest → ~20 tasks. Serve the 14B (PATH prefix), run `scripts/run_arm_detached.sh B_symmem runs/c-smoke runs/smoke-c-tasks.txt`. Checks: DONE=0; 0 infra/400s; counter…not visible in records — instead: every second-exposure record of a SOLVED class has non-empty `retrieved_ids`; novel records: block present ONLY where a τ-passing match existed (spot-read the log); wall s/task noted.
- [ ] `docs/LOCK-C.md` per spec §9 (both Δ_min values + derivations, τ + rule + pair counts + distribution summaries + sanity verdict, comparator lens sha256s + recomputed pool rates from records, stream hash, model/serving/vLLM verbatim, arms + task set, smoke outcome). Commit, tag `prereg-lock-c`, push master + tag, verify sync + `git ls-remote --tags`.

### Task 9 (ops, controller): the runs

- [ ] `scripts/run_arm_detached.sh B_symmem runs/gate-c-symmem all` (14B already serving) — single-driver check, Monitor marker-or-death, ETA from smoke. On DONE=0 do NOT read numbers yet.
- [ ] Server swap (kill EngineCore by pid, port-down, ≥13 GiB free, serve 1.5B), `scripts/run_arm_detached.sh A_symmem runs/abl-symmem-15b all`, Monitor. Infra kill on either → archive dir, fix infra only, clean rerun (R-S4-1).

### Task 10 (ops, controller): verdict + closeout

- [ ] After BOTH complete: write `lens.json` per arm + compute pool = mean hidden_pass over measured phase-1∪novel records (from records, not lens fields). C3/C4 derivation: a record's block was exact-path iff every retrieved item's `class_id` (joined from the arm's own memory DB) equals the task's `class_id`; non-exact tasks split into matched (non-empty `retrieved_ids`) vs silent for C3/C4. C5: `symptom_probes.txt` in each arm dir must read 450. Evaluate exactly as locked: saturation clause → CONFOUNDED checks (infra > 0.02, landing < 0.98) → repeat guard |succ_second − 0.9050| ≤ 0.0586 (violation → CONFOUNDED + the §5 mandatory solved-at-phase-1 subset diagnosis) → C1 pool ≥ 0.8364 → C2–C5 + exploratory reads + the counter == 450 check per arm.
- [ ] `docs/findings/GATE-C.md` in GATE-B's shape (run ledger, endpoint table, verdict per §6 verbatim, exploratory table, silence/match stats, GPU accounting). Teardown (EngineCore pid, port down, `systemctl is-active ollama`). Secret-scan, commit, push, verify sync. Update `crucible-project.md` + `MEMORY.md` + session handoff. Report to Brice with the endpoint table and the §6-prescribed next step.
