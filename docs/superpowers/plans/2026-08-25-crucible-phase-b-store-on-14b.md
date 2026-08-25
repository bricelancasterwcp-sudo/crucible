# Phase-B: Store-on-14B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the pre-registered Phase-B experiment: does retrieval-only memory (MemHooks) lift the 14B's second-exposure rate, with frozen B_search as control, plus the A_mem_exactonly exploratory arm.

**Architecture:** Two new arms wired through the existing hook seam. `MemHooks` (store-only, no value model, no calibrator) rides `run_arm`'s existing hook calls for B_mem; `FULL_FAMILY` grows a retrieval *mode* (`"full"|"exact"|"off"`) so A_mem_exactonly reuses FullHooks with the family-wide lesson fallback severed. No driver, search, or prompt changes.

**Tech Stack:** Python 3.12 (`.venv/bin/python`), pytest, vLLM 0.27.1 serving Qwen2.5-Coder-14B-AWQ / 1.5B on port 8010.

**Spec:** `docs/superpowers/specs/2026-08-25-crucible-phase-b-prereg.md` — the plan argues from it; read it first. §5–§7 freeze at the lock tag (Task 9); after that NOTHING in the endpoint/verdict path may change.

## Global Constraints

- PUBLIC repo: secret-scan every staged diff before ANY push (`git diff --cached | grep -inE "api[_-]?key|secret|token|password|BEGIN [A-Z]+ PRIVATE|aws_|hf_[A-Za-z0-9]{20}"`); verify origin sync after ANY push (`git fetch -q && git rev-parse master origin/master | uniq -c` → one line, count 2).
- Full pytest ALWAYS under `systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q` (R-T2-6). `pytest -q` prints no tail summary on this box — check `$?` and pipe `tr '\r' '\n'`.
- Mutation checks: purge `__pycache__` before every run, `PYTHONDONTWRITEBYTECODE=1`, restore from a `cp` backup — **NEVER `git checkout`** (it wipes uncommitted edits; burned 2026-08-25).
- `runs/` and `streams/` are gitignored; never commit them. `.superpowers/` ledgers stay private.
- The frozen control (`runs/gate-b-search/B_search/lens.json`) is READ-ONLY evidence: never re-run B_search, never regenerate its lens.
- After the lock tag: no endpoint, threshold, stream, or arm change; an infra-killed run is archived and cleanly rerun into a FRESH dir (R-S4-1), never resumed.
- Box gotchas: bare `vllm` needs `.venv/bin` on PATH; kill vLLM by the EngineCore pid from `nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader` (pkill misses it / pgrep self-matches); exit 144 from a compound Bash command = harness kill, re-verify state before retrying.

---

### Task 1: 14B serve window 8192 → 16384

**Files:**
- Modify: `crucible/run/serving.py` (SERVE entry `"Qwen/Qwen2.5-Coder-14B-Instruct"`)
- Test: `tests/run/test_serving.py::test_serve_14b_fallback_carries_awq_repo_and_eager`

**Interfaces:** Produces: the SERVE table entry Task 8's server launch reads via `scripts/serve_model.sh "Qwen/Qwen2.5-Coder-14B-Instruct"`. Nothing else changes: hf_id stays `Qwen/Qwen2.5-Coder-14B-Instruct-AWQ`, `--enforce-eager` and util 0.90 stay.

- [ ] **Step 1: Flip the pin test.** In `test_serve_14b_fallback_carries_awq_repo_and_eager`, change the expectation line to

```python
    assert spec.extra_args[spec.extra_args.index("--max-model-len") + 1] == "16384"
```

and extend the docstring: `8192 → 16384 (2026-08-25, Phase-B prereg §4.4): B_mem's memory-augmented refinement prompts exceed 8192−2048 — the same overflow that infra-killed A_full's first gate attempt at task 184. Pure KV capacity; sampling-neutral.`

- [ ] **Step 2: Run it — must FAIL** (spec still says 8192): `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/run/test_serving.py -q` → 1 failed.
- [ ] **Step 3: Implement.** In `crucible/run/serving.py`, in the 14B `ServeSpec`, change `"--max-model-len", "8192"` to `"--max-model-len", "16384"` and add a comment line citing Phase-B prereg §4.4 and the A_full task-184 overflow.
- [ ] **Step 4: Run `tests/run/test_serving.py` — all pass.**
- [ ] **Step 5: Commit:** `git add -A && git commit -m "fix: 14B serve window 8192 -> 16384 (Phase-B prereg §4.4)"`

### Task 2: ARMS entries B_mem + A_mem_exactonly

**Files:**
- Modify: `crucible/run/arm.py` (ARMS dict)
- Test: `tests/run/test_arm.py`

**Interfaces:** Produces: `ARMS["B_mem"] == ArmConfig("B_mem", "Qwen/Qwen2.5-Coder-14B-Instruct", True, chat=True)` and `ARMS["A_mem_exactonly"] == ArmConfig("A_mem_exactonly", "Qwen/Qwen2.5-Coder-1.5B-Instruct", True, chat=True)`. Tasks 4/6 gate on these names.

- [ ] **Step 1: Write the failing test** in `tests/run/test_arm.py`, after `test_ablation_arms_are_a_full_s_serving_identity`:

```python
def test_phase_b_arms_carry_their_controls_serving_identity():
    """Phase-B (prereg §3): B_mem must reach the server exactly as the frozen B_search
    did, and A_mem_exactonly exactly as A_full/A_mem_nosleep — or a rate difference is
    confounded by serving instead of isolating the store / the retrieval policy."""
    assert ARMS["B_mem"] == ArmConfig("B_mem", ARMS["B_search"].model, True, chat=True)
    assert (ARMS["B_mem"].k, ARMS["B_mem"].width, ARMS["B_mem"].seed) == (
        ARMS["B_search"].k, ARMS["B_search"].width, ARMS["B_search"].seed)
    assert ARMS["A_mem_exactonly"] == ArmConfig(
        "A_mem_exactonly", ARMS["A_full"].model, True, chat=True)
```

- [ ] **Step 2: Run it — FAIL** (KeyError): `.venv/bin/python -m pytest tests/run/test_arm.py -q`
- [ ] **Step 3: Implement** — two entries at the end of `ARMS` in `crucible/run/arm.py`, with a comment citing Phase-B prereg §3 and noting (as the existing ablation comment does) that the arm's difference lives in the HOOKS, never here:

```python
    "B_mem": ArmConfig("B_mem", "Qwen/Qwen2.5-Coder-14B-Instruct", True, chat=True),
    "A_mem_exactonly": ArmConfig("A_mem_exactonly", "Qwen/Qwen2.5-Coder-1.5B-Instruct", True, chat=True),
```

- [ ] **Step 4: Run `tests/run/test_arm.py` — all pass.**
- [ ] **Step 5: Commit:** `git commit -am "feat: Phase-B arm configs B_mem + A_mem_exactonly (prereg §3)"`

### Task 3: `retrieve(..., exact_only=True)`

**Files:**
- Modify: `crucible/memory/retrieve.py` (`retrieve` signature + one branch)
- Test: `tests/memory/test_retrieve.py`

**Interfaces:** Produces: `retrieve(store, unit_id, family, *, exact_only: bool = False) -> RetrievedBlock`. INVARIANT: `exact_only` changes exactly one decision — when the exact-class pool has no live lesson, the eligible-lesson pool is EMPTY instead of the family-wide pool. The exemplar path (`_pick_exemplar`, already class-exact) is untouched, so a class with a live exact lesson returns a byte-identical block under both flags.

- [ ] **Step 1: Write the failing tests.** `tests/memory/test_retrieve.py` already has tests covering the family-wide fallback — reuse that file's existing store/lesson fixture helpers (read the neighbouring tests and build state the same way). Two tests:

```python
def test_exact_only_gives_a_stranger_silence_not_family_lessons(...):
    """Seed ONLY a family-wide lesson (different unit, same family) plus its episode.
    Full policy returns a block citing it; exact_only must return
    RetrievedBlock(None, ()) for the stranger unit — silence, never "" (prereg §3)."""
    # arrange: same seeding as this file's family-fallback test, target a DIFFERENT unit_id
    assert retrieve(store, stranger_unit, family).block is not None      # full policy: fallback fires
    got = retrieve(store, stranger_unit, family, exact_only=True)
    assert got.block is None and got.item_ids == ()

def test_exact_only_is_identical_when_an_exact_class_lesson_exists(...):
    """Seed an exact-class lesson for (unit, family). exact_only and full policy must
    return equal blocks and item_ids — the flag only severs the FALLBACK."""
    assert retrieve(store, unit, family, exact_only=True) == retrieve(store, unit, family)
```

- [ ] **Step 2: Run them — FAIL** (unexpected keyword `exact_only`).
- [ ] **Step 3: Implement.** In `retrieve`, add `*, exact_only: bool = False` and change the fallback branch:

```python
    if live_exact:
        eligible = live_exact
    elif exact_only:
        eligible = []   # A_mem_exactonly (Phase-B §3): strangers get silence, never family-wide lessons
    else:
        family_pool = store.semantic_family(family)
        eligible = [item for item in family_pool if item.falsified_by is None]
```

Document the flag in the docstring: one sentence, citing Phase-B prereg §3 and that the exemplar path is deliberately untouched.

- [ ] **Step 4: Run `tests/memory/test_retrieve.py` — all pass.**
- [ ] **Step 5: Commit:** `git commit -am "feat: exact_only retrieval policy (Phase-B prereg §4.3)"`

### Task 4: FULL_FAMILY retrieval modes + A_mem_exactonly wiring

**Files:**
- Modify: `crucible/run/full.py` (FULL_FAMILY, FullHooks, build_full_hooks), `crucible/cli.py` (`_arm_run` unpack)
- Test: `tests/run/test_full.py`, `tests/test_cli.py`

**Interfaces:** Produces: `FULL_FAMILY: dict[str, tuple[str, bool]]` mapping arm → `(retrieval_mode, sleep_enabled)` with retrieval_mode ∈ `{"full","exact","off"}`; `FullHooks(..., retrieval: str = "full", sleep_enabled: bool = True)`; property `retrieval_mode -> str`; property `retrieval_enabled -> bool` (== `mode != "off"`, kept so existing assertions survive); `build_full_hooks(..., retrieval: str = "full", sleep_enabled: bool = True)`. Consumes Task 3's `exact_only` flag.

- [ ] **Step 1: Update the pin test** `test_full_family_maps_each_ablation_to_a_full_minus_exactly_one_mechanism` in `tests/run/test_full.py` to:

```python
    assert FULL_FAMILY == {"A_full": ("full", True),
                           "A_mem_nosleep": ("full", False),
                           "A_sleep_nomem": ("off", True),
                           "A_mem_exactonly": ("exact", False)}
```

and add one new hooks-level test (reuse `_Rig`; it forwards kwargs to FullHooks — update `_Rig.__init__` to take `retrieval="full"` instead of `retrieval_enabled=True` and pass it through; migrate the two existing ablation tests that pass `retrieval_enabled=False` to `retrieval="off"`):

```python
def test_exact_mode_serves_repeats_but_silences_strangers(tmp_path):
    """retrieval="exact": after a verified attempt mints a lesson for SPEC's class, a
    second exposure of the SAME class still gets a block, but a task of the same family
    on a DIFFERENT unit gets None with item_ids=() (Phase-B §3 reading guide)."""
    rig = _Rig(tmp_path, retrieval="exact")
    _open_task(rig.hooks)
    rig.hooks.after_task(U, SPEC, _record(), _result(), NOW)          # mints exact-class lesson
    same = rig.hooks.before_task(U, replace(SPEC, task_key="k2", phase=2, kind="second"))
    assert same is not None                                            # exact class: served
    rig.hooks.task_confidence().calibrate(0.6)
    rig.hooks.after_task(U, replace(SPEC, task_key="k2", phase=2, kind="second"), _record(), _result(), NOW)
    stranger = replace(SPEC, task_key="k3", unit_id="Y/9", class_id="Y/9|ARITH", kind="novel", phase=2)
    assert rig.hooks.before_task(U, stranger) is None                  # stranger: silence
    assert rig.value.begun[-1] == (SPEC.family, False)
```

- [ ] **Step 2: Run `tests/run/test_full.py` — FAILs** (old tuple shape / unknown kwarg).
- [ ] **Step 3: Implement** in `crucible/run/full.py`: new FULL_FAMILY dict exactly as the pin; FullHooks `__init__` takes `retrieval: str = "full"` (store as `self._retrieval`; keep `sleep_enabled` as-is; delete `retrieval_enabled` param); `before_task` becomes

```python
        if self._retrieval != "off":
            block = retrieve(self._store, taskspec.unit_id, taskspec.family,
                             exact_only=(self._retrieval == "exact"))
        else:
            block = RetrievedBlock(None, ())
```

properties: `retrieval_mode` returns `self._retrieval`; `retrieval_enabled` returns `self._retrieval != "off"` (docstring: kept for the Phase-A assertions). `build_full_hooks` mirrors the signature change. In `cli.py` `_arm_run`: `retrieval_mode, sleep_enabled = FULL_FAMILY[cfg.name]` and pass `retrieval=retrieval_mode`.
- [ ] **Step 4: Extend the CLI parametrize** in `tests/test_cli.py::test_cli_arm_run_full_family_wires_the_declared_switches` with `("A_mem_exactonly", True, False)` and add inside the test body: `assert hooks.retrieval_mode == {"A_full": "full", "A_mem_nosleep": "full", "A_sleep_nomem": "off", "A_mem_exactonly": "exact"}[arm]`.
- [ ] **Step 5: Run `tests/run/test_full.py tests/test_cli.py` — all pass.**
- [ ] **Step 6: Commit:** `git commit -am "feat: retrieval modes in FULL_FAMILY + A_mem_exactonly arm (prereg §4.3)"`

### Task 5: extract `build_episode`

**Files:**
- Modify: `crucible/run/full.py` (`FullHooks._episode` → module-level `build_episode`)
- Test: existing episode tests in `tests/run/test_full.py` are the net — no new test

**Interfaces:** Produces: module-level `build_episode(taskspec: TaskSpec, record: TaskRecord, result: SearchResult, memory_item_ids: tuple[str, ...], now: str, confidence: float) -> EpisodicRecord` — the body of today's `FullHooks._episode` verbatim with `pending.item_ids` → `memory_item_ids`. `FullHooks._episode` becomes a one-line delegate. Task 6's MemHooks consumes it.

- [ ] **Step 1: Refactor** exactly as above (move the docstring to the function; `_episode`'s own docstring shrinks to "delegates — see build_episode").
- [ ] **Step 2: Run `tests/run/test_full.py` — all pass** (episode-writing tests prove the extraction changed nothing).
- [ ] **Step 3: Commit:** `git commit -am "refactor: extract build_episode for MemHooks reuse"`

### Task 6: MemHooks + build_mem_hooks + CLI gate

**Files:**
- Modify: `crucible/run/full.py` (MemHooks, MEM_ARMS, build_mem_hooks), `crucible/cli.py` (`_arm_run` branch)
- Test: `tests/run/test_full.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: `build_episode` (Task 5), `retrieve` (Task 3, full policy), `distill`, `MemoryStore`, `_assert_resume_coherent`, `MEMORY_DB_FILE`.
- Produces: `MEM_ARMS = frozenset({"B_mem"})`; `class MemHooks` with `before_task/task_confidence/after_task/between_tasks`; `build_mem_hooks(cfg: ArmConfig, stream_dir: Path, out_dir: Path, *, memory_db: Path | None = None, log=print) -> MemHooks`.

The class, verbatim (this is the guess-wrong spot — the driver calls `task_confidence()` unconditionally when hooks are present, and returning `None` is what keeps B_mem's status rule byte-identical to B_search's):

```python
class MemHooks:
    """B_mem's hooks: the store and NOTHING else (Phase-B prereg §3).

    The one Phase-B treatment is the memory block in the prompt. Everything else that
    FullHooks carries is deliberately ABSENT: ``task_confidence()`` returns ``None`` so
    the search runs the S2 structural status rule the frozen B_search control ran (the
    driver threads the return unconditionally -- None IS the S2 configuration, see
    ``attempt_task``); there is no value model to train (the caller passes
    ``ConstantValue``, as B_search's run did) and no calibrator to observe. Episodes are
    written for every attempt and lessons distilled from verified ones -- the same two
    gates as FullHooks.after_task -- because the store must FILL during the run for
    retrieval to have anything to say by phase 2.
    """

    def __init__(self, store: MemoryStore, *, log=print) -> None:
        self._store = store
        self._log = log
        self._pending: tuple[str, tuple[str, ...]] | None = None   # (task_key, item_ids)

    def before_task(self, unit: Unit, taskspec: TaskSpec) -> str | None:
        block = retrieve(self._store, taskspec.unit_id, taskspec.family)
        self._pending = (taskspec.task_key, block.item_ids)
        return block.block

    def task_confidence(self) -> None:
        return None                      # S2 status rule -- byte-identical to B_search

    def after_task(self, unit: Unit, taskspec: TaskSpec, record: TaskRecord,
                   result: SearchResult, now: str) -> tuple[tuple[str, ...], str | None]:
        pending = self._pending
        if pending is None or pending[0] != taskspec.task_key:
            raise ValueError(
                f"after_task for {taskspec.task_key!r} without a matching before_task "
                f"(pending={pending[0] if pending else None!r}) -- guessing retrieved_ids "
                f"would fabricate the record's memory column")
        self._pending = None
        episode = build_episode(taskspec, record, result, pending[1], now, record.confidence)
        self._store.write_episode(episode)
        if episode.verified and episode.landed_module is not None and result.symptom_failed:
            spans = (taskspec.span,) if taskspec.span2 is None else (taskspec.span, taskspec.span2)
            self._store.write_semantic(distill(
                episode, mutated_src=unit.module_src, spans=spans,
                flipped_tests=result.symptom_failed, killing_tests=result.symptom_failed,
                now=now,
            ))
        return pending[1], None          # adapter_id is always None: nothing ever trains

    def between_tasks(self, solved_task_keys: list[str], now: str) -> None:
        return None                      # no sleep, structurally
```

`build_mem_hooks`: mirrors `build_full_hooks`'s store setup ONLY — arm_dir mkdir, db path (`memory_db` override else `arm_dir / MEMORY_DB_FILE`), `MemoryStore`, `store.bind_identity(cfg.name, stream_hash)`, `_assert_resume_coherent(store, arm_dir)`, return `MemHooks(store, log=log)`. No registry, no controller, no proposer wrap.

CLI `_arm_run`: extend the existing local import to `from crucible.run.full import FULL_FAMILY, MEM_ARMS, build_full_hooks`, then after the FULL_FAMILY branch add

```python
    elif cfg.name in MEM_ARMS:
        from crucible.run.full import build_mem_hooks
        hooks = build_mem_hooks(cfg, a.stream_dir, a.out, memory_db=a.memory_db)
```

(`value` stays `ConstantValue`; `proposer` stays the plain client — no AdapterProposer.)

- [ ] **Step 1: Write the failing tests.** In `tests/run/test_full.py`:

```python
def test_mem_hooks_write_episodes_and_verified_lessons_but_never_train(tmp_path):
    """MemHooks (Phase-B §3): episode per attempt, lesson on verified only, confidence
    hook is None (the S2 status rule), adapter stamp always None, between_tasks inert."""
    store = MemoryStore(tmp_path / "memory.sqlite3")
    hooks = MemHooks(store)
    assert hooks.task_confidence() is None
    hooks.before_task(U, SPEC)
    ids, adapter = hooks.after_task(U, SPEC, _record(), _result(), NOW)
    assert (ids, adapter) == ((), None)
    assert len(store.episodes()) == 1 and store.semantic_for(SPEC.unit_id, SPEC.family)
    spec2 = replace(SPEC, task_key="k2", phase=2, kind="second")
    assert hooks.before_task(U, spec2) is not None        # the store FILLED and now serves
    hooks.after_task(U, spec2, _record(hidden_pass=False, status="believed"), _result(), NOW)
    assert len(store.episodes()) == 2
    assert len(store.semantic_for(SPEC.unit_id, SPEC.family)) == 1   # no lesson from a failure
    hooks.between_tasks(["k1"], NOW)                       # must be a no-op, nothing raises

def test_mem_hooks_after_task_without_before_task_raises(tmp_path):
    hooks = MemHooks(MemoryStore(tmp_path / "memory.sqlite3"))
    with pytest.raises(ValueError, match="without a matching before_task"):
        hooks.after_task(U, SPEC, _record(), _result(), NOW)
```

In `tests/test_cli.py`:

```python
def test_cli_arm_run_b_mem_wires_store_only_hooks(_stream, tmp_path, monkeypatch):
    """B_mem differs from the frozen B_search by the store ALONE (prereg §3): MemHooks,
    ConstantValue (never OnlineValue), the plain proposer (never AdapterProposer)."""
    from crucible.cli import main
    from crucible.run.full import AdapterProposer, MemHooks
    from crucible.value.model import ConstantValue
    seen = {}
    _stub_run(monkeypatch, seen)
    rc = main(["arm", "run", str(_stream), "--arm", "B_mem", "--base-url", "http://x",
               "--out", str(tmp_path / "runs")])
    assert rc == 0
    assert isinstance(seen["hooks"], MemHooks)
    assert isinstance(seen["value"], ConstantValue)
    assert not isinstance(seen["proposer"], AdapterProposer)
    assert (tmp_path / "runs" / "B_mem" / "memory.sqlite3").exists()
```

- [ ] **Step 2: Run both files — FAIL** (ImportError: MemHooks).
- [ ] **Step 3: Implement** as specified above.
- [ ] **Step 4: Run `tests/run/test_full.py tests/test_cli.py` — all pass.**
- [ ] **Step 5: Commit:** `git commit -am "feat: MemHooks + B_mem CLI wiring — the store and nothing else (prereg §4.2)"`

### Task 7: mutation checks + full suite

**Files:** none created — verification only. Run from repo root.

- [ ] **Step 1: Define the cp-backup harness** (shell function; NEVER git checkout):

```bash
mut2() { cp "$1" "$1.mutbak"; sed -i "$2" "$1";
  if cmp -s "$1" "$1.mutbak"; then echo "NOOP(BAD): $4"; else
    find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
    if PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest "$3" -q >/dev/null 2>&1; then echo "SURVIVED(BAD): $4"; else echo "killed: $4"; fi; fi
  mv "$1.mutbak" "$1"; find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null; }
```

- [ ] **Step 2: Run the five mutations — every one must print `killed:`** (a NOOP or SURVIVED is a defect in the test, fix before proceeding):

```bash
mut2 crucible/memory/retrieve.py 's/elif exact_only:/elif False:/' tests/memory/test_retrieve.py "exact_only branch disabled"
mut2 crucible/run/full.py 's/"A_mem_exactonly": ("exact", False)/"A_mem_exactonly": ("full", False)/' tests/test_cli.py::test_cli_arm_run_full_family_wires_the_declared_switches "FULL_FAMILY exact mode flip"
mut2 crucible/run/full.py 's/exact_only=(self._retrieval == "exact")/exact_only=False/' tests/run/test_full.py::test_exact_mode_serves_repeats_but_silences_strangers "before_task exact flag dropped"
mut2 crucible/run/full.py 's/if episode.verified and episode.landed_module is not None and result.symptom_failed:/if episode.landed_module is not None and result.symptom_failed:/2' tests/run/test_full.py::test_mem_hooks_write_episodes_and_verified_lessons_but_never_train "MemHooks verified gate dropped"
mut2 crucible/run/serving.py 's/"--max-model-len", "16384", "--gpu-memory-utilization", "0.90"/"--max-model-len", "8192", "--gpu-memory-utilization", "0.90"/' tests/run/test_serving.py "14B window reverted"
```

(For the 4th: `sed` `/2` targets the second occurrence — FullHooks' gate is the first. If occurrence order differs on disk, adapt the address but verify with `git diff` that ONLY MemHooks' line changed before running, then restore.)

- [ ] **Step 3: Full suite under the scope** (Global Constraints command) → exit 0.
- [ ] **Step 4: Secret-scan, commit anything outstanding, push, verify sync.**

### Task 8: pre-lock smoke (GPU, non-gating)

**Files:** creates `runs/b-smoke*` (gitignored), `runs/smoke-tasks.txt`.

- [ ] **Step 1: Build the smoke task file** — 8 firsts plus the second exposures of the SAME classes, so retrieval fires with real content (the stream is phase-ordered: indices 0–199 are `first`, 200+ hold `second`/`novel`):

```bash
.venv/bin/python - <<'EOF'
import pathlib
from crucible.stream import store
m = store.read_manifest(pathlib.Path("streams/1158e92f40ad"))
firsts = m.tasks[:8]
classes = {t.class_id for t in firsts}
seconds = [t for t in m.tasks if t.kind == "second" and t.class_id in classes]
keys = [t.task_key for t in firsts] + [t.task_key for t in seconds]
pathlib.Path("runs/smoke-tasks.txt").write_text("\n".join(keys) + "\n")
print(len(keys), "tasks;", len(seconds), "second exposures")
EOF
```

- [ ] **Step 2: Serve the 14B** (PATH gotcha; ~2–4 min to load):

```bash
PATH="$PWD/.venv/bin:$PATH" setsid nohup scripts/serve_model.sh "Qwen/Qwen2.5-Coder-14B-Instruct" > runs/b-serve-14b.log 2>&1 &
# poll: curl -sf http://127.0.0.1:8010/v1/models until UP; assert the served id is
# Qwen/Qwen2.5-Coder-14B-Instruct; then nvidia-smi — expect ~13-14 GiB used (weights+KV at util 0.90).
```

Precondition before serving: free VRAM ≥ 13 GiB (`nvidia-smi --query-gpu=memory.free --format=csv`).

- [ ] **Step 3: Run the smoke:** `scripts/run_arm_detached.sh B_mem runs/b-smoke runs/smoke-tasks.txt` → watch `runs/b-smoke.DONE`.
- [ ] **Step 4: Check, in order:** (a) `cat runs/b-smoke.DONE` = 0; (b) `grep -c "HTTP\|400\|Error" runs/b-smoke.log` = 0 context overflows; (c) every smoke record has `hidden_pass` non-null and the second-exposure records carry non-empty `retrieved_ids`; (d) `grep -iE "preempt|KV cache" runs/b-serve-14b.log` shows no repeated preemption churn; (e) note s/task for the Task 10 ETA.
- [ ] **Step 5: Judgment gate:** if (b) or (d) fails → fall back to 12288 per prereg §4.4 (edit SERVE + pin test, commit as a PRE-LOCK amendment noted in prereg §11, restart server, re-smoke). Two failures → STOP per prereg §7, report to Brice.
- [ ] **Step 6:** Leave the server running (Task 10 uses it). Commit nothing (runs/ is gitignored).

### Task 9: lock — `docs/LOCK-B.md` + tag

**Files:**
- Create: `docs/LOCK-B.md`

- [ ] **Step 1: Compute the lock values** (each printed by command, none typed from memory):

```bash
.venv/bin/python - <<'EOF'
import hashlib, json, math, pathlib
ctl = pathlib.Path("runs/gate-b-search/B_search/lens.json")
abl = pathlib.Path("runs/abl-mem-nosleep/A_mem_nosleep/lens.json")
p = json.loads(ctl.read_text())["succ_phase1"]
print("p_bar =", p, "Delta_min =", round(2*math.sqrt(2*p*(1-p)/200), 4))
for f in (ctl, abl): print(f, hashlib.sha256(f.read_bytes()).hexdigest())
EOF
```

- [ ] **Step 2: Write `docs/LOCK-B.md`** following LOCK-A.md's shape: spec commit hash; Δ_min = 0.0807 WITH the printed derivation inputs (p̄=0.795, C=200); both lens sha256 digests; stream hash (copy the full 64-hex from LOCK-A); 14B AWQ revision + 3 shard digests and 1.5B revision (copy from LOCK-A — same artifacts); vLLM 0.27.1; the two SERVE entries verbatim as post-Task-1 code; arms `B_mem`, `A_mem_exactonly`; task set `--tasks all`; smoke outcome one-liner (EXIT, s/task, window verdict 16384-or-12288).
- [ ] **Step 3: Commit, tag, push:** `git add docs/LOCK-B.md && git commit -m "docs: Phase-B lock record" && git tag prereg-lock-b && git push -q origin master prereg-lock-b` → verify sync AND `git ls-remote --tags origin | grep prereg-lock-b`.

### Task 10: gating run — B_mem (450 tasks)

- [ ] **Step 1: Launch:** `scripts/run_arm_detached.sh B_mem runs/gate-b2-mem all` — fresh dir (launcher refuses non-empty). Confirm exactly ONE driver: `pgrep -af "crucible arm run" | grep -v pgrep` shows one bash wrapper + one python.
- [ ] **Step 2: Arm the watcher** (Monitor tool, persistent; harness-tracked `Bash run_in_background` waiters get reaped): poll `runs/gate-b2-mem.DONE` OR pid death every 30 s, emit which one fired. ETA ≈ 450 × smoke s/task.
- [ ] **Step 3: On DONE=0:** write the lens —

```bash
.venv/bin/python -c "
import json, pathlib
from crucible.run.lens import build_lens
from crucible.run.records import read_task_records
d = pathlib.Path('runs/gate-b2-mem/B_mem')
lens = build_lens(read_task_records(d))
d.joinpath('lens.json').write_text(json.dumps(lens.to_dict(), indent=1, sort_keys=True))
print(json.dumps({k: v for k, v in lens.to_dict().items() if k != 'adapter_ids'}, sort_keys=True))"
```

- [ ] **Step 4: On a non-zero DONE or death (infra kill):** diagnose from `runs/gate-b2-mem.log`, fix ONLY infra (R-S4-1), `mv runs/gate-b2-mem runs/gate-b2-mem-infra1` (and its `.log/.pid/.DONE`), clean rerun into a fresh `runs/gate-b2-mem`. Never resume, never splice. A completed run is NEVER rerun.

### Task 11: exploratory run — A_mem_exactonly (450 tasks)

- [ ] **Step 1: Swap servers:** kill the 14B EngineCore by pid (`nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader`), wait for port 8010 DOWN, confirm free VRAM ≥ 13 GiB, then serve the 1.5B: `PATH="$PWD/.venv/bin:$PATH" setsid nohup scripts/serve_model.sh "Qwen/Qwen2.5-Coder-1.5B-Instruct" > runs/b-serve-1p5b.log 2>&1 &`, poll UP + identity.
- [ ] **Step 2: Launch:** `scripts/run_arm_detached.sh A_mem_exactonly runs/abl-mem-exactonly all`; single-driver check; Monitor watcher as in Task 10 (ETA ~85 min).
- [ ] **Step 3: On DONE=0:** write `lens.json` (same snippet, path `runs/abl-mem-exactonly/A_mem_exactonly`). Structural invariants: no sleep-records file; `retrieved_ids` non-empty on (most) second exposures but EMPTY on novel records whose class had no exact content.

### Task 12: verdict + GATE-B.md + closeout

- [ ] **Step 1: Compute §5 exactly as locked** — B1: Δ_B = succ_second − succ_phase1 of `runs/gate-b2-mem/B_mem/lens.json`, pass iff Δ_B ≥ Δ_min(LOCK-B) AND −0.0100 ∈ ±Δ_min (frozen, true), with the saturation clause checked FIRST (succ_phase1 > 1 − Δ_min → not exercisable); CONFOUNDED checks: infra_rate > 0.02, landing < 0.98. B2/B3/B4 + the exploratory reading-guide comparisons, all against the LOCK-B digested lenses.
- [ ] **Step 2: Write `docs/findings/GATE-B.md`** in GATE-A.md's shape: run ledger (incl. any infra kill), endpoint table with bars, verdict GO_B / NO-GO / CONFOUNDED per §6 verbatim, exploratory table with its pre-stated reading guide answered, GPU accounting.
- [ ] **Step 3: Teardown:** kill the 1.5B EngineCore by pid, wait port DOWN, `nvidia-smi` ≈ desktop-idle; ollama is a system unit and manages itself — just verify `systemctl is-active ollama`.
- [ ] **Step 4: Secret-scan, commit findings, push, verify sync.**
- [ ] **Step 5: Update memory** (`crucible-project.md` + `MEMORY.md` line) and append to the session handoff; report to Brice with the endpoint table and the §6-prescribed next step (GO_B → Phase-C retrieval policy; NO-GO → ship the findings arc).
