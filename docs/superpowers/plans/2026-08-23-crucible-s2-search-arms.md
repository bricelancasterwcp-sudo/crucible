# crucible S2 — Search, Arms, Ceiling Pilot — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the proposer→search→verify loop and the arm driver, then run the 30-task ceiling pilot that tells us whether the S1 stream is calibrated for Phase A.

**Architecture:** A frozen small LLM (Qwen3.5-2B, served by vLLM — proven in S1) proposes a **full-module rewrite** to repair a mutated function; a ported **REx Thompson-sampling scheduler** spends a budget of **K=8** visible-suite executions across a refinement tree, re-prompting with test feedback; a **value-fn v0** (untrained constant) orders candidates; a **driver** runs an *arm* (A_noMem = 2B+search, B_search = 9B+search, B_naive = 9B single-shot) over a task set, scores the final submission against the **hidden** suite, and writes per-execution + per-task records; a **lens** reduces records to success-by-phase for E1/E2. This slice has **no memory, no A_full, no sleep, no uncertainty** — those are S3. Arms serve one model at a time (2B and 9B do not co-reside in 16 GiB), swapping via a serving harness.

**Tech Stack:** Python 3.12, uv venv, pytest + pytest-timeout, vLLM 0.27.1 (served with `VLLM_USE_FLASHINFER_SAMPLER=0`), the S1 `crucible/` package (stream, sandbox, proposer/identity). REx (MIT, ported). No new heavyweight deps beyond what S1 + the `serve` extra already declare.

**Spec:** `docs/superpowers/specs/2026-08-23-crucible-phase-a-prereg.md` — read §4.4 (agent I/O), §4.5 (sandbox), §4.6 (budget), §4.7 (codec landing), §4.8.4 (ceiling pilot), §5 (search/value contracts — memory/uncertainty parts are S3), §6 (endpoints), §9 (architecture/interfaces), §10 (slice 2 scope).

**Serving facts (measured in S1, `docs/findings/S1-serving.md`):** vLLM serves `Qwen/Qwen3.5-2B` with n-best + logprobs; the FlashInfer sampler JIT needs `ninja`+`nvcc` (absent) so launch with `VLLM_USE_FLASHINFER_SAMPLER=0`. `Qwen3.5-2B` bf16 = 9.6 GiB @ 138 tok/s; `lovedheart/Qwen3.5-9B-FP8` = 14.2 GiB @ 57 tok/s, needs `--enforce-eager --gpu-memory-utilization 0.90 --max-model-len 4096`. Port 8001 is taken on this box — use **8010**. Real stream at `streams/full/dd5912cddedc` (450 tasks, passes all pre-checks).

## Global Constraints

- Python `>=3.12,<3.13`; venv via `uv` at `.venv`; never the system Python. Tests run with `.venv/bin/python -m pytest`.
- **R-T2-6 (memory safety, non-negotiable):** every pytest run that touches the sandbox (`tests/sandbox`, `tests/search`, `tests/run`, or any test that calls `run_tests`/`sandbox.run`) MUST be wrapped: `systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 -- .venv/bin/python -m pytest …`. An uncapped sandbox test OOM-killed a session. Pure-unit tests (proposer prompt/codec parsing, REx math, records, lens) spawn no sandbox and run unwrapped.
- **Any live-serving or arm run** (a real vLLM server + real task execution) runs the vLLM server in its own scope and the driver under a cgroup cap; never launch an unbounded full run in the session's terminal scope. Model swaps stop the previous server (by recorded PID, never `pkill -f 'vllm serve'` which self-matches).
- Licensing: only Apache-2.0 / MIT / BSD artifacts enter the tree; every ported artifact (REx) gets a `THIRD_PARTY.md` row with its verification command. No copyleft.
- **Instrument honesty (spec §8), enforced everywhere:** the agent NEVER sees or executes hidden tests; a submission that alters the test files or touches anything but the module scores `tampered` (failure); `infra_error` is never a measurement and never charged to the budget; **K=8** visible executions per task after **one free** symptom run; cached (identical patch+subset) results are free; hidden-suite pass is the outcome, computed by the driver, never by the agent.
- Determinism: all randomness via `random.Random(f"{seed}:<purpose>")` or an explicit `seed=` passed to the proposer; REx uses a seeded `random.Random`; no module-level randomness. Same (arm, task, seed) ⇒ same record modulo genuine model nondeterminism, which is pinned by `seed=` + `temperature` in the proposer call.
- Keys are content hashes (sha256) where a record needs an identity; task identity = `TaskSpec.task_key` (= mutant key) from S1.
- Every load-bearing test is mutation-checked (break the pinned line → test FAILS → restore) with `__pycache__` purged and `PYTHONDONTWRITEBYTECODE=1` before and after restore.
- Files ≤ 400 lines typical, 800 max; functions < 50 lines; organise by feature (`proposer/`, `search/`, `value/`, `run/`). Data types crossing a file boundary have `to_dict()`/`from_dict()` round-trips with a completeness test.
- Commit after every green step; messages `<type>: <description>`; no trailers.

---

### Task 1: `Candidate` type + `run(unit, patch, subset)` sandbox wrapper

**Files:**
- Create: `crucible/run/types.py`, `crucible/sandbox/task_run.py`, `tests/run/__init__.py`, `tests/run/test_types.py`, `tests/sandbox/test_task_run.py`

**Interfaces:**
- Consumes: `crucible/sandbox/runner.py::run_tests(module_name, module_src, test_src, *, subset, per_test_timeout_s, wall_cap_s, mem_limit_bytes, python) -> TestReport`; `crucible/stream/units.py::Unit`.
- Produces:
  ```python
  # crucible/run/types.py
  @dataclass(frozen=True)
  class Candidate:
      text: str                 # the model's full-module rewrite (post-codec-extraction)
      mean_logprob: float | None
      self_certainty: float | None
      def to_dict(self) -> dict; @classmethod from_dict(cls, d) -> "Candidate"
  # crucible/sandbox/task_run.py
  def run(unit: Unit, patch: str, subset: list[str] | None, *,
          per_test_timeout_s: float = 5.0, wall_cap_s: float = 60.0) -> TestReport
      # applies `patch` as the module source, runs the VISIBLE suite (optionally a subset).
      # Never runs the hidden suite. Thin wrapper over run_tests(unit.module_name, patch, unit.visible_test_src, ...).
  def run_hidden(unit: Unit, patch: str, *, per_test_timeout_s: float = 5.0, wall_cap_s: float = 60.0) -> TestReport
      # the OUTCOME oracle: runs the HIDDEN suite against a final submission. Driver-only; the agent never calls this.
  ```

- [ ] **Step 1: Failing tests** — `tests/run/test_types.py`:
```python
from crucible.run.types import Candidate

def test_candidate_round_trips_through_json():
    import json
    c = Candidate("def f():\n    return 1\n", -0.42, 0.83)
    assert Candidate.from_dict(json.loads(json.dumps(c.to_dict()))) == c

def test_candidate_allows_none_scores():
    c = Candidate("x", None, None)
    assert c.mean_logprob is None and Candidate.from_dict(c.to_dict()) == c
```
`tests/sandbox/test_task_run.py` (WRAP with systemd-run — touches the sandbox):
```python
from crucible.sandbox.task_run import run, run_hidden
from crucible.stream.units import Unit, sha256_text

SRC = "def add(a, b):\n    return a + b\n"
VIS = "from unit_x import add as candidate\nimport math\nATOL = 1e-06\ndef _eq(a, b):\n    return a == b\ndef test_v0():\n    assert _eq(candidate(1, 2), 3)\n"
HID = "from unit_x import add as candidate\nimport math\nATOL = 1e-06\ndef _eq(a, b):\n    return a == b\ndef test_h0():\n    assert _eq(candidate(0, 0), 0)\n"
U = Unit("X/0", "unit_x", "add", SRC, VIS, HID, sha256_text(SRC), 1, 1, ())

def test_run_executes_visible_only():
    r = run(U, SRC, None)
    assert r.all_passed and r.infra_error is None

def test_run_reports_failure_for_a_bad_patch():
    r = run(U, "def add(a, b):\n    return a - b\n", None)
    assert not r.all_passed and "test_v0" in r.failed

def test_run_hidden_is_the_outcome_oracle():
    assert run_hidden(U, SRC).all_passed
    assert not run_hidden(U, "def add(a, b):\n    return a - b\n").all_passed
```
- [ ] **Step 2: RED** — `systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 -- .venv/bin/python -m pytest tests/sandbox/test_task_run.py`; unwrapped for `tests/run/test_types.py`. Expect import errors.
- [ ] **Step 3: Implement** `crucible/run/types.py` (dataclass with `dataclasses.asdict` in `to_dict`, `cls(**d)` in `from_dict`) and `crucible/sandbox/task_run.py`:
```python
from ..stream.units import Unit
from .runner import run_tests, TestReport

def run(unit, patch, subset, *, per_test_timeout_s=5.0, wall_cap_s=60.0):
    return run_tests(unit.module_name, patch, unit.visible_test_src,
                     subset=subset, per_test_timeout_s=per_test_timeout_s, wall_cap_s=wall_cap_s)

def run_hidden(unit, patch, *, per_test_timeout_s=5.0, wall_cap_s=60.0):
    return run_tests(unit.module_name, patch, unit.hidden_test_src,
                     subset=None, per_test_timeout_s=per_test_timeout_s, wall_cap_s=wall_cap_s)
```
- [ ] **Step 4: GREEN** — both test files pass (sandbox one wrapped).
- [ ] **Step 5: Mutation check** — in `task_run.run`, swap `unit.visible_test_src` → `unit.hidden_test_src`; `test_run_executes_visible_only` still passes but `test_run_reports_failure_for_a_bad_patch` behaviour changes — if neither fails, add an assertion that `run` and `run_hidden` give DIFFERENT results for a patch that passes visible but fails hidden (craft `return a + b if a else 1`). Purge `__pycache__`, restore, re-green.
- [ ] **Step 6: Commit** — `feat(run): Candidate type and visible/hidden sandbox run wrappers`

---

### Task 2: Proposer prompt builder (full-module-rewrite codec)

**Files:**
- Create: `crucible/proposer/prompt.py`, `tests/proposer/test_prompt.py`

**Interfaces:**
- Consumes: `crucible/stream/units.py::Unit`; `crucible/sandbox/runner.py::TestReport` (the symptom).
- Produces:
  ```python
  def build_prompt(unit: Unit, symptom: TestReport, *, feedback: str | None = None) -> str
      # Constructs the repair prompt: the mutated module source, the visible test file, the symptom
      # (which visible tests failed + short reason), an instruction to emit the COMPLETE corrected
      # module inside a single ```python fenced block, and (on refinement) the prior attempt's feedback.
  MODULE_FENCE = "```python"   # the codec: model must return one fenced python block = the full module
  ```

- [ ] **Step 1: Failing test** — `tests/proposer/test_prompt.py` (no sandbox, unwrapped):
```python
from crucible.proposer.prompt import build_prompt
from crucible.sandbox.runner import TestReport
from crucible.stream.units import Unit, sha256_text

SRC = "def add(a, b):\n    return a - b\n"  # mutated (bug)
U = Unit("X/0","unit_x","add",SRC,"from unit_x import add as candidate\ndef test_v0():\n    assert candidate(1,2)==3\n","h",sha256_text(SRC),1,0,())
SYM = TestReport((), ("test_v0",), (), (), 0.1, None)

def test_prompt_contains_source_tests_symptom_and_codec_instruction():
    p = build_prompt(U, SYM)
    assert "def add(a, b):" in p and "test_v0" in p
    assert "complete" in p.lower() and "```python" in p
    assert "hidden" not in p.lower()      # never leak the notion of hidden tests as runnable

def test_feedback_is_included_on_refinement():
    p = build_prompt(U, SYM, feedback="attempt 1 still failed test_v0")
    assert "attempt 1 still failed test_v0" in p
```
- [ ] **Step 2: RED** (unwrapped). **Step 3: Implement** `build_prompt` — a deterministic f-string template: system-style preamble ("You are repairing a single Python function. Return the COMPLETE corrected module in one ```python block, nothing else."), then the module source, the visible test file, a rendered symptom line (`failed: test_v0`), and the optional feedback block. No randomness. **Step 4: GREEN.**
- [ ] **Step 5: Mutation check** — delete the `feedback` interpolation → `test_feedback_is_included_on_refinement` FAILS; restore. **Step 6: Commit** — `feat(proposer): full-module-rewrite repair prompt`

---

### Task 3: Codec — extract the full module + landing check

**Files:**
- Create: `crucible/proposer/codec.py`, `tests/proposer/test_codec.py`

**Interfaces:**
- Produces:
  ```python
  @dataclass(frozen=True)
  class Landed:
      ok: bool; module_src: str | None; reason: str | None   # reason ∈ {"no-fence","empty","syntax","multiple-fences-ok"}
  def extract_module(text: str) -> Landed
      # Pulls the LAST ```python … ``` block (models often restate then correct); if no fence, tries the
      # whole text if it parses. `ok` iff the extracted text is non-empty AND compiles (ast.parse). This is
      # the §4.7 "landing" definition: a submission LANDS iff it yields a parseable module.
  def landing_rate(texts: list[str]) -> float   # fraction of `texts` that land — the §4.7 pre-check statistic
  ```

- [ ] **Step 1: Failing test** — `tests/proposer/test_codec.py`:
```python
from crucible.proposer.codec import extract_module, landing_rate

def test_extracts_fenced_module_and_lands():
    t = "Here is the fix:\n```python\ndef add(a, b):\n    return a + b\n```\n"
    L = extract_module(t); assert L.ok and "return a + b" in L.module_src

def test_takes_last_fence_when_multiple():
    t = "```python\nWRONG\n```\nnow correct:\n```python\ndef f():\n    return 2\n```"
    assert extract_module(t).module_src.strip() == "def f():\n    return 2"

def test_unparseable_does_not_land():
    assert not extract_module("```python\ndef f(:\n```").ok
def test_no_fence_falls_back_to_whole_text_if_it_parses():
    assert extract_module("def f():\n    return 1\n").ok
def test_landing_rate():
    assert landing_rate(["```python\ndef f():\n    return 1\n```", "prose only, no code {"]) == 0.5
```
- [ ] **Step 2: RED. Step 3: Implement** — regex for ```` ```python … ``` ```` blocks (take the last), fall back to whole text; `ok = bool(stripped) and compiles`, guarding `compile()` under `warnings.catch_warnings(); simplefilter("ignore", SyntaxWarning)` (R-T11-1 lesson — do not let a SyntaxWarning become a landing failure under `-W error`). `landing_rate` = mean of `extract_module(t).ok`. **Step 4: GREEN.**
- [ ] **Step 5: Mutation check** — change "last fence" to "first fence" → `test_takes_last_fence_when_multiple` FAILS; drop the `compile()` guard → `test_unparseable_does_not_land` FAILS. Restore. **Step 6: Commit** — `feat(proposer): codec extraction and landing check`

---

### Task 4: `Proposer` protocol + vLLM adapter

**Files:**
- Create: `crucible/proposer/client.py`, `tests/proposer/test_client.py`

**Interfaces:**
- Consumes: `crucible/proposer/identity.py::assert_identity`; `crucible/proposer/codec.py::extract_module`; `crucible/run/types.py::Candidate`.
- Produces:
  ```python
  class Proposer(Protocol):
      def generate(self, prompt: str, *, n: int, seed: int, max_tokens: int = 1024,
                   temperature: float = 0.7) -> list[Candidate]: ...
      model: str
  class VLLMProposer:
      def __init__(self, base_url: str, model: str): ...   # asserts served identity on construction
      def generate(self, prompt, *, n, seed, max_tokens=1024, temperature=0.7) -> list[Candidate]
      # POSTs /v1/completions with n, logprobs=1, seed, temperature; for each choice: extract_module(text)
      # → Candidate(landed_src_or_raw, mean_logprob, self_certainty). mean_logprob = mean of token_logprobs;
      # self_certainty = mean over tokens of the top-token prob (from logprobs) — a cheap pre-execution prior.
  ```

- [ ] **Step 1: Failing test** — `tests/proposer/test_client.py` uses a **fake HTTP server** (like `tests/proposer/test_identity.py`) serving `/v1/models` (for the identity assert) and `/v1/completions` returning a canned 2-choice response with `logprobs.token_logprobs`. Assert: `generate(...,n=2)` returns 2 `Candidate`s, each `text` is the codec-extracted module, `mean_logprob` equals the mean of the canned logprobs, `self_certainty` in [0,1]. No GPU, no sandbox → unwrapped.
- [ ] **Step 2: RED. Step 3: Implement** `VLLMProposer` with `urllib` (stdlib, like identity.py); constructor calls `assert_identity(base_url, model)`. **Step 4: GREEN.**
- [ ] **Step 5: Mutation check** — make `generate` ignore `n` (always request 1) → the 2-candidate test FAILS; drop `extract_module` (return raw `text`) → a test asserting the fenced block is stripped FAILS. Restore. **Step 6: Commit** — `feat(proposer): Proposer protocol and vLLM adapter with logprob capture`

---

### Task 5: Port REx Thompson scheduler (verbatim, MIT)

**Files:**
- Create: `crucible/search/rex.py` (ported), `tests/search/__init__.py`, `tests/search/test_rex.py`
- Modify: `THIRD_PARTY.md` (add the REx row)

**Interfaces:**
- Produces:
  ```python
  # Ported verbatim from haotang1995/REx acr/scheduler/rex.py (MIT), adapted only to take an injected
  # random.Random for determinism. Beta(alpha,beta) Thompson sampling over "arms".
  class RexScheduler:
      def __init__(self, *, smoothing: float = 1.0, heuristic_weight: float = 1.0, rng: random.Random): ...
      def add_arm(self, arm_id, heuristic_reward: float = 0.0) -> None   # alpha = smoothing + heuristic_weight*heuristic_reward, beta = smoothing
      def select(self) -> arm_id                                          # argmax over arms of rng.beta(alpha, beta)
      def update(self, arm_id, reward: float) -> None                     # alpha += reward ; beta += (1 - reward)
  ```

- [ ] **Step 1: Failing test** — `tests/search/test_rex.py` (pure math, unwrapped):
```python
import random
from crucible.search.rex import RexScheduler

def test_selection_is_deterministic_under_a_seeded_rng():
    def run():
        s = RexScheduler(rng=random.Random("t"))
        for a in ("x","y","z"): s.add_arm(a)
        return [s.select() for _ in range(5)]
    assert run() == run()

def test_reward_shifts_selection_toward_the_rewarded_arm():
    s = RexScheduler(rng=random.Random("t"))
    for a in ("x","y"): s.add_arm(a)
    for _ in range(30): s.update("x", 1.0); s.update("y", 0.0)
    picks = [s.select() for _ in range(200)]
    assert picks.count("x") > picks.count("y") * 3
```
- [ ] **Step 2: RED. Step 3: Implement** the 45-line Beta-Thompson scheduler verbatim (only injecting `rng`), add the `THIRD_PARTY.md` row (`REx | github.com/haotang1995/REx | MIT | gh api repos/haotang1995/REx --jq .license.spdx_id | <date> | Thompson scheduler ported to crucible/search/rex.py`). **Step 4: GREEN.**
- [ ] **Step 5: Mutation check** — change `update` to `alpha += reward` only (drop the `beta += 1-reward`) → `test_reward_shifts_selection_toward_the_rewarded_arm` still passes (both grow) — so ALSO assert that after only-failures on "y", selecting favours "x"; break the beta update → that assertion FAILS. Restore. **Step 6: Commit** — `feat(search): port REx Thompson scheduler (MIT)`

---

### Task 6: Search `Node` + tree

**Files:**
- Create: `crucible/search/node.py`, `tests/search/test_node.py`

**Interfaces:**
- Consumes: `crucible/run/types.py::Candidate`; `crucible/sandbox/runner.py::TestReport`.
- Produces:
  ```python
  @dataclass
  class Node:
      node_id: str                  # sha256(candidate.text) — cached-result key
      candidate: Candidate
      parent_id: str | None
      report: TestReport | None     # visible-suite result once executed (None = unexecuted)
      status: str                   # "unexecuted" | "visible_partial" | "visible_pass" | "visible_fail"
      def visible_reward(self) -> float   # fraction of visible tests passing; 0.0 if unexecuted
  class Tree:
      def __init__(self, root: Node): ...
      def add(self, node: Node) -> None
      def children(self, node_id: str) -> list[Node]
      def best_visible(self) -> Node        # highest visible_reward, ties → highest self_certainty
  ```

- [ ] **Step 1: Failing tests** — `test_node.py` (unwrapped): `visible_reward` = passed / (passed+failed+timed_out+errored) from a `TestReport`; `status` classification; `best_visible` tie-break. **Step 2: RED. Step 3: Implement. Step 4: GREEN.**
- [ ] **Step 5: Mutation check** — make `visible_reward` count only `passed` in the numerator but total tests wrong (e.g. divide by passed) → a mixed-report test FAILS. Restore. **Step 6: Commit** — `feat(search): search node and refinement tree`

---

### Task 7: Search loop — propose→execute→refine under the budget

**Files:**
- Create: `crucible/search/loop.py`, `tests/search/test_loop.py`

**Interfaces:**
- Consumes: `RexScheduler`, `Tree`, `Node`, `sandbox.task_run.run`, `sandbox.budget.BudgetMeter`, `proposer.Proposer`, `proposer.prompt.build_prompt`, `value.Value` (Task 8).
- Produces:
  ```python
  @dataclass(frozen=True)
  class SearchResult:
      best_patch: str; best_node_id: str; visible_reward: float
      executions_charged: int; landed: bool; nodes: int
      status: str; confidence: float     # status ∈ {"verified_visible","believed","abstain"} ; confidence from value/self_certainty
  def search(unit, proposer, value, *, seed: int, k: int = 8, width: int = 4,
             per_test_timeout_s=5.0, wall_cap_s=60.0) -> SearchResult
      # 1) one FREE symptom run (visible suite on the mutated source) — not charged.
      # 2) seed the tree with `width` root candidates from the proposer; add each as a REx arm with
      #    heuristic_reward = value.score(node).
      # 3) loop while BudgetMeter has budget (k): REx.select() an arm; if unexecuted, run its visible
      #    subset (charge the report); reward = node.visible_reward(); REx.update. If reward==1.0 → stop
      #    (verified_visible). Else expand: build_prompt(...feedback from failing tests...), propose
      #    `width` children, add as arms.
      # 4) final submission = tree.best_visible(). status: verified_visible if reward==1.0; abstain if the
      #    best landed reward is below an abstain threshold AND value confidence low; else believed.
  ```

- [ ] **Step 1: Failing test** — `tests/search/test_loop.py` (WRAP — touches the sandbox). Use a **fake in-process Proposer** (returns scripted candidates, no network) + a real `Unit` fixture whose canonical repair is known. Assert: (a) a proposer that returns the correct module on the first candidate yields `status=="verified_visible"`, `visible_reward==1.0`, `executions_charged<=k`; (b) a proposer that never lands a passing patch exhausts at most `k` charged executions and returns `believed`/`abstain` with `executions_charged<=k` (the budget is never exceeded — pin `BudgetExhausted` is not raised past k); (c) a proposer whose FIRST candidate fails and SECOND (a refinement child) passes yields `verified_visible` with `executions_charged>=2` (proves refinement + feedback works). **Step 2: RED. Step 3: Implement. Step 4: GREEN.**
- [ ] **Step 5: Mutation check** — remove the `k`-budget guard (loop forever until solved) → test (b) FAILS (executions exceed k) — pin `executions_charged<=k` hard. Remove the "free symptom run is not charged" exemption → test (a)'s `executions_charged` count FAILS. Restore. **Step 6: Commit** — `feat(search): budgeted propose-execute-refine loop`

---

### Task 8: Value function v0 (constant scaffold)

**Files:**
- Create: `crucible/value/model.py`, `tests/value/__init__.py`, `tests/value/test_model.py`

**Interfaces:**
- Produces:
  ```python
  class Value(Protocol):
      def score(self, node) -> float: ...          # P(hidden pass | features), used to ORDER candidates
      def update(self, node, outcome: bool) -> None: ...
  class ConstantValue:
      def __init__(self, c: float = 0.5): ...
      def score(self, node) -> float                # returns c, always (v0 — untrained)
      def update(self, node, outcome) -> None        # no-op (training is S3); records the call count for a test
  ```

- [ ] **Step 1: Failing test** — `score` returns the constant regardless of node; `update` is a no-op but increments an observable counter (so S3 can prove it's wired). Unwrapped. **Step 2: RED. Step 3: Implement. Step 4: GREEN.**
- [ ] **Step 5: Mutation check** — make `score` return `node.something` instead of the constant → the constancy test FAILS. Restore. **Step 6: Commit** — `feat(value): v0 constant value scaffold`

---

### Task 9: Record schema + writers

**Files:**
- Create: `crucible/run/records.py`, `tests/run/test_records.py`

**Interfaces:**
- Produces:
  ```python
  @dataclass(frozen=True)
  class ExecRecord:               # one charged execution (a node run)
      task_key: str; arm: str; node_id: str; visible_reward: float; charged: bool
      wall_s: float; infra_error: str | None
      def to_dict/from_dict
  @dataclass(frozen=True)
  class TaskRecord:               # one task attempt by one arm — the E1/E2 unit
      task_key: str; arm: str; unit_id: str; family: str; phase: int; kind: str   # kind from TaskSpec
      landed: bool; status: str; confidence: float
      visible_reward: float; executions_charged: int
      hidden_pass: bool | None    # THE OUTCOME (driver-computed). None iff infra/not-measured.
      tampered: bool; infra_error: str | None
      tokens: int | None; wall_s: float; gpu_s: float | None
      def to_dict/from_dict
  def write_records(path: Path, task_recs: list[TaskRecord], exec_recs: list[ExecRecord]) -> None
      # task_records.jsonl + exec_records.jsonl under `path`, sort_keys, utf-8, one object per line.
  def read_task_records(path: Path) -> list[TaskRecord]
  ```

- [ ] **Step 1: Failing tests** — round-trip both dataclasses through JSON; `hidden_pass` may be None; `tampered`/`infra_error` present; `write_records`/`read_task_records` round-trip through a real `tmp_path`. Unwrapped. **Step 2: RED. Step 3: Implement. Step 4: GREEN.**
- [ ] **Step 5: Mutation check** — drop `hidden_pass` from `to_dict` → completeness test FAILS. Restore. **Step 6: Commit** — `feat(run): per-execution and per-task record schema`

---

### Task 10: Arm configs + the agent (task → submission)

**Files:**
- Create: `crucible/run/arm.py`, `tests/run/test_arm.py`

**Interfaces:**
- Consumes: `search.loop.search`, `proposer.Proposer`, `value.Value`, `sandbox.task_run.run_hidden`, `run.records.*`.
- Produces:
  ```python
  @dataclass(frozen=True)
  class ArmConfig:
      name: str                 # "A_noMem" | "B_search" | "B_naive"
      model: str                # served-model id to assert
      use_search: bool          # A_noMem/B_search = True ; B_naive = False (single-shot, 1 candidate, no refinement)
      k: int = 8; width: int = 4; seed: int = 0
  ARMS = {"A_noMem": ArmConfig("A_noMem","Qwen/Qwen3.5-2B",True),
          "B_search": ArmConfig("B_search","Qwen/Qwen3.5-9B",True),
          "B_naive":  ArmConfig("B_naive","Qwen/Qwen3.5-9B",False)}
  def attempt_task(cfg, unit, taskspec, proposer, value) -> tuple[TaskRecord, list[ExecRecord]]
      # runs search (or a single generate for B_naive), takes the final submission, computes the OUTCOME
      # via sandbox.task_run.run_hidden (driver-side; not charged; never seen by the agent). Detects
      # `tampered` (submission != a pure module — already guaranteed by the codec, but re-assert the test
      # files are untouched). Fills the TaskRecord (hidden_pass from run_hidden.all_passed).
  ```

- [ ] **Step 1: Failing test** — WRAP. Fake in-process proposer + a real `Unit`+`TaskSpec`. Assert: (a) A_noMem attempt on a unit whose repair the fake proposer knows → `hidden_pass True`, `status verified_visible`; (b) B_naive (single-shot) makes exactly ONE generate call and no refinement (`executions_charged<=1`); (c) a submission that passes VISIBLE but fails HIDDEN → `hidden_pass False` while `visible_reward==1.0` (the honest-measurement core: visible pass ≠ hidden pass). **Step 2: RED. Step 3: Implement. Step 4: GREEN.**
- [ ] **Step 5: Mutation check** — compute `hidden_pass` from the VISIBLE report instead of `run_hidden` → test (c) FAILS (fabricated success). This is the single most important mutation in S2 — pin it hard. Restore. **Step 6: Commit** — `feat(run): arm configs and per-task attempt with hidden-suite outcome`

---

### Task 11: Serving harness (model swap between arms)

**Files:**
- Create: `crucible/run/serving.py`, `tests/run/test_serving.py`, `scripts/serve_model.sh`

**Interfaces:**
- Produces:
  ```python
  @dataclass(frozen=True)
  class ServeSpec:
      served_name: str; hf_id: str; extra_args: list[str]; port: int = 8010
  SERVE = {"Qwen/Qwen3.5-2B": ServeSpec("Qwen/Qwen3.5-2B","Qwen/Qwen3.5-2B",
              ["--max-model-len","8192","--gpu-memory-utilization","0.6","--enable-lora","--max-lora-rank","32"]),
           "Qwen/Qwen3.5-9B": ServeSpec("Qwen/Qwen3.5-9B","lovedheart/Qwen3.5-9B-FP8",
              ["--max-model-len","4096","--gpu-memory-utilization","0.90","--enforce-eager"])}
  def wait_ready(base_url: str, pid: int, *, timeout_s: float) -> bool   # poll /v1/models until up or pid dies
  # scripts/serve_model.sh: launches `VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve <hf_id> --served-model-name <name> <args> --port <port>`
  ```
- Note: the actual start/stop of a live server is an OPERATIONAL step run by the ceiling-pilot runner (Task 14), NOT inside the unit tests. `test_serving.py` covers `wait_ready`'s polling logic against a **fake** server (readiness detection + pid-death detection), and asserts `SERVE` carries the S1-measured launch flags verbatim (incl. `VLLM_USE_FLASHINFER_SAMPLER=0` in the script). Unwrapped (fake server, no GPU).

- [ ] **Step 1: Failing test** → **Step 4: GREEN** as above. **Step 5: Mutation check** — remove `VLLM_USE_FLASHINFER_SAMPLER=0` from the script → a test asserting the env var is present FAILS (this is the S1 gotcha; must not regress). Restore. **Step 6: Commit** — `feat(run): model-serving harness with S1 launch flags`

---

### Task 12: Driver — run an arm over a task set

**Files:**
- Create: `crucible/run/driver.py`, `tests/run/test_driver.py`

**Interfaces:**
- Consumes: `arm.attempt_task`, `stream.store.read_manifest/read_unit/read_mutant`, `run.records.write_records`, `proposer`, `value`.
- Produces:
  ```python
  def run_arm(cfg: ArmConfig, stream_dir: Path, task_keys: list[str], proposer, value,
              out_dir: Path, *, log=print) -> Path
      # for each task_key: load unit+mutant+taskspec from the stream store, attempt_task, accumulate
      # records; write task_records.jsonl + exec_records.jsonl under out_dir/<arm>/ ; write a `.DONE`
      # marker with the arm+stream_hash+seed. Deterministic order (task_keys as given). Resumable: skip
      # task_keys already present in an existing partial task_records.jsonl.
  def select_pilot_tasks(stream_dir: Path, n: int, *, seed: int) -> list[str]
      # n phase-1 task_keys drawn with random.Random(f"{seed}:pilot") — never "first N".
  ```

- [ ] **Step 1: Failing test** — WRAP. Build a tiny real stream in `tmp_path` (reuse `stream.pipeline.build_stream` on the 3-record fixture from `tests/stream/test_pipeline.py`), run `run_arm` with a fake in-process proposer over 2 task_keys; assert both task records written, `.DONE` present, resumability (second call skips done tasks — assert the proposer isn't called again). `select_pilot_tasks` is seeded (two calls same seed → same list; different seed → different) and only returns phase-1 keys. **Step 2: RED. Step 3: Implement. Step 4: GREEN.**
- [ ] **Step 5: Mutation check** — make `select_pilot_tasks` return `phase1[:n]` (first-N) → the seed-sensitivity test FAILS; drop the resumability skip → the "proposer not called again" test FAILS. Restore. **Step 6: Commit** — `feat(run): arm driver with resumable records and seeded pilot sampling`

---

### Task 13: Lens — records → success by phase/exposure (E1/E2 inputs)

**Files:**
- Create: `crucible/run/lens.py`, `tests/run/test_lens.py`

**Interfaces:**
- Consumes: `run.records.TaskRecord`.
- Produces:
  ```python
  @dataclass(frozen=True)
  class ArmLens:
      arm: str; n: int
      succ_overall: float                        # mean hidden_pass over MEASURED tasks (infra excluded)
      succ_phase1: float; succ_second: float; succ_novel: float
      landing_rate: float; abstain_rate: float; infra_rate: float
      def to_dict/from_dict
  def build_lens(task_recs: list[TaskRecord]) -> ArmLens
      # success = mean(hidden_pass) over records where hidden_pass is not None (infra excluded — honest).
      # phase1 = kind=="first"; second = kind=="second"; novel = kind=="novel".
  ```

- [ ] **Step 1: Failing test** — a mixed list of `TaskRecord`s (some infra/None hidden_pass, some pass/fail across kinds) → assert `succ_overall` EXCLUDES the infra ones, and the per-kind rates match hand-computed values. Unwrapped. **Step 2: RED. Step 3: Implement. Step 4: GREEN.**
- [ ] **Step 5: Mutation check** — include `hidden_pass is None` as `False` in `succ_overall` (charging infra as a failure) → the infra-exclusion test FAILS. Restore. **Step 6: Commit** — `feat(run): lens computing success by phase/exposure`

---

### Task 14: Ceiling-pilot runner + CLI wiring

**Files:**
- Create: `crucible/run/pilot.py`, `tests/run/test_pilot.py`
- Modify: `crucible/cli.py` (add `crucible arm pilot` and `crucible arm run` subcommands)

**Interfaces:**
- Produces:
  ```python
  @dataclass(frozen=True)
  class PilotVerdict:
      p0: float; n: int; too_easy: bool; recommendation: str   # too_easy iff p0 > 0.70 (§4.8.4)
      def to_dict
  def ceiling_pilot(stream_dir: Path, out_dir: Path, proposer, value, *, n=30, seed=0, log=print) -> PilotVerdict
      # select_pilot_tasks(n) → run_arm(A_noMem) over them → build_lens → p0 = succ_overall.
      # too_easy = p0 > 0.70 ; recommendation names the first hardening-ladder rung if too_easy, else "proceed".
  # cli.py:
  #   crucible arm pilot <stream_dir> --base-url URL --model Qwen/Qwen3.5-2B [--n 30 --seed 0 --out DIR]
  #   crucible arm run <stream_dir> --arm A_noMem --base-url URL [--tasks all|phase1|<file>] [--out DIR]
  ```

- [ ] **Step 1: Failing test** — WRAP. With a fake in-process proposer scripted so a known fraction of the pilot tasks get a hidden-passing repair, assert `ceiling_pilot` computes `p0` = that fraction and sets `too_easy` correctly at the 0.70 boundary (test both a 0.5 case → proceed and a 0.8 case → too_easy with a rung recommendation). CLI: `python -m crucible.cli arm --help` exits 0 and lists `pilot`/`run` (subprocess test, no sandbox). **Step 2: RED. Step 3: Implement. Step 4: GREEN.**
- [ ] **Step 5: Mutation check** — change the threshold to `>= 0.70` vs `> 0.70` and pin the boundary at exactly 0.70 (spec says >0.70 → too easy, so 0.70 exactly = proceed) with a test; flip it → FAILS. Restore. **Step 6: Commit** — `feat(run): ceiling-pilot runner and arm CLI`

---

### Task 15: Codec-landing pre-check (§4.7) via assay

**Files:**
- Create: `crucible/run/landing_check.py`, `tests/run/test_landing_check.py`

**Interfaces:**
- Consumes: `proposer.Proposer`, `proposer.codec.landing_rate`, `stream.store`.
- Produces:
  ```python
  @dataclass(frozen=True)
  class LandingResult:
      model: str; n: int; rate: float; passes: bool   # passes iff rate >= 0.95 (§4.7)
      def to_dict
  def landing_precheck(stream_dir: Path, proposer, *, n=30, seed=0) -> LandingResult
      # draw n smoke tasks, build_prompt for each, proposer.generate(n=1), codec-landing each output,
      # rate = landing_rate(outputs); passes iff >= 0.95. On failure the operator falls back to the §2
      # baseline proposer (Qwen2.5-Coder-1.5B) — recorded, not automated here.
  ```

- [ ] **Step 1: Failing test** — fake proposer that lands a set fraction; assert `rate` and the 0.95 gate (0.96 → passes, 0.9 → fails). WRAP only if it touches the sandbox (it doesn't — landing is pure parse; unwrapped, fake proposer). **Step 2–4.** **Step 5: Mutation check** — flip the `>= 0.95` gate → boundary test FAILS. Restore. **Step 6: Commit** — `feat(run): codec-landing pre-check`

---

### Task 16: Live integration smoke + the actual ceiling pilot run (operational)

**Files:**
- Create: `docs/findings/S2-ceiling-pilot.md`
- Modify: `docs/CARRIED-DEBT.md`

This task is **operational, not unit-tested** — it runs the real vLLM server + real proposer against the real stream `streams/full/dd5912cddedc`, in capped scopes.

- [ ] **Step 1:** Serve `Qwen/Qwen3.5-2B` (S1 flags, `VLLM_USE_FLASHINFER_SAMPLER=0`, port 8010) in its own scope; `assert_identity`.
- [ ] **Step 2:** Run `landing_precheck` on 30 smoke tasks → record the landing rate; require ≥ 95% (else fall back to the 1.5B baseline and record it).
- [ ] **Step 3:** Run `crucible arm pilot streams/full/dd5912cddedc --base-url http://127.0.0.1:8010 --model Qwen/Qwen3.5-2B --n 30` under a cgroup cap (`systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0`), detached, monitored. This does 30 tasks × up to K=8 sandboxed executions each — expect tens of minutes.
- [ ] **Step 4:** Record in `docs/findings/S2-ceiling-pilot.md`: landing rate, **p0**, per-family success, whether `p0 > 0.70` (too-easy → name the hardening-ladder rung to apply, which loops back to the S1 builder), tokens/wall/GPU-minutes. No fabricated numbers.
- [ ] **Step 5:** Stop the server (by recorded PID), restart ollama, free VRAM.
- [ ] **Step 6: Commit** — `docs(run): S2 ceiling-pilot findings (p0=<measured>)`

---

## Self-Review notes (for the executor)

- **Spec coverage:** §4.4 codec = Task 2/3/10; §4.5 sandbox = Task 1 (reuses S1); §4.6 budget K=8 + free symptom = Task 7 (pinned by mutation); §4.7 landing = Task 3/15/16; §4.8.4 ceiling pilot + 0.70 gate = Task 14/16; §5 search/value contracts (memory/uncertainty EXCLUDED — S3) = Task 7/8; §6 E1/E2 inputs = Task 13 lens; §9 interfaces = each task's Produces block; §10 slice-2 scope (A_noMem + B arms, no A_full) = Task 10 ARMS. **Out of scope (S3), do not build:** `memory/`, `sleep/`, `uncertainty/`, the A_full arm, value training.
- **The one mutation that matters most:** Task 10 — `hidden_pass` MUST come from `run_hidden`, never the visible report. A regression there fabricates the entire experiment's primary endpoint with green unit tests.
- **R-T2-6 wrapping:** Tasks 1,7,10,12,14,16 touch the sandbox → wrap every pytest run. Tasks 2,3,4,5,6,8,9,11,13,15 are pure → unwrapped.
- **Serving gotchas carried from S1 (do not rediscover):** `VLLM_USE_FLASHINFER_SAMPLER=0`, port 8010, 9B needs `--enforce-eager --gpu-memory-utilization 0.90`, stop servers by recorded PID (never `pkill -f 'vllm serve'`).
