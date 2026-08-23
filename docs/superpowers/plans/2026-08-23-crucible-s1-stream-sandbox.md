# crucible S1 — Environment, Sandbox, Task Stream, Pre-checks — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the deterministic mutation-injected task stream, the isolated test sandbox with a verification-budget meter, the §4.8.1–3 structural pre-checks, and settle the serving/LoRA environment — everything Phase A's arms stand on.

**Architecture:** A `crucible/` Python package (uv-managed, Python 3.12). `sandbox/` executes code in a fresh temp dir with no network, memory limit, per-test and per-run wall-clock kills, and returns an honest `TestReport`. `stream/` turns EvalPlus records into units (module + visible/hidden pytest files), enumerates mutants with cosmic-ray (+ a ported statement-deletion operator), validates them in the sandbox (valid = killed by the *visible* suite), composes phase-1/phase-2/novel tasks into a content-hashed `StreamManifest`, and runs the pre-checks. A `cli.py` drives it. One exploratory task settles torch/vLLM-or-llama.cpp serving and whether LoRA attaches to Qwen3.5-2B.

**Tech Stack:** Python 3.12, uv, pytest + pytest-timeout, cosmic-ray 8.7 (MIT) + parso, numpy (MBPP+ tests), stdlib only otherwise. Environment task: torch (cu128+), vLLM or llama.cpp, transformers + peft.

**Spec:** `docs/superpowers/specs/2026-08-23-crucible-phase-a-prereg.md` — read §4 (task stream), §4.5 (sandbox), §4.6 (budget), §4.8 (pre-checks), §8 (instrument honesty), §9 (architecture), §10 S1 exit criteria.

## Global Constraints

- Python `>=3.12,<3.13` (torch/vLLM wheel coverage; sensorium needs 3.12+). Venv via `uv`. Never install into the system Python.
- Licensing: only Apache-2.0 / MIT / BSD artifacts enter the tree; every adopted or ported artifact gets a row in `THIRD_PARTY.md` with the verification command. No copyleft, ever.
- Every sandbox execution: fresh temp dir, `PYTHONDONTWRITEBYTECODE=1`, `__pycache__` purged, outbound sockets blocked, `RLIMIT_AS` = 4 GiB (see amendment A1), per-test timeout **5 s** (`--timeout-method=signal`), per-execution wall cap **60 s**, subprocess in its own session (killable as a group).
- `TestReport.infra_error` set ⇒ not a measurement: never charged to the budget, never scored; counted.
- **None-vs-zero:** unmeasured fields are `None`; a `dropped` list names every excluded thing with its reason. No field may default to something that looks like a measurement.
- Keys are content hashes (sha256), never descriptions. Task key = mutant key.
- Mutation-test every load-bearing test (steps say exactly which line to break). Every mutation-check run: `find . -name __pycache__ -prune -exec rm -rf {} +` first and `PYTHONDONTWRITEBYTECODE=1` set (same-length edits restored within one second otherwise run the mutant from stale bytecode).
- Files ≤ 400 lines typical, 800 max. Functions < 50 lines. Organise by feature (`sandbox/`, `stream/`, `proposer/`).
- Commit after every green step. Commit messages: `<type>: <description>` (feat/fix/test/docs/chore).

## Pre-lock spec amendments recorded by this plan (footnote in the spec at lock)

- **A1** §4.5: `RLIMIT_AS` is **4 GiB**, not 1 GiB — numpy/OpenBLAS reserve several GiB of *virtual* address space at import and a 1 GiB cap breaks MBPP+ tests that import numpy. It remains a runaway guard, not a tight bound.
- **A2** §4.2: the **VAR** family (`VariableReplacer`, `VariableInserter`) is **excluded** — those two are the only cosmic-ray operators that require per-variable constructor arguments (`cause_variable`, `effect_variable`); 8 families remain. The family map asserts every one of the 213 core operator names is either mapped or explicitly excluded.
- **A3** §4.7: the codec-landing probe moves to S2 (it needs the proposer adapter and prompt format that S2 builds); it still runs before lock.
- **A4** §4.1: hidden tests are capped at **100 per unit** (plus_input has up to 1000 inputs; median 972 for HumanEval+), sampled deterministically with the stream seed; inputs whose `repr` exceeds 2000 chars, whose value does not round-trip through `eval(repr(x)) == x`, or on which the canonical solution raises or times out are **dropped and counted** per unit.

## Facts verified on 2026-08-23 (do not re-derive; cite)

- cosmic-ray API (verified by running): `from cosmic_ray.mutating import mutate_code` — `mutate_code(code: str, operator_instance, occurrence: int) -> str | None`; `from cosmic_ray import plugins` — `plugins.operator_names()` returns 213 names prefixed `core/`; `plugins.get_operator(name)` returns a **class** (instantiate with `()`; only `VariableReplacer`/`VariableInserter` need args); `from cosmic_ray.ast import get_ast, ast_nodes` — `ast_nodes(get_ast(src))` yields parso nodes in the **same pre-order** `MutationVisitor` walks, so occurrence *i* of operator `op` is `[p for node in ast_nodes(tree) for p in op.mutation_positions(node)][i]`. `Operator` ABC (`cosmic_ray.operators.operator.Operator`) requires `mutation_positions(self, node)`, `mutate(self, node, index)`, and classmethod `examples(cls)`.
- Operator name groups (213): `ReplaceBinaryOperator_*` 132, `ReplaceComparisonOperator_*` 56, `ReplaceUnaryOperator_*` 12, and singletons `AddNot ExceptionReplacer NumberReplacer RemoveDecorator ReplaceAndWithOr ReplaceOrWithAnd ReplaceTrueWithFalse ReplaceFalseWithTrue ReplaceBreakWithContinue ReplaceContinueWithBreak ZeroIterationForLoop VariableInserter VariableReplacer`.
- parso shapes: a function body is a `suite` node whose children are `newline` then `simple_stmt` nodes; a `simple_stmt`'s first leaf `.prefix` is its indentation (e.g. `'    '`); `parso.parse("pass\n").children[0]` is a `simple_stmt` with children `[keyword, newline]`.
- EvalPlus data (URLs return 200; digests verified):
  - `https://github.com/evalplus/humanevalplus_release/releases/download/v0.1.10/HumanEvalPlus.jsonl.gz` sha256 `272720b90ac375502c8ed23cd791c2a93dfb22a911641a494da74a426c09f101` — 164 records, keys `atol base_input canonical_solution contract entry_point plus_input prompt task_id test`; `prompt` = imports + signature + docstring, `canonical_solution` = indented body ⇒ full source = `prompt + canonical_solution`.
  - `https://github.com/evalplus/mbppplus_release/releases/download/v0.2.0/MbppPlus.jsonl.gz` sha256 `af43697e8791c4c149bdfd6b489d8b5412507551ac20e28a439f650b8225db63` — 378 records, keys `assertion atol base_input canonical_solution contract entry_point plus_input prompt task_id`; `prompt` = a docstring string, `canonical_solution` = full `def` ⇒ full source = `prompt + canonical_solution` (the docstring is then stripped).
  - `base_input` / `plus_input` are lists of argument lists (call as `fn(*args)`); `atol` is 0 or a float tolerance.
- Box: 16 cores, RTX 5080 (sm_120), driver 595.84, **no torch anywhere**, `uv` at `~/.local/bin/uv`, `~/llama.cpp` checkout at `4988f6e` (2026-06-13, has `qwen35`), Ollama 0.32.13 installed (must be stopped before serving: it holds VRAM).

## File structure

```
crucible/
├── pyproject.toml
├── README.md                       (exists)
├── THIRD_PARTY.md                  (new — license ledger)
├── docs/CARRIED-DEBT.md            (new)
├── docs/findings/S1-serving.md     (new — Task 16 findings)
├── crucible/
│   ├── __init__.py
│   ├── cli.py                      — `crucible stream build|precheck|smoke`
│   ├── sandbox/
│   │   ├── __init__.py
│   │   ├── exec.py                 — isolated subprocess primitive (tmpdir, sitecustomize, rlimit, kill group)
│   │   ├── report.py               — TestReport + junit parsing
│   │   ├── runner.py               — run_tests(): pytest in the sandbox → TestReport
│   │   └── budget.py               — BudgetMeter / BudgetExhausted
│   ├── stream/
│   │   ├── __init__.py
│   │   ├── evalplus.py             — fetch/verify/load the two JSONL.gz; full_source()
│   │   ├── units.py                — Unit dataclass, strip_docstrings, module_name_for
│   │   ├── oracle.py               — expected outputs from the canonical solution (in sandbox)
│   │   ├── testgen.py              — visible/hidden pytest files from (inputs, expected)
│   │   ├── build.py                — build_unit(): assemble + self-check + drop reasons
│   │   ├── families.py             — operator name → family; EXCLUDED; completeness assert
│   │   ├── sdl.py                  — StatementDeletion operator (cosmic-ray Operator ABC)
│   │   ├── mutants.py              — enumerate/apply/make mutants; MutantSpec/Mutant
│   │   ├── validate.py             — sandbox-kill validation; Validation
│   │   ├── compose.py              — classes/phases/novel → StreamManifest; hashing
│   │   ├── store.py                — write/read a stream directory
│   │   └── precheck.py             — §4.8.1–3 checks → PrecheckReport
│   └── proposer/
│       ├── __init__.py
│       └── identity.py             — served-model identity assertion (vLLM + llama.cpp)
└── tests/
    ├── conftest.py
    ├── sandbox/test_exec.py test_report.py test_runner.py test_budget.py
    ├── stream/test_evalplus.py test_units.py test_oracle.py test_testgen.py test_build.py
    │          test_families.py test_sdl.py test_mutants.py test_validate.py test_compose.py
    │          test_store.py test_precheck.py
    ├── proposer/test_identity.py
    └── fixtures/mini_humaneval.jsonl.gz  mini_mbpp.jsonl.gz
```

---

### Task 1: Project scaffold, venv, license ledger

**Files:**
- Create: `pyproject.toml`, `crucible/__init__.py`, `crucible/sandbox/__init__.py`, `crucible/stream/__init__.py`, `crucible/proposer/__init__.py`, `tests/conftest.py`, `tests/test_smoke.py`, `THIRD_PARTY.md`, `docs/CARRIED-DEBT.md`
- Modify: `.gitignore` (add `streams/`, `.cache/`, `*.xml`)

**Interfaces:**
- Produces: an importable `crucible` package; `uv run pytest` works; `THIRD_PARTY.md` table shape used by every later task.

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "crucible"
version = "0.0.1"
description = "Research spike: small frozen proposer + structured memory + verify-by-execution search"
requires-python = ">=3.12,<3.13"
dependencies = [
  "cosmic-ray>=8.7",
  "parso>=0.8",
  "pytest>=8.0",
  "pytest-timeout>=2.3",
  "numpy>=1.26",
]

[project.scripts]
crucible = "crucible.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["crucible"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-p no:cacheprovider -q"
```

- [ ] **Step 2: Create the venv and install**

Run:
```bash
cd ~/workspace/crucible && uv venv -p 3.12 .venv && uv pip install -e . --python .venv/bin/python && .venv/bin/python -c "import cosmic_ray, parso, pytest, numpy; print('ok')"
```
Expected: `ok`.

- [ ] **Step 3: Write the smoke test and conftest**

`tests/conftest.py`:
```python
import os

# The pyc rule: never let stale bytecode survive a mutation-check run.
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
```

`tests/test_smoke.py`:
```python
def test_package_imports():
    import crucible  # noqa: F401
```

`crucible/__init__.py`:
```python
"""crucible — research spike on learning-in-the-loop AI (see docs/superpowers/specs)."""
__version__ = "0.0.1"
```
Empty `__init__.py` for `sandbox/`, `stream/`, `proposer/`. **Also** create empty `tests/__init__.py`, `tests/sandbox/__init__.py`, `tests/stream/__init__.py`, `tests/proposer/__init__.py` — Task 15's tests import a helper from `tests.stream.test_compose`, which needs the tests tree to be a package.

- [ ] **Step 4: Run the tests**

Run: `cd ~/workspace/crucible && .venv/bin/python -m pytest`
Expected: `1 passed`.

- [ ] **Step 5: Write the license ledger and debt file**

`THIRD_PARTY.md`:
```markdown
# Third-party artifacts in crucible

Every adopted, vendored, or ported artifact. License is what the command returned on the date shown, not what anyone remembered.

| Artifact | Source | License | Verified by | Date | What we take |
|---|---|---|---|---|---|
| cosmic-ray | github.com/sixty-north/cosmic-ray | MIT | `gh api repos/sixty-north/cosmic-ray --jq .license.spdx_id` | 2026-08-23 | library dependency: `mutate_code`, operators, parso AST helpers |
| parso | pypi.org/project/parso | MIT | PyPI `.info.license` | 2026-08-23 | library dependency (cosmic-ray's parser) |
| pytest-timeout | pypi.org/project/pytest-timeout | MIT | PyPI classifier | 2026-08-23 | per-test wall-clock kill in the sandbox |
| EvalPlus HumanEval+ v0.1.10 | github.com/evalplus/humanevalplus_release | Apache-2.0 | HF API `cardData.license` on `evalplus/humanevalplus` | 2026-08-23 | seed units (data) |
| EvalPlus MBPP+ v0.2.0 | github.com/evalplus/mbppplus_release | Apache-2.0 | HF API `cardData.license` on `evalplus/mbppplus` | 2026-08-23 | seed units (data) |
| MutPy `StatementDeletion` (idea) | github.com/mutpy/mutpy | Apache-2.0 | LICENSE file fetched (API said NOASSERTION) | 2026-08-23 | operator *design* reimplemented in `crucible/stream/sdl.py` on cosmic-ray's ABC; no code copied |
| mini-swe-agent `LocalEnvironment` (pattern) | github.com/SWE-agent/mini-swe-agent | MIT | `gh api` | 2026-08-23 | subprocess-isolation pattern in `crucible/sandbox/exec.py`; no code copied |
```

`docs/CARRIED-DEBT.md`:
```markdown
# CARRIED-DEBT

Appended at every slice merge: what the slice settled → deferred, with rulings → process lessons. Resolved items are struck through, never deleted.

## S1 (in progress)
### Settled
- (fill at merge)
### Deferred, with rulings
- (fill at merge)
### Process lessons
- (fill at merge)
```

- [ ] **Step 6: Commit**

```bash
cd ~/workspace/crucible && printf 'streams/\n.cache/\n*.xml\n' >> .gitignore && git add -A && git commit -m "chore: scaffold crucible package, venv, license ledger, debt file"
```

---

### Task 2: Isolated execution primitive (`sandbox/exec.py`)

**Files:**
- Create: `crucible/sandbox/exec.py`, `tests/sandbox/test_exec.py`

**Interfaces:**
- Produces:
  ```python
  @dataclass(frozen=True)
  class ExecResult:
      returncode: int | None   # None when the wall cap killed it
      stdout: str
      stderr: str
      wall_s: float
      timed_out: bool
      workdir: str             # the temp dir (kept only if keep=True)
  def execute(argv: list[str], files: dict[str, str], *, wall_cap_s: float = 60.0,
              mem_limit_bytes: int = 4 << 30, env_extra: dict[str, str] | None = None,
              keep: bool = False) -> ExecResult
  SITECUSTOMIZE_SRC: str   # the socket-blocking shim written into every workdir
  ```
- Invariants: fresh temp dir per call; `files` written before exec; `PYTHONDONTWRITEBYTECODE=1`; `PYTHONPATH` = workdir (so `sitecustomize.py` loads); `PYTHONHASHSEED=0`; `PYTEST_ADDOPTS` never inherited; process group killed on wall cap; `RLIMIT_AS` applied in the child.

- [ ] **Step 1: Write the failing tests**

`tests/sandbox/test_exec.py`:
```python
import sys, time
from crucible.sandbox.exec import execute, ExecResult

PY = sys.executable

def test_runs_a_script_and_captures_stdout():
    r = execute([PY, "main.py"], {"main.py": "print('hi')"})
    assert isinstance(r, ExecResult)
    assert r.returncode == 0 and r.stdout.strip() == "hi" and not r.timed_out

def test_wall_cap_kills_infinite_loop():
    t0 = time.monotonic()
    r = execute([PY, "-c", "while True: pass"], {}, wall_cap_s=1.0)
    assert r.timed_out and r.returncode is None
    assert time.monotonic() - t0 < 5.0

def test_network_is_blocked():
    src = "import socket\ntry:\n    socket.create_connection(('93.184.216.34', 80), timeout=2)\n    print('OPEN')\nexcept OSError as e:\n    print('BLOCKED', e)\n"
    r = execute([PY, "net.py"], {"net.py": src})
    assert "BLOCKED" in r.stdout and "OPEN" not in r.stdout

def test_no_bytecode_written_and_addopts_not_inherited(monkeypatch):
    monkeypatch.setenv("PYTEST_ADDOPTS", "--this-would-break")
    src = "import os,sys\nprint(os.environ.get('PYTHONDONTWRITEBYTECODE'), 'PYTEST_ADDOPTS' in os.environ, sys.flags.dont_write_bytecode)"
    r = execute([PY, "e.py"], {"e.py": src})
    assert r.stdout.split() == ["1", "False", "1"]

def test_memory_limit_is_applied():
    src = "import resource\nprint(resource.getrlimit(resource.RLIMIT_AS)[0])"
    r = execute([PY, "m.py"], {"m.py": src}, mem_limit_bytes=4 << 30)
    assert int(r.stdout.strip()) == 4 << 30

def test_workdir_is_removed_unless_keep():
    import os
    r = execute([PY, "-c", "pass"], {})
    assert not os.path.exists(r.workdir)
    r2 = execute([PY, "-c", "pass"], {}, keep=True)
    assert os.path.exists(r2.workdir)
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/sandbox/test_exec.py -q`
Expected: `ImportError` / collection error (module missing).

- [ ] **Step 3: Implement `crucible/sandbox/exec.py`**

```python
"""Isolated subprocess execution: the one place crucible runs untrusted code.

Pattern borrowed from mini-swe-agent's LocalEnvironment (MIT): subprocess + timeout, but with
a fresh workdir, socket blocking, an address-space limit, and process-group kill.
"""
from __future__ import annotations

import os
import resource
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass

SITECUSTOMIZE_SRC = '''\
# crucible sandbox: block outbound network without breaking socket class hierarchy.
import socket as _s
def _blocked(*a, **k):
    raise OSError("network disabled in crucible sandbox")
_s.socket.connect = _blocked
_s.socket.connect_ex = _blocked
_s.create_connection = _blocked
_s.getaddrinfo = _blocked
'''

_CLEAN_ENV_KEYS = ("PATH", "LANG", "LC_ALL", "TERM")


@dataclass(frozen=True)
class ExecResult:
    returncode: int | None
    stdout: str
    stderr: str
    wall_s: float
    timed_out: bool
    workdir: str


def _child_limits(mem_limit_bytes: int):
    def _apply() -> None:
        resource.setrlimit(resource.RLIMIT_AS, (mem_limit_bytes, mem_limit_bytes))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    return _apply


def _env(workdir: str, extra: dict[str, str] | None) -> dict[str, str]:
    env = {k: os.environ[k] for k in _CLEAN_ENV_KEYS if k in os.environ}
    env.update({
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONPATH": workdir,
        "HOME": workdir,
        "TMPDIR": workdir,
    })
    if extra:
        env.update(extra)
    env.pop("PYTEST_ADDOPTS", None)
    return env


def execute(argv: list[str], files: dict[str, str], *, wall_cap_s: float = 60.0,
            mem_limit_bytes: int = 4 << 30, env_extra: dict[str, str] | None = None,
            keep: bool = False) -> ExecResult:
    workdir = tempfile.mkdtemp(prefix="crucible-")
    try:
        for name, src in {**files, "sitecustomize.py": SITECUSTOMIZE_SRC}.items():
            path = os.path.join(workdir, name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(src)
        t0 = time.monotonic()
        proc = subprocess.Popen(
            argv, cwd=workdir, env=_env(workdir, env_extra),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            start_new_session=True, preexec_fn=_child_limits(mem_limit_bytes),
        )
        try:
            out, err = proc.communicate(timeout=wall_cap_s)
            timed_out, rc = False, proc.returncode
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            out, err = proc.communicate()
            timed_out, rc = True, None
        return ExecResult(rc, out, err, time.monotonic() - t0, timed_out, workdir)
    finally:
        if not keep:
            shutil.rmtree(workdir, ignore_errors=True)
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/sandbox/test_exec.py -q`
Expected: `6 passed`. (If `test_network_is_blocked` fails because the host has no route and raises before our shim, that is still `BLOCKED` — the assert is on the string; if it prints `OPEN`, the shim is not loading: check `PYTHONPATH` is the workdir.)

- [ ] **Step 5: Mutation check (load-bearing: wall cap and env hygiene)**

Break: in `_env`, delete the line `env.pop("PYTEST_ADDOPTS", None)` and change the `env` seed to `dict(os.environ)`. Run: `find . -name __pycache__ -prune -exec rm -rf {} +; PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/sandbox/test_exec.py -q`. Expected: `test_no_bytecode_written_and_addopts_not_inherited` FAILS. Restore. Break: replace `os.killpg(proc.pid, signal.SIGKILL)` with `pass`. Expected: `test_wall_cap_kills_infinite_loop` FAILS (hangs past 5 s or returncode not None). Restore; purge pycache; rerun: `6 passed`.

- [ ] **Step 6: Commit**

```bash
git add crucible/sandbox/exec.py tests/sandbox/test_exec.py && git commit -m "feat(sandbox): isolated execution primitive with network block, rlimit, wall cap"
```

> **Amended after Task 2 review (rulings R-T2-1..5 in the SDD ledger):** the shipped `exec.py` differs from the Step 3 code above — stdout/stderr are captured to files in the workdir (not pipes) and read back capped at 1 MiB with `errors="replace"`; the child is awaited with `proc.wait(timeout)` and `killpg`'d (a `setsid` grandchild can no longer stall the cap); resource limits are applied through a `prlimit` wrapper (`--as`, `--fsize=16MiB`, `--core=0`) with `preexec_fn` only as a fallback (thread-safety); `env_extra` cannot override sandbox keys. The Python-level socket shim is accepted for S1 with its threat model documented (accidental network use, not adversarial); OS-level isolation is deferred in CARRIED-DEBT.

---

### Task 3: `TestReport` + junit parsing + pytest runner (`sandbox/report.py`, `sandbox/runner.py`)

**Files:**
- Create: `crucible/sandbox/report.py`, `crucible/sandbox/runner.py`, `tests/sandbox/test_report.py`, `tests/sandbox/test_runner.py`

**Interfaces:**
- Consumes: `execute()` from Task 2.
- Produces:
  ```python
  @dataclass(frozen=True)
  class TestReport:
      passed: tuple[str, ...]; failed: tuple[str, ...]; timed_out: tuple[str, ...]; errored: tuple[str, ...]
      wall_s: float; infra_error: str | None
      @property
      def all_passed(self) -> bool       # infra_error is None and passed non-empty and no failed/timed_out/errored
      @property
      def killed(self) -> bool           # infra_error is None and (failed or timed_out or errored)
      def to_dict(self) -> dict; @classmethod from_dict(d)
  def parse_junit(xml_text: str) -> tuple[tuple[str,...], tuple[str,...], tuple[str,...], tuple[str,...]]  # passed, failed, timed_out, errored (test *names*)
  TEST_FILE = "test_unit.py"
  def run_tests(module_name: str, module_src: str, test_src: str, *, subset: list[str] | None = None,
                per_test_timeout_s: float = 5.0, wall_cap_s: float = 60.0, mem_limit_bytes: int = 4 << 30,
                python: str = sys.executable) -> TestReport
  ```
- Invariants: test ids are bare function names (`test_v0`, `test_h3`); `subset` selects by name; a suite hang past the wall cap ⇒ `timed_out=("__suite__",)` (a failure, not infra); a collection error (module does not import) ⇒ `errored=("__collection__",)` (a failure, not infra); missing junit with no timeout, or pytest exit 5 (no tests collected), or exit 3/4 ⇒ `infra_error`.

- [ ] **Step 1: Write the failing tests for the report/parser**

`tests/sandbox/test_report.py`:
```python
from crucible.sandbox.report import TestReport, parse_junit

JUNIT = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest" tests="4">
<testcase classname="test_unit" name="test_v0" time="0.001"/>
<testcase classname="test_unit" name="test_v1" time="0.001"><failure message="assert 1 == 2">x</failure></testcase>
<testcase classname="test_unit" name="test_v2" time="5.0"><failure message="Failed: Timeout &gt;5.0s">x</failure></testcase>
<testcase classname="test_unit" name="test_v3" time="0.0"><error message="boom">x</error></testcase>
</testsuite></testsuites>"""

def test_parse_junit_buckets_by_outcome():
    p, f, t, e = parse_junit(JUNIT)
    assert p == ("test_v0",) and f == ("test_v1",) and t == ("test_v2",) and e == ("test_v3",)

def test_report_flags():
    ok = TestReport(("a",), (), (), (), 0.1, None)
    assert ok.all_passed and not ok.killed
    k = TestReport((), ("a",), (), (), 0.1, None)
    assert k.killed and not k.all_passed
    infra = TestReport((), (), (), (), 0.1, "server down")
    assert not infra.killed and not infra.all_passed
    empty = TestReport((), (), (), (), 0.1, None)
    assert not empty.all_passed, "zero tests passed must not count as all_passed"

def test_report_round_trip():
    r = TestReport(("a", "b"), ("c",), ("__suite__",), (), 1.5, None)
    assert TestReport.from_dict(r.to_dict()) == r
    assert set(r.to_dict()) == {"passed", "failed", "timed_out", "errored", "wall_s", "infra_error"}
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/sandbox/test_report.py -q` → collection error (missing module).

- [ ] **Step 3: Implement `crucible/sandbox/report.py`**

```python
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TestReport:
    passed: tuple[str, ...]
    failed: tuple[str, ...]
    timed_out: tuple[str, ...]
    errored: tuple[str, ...]
    wall_s: float
    infra_error: str | None

    @property
    def all_passed(self) -> bool:
        return (self.infra_error is None and bool(self.passed)
                and not self.failed and not self.timed_out and not self.errored)

    @property
    def killed(self) -> bool:
        return self.infra_error is None and bool(self.failed or self.timed_out or self.errored)

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("passed", "failed", "timed_out", "errored"):
            d[k] = list(d[k])
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TestReport":
        return cls(tuple(d["passed"]), tuple(d["failed"]), tuple(d["timed_out"]),
                   tuple(d["errored"]), float(d["wall_s"]), d["infra_error"])


def _is_timeout(msg: str) -> bool:
    return "Timeout" in msg and (">" in msg or "timeout" in msg.lower())


def parse_junit(xml_text: str):
    root = ET.fromstring(xml_text)
    passed, failed, timed_out, errored = [], [], [], []
    for tc in root.iter("testcase"):
        name = tc.get("name", "")
        fail = tc.find("failure")
        err = tc.find("error")
        if err is not None:
            errored.append(name)
        elif fail is not None:
            (timed_out if _is_timeout(fail.get("message", "")) else failed).append(name)
        elif tc.find("skipped") is not None:
            continue
        else:
            passed.append(name)
    return tuple(passed), tuple(failed), tuple(timed_out), tuple(errored)
```

- [ ] **Step 4: Run: `.venv/bin/python -m pytest tests/sandbox/test_report.py -q` → `3 passed`. Commit.**

```bash
git add crucible/sandbox/report.py tests/sandbox/test_report.py && git commit -m "feat(sandbox): TestReport and junit parsing"
```

- [ ] **Step 5: Write the failing runner tests**

`tests/sandbox/test_runner.py`:
```python
from crucible.sandbox.runner import run_tests

MOD = "def add(a, b):\n    return a + b\n"
TESTS = ("import pytest\nfrom unit_x import add as candidate\n"
         "def test_v0():\n    assert candidate(1, 2) == 3\n"
         "def test_v1():\n    assert candidate(2, 2) == 4\n")

def test_passing_module():
    r = run_tests("unit_x", MOD, TESTS)
    assert r.infra_error is None and set(r.passed) == {"test_v0", "test_v1"} and r.all_passed

def test_mutant_is_killed():
    r = run_tests("unit_x", MOD.replace("a + b", "a - b"), TESTS)
    assert r.killed and set(r.failed) == {"test_v0", "test_v1"} and not r.passed

def test_subset_runs_only_named_tests():
    r = run_tests("unit_x", MOD, TESTS, subset=["test_v1"])
    assert r.passed == ("test_v1",) and not r.failed

def test_hang_is_timed_out_not_infra():
    r = run_tests("unit_x", "def add(a, b):\n    while True: pass\n", TESTS, per_test_timeout_s=1.0, wall_cap_s=20.0)
    assert r.infra_error is None and set(r.timed_out) == {"test_v0", "test_v1"} and r.killed

def test_syntax_error_module_is_collection_error_not_infra():
    r = run_tests("unit_x", "def add(a, b)\n    return a + b\n", TESTS)
    assert r.infra_error is None and r.errored == ("__collection__",) and r.killed

def test_broken_test_file_is_infra():
    r = run_tests("unit_x", MOD, "this is not python\n")
    assert r.infra_error is not None and not r.killed

def test_no_tests_collected_is_infra():
    r = run_tests("unit_x", MOD, "from unit_x import add\n")
    assert r.infra_error is not None and not r.killed
```

- [ ] **Step 6: Run → collection error. Implement `crucible/sandbox/runner.py`**

```python
from __future__ import annotations

import os
import sys

from .exec import execute
from .report import TestReport, parse_junit

TEST_FILE = "test_unit.py"
_JUNIT = "junit.xml"


def run_tests(module_name: str, module_src: str, test_src: str, *, subset: list[str] | None = None,
              per_test_timeout_s: float = 5.0, wall_cap_s: float = 60.0, mem_limit_bytes: int = 4 << 30,
              python: str = sys.executable) -> TestReport:
    targets = [f"{TEST_FILE}::{name}" for name in subset] if subset else [TEST_FILE]
    argv = [python, "-m", "pytest", *targets, "-q", "-p", "no:cacheprovider", "--tb=line",
            "-W", "ignore", f"--timeout={per_test_timeout_s}", "--timeout-method=signal",
            f"--junitxml={_JUNIT}", "-o", "junit_logging=no"]
    files = {f"{module_name}.py": module_src, TEST_FILE: test_src}
    res = execute(argv, files, wall_cap_s=wall_cap_s, mem_limit_bytes=mem_limit_bytes, keep=True)
    try:
        junit_path = os.path.join(res.workdir, _JUNIT)
        xml = open(junit_path, encoding="utf-8").read() if os.path.exists(junit_path) else None
    finally:
        import shutil
        shutil.rmtree(res.workdir, ignore_errors=True)
    return _classify(res.returncode, res.timed_out, xml, res.stderr, res.wall_s)


def _classify(rc: int | None, timed_out: bool, xml: str | None, stderr: str, wall_s: float) -> TestReport:
    if timed_out:
        return TestReport((), (), ("__suite__",), (), wall_s, None)
    if xml is None:
        return TestReport((), (), (), (), wall_s, f"no junit written (rc={rc}): {stderr[-400:]}")
    passed, failed, t_out, errored = parse_junit(xml)
    if rc == 2 and not passed and not failed and not t_out:
        # pytest exit 2 = interrupted (collection errors): the module under test failed to import.
        return TestReport((), (), (), ("__collection__",), wall_s, None)
    if rc in (3, 4, 5) or (rc not in (0, 1, 2)):
        return TestReport((), (), (), (), wall_s, f"pytest rc={rc}: {stderr[-400:]}")
    if not (passed or failed or t_out or errored):
        return TestReport((), (), (), (), wall_s, "no tests collected")
    return TestReport(passed, failed, t_out, errored, wall_s, None)
```

Note the subtlety the tests pin: a **broken test file** also collects with errors (rc=2) — but then the junit `<error>` is on the test module, not the unit. Distinguish by checking the error's classname: if `parse_junit` reports errored entries whose name equals the test module (`test_unit`), that is an infra error. Update `parse_junit` consumers: in `_classify`, before the rc==2 branch add:
```python
    if rc == 2 and errored and all(n in ("test_unit", "test_unit.py") for n in errored) and "unit_" not in (stderr or ""):
        pass  # fallthrough; see below
```
Rather than string-sniffing, use a deterministic rule: **run the test file's own import first.** In `run_tests`, before pytest, execute `python -c "import ast,sys; ast.parse(open('test_unit.py').read())"` via `execute` (cheap) — if it fails, return `infra_error="test file does not parse"`. Then an rc=2 with a collection error can only come from the unit module. Add that pre-step:

```python
    probe = execute([python, "-c", "import ast; ast.parse(open('test_unit.py', encoding='utf-8').read())"],
                    {TEST_FILE: test_src}, wall_cap_s=10.0)
    if probe.returncode != 0:
        return TestReport((), (), (), (), probe.wall_s, f"test file does not parse: {probe.stderr[-300:]}")
```
(and `rc == 5` → `no tests collected` is already an infra branch.)

- [ ] **Step 7: Run: `.venv/bin/python -m pytest tests/sandbox/test_runner.py -q` → `7 passed`.** If `test_hang_is_timed_out_not_infra` reports the names as `failed` rather than `timed_out`, print the junit message text and widen `_is_timeout` to match it exactly (pytest-timeout's message is `Failed: Timeout >1.0s`).

- [ ] **Step 8: Mutation check.** Break: in `_classify`, change the `if timed_out:` branch to return `infra_error="hang"`. Purge pycache; run; expected `test_hang_is_timed_out_not_infra` FAILS. Restore. Break: in `TestReport.all_passed` drop `and bool(self.passed)`. Expected `test_report_flags` FAILS. Restore; purge; rerun both files green.

- [ ] **Step 9: Commit**

```bash
git add crucible/sandbox/runner.py tests/sandbox/test_runner.py crucible/sandbox/report.py && git commit -m "feat(sandbox): pytest runner with honest hang/collection/infra classification"
```

> **Amended after Task 3 implementation (rulings R-T3-1..2 in the SDD ledger):** the shipped `runner.py` probe is `import test_unit` run in the sandbox with the unit replaced by a PEP 562 `__getattr__` stub (infra message `"test file does not load: …"`), NOT `ast.parse` — the plan's own test case `this is not python` parses as a `Name is not Name` comparison and only fails at import, so the `ast.parse` probe let it through and charged the unit with `__collection__`. Any test file that fails to LOAD standalone is infra; an rc=2 collection error after a passing probe can then only be the unit's fault. `test_runner.py` has 8 tests, not 7: the 8th pins the wall-cap `__suite__` branch, which the 7 left mutation-survivable (the hang test exercises pytest-timeout, not `execute()`'s cap). `_classify`'s rc condition is written as `rc not in (0, 1, 2)` (identical to the Step 6 text).

---

### Task 4: Verification budget meter (`sandbox/budget.py`)

**Files:**
- Create: `crucible/sandbox/budget.py`, `tests/sandbox/test_budget.py`

**Interfaces:**
- Produces:
  ```python
  class BudgetExhausted(RuntimeError): ...
  class BudgetMeter:
      def __init__(self, k: int = 8): ...
      k: int; charged: int; infra: int
      def check(self) -> None            # raises BudgetExhausted if charged >= k
      def charge(self, report: TestReport) -> None   # charged += 1 unless report.infra_error; infra += 1 otherwise
      def remaining(self) -> int
      def to_dict(self) -> dict          # {"k", "charged", "infra"}
  ```

- [ ] **Step 1: Failing tests**

`tests/sandbox/test_budget.py`:
```python
import pytest
from crucible.sandbox.budget import BudgetMeter, BudgetExhausted
from crucible.sandbox.report import TestReport

OK = TestReport(("t",), (), (), (), 0.1, None)
INFRA = TestReport((), (), (), (), 0.1, "server down")

def test_ninth_execution_is_refused():
    m = BudgetMeter(k=8)
    for _ in range(8):
        m.check(); m.charge(OK)
    assert m.remaining() == 0
    with pytest.raises(BudgetExhausted):
        m.check()

def test_infra_is_counted_but_not_charged():
    m = BudgetMeter(k=2)
    m.charge(INFRA); m.charge(INFRA); m.charge(INFRA)
    assert m.charged == 0 and m.infra == 3 and m.remaining() == 2
    m.check()  # still allowed

def test_to_dict_complete():
    assert set(BudgetMeter(3).to_dict()) == {"k", "charged", "infra"}
```

- [ ] **Step 2: Run → fails. Implement `crucible/sandbox/budget.py`**

```python
from __future__ import annotations
from .report import TestReport

class BudgetExhausted(RuntimeError):
    pass

class BudgetMeter:
    """Counts test executions that are measurements. Infra failures are counted separately, never charged."""
    def __init__(self, k: int = 8):
        self.k, self.charged, self.infra = k, 0, 0
    def check(self) -> None:
        if self.charged >= self.k:
            raise BudgetExhausted(f"verification budget exhausted: {self.charged}/{self.k}")
    def charge(self, report: TestReport) -> None:
        if report.infra_error is None:
            self.charged += 1
        else:
            self.infra += 1
    def remaining(self) -> int:
        return max(0, self.k - self.charged)
    def to_dict(self) -> dict:
        return {"k": self.k, "charged": self.charged, "infra": self.infra}
```

- [ ] **Step 3: Run → `3 passed`. Mutation check: change `>=` to `>` in `check`; purge; expect `test_ninth_execution_is_refused` FAILS; restore. Change `charge` to always increment `charged`; expect `test_infra_is_counted_but_not_charged` FAILS; restore; rerun green.**

- [ ] **Step 4: Commit** — `git add crucible/sandbox/budget.py tests/sandbox/test_budget.py && git commit -m "feat(sandbox): verification budget meter"`

---

### Task 5: EvalPlus loader (`stream/evalplus.py`)

**Files:**
- Create: `crucible/stream/evalplus.py`, `tests/stream/test_evalplus.py`, `tests/fixtures/mini_humaneval.jsonl.gz`, `tests/fixtures/mini_mbpp.jsonl.gz`

**Interfaces:**
- Produces:
  ```python
  DATASETS: dict[str, Dataset]   # "humaneval", "mbpp" → Dataset(url, sha256, filename)
  def cache_dir() -> Path        # $CRUCIBLE_CACHE or ~/.cache/crucible
  def fetch(name: str, *, cache: Path | None = None) -> Path      # download if missing; verify sha256; raise on mismatch
  def load(name: str, *, cache: Path | None = None) -> list[dict]  # records, in file order
  def full_source(rec: dict) -> str   # prompt + canonical_solution
  def source_of(task_id: str) -> str  # "humaneval" | "mbpp" from "HumanEval/0" / "Mbpp/2"
  ```
- Tests never touch the network: they point `cache` at a tmp dir pre-seeded with the mini fixtures (and a matching sha256 computed at test time via monkeypatching `DATASETS`).

- [ ] **Step 1: Create the fixtures** (two records each, real shapes, tiny inputs)

Run this once to write fixtures:
```bash
.venv/bin/python - <<'EOF'
import gzip, json, pathlib
he = [{"task_id":"HumanEval/0","entry_point":"add2","atol":0,"contract":"",
       "prompt":"def add2(a: int, b: int) -> int:\n    \"\"\"Return a plus b.\"\"\"\n",
       "canonical_solution":"    if a > b:\n        return a + b\n    return b + a\n","test":"",
       "base_input":[[1,2],[0,0],[-1,1],[3,1]],"plus_input":[[10,20],[5,5],[100,-1],[3,4]]},
      {"task_id":"HumanEval/1","entry_point":"is_pos","atol":0,"contract":"",
       "prompt":"def is_pos(x: float) -> bool:\n    \"\"\"True if x > 0.\"\"\"\n",
       "canonical_solution":"    if x > 0:\n        return True\n    return False\n","test":"",
       "base_input":[[1.5],[-2.0],[0.0]],"plus_input":[[2.0],[-0.5]]}]
mb = [{"task_id":"Mbpp/2","entry_point":"first","atol":0,"contract":"","assertion":"",
       "prompt":"\"\"\"\nReturn first element.\n\"\"\"\n",
       "canonical_solution":"\ndef first(xs):\n  return xs[0]\n",
       "base_input":[[[1,2,3]],[[9]]],"plus_input":[[[4,5]],[["a","b"]]]}]
d = pathlib.Path("tests/fixtures"); d.mkdir(parents=True, exist_ok=True)
for name, rows in (("mini_humaneval", he), ("mini_mbpp", mb)):
    with gzip.open(d / f"{name}.jsonl.gz", "wt") as fh:
        for r in rows: fh.write(json.dumps(r) + "\n")
print("fixtures written")
EOF
```

- [ ] **Step 2: Failing tests**

`tests/stream/test_evalplus.py`:
```python
import gzip, hashlib, json, pathlib, shutil
import pytest
from crucible.stream import evalplus as ep

FIX = pathlib.Path(__file__).resolve().parents[1] / "fixtures"

def _seed(tmp_path, monkeypatch):
    ds = {}
    for name, fx in (("humaneval", "mini_humaneval.jsonl.gz"), ("mbpp", "mini_mbpp.jsonl.gz")):
        src = FIX / fx
        dst = tmp_path / ep.DATASETS[name].filename
        shutil.copy(src, dst)
        ds[name] = ep.Dataset(url="file://unused", sha256=hashlib.sha256(dst.read_bytes()).hexdigest(),
                              filename=ep.DATASETS[name].filename)
    monkeypatch.setattr(ep, "DATASETS", ds)
    return tmp_path

def test_load_reads_records_from_cache(tmp_path, monkeypatch):
    cache = _seed(tmp_path, monkeypatch)
    rows = ep.load("humaneval", cache=cache)
    assert [r["task_id"] for r in rows] == ["HumanEval/0", "HumanEval/1"]
    assert ep.load("mbpp", cache=cache)[0]["entry_point"] == "first"

def test_fetch_rejects_bad_digest(tmp_path, monkeypatch):
    cache = _seed(tmp_path, monkeypatch)
    bad = dict(ep.DATASETS); bad["humaneval"] = ep.Dataset("file://unused", "0" * 64, bad["humaneval"].filename)
    monkeypatch.setattr(ep, "DATASETS", bad)
    with pytest.raises(ep.DigestMismatch):
        ep.fetch("humaneval", cache=cache)

def test_full_source_and_source_of(tmp_path, monkeypatch):
    cache = _seed(tmp_path, monkeypatch)
    he = ep.load("humaneval", cache=cache)[0]
    assert ep.full_source(he).startswith("def add2(") and "return a + b" in ep.full_source(he)
    assert ep.source_of("HumanEval/0") == "humaneval" and ep.source_of("Mbpp/2") == "mbpp"

def test_real_dataset_table_is_pinned():
    # The pinned digests are the provenance of the whole stream; they must not drift silently.
    assert ep.DATASETS["humaneval"].sha256 == "272720b90ac375502c8ed23cd791c2a93dfb22a911641a494da74a426c09f101"
    assert ep.DATASETS["mbpp"].sha256 == "af43697e8791c4c149bdfd6b489d8b5412507551ac20e28a439f650b8225db63"
```

- [ ] **Step 3: Run → fails. Implement `crucible/stream/evalplus.py`**

```python
from __future__ import annotations

import gzip
import hashlib
import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Dataset:
    url: str
    sha256: str
    filename: str


DATASETS: dict[str, Dataset] = {
    "humaneval": Dataset(
        "https://github.com/evalplus/humanevalplus_release/releases/download/v0.1.10/HumanEvalPlus.jsonl.gz",
        "272720b90ac375502c8ed23cd791c2a93dfb22a911641a494da74a426c09f101", "HumanEvalPlus-v0.1.10.jsonl.gz"),
    "mbpp": Dataset(
        "https://github.com/evalplus/mbppplus_release/releases/download/v0.2.0/MbppPlus.jsonl.gz",
        "af43697e8791c4c149bdfd6b489d8b5412507551ac20e28a439f650b8225db63", "MbppPlus-v0.2.0.jsonl.gz"),
}


class DigestMismatch(RuntimeError):
    pass


def cache_dir() -> Path:
    return Path(os.environ.get("CRUCIBLE_CACHE", Path.home() / ".cache" / "crucible"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(name: str, *, cache: Path | None = None) -> Path:
    ds = DATASETS[name]
    cache = cache or cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / ds.filename
    if not path.exists():
        with urllib.request.urlopen(ds.url, timeout=60) as resp, open(path, "wb") as out:
            out.write(resp.read())
    got = _sha256(path)
    if got != ds.sha256:
        raise DigestMismatch(f"{name}: expected {ds.sha256}, got {got} at {path}")
    return path


def load(name: str, *, cache: Path | None = None) -> list[dict]:
    path = fetch(name, cache=cache)
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def full_source(rec: dict) -> str:
    return rec["prompt"] + rec["canonical_solution"]


def source_of(task_id: str) -> str:
    head = task_id.split("/")[0].lower()
    return {"humaneval": "humaneval", "mbpp": "mbpp"}[head]
```

- [ ] **Step 4: Run → `4 passed`. Mutation check: in `fetch`, replace `if got != ds.sha256:` with `if False:`; purge; expect `test_fetch_rejects_bad_digest` FAILS; restore; rerun green.**

- [ ] **Step 5: One real fetch, outside the test suite, to populate the cache (network):**
Run: `.venv/bin/python -c "from crucible.stream import evalplus as ep; print(len(ep.load('humaneval')), len(ep.load('mbpp')))"` → `164 378`.

- [ ] **Step 6: Commit** — `git add crucible/stream/evalplus.py tests/stream/test_evalplus.py tests/fixtures && git commit -m "feat(stream): EvalPlus loader with pinned digests"`

---

### Task 6: Unit dataclass, docstring stripping, naming (`stream/units.py`)

**Files:**
- Create: `crucible/stream/units.py`, `tests/stream/test_units.py`

**Interfaces:**
- Produces:
  ```python
  @dataclass(frozen=True)
  class Unit:
      unit_id: str          # EvalPlus task_id, e.g. "HumanEval/0"
      module_name: str      # "unit_humaneval_0"
      entry_point: str
      module_src: str       # docstring-stripped canonical source
      visible_test_src: str
      hidden_test_src: str
      src_hash: str         # sha256(module_src)
      n_visible: int; n_hidden: int
      dropped_inputs: tuple[tuple[str, str], ...]   # (input index as "v3"/"h17", reason)
      def to_dict(self) -> dict; @classmethod from_dict(d)
  def module_name_for(task_id: str) -> str
  def strip_docstrings(src: str) -> str     # removes module/class/function docstrings; ast.unparse output
  def sha256_text(s: str) -> str
  ```

- [ ] **Step 1: Failing tests**

`tests/stream/test_units.py`:
```python
import ast
from crucible.stream.units import Unit, module_name_for, strip_docstrings, sha256_text

def test_module_name_for():
    assert module_name_for("HumanEval/0") == "unit_humaneval_0"
    assert module_name_for("Mbpp/2") == "unit_mbpp_2"

def test_strip_docstrings_removes_all_three_kinds_and_keeps_behaviour():
    src = '"""mod doc"""\nclass C:\n    """cdoc"""\n    def m(self):\n        """mdoc"""\n        return 1\n\ndef f(x):\n    """fdoc"""\n    return x + 1\n'
    out = strip_docstrings(src)
    assert "doc" not in out
    ns = {}; exec(out, ns)
    assert ns["f"](1) == 2 and ns["C"]().m() == 1

def test_strip_docstrings_keeps_function_with_only_docstring_valid():
    assert ast.parse(strip_docstrings('def g():\n    """only"""\n'))

def test_unit_round_trip():
    u = Unit("HumanEval/0", "unit_humaneval_0", "f", "def f():\n    return 1\n", "t", "h",
             sha256_text("def f():\n    return 1\n"), 1, 1, (("h0", "repr too long"),))
    assert Unit.from_dict(u.to_dict()) == u
    assert set(u.to_dict()) == {"unit_id","module_name","entry_point","module_src","visible_test_src",
                                "hidden_test_src","src_hash","n_visible","n_hidden","dropped_inputs"}
```

- [ ] **Step 2: Run → fails. Implement `crucible/stream/units.py`**

```python
from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Unit:
    unit_id: str
    module_name: str
    entry_point: str
    module_src: str
    visible_test_src: str
    hidden_test_src: str
    src_hash: str
    n_visible: int
    n_hidden: int
    dropped_inputs: tuple[tuple[str, str], ...]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["dropped_inputs"] = [list(x) for x in self.dropped_inputs]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Unit":
        d = dict(d)
        d["dropped_inputs"] = tuple(tuple(x) for x in d["dropped_inputs"])
        return cls(**d)


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def module_name_for(task_id: str) -> str:
    return "unit_" + re.sub(r"[^a-z0-9]+", "_", task_id.lower()).strip("_")


class _DocstringStripper(ast.NodeTransformer):
    def _strip(self, node):
        body = getattr(node, "body", None)
        if (body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            body = body[1:] or [ast.Pass()]
            node.body = body
        self.generic_visit(node)
        return node

    visit_Module = visit_ClassDef = visit_FunctionDef = visit_AsyncFunctionDef = _strip


def strip_docstrings(src: str) -> str:
    tree = _DocstringStripper().visit(ast.parse(src))
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"
```

- [ ] **Step 3: Run → `4 passed`. Commit** — `git add crucible/stream/units.py tests/stream/test_units.py && git commit -m "feat(stream): Unit dataclass, docstring stripping, module naming"`

---

### Task 7: Oracle (expected outputs) + test-file generation (`stream/oracle.py`, `stream/testgen.py`)

**Files:**
- Create: `crucible/stream/oracle.py`, `crucible/stream/testgen.py`, `tests/stream/test_oracle.py`, `tests/stream/test_testgen.py`

**Interfaces:**
- Consumes: `execute()` (Task 2).
- Produces:
  ```python
  @dataclass(frozen=True)
  class Expected:
      index: int; ok: bool; value_repr: str | None; reason: str | None   # reason ∈ {"raised:<ExcName>", "timeout", "no-roundtrip", "repr-too-long", "unpicklable"}
  def compute_expected(module_name: str, module_src: str, entry_point: str, inputs: list[list],
                       *, per_input_timeout_s: float = 5.0, wall_cap_s: float = 60.0,
                       max_repr: int = 2000) -> list[Expected]
  def render_tests(module_name: str, entry_point: str, inputs: list[list], expected: list[Expected],
                   *, prefix: str, atol: float) -> tuple[str, list[tuple[str, str]]]
      # returns (test file source, dropped [(f"{prefix}{i}", reason)]) ; test names f"test_{prefix}{i}"
  ```
- Invariants: the oracle runs inside the sandbox (`execute`), one subprocess per unit, per-input `signal.alarm`; only values whose `repr` round-trips (`eval(repr(v)) == v`) and is ≤ `max_repr` chars become tests; floats (or `atol > 0`) compare with `pytest.approx(expected, abs=atol or 1e-9)`.

- [ ] **Step 1: Failing oracle tests**

`tests/stream/test_oracle.py`:
```python
from crucible.stream.oracle import compute_expected

MOD = "def f(x):\n    if x == 'boom':\n        raise ValueError('no')\n    if x == 'hang':\n        while True: pass\n    return [x, x]\n"

def test_expected_values_and_reasons():
    exp = compute_expected("unit_t", MOD, "f", [[1], ["boom"], ["hang"], [2.5]], per_input_timeout_s=1.0)
    by = {e.index: e for e in exp}
    assert by[0].ok and by[0].value_repr == "[1, 1]"
    assert not by[1].ok and by[1].reason == "raised:ValueError"
    assert not by[2].ok and by[2].reason == "timeout"
    assert by[3].ok and by[3].value_repr == "[2.5, 2.5]"

def test_non_roundtrip_and_long_values_are_dropped():
    mod = "def g(n):\n    if n == 0:\n        return float('nan')\n    return list(range(n))\n"
    exp = compute_expected("unit_g", mod, "g", [[0], [5000]], max_repr=100)
    assert exp[0].reason == "no-roundtrip" and exp[1].reason == "repr-too-long"
```

- [ ] **Step 2: Run → fails. Implement `crucible/stream/oracle.py`**

```python
from __future__ import annotations

import json
from dataclasses import dataclass

from ..sandbox.exec import execute

_DRIVER = r'''
import json, signal, sys, math
import {module} as M
fn = getattr(M, {entry!r})
inputs = json.load(open("inputs.json"))
TO = {timeout}
MAXR = {max_repr}
def _alarm(*a): raise TimeoutError()
signal.signal(signal.SIGALRM, _alarm)
out = []
for i, args in enumerate(inputs):
    rec = {{"index": i, "ok": False, "value_repr": None, "reason": None}}
    try:
        signal.setitimer(signal.ITIMER_REAL, TO)
        try:
            v = fn(*args)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        r = repr(v)
        if len(r) > MAXR:
            rec["reason"] = "repr-too-long"
        else:
            try:
                back = eval(r)
                same = (back == v) if not isinstance(v, float) else (back == v and not math.isnan(v))
            except Exception:
                same = False
            if same:
                rec["ok"] = True; rec["value_repr"] = r
            else:
                rec["reason"] = "no-roundtrip"
    except TimeoutError:
        rec["reason"] = "timeout"
    except BaseException as e:
        rec["reason"] = "raised:" + type(e).__name__
    out.append(rec)
print(json.dumps(out))
'''


@dataclass(frozen=True)
class Expected:
    index: int
    ok: bool
    value_repr: str | None
    reason: str | None


def compute_expected(module_name: str, module_src: str, entry_point: str, inputs: list[list], *,
                     per_input_timeout_s: float = 5.0, wall_cap_s: float = 60.0,
                     max_repr: int = 2000) -> list[Expected]:
    driver = _DRIVER.format(module=module_name, entry=entry_point, timeout=per_input_timeout_s, max_repr=max_repr)
    files = {f"{module_name}.py": module_src, "driver.py": driver, "inputs.json": json.dumps(inputs)}
    import sys
    res = execute([sys.executable, "driver.py"], files, wall_cap_s=wall_cap_s)
    if res.timed_out or res.returncode != 0:
        # Whole-driver failure: every input is unmeasured, named as such (None-vs-zero).
        reason = "timeout" if res.timed_out else f"driver-failed:{res.stderr[-200:]}"
        return [Expected(i, False, None, reason) for i in range(len(inputs))]
    return [Expected(r["index"], r["ok"], r["value_repr"], r["reason"]) for r in json.loads(res.stdout)]
```

- [ ] **Step 3: Run → `2 passed`. Commit** — `git add crucible/stream/oracle.py tests/stream/test_oracle.py && git commit -m "feat(stream): sandboxed oracle for expected outputs"`

- [ ] **Step 4: Failing testgen tests**

`tests/stream/test_testgen.py`:
```python
from crucible.stream.oracle import Expected
from crucible.stream.testgen import render_tests
from crucible.sandbox.runner import run_tests

MOD = "def add(a, b):\n    return a + b\n"

def test_render_and_run_generated_tests():
    exp = [Expected(0, True, "3", None), Expected(1, False, None, "raised:TypeError"), Expected(2, True, "0.30000000000000004", None)]
    src, dropped = render_tests("unit_x", "add", [[1, 2], [None, 1], [0.1, 0.2]], exp, prefix="v", atol=0)
    assert dropped == [("v1", "raised:TypeError")]
    assert "def test_v0" in src and "def test_v2" in src and "def test_v1" not in src
    r = run_tests("unit_x", MOD, src)
    assert r.all_passed and set(r.passed) == {"test_v0", "test_v2"}
    r2 = run_tests("unit_x", MOD.replace("a + b", "a - b"), src)
    assert r2.killed

def test_float_uses_approx_with_atol():
    exp = [Expected(0, True, "0.3", None)]
    src, _ = render_tests("unit_x", "add", [[0.1, 0.2]], exp, prefix="h", atol=1e-6)
    assert "pytest.approx" in src
    assert run_tests("unit_x", MOD, src).all_passed
```

- [ ] **Step 5: Run → fails. Implement `crucible/stream/testgen.py`**

```python
from __future__ import annotations

from .oracle import Expected

_HEADER = "import pytest\nfrom {module} import {entry} as candidate\n\n"


def _is_floaty(value_repr: str) -> bool:
    try:
        v = eval(value_repr)  # repr round-trips by construction (oracle checked)
    except Exception:
        return False
    return isinstance(v, float) or (isinstance(v, (list, tuple)) and any(isinstance(x, float) for x in v))


def render_tests(module_name: str, entry_point: str, inputs: list[list], expected: list[Expected], *,
                 prefix: str, atol: float) -> tuple[str, list[tuple[str, str]]]:
    lines = [_HEADER.format(module=module_name, entry=entry_point)]
    dropped: list[tuple[str, str]] = []
    for e, args in zip(expected, inputs):
        name = f"{prefix}{e.index}"
        if not e.ok:
            dropped.append((name, e.reason or "unknown"))
            continue
        call = f"candidate(*{args!r})"
        if atol > 0 or _is_floaty(e.value_repr or ""):
            cmp = f"assert {call} == pytest.approx({e.value_repr}, abs={atol or 1e-9!r})"
        else:
            cmp = f"assert {call} == {e.value_repr}"
        lines.append(f"def test_{name}():\n    {cmp}\n\n")
    return "".join(lines), dropped
```

- [ ] **Step 6: Run → `2 passed`. Mutation check: in `render_tests`, make `if not e.ok:` into `if False:`; purge; expect the first test FAILS (test_v1 present / dropped wrong). Restore; rerun green. Commit** — `git add crucible/stream/testgen.py tests/stream/test_testgen.py && git commit -m "feat(stream): pytest file generation from oracle outputs"`

---

### Task 8: Build a unit end-to-end with self-check (`stream/build.py`)

**Files:**
- Create: `crucible/stream/build.py`, `tests/stream/test_build.py`

**Interfaces:**
- Consumes: `evalplus.full_source`, `units.*`, `oracle.compute_expected`, `testgen.render_tests`, `runner.run_tests`.
- Produces:
  ```python
  @dataclass(frozen=True)
  class Dropped: unit_id: str; reason: str
  def build_unit(rec: dict, *, seed: int, max_hidden: int = 100) -> Unit | Dropped
  def build_units(recs: list[dict], *, seed: int, max_hidden: int = 100, jobs: int = 8,
                  progress=None) -> tuple[list[Unit], list[Dropped]]
  ```
- Invariants: hidden inputs sampled with `random.Random(f"{seed}:{task_id}:hidden")` when more than `max_hidden`; a unit is dropped (with reason) if the canonical module does not import, if it has < 1 visible test after dropping, if the canonical does not pass visible ∪ hidden in the sandbox, or if the visible suite takes > 20 s on the canonical (too slow for K=8). Deterministic: same inputs ⇒ same `Unit` (tests assert it).

- [ ] **Step 1: Failing tests**

`tests/stream/test_build.py`:
```python
import gzip, json, pathlib
from crucible.stream.build import build_unit, build_units, Dropped
from crucible.stream.units import Unit

FIX = pathlib.Path(__file__).resolve().parents[1] / "fixtures"
def _recs(name):
    with gzip.open(FIX / name, "rt") as fh:
        return [json.loads(l) for l in fh]

def test_build_unit_humaneval_is_deterministic_and_self_checked():
    rec = _recs("mini_humaneval.jsonl.gz")[0]
    u1 = build_unit(rec, seed=0, max_hidden=2)
    u2 = build_unit(rec, seed=0, max_hidden=2)
    assert isinstance(u1, Unit) and u1 == u2
    assert u1.module_name == "unit_humaneval_0" and u1.n_visible == 4 and u1.n_hidden == 2
    assert '"""' not in u1.module_src and "return a + b" in u1.module_src

def test_build_unit_mbpp_strips_prompt_docstring():
    rec = _recs("mini_mbpp.jsonl.gz")[0]
    u = build_unit(rec, seed=0)
    assert isinstance(u, Unit) and "Return first element" not in u.module_src and u.n_visible == 2

def test_nondeterministic_canonical_is_dropped_with_reason():
    # The oracle derives expected values FROM the canonical, so a merely-wrong canonical is self-consistent
    # and cannot be detected here. What the self-check catches is a canonical whose outputs differ between
    # the oracle process and the pytest process: pid-dependent output does exactly that, deterministically.
    rec = dict(_recs("mini_humaneval.jsonl.gz")[0]); rec["canonical_solution"] = "    import os\n    return os.getpid()\n"
    d = build_unit(rec, seed=0)
    assert isinstance(d, Dropped) and d.reason.startswith("canonical-fails-visible")

def test_build_units_partitions_units_and_dropped():
    recs = _recs("mini_humaneval.jsonl.gz") + _recs("mini_mbpp.jsonl.gz")
    bad = dict(recs[0]); bad["task_id"] = "HumanEval/999"; bad["canonical_solution"] = "    return a -\n"
    units, dropped = build_units(recs + [bad], seed=0, jobs=2)
    assert {u.unit_id for u in units} == {"HumanEval/0", "HumanEval/1", "Mbpp/2"}
    assert [d.unit_id for d in dropped] == ["HumanEval/999"]
```

- [ ] **Step 2: Run → fails. Implement `crucible/stream/build.py`**

```python
from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from ..sandbox.runner import run_tests
from .evalplus import full_source
from .oracle import compute_expected
from .testgen import render_tests
from .units import Unit, module_name_for, sha256_text, strip_docstrings

MAX_CANONICAL_VISIBLE_WALL_S = 20.0


@dataclass(frozen=True)
class Dropped:
    unit_id: str
    reason: str


def _select_hidden(rec: dict, seed: int, max_hidden: int) -> list[list]:
    plus = list(rec["plus_input"])
    if len(plus) <= max_hidden:
        return plus
    rng = random.Random(f"{seed}:{rec['task_id']}:hidden")
    idx = sorted(rng.sample(range(len(plus)), max_hidden))
    return [plus[i] for i in idx]


def build_unit(rec: dict, *, seed: int, max_hidden: int = 100) -> Unit | Dropped:
    uid, entry = rec["task_id"], rec["entry_point"]
    try:
        module_src = strip_docstrings(full_source(rec))
        compile(module_src, uid, "exec")
    except SyntaxError as e:
        return Dropped(uid, f"canonical-syntax:{e.msg}")
    mod = module_name_for(uid)
    atol = float(rec.get("atol") or 0)
    vis_in, hid_in = list(rec["base_input"]), _select_hidden(rec, seed, max_hidden)
    exp_v = compute_expected(mod, module_src, entry, vis_in)
    exp_h = compute_expected(mod, module_src, entry, hid_in)
    vis_src, drop_v = render_tests(mod, entry, vis_in, exp_v, prefix="v", atol=atol)
    hid_src, drop_h = render_tests(mod, entry, hid_in, exp_h, prefix="h", atol=atol)
    n_v, n_h = len(vis_in) - len(drop_v), len(hid_in) - len(drop_h)
    if n_v < 1:
        return Dropped(uid, "no-visible-tests")
    rv = run_tests(mod, module_src, vis_src)
    if not rv.all_passed:
        return Dropped(uid, f"canonical-fails-visible:{rv.infra_error or (rv.failed + rv.timed_out + rv.errored)}")
    if rv.wall_s > MAX_CANONICAL_VISIBLE_WALL_S:
        return Dropped(uid, f"visible-too-slow:{rv.wall_s:.1f}s")
    if n_h:
        rh = run_tests(mod, module_src, hid_src)
        if not rh.all_passed:
            return Dropped(uid, f"canonical-fails-hidden:{rh.infra_error or (rh.failed + rh.timed_out + rh.errored)}")
    return Unit(uid, mod, entry, module_src, vis_src, hid_src, sha256_text(module_src), n_v, n_h,
                tuple(drop_v + drop_h))


def build_units(recs: list[dict], *, seed: int, max_hidden: int = 100, jobs: int = 8,
                progress=None) -> tuple[list[Unit], list[Dropped]]:
    units, dropped = [], []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for res in pool.map(lambda r: build_unit(r, seed=seed, max_hidden=max_hidden), recs):
            (units if isinstance(res, Unit) else dropped).append(res)
            if progress:
                progress(res)
    units.sort(key=lambda u: u.unit_id)
    dropped.sort(key=lambda d: d.unit_id)
    return units, dropped
```

- [ ] **Step 3: Run → `4 passed` (the canonical-fails test: `add2` with `a - b` fails visible `[1,2]→3`). Commit** — `git add crucible/stream/build.py tests/stream/test_build.py && git commit -m "feat(stream): build units with sandboxed self-check and drop reasons"`

- [ ] **Step 4: Real-data smoke (not a test): build 10 real units and report drops.**
Run:
```bash
.venv/bin/python - <<'EOF'
from crucible.stream import evalplus as ep
from crucible.stream.build import build_units
recs = ep.load("humaneval")[:5] + ep.load("mbpp")[:5]
units, dropped = build_units(recs, seed=0, jobs=4)
print(len(units), "units;", [ (d.unit_id, d.reason) for d in dropped ])
for u in units[:3]: print(u.unit_id, u.n_visible, u.n_hidden, len(u.dropped_inputs))
EOF
```
Expected: ≥ 8 units; any drops have a named reason. If MBPP units drop with `canonical-fails-*` because their solutions import `math`/`re`/numpy at module level and the oracle driver cannot import them — inspect `reason`; numpy is installed in the venv so it should be available inside the sandbox (same interpreter). Record anything surprising in `docs/CARRIED-DEBT.md` under S1 → Process lessons.

---

### Task 9: Operator families (`stream/families.py`)

**Files:**
- Create: `crucible/stream/families.py`, `tests/stream/test_families.py`

**Interfaces:**
- Produces:
  ```python
  FAMILIES: tuple[str, ...] = ("ARITH","CMP","BOOL","UNARY","CONST","FLOW","EXC","SDL")
  EXCLUDED: frozenset[str] = {"VariableReplacer", "VariableInserter"}   # need per-variable ctor args (amendment A2)
  SDL_OPERATOR = "StatementDeletion"          # our own, not a cosmic-ray core op
  def family_of(op_name: str) -> str | None   # op_name WITHOUT "core/"; None if excluded/unknown
  def all_operator_names() -> list[str]       # 213 core names (without prefix) + SDL_OPERATOR, sorted
  def operators_by_family() -> dict[str, list[str]]
  def check_complete() -> list[str]           # names that are neither mapped nor excluded (must be empty)
  ```

- [ ] **Step 1: Failing tests**

`tests/stream/test_families.py`:
```python
from crucible.stream import families as F

def test_every_core_operator_is_mapped_or_excluded():
    assert F.check_complete() == []

def test_family_examples():
    assert F.family_of("ReplaceBinaryOperator_Add_Sub") == "ARITH"
    assert F.family_of("ReplaceComparisonOperator_Lt_GtE") == "CMP"
    assert F.family_of("ReplaceUnaryOperator_Delete_Not") == "UNARY"
    assert F.family_of("AddNot") == "BOOL" and F.family_of("ReplaceOrWithAnd") == "BOOL"
    assert F.family_of("NumberReplacer") == "CONST"
    assert F.family_of("ZeroIterationForLoop") == "FLOW"
    assert F.family_of("RemoveDecorator") == "EXC" and F.family_of("ExceptionReplacer") == "EXC"
    assert F.family_of("StatementDeletion") == "SDL"
    assert F.family_of("VariableReplacer") is None and F.family_of("Nonsense") is None

def test_operators_by_family_covers_all_families_and_counts():
    by = F.operators_by_family()
    assert set(by) == set(F.FAMILIES)
    assert len(by["ARITH"]) == 132 and len(by["CMP"]) == 56 and len(by["UNARY"]) == 12
    assert sum(len(v) for v in by.values()) == 213 - len(F.EXCLUDED) + 1
```

- [ ] **Step 2: Run → fails. Implement `crucible/stream/families.py`**

```python
"""Operator → family map, frozen at lock (spec §4.2, amendment A2)."""
from __future__ import annotations

from cosmic_ray import plugins

FAMILIES: tuple[str, ...] = ("ARITH", "CMP", "BOOL", "UNARY", "CONST", "FLOW", "EXC", "SDL")
EXCLUDED: frozenset[str] = frozenset({"VariableReplacer", "VariableInserter"})
SDL_OPERATOR = "StatementDeletion"

_PREFIX = {"ReplaceBinaryOperator_": "ARITH", "ReplaceComparisonOperator_": "CMP", "ReplaceUnaryOperator_": "UNARY"}
_EXACT = {
    "AddNot": "BOOL", "ReplaceTrueWithFalse": "BOOL", "ReplaceFalseWithTrue": "BOOL",
    "ReplaceAndWithOr": "BOOL", "ReplaceOrWithAnd": "BOOL",
    "NumberReplacer": "CONST",
    "ReplaceBreakWithContinue": "FLOW", "ReplaceContinueWithBreak": "FLOW", "ZeroIterationForLoop": "FLOW",
    "ExceptionReplacer": "EXC", "RemoveDecorator": "EXC",
    SDL_OPERATOR: "SDL",
}


def family_of(op_name: str) -> str | None:
    if op_name in EXCLUDED:
        return None
    if op_name in _EXACT:
        return _EXACT[op_name]
    for pre, fam in _PREFIX.items():
        if op_name.startswith(pre):
            return fam
    return None


def core_operator_names() -> list[str]:
    return sorted(n.removeprefix("core/") for n in plugins.operator_names())


def all_operator_names() -> list[str]:
    return sorted(core_operator_names() + [SDL_OPERATOR])


def operators_by_family() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {f: [] for f in FAMILIES}
    for n in all_operator_names():
        fam = family_of(n)
        if fam is not None:
            out[fam].append(n)
    return out


def check_complete() -> list[str]:
    return [n for n in core_operator_names() if family_of(n) is None and n not in EXCLUDED]
```

- [ ] **Step 3: Run → `3 passed`. Mutation check: remove `"RemoveDecorator": "EXC"` from `_EXACT`; purge; expect `test_every_core_operator_is_mapped_or_excluded` FAILS; restore; rerun green. Commit** — `git add crucible/stream/families.py tests/stream/test_families.py && git commit -m "feat(stream): operator family map with completeness check"`

---

### Task 10: Statement-deletion operator (`stream/sdl.py`)

**Files:**
- Create: `crucible/stream/sdl.py`, `tests/stream/test_sdl.py`

**Interfaces:**
- Produces: `class StatementDeletion(cosmic_ray.operators.operator.Operator)` — deletes one `simple_stmt` inside an indented `suite` by replacing it with `pass` (keeps indentation), usable with `cosmic_ray.mutating.mutate_code(src, StatementDeletion(), occurrence)`.
- Invariants: never targets docstrings, statements that are already `pass`, or statements outside a `suite` (module-level imports/defs); mutated source always compiles.

- [ ] **Step 1: Failing tests**

`tests/stream/test_sdl.py`:
```python
from cosmic_ray.mutating import mutate_code
from cosmic_ray.ast import get_ast, ast_nodes
from crucible.stream.sdl import StatementDeletion

SRC = "import os\n\ndef f(x):\n    y = x + 1\n    z = y * 2\n    return z\n"

def _positions(src):
    op = StatementDeletion()
    return [p for n in ast_nodes(get_ast(src)) for p in op.mutation_positions(n)]

def test_positions_are_only_suite_statements():
    pos = _positions(SRC)
    assert len(pos) == 3 and pos[0][0][0] == 4 and pos[2][0][0] == 6  # lines 4,5,6; not the import

def test_mutate_replaces_with_pass_keeping_indent_and_compiles():
    out = mutate_code(SRC, StatementDeletion(), 0)
    assert out == "import os\n\ndef f(x):\n    pass\n    z = y * 2\n    return z\n"
    compile(out, "m", "exec")
    out2 = mutate_code(SRC, StatementDeletion(), 2)
    assert out2.endswith("    z = y * 2\n    pass\n")

def test_skips_docstring_and_pass():
    src = 'def g():\n    """doc"""\n    pass\n'
    assert _positions(src) == []

def test_occurrence_out_of_range_returns_none():
    assert mutate_code(SRC, StatementDeletion(), 99) is None
```

- [ ] **Step 2: Run → fails. Implement `crucible/stream/sdl.py`**

```python
"""StatementDeletion: MutPy's SDL idea (Apache-2.0, reimplemented) on cosmic-ray's Operator ABC."""
from __future__ import annotations

import parso
from cosmic_ray.operators.operator import Operator


def _is_docstring(stmt) -> bool:
    first = stmt.children[0]
    return getattr(first, "type", "") == "string"


def _is_pass(stmt) -> bool:
    first = stmt.children[0]
    return getattr(first, "type", "") == "keyword" and first.value == "pass"


def _deletable(node) -> bool:
    return (node.type == "simple_stmt" and node.parent is not None and node.parent.type == "suite"
            and not _is_docstring(node) and not _is_pass(node))


class StatementDeletion(Operator):
    def mutation_positions(self, node):
        if _deletable(node):
            yield (node.start_pos, node.end_pos)

    def mutate(self, node, index):
        prefix = node.get_first_leaf().prefix
        new = parso.parse("pass\n").children[0]
        new.get_first_leaf().prefix = prefix
        new.parent = node.parent
        return new

    @classmethod
    def examples(cls):
        return ()
```

- [ ] **Step 3: Run → `4 passed`. Add the THIRD_PARTY row check: the ledger already lists MutPy (Task 1). Commit** — `git add crucible/stream/sdl.py tests/stream/test_sdl.py && git commit -m "feat(stream): statement-deletion operator on cosmic-ray ABC"`

---

### Task 11: Mutant enumeration and application (`stream/mutants.py`)

**Files:**
- Create: `crucible/stream/mutants.py`, `tests/stream/test_mutants.py`

**Interfaces:**
- Consumes: `families.*`, `sdl.StatementDeletion`, `units.Unit`, `units.sha256_text`.
- Produces:
  ```python
  Span = tuple[tuple[int, int], tuple[int, int]]
  @dataclass(frozen=True)
  class MutantSpec: operator: str; occurrence: int; family: str; span: Span
  @dataclass(frozen=True)
  class Mutant:
      unit_id: str; key: str; operator: str; occurrence: int; family: str; span: Span
      mutated_src: str; diff: str
      def to_dict(self) -> dict; @classmethod from_dict(d)
  def operator_instance(name: str)                      # "StatementDeletion" → sdl; else plugins.get_operator("core/"+name)()
  def enumerate_specs(src: str, operators: list[str]) -> list[MutantSpec]     # all occurrences, in operator order then occurrence order
  def apply_spec(src: str, spec: MutantSpec) -> str | None
  def make_mutant(unit: Unit, spec: MutantSpec) -> Mutant | None   # None if no change, identical to original, or SyntaxError
  def sample_specs(specs: list[MutantSpec], *, per_family: int, rng: random.Random) -> list[MutantSpec]
      # at most `per_family` specs per family, chosen by seeded sampling; distinct spans preferred
  ```
- Invariants: `key = sha256_text(unit.src_hash + "\n" + diff)`; `diff` is a unified diff with fixed headers `a/<module>.py` → `b/<module>.py` (no timestamps).

- [ ] **Step 1: Failing tests**

`tests/stream/test_mutants.py`:
```python
import random
from crucible.stream.mutants import enumerate_specs, apply_spec, make_mutant, sample_specs, Mutant
from crucible.stream.units import Unit, sha256_text

SRC = "def f(a, b):\n    if a < b:\n        return a + b\n    return a - b\n"
UNIT = Unit("HumanEval/0", "unit_humaneval_0", "f", SRC, "", "", sha256_text(SRC), 1, 0, ())

def test_enumerate_add_sub_and_lt_gte():
    specs = enumerate_specs(SRC, ["ReplaceBinaryOperator_Add_Sub", "ReplaceComparisonOperator_Lt_GtE", "StatementDeletion"])
    ops = [(s.operator, s.occurrence, s.family) for s in specs]
    assert ("ReplaceBinaryOperator_Add_Sub", 0, "ARITH") in ops
    assert ("ReplaceComparisonOperator_Lt_GtE", 0, "CMP") in ops
    assert sum(1 for s in specs if s.family == "SDL") == 3   # if-stmt body return, outer return, ... (see note)

def test_apply_and_make_mutant_key_is_content_hash():
    spec = enumerate_specs(SRC, ["ReplaceBinaryOperator_Add_Sub"])[0]
    assert apply_spec(SRC, spec) == SRC.replace("a + b", "a - b")
    m = make_mutant(UNIT, spec)
    assert isinstance(m, Mutant) and m.key == sha256_text(UNIT.src_hash + "\n" + m.diff)
    assert m.diff.startswith("--- a/unit_humaneval_0.py\n+++ b/unit_humaneval_0.py\n")
    assert Mutant.from_dict(m.to_dict()) == m

def test_make_mutant_returns_none_when_unchanged_or_invalid():
    # Sub_Add on a source with no subtraction inside f? There is one ("a - b") -> changes. Use Mul_Div: no '*' present.
    specs = enumerate_specs(SRC, ["ReplaceBinaryOperator_Mul_Div"])
    assert specs == []

def test_sample_specs_caps_per_family_and_is_seeded():
    specs = enumerate_specs(SRC, ["ReplaceBinaryOperator_Add_Sub", "ReplaceBinaryOperator_Sub_Add",
                                  "ReplaceComparisonOperator_Lt_GtE", "ReplaceComparisonOperator_Lt_Gt", "StatementDeletion"])
    s1 = sample_specs(specs, per_family=1, rng=random.Random(7))
    s2 = sample_specs(specs, per_family=1, rng=random.Random(7))
    assert s1 == s2 and len(s1) == 3 and len({s.family for s in s1}) == 3
```
Note on the SDL count: the statements inside `f`'s suite are `if ...:` (a compound statement, **not** a `simple_stmt`), and `return a - b`; inside the `if` suite: `return a + b`. So deletable simple statements = `return a + b`, `return a - b` = **2**, not 3. Fix the assertion to `== 2` before running (the plan author's count above was wrong; the test pins the true behaviour).

- [ ] **Step 2: Run → fails. Implement `crucible/stream/mutants.py`**

```python
from __future__ import annotations

import difflib
import random
from dataclasses import asdict, dataclass

from cosmic_ray import plugins
from cosmic_ray.ast import ast_nodes, get_ast
from cosmic_ray.mutating import mutate_code

from .families import SDL_OPERATOR, family_of
from .sdl import StatementDeletion
from .units import Unit, sha256_text

Span = tuple[tuple[int, int], tuple[int, int]]


@dataclass(frozen=True)
class MutantSpec:
    operator: str
    occurrence: int
    family: str
    span: Span


@dataclass(frozen=True)
class Mutant:
    unit_id: str
    key: str
    operator: str
    occurrence: int
    family: str
    span: Span
    mutated_src: str
    diff: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["span"] = [list(d["span"][0]), list(d["span"][1])]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Mutant":
        d = dict(d)
        d["span"] = (tuple(d["span"][0]), tuple(d["span"][1]))
        return cls(**d)


def operator_instance(name: str):
    if name == SDL_OPERATOR:
        return StatementDeletion()
    return plugins.get_operator(f"core/{name}")()


def enumerate_specs(src: str, operators: list[str]) -> list[MutantSpec]:
    tree = get_ast(src)
    nodes = list(ast_nodes(tree))
    out: list[MutantSpec] = []
    for name in operators:
        fam = family_of(name)
        if fam is None:
            continue
        op = operator_instance(name)
        positions = [p for node in nodes for p in op.mutation_positions(node)]
        for i, span in enumerate(positions):
            out.append(MutantSpec(name, i, fam, (tuple(span[0]), tuple(span[1]))))
    return out


def apply_spec(src: str, spec: MutantSpec) -> str | None:
    return mutate_code(src, operator_instance(spec.operator), spec.occurrence)


def _unified(module_name: str, a: str, b: str) -> str:
    return "".join(difflib.unified_diff(a.splitlines(keepends=True), b.splitlines(keepends=True),
                                        fromfile=f"a/{module_name}.py", tofile=f"b/{module_name}.py", n=3))


def make_mutant(unit: Unit, spec: MutantSpec) -> Mutant | None:
    mutated = apply_spec(unit.module_src, spec)
    if mutated is None or mutated == unit.module_src:
        return None
    try:
        compile(mutated, unit.module_name, "exec")
    except SyntaxError:
        return None
    diff = _unified(unit.module_name, unit.module_src, mutated)
    return Mutant(unit.unit_id, sha256_text(unit.src_hash + "\n" + diff), spec.operator, spec.occurrence,
                  spec.family, spec.span, mutated, diff)


def sample_specs(specs: list[MutantSpec], *, per_family: int, rng: random.Random) -> list[MutantSpec]:
    by: dict[str, list[MutantSpec]] = {}
    for s in specs:
        by.setdefault(s.family, []).append(s)
    out: list[MutantSpec] = []
    for fam in sorted(by):
        pool = list(by[fam])
        rng.shuffle(pool)
        chosen, seen_spans = [], set()
        for s in pool:                      # distinct spans first
            if s.span not in seen_spans and len(chosen) < per_family:
                chosen.append(s); seen_spans.add(s.span)
        for s in pool:                      # then fill
            if len(chosen) >= per_family:
                break
            if s not in chosen:
                chosen.append(s)
        out.extend(chosen)
    return out
```

- [ ] **Step 3: Run → `4 passed`. Mutation check: in `make_mutant`, drop the `mutated == unit.module_src` guard; purge; run; if nothing fails, add a test that asserts `make_mutant` returns `None` for a spec whose operator is a no-op on the source (e.g. `ReplaceTrueWithFalse` has no positions → enumerate returns []; so instead mutate `_unified` to return `""` and assert the key test fails). Restore; rerun green. Commit** — `git add crucible/stream/mutants.py tests/stream/test_mutants.py && git commit -m "feat(stream): mutant enumeration, application, content-hash keys"`

---

### Task 12: Mutant validation in the sandbox (`stream/validate.py`)

**Files:**
- Create: `crucible/stream/validate.py`, `tests/stream/test_validate.py`

**Interfaces:**
- Consumes: `run_tests`, `Unit`, `Mutant`.
- Produces:
  ```python
  @dataclass(frozen=True)
  class Validation:
      mutant_key: str; valid: bool; reason: str     # reason ∈ {"killed-visible","hidden-only","equivalent","infra","syntax"}
      kills_by_timeout: bool; n_killing_visible: int; visible_failed: tuple[str, ...]
      def to_dict(self) -> dict; @classmethod from_dict(d)
  def validate_mutant(unit: Unit, mutant: Mutant, *, per_test_timeout_s: float = 5.0, wall_cap_s: float = 60.0) -> Validation
  def validate_many(unit: Unit, mutants: list[Mutant], *, jobs: int = 8) -> list[Validation]   # order preserved
  ```
- Invariants: `valid` ⇔ `reason == "killed-visible"`; hidden suite is run **only** when the visible suite did not kill (to label `hidden-only` vs `equivalent`); `infra` is never `valid` and is counted by the caller.

- [ ] **Step 1: Failing tests**

`tests/stream/test_validate.py`:
```python
from crucible.stream.validate import validate_mutant, validate_many, Validation
from crucible.stream.units import Unit, sha256_text
from crucible.stream.mutants import enumerate_specs, make_mutant

SRC = "def f(a, b):\n    return a + b\n"
VIS = "from unit_x import f as candidate\ndef test_v0():\n    assert candidate(1, 2) == 3\n"
HID = "from unit_x import f as candidate\ndef test_h0():\n    assert candidate(0, 0) == 0\n"
U = Unit("X/0", "unit_x", "f", SRC, VIS, HID, sha256_text(SRC), 1, 1, ())

def _mut(op):
    spec = enumerate_specs(SRC, [op])[0]
    return make_mutant(U, spec)

def test_killed_by_visible_is_valid():
    v = validate_mutant(U, _mut("ReplaceBinaryOperator_Add_Sub"))
    assert v.valid and v.reason == "killed-visible" and v.n_killing_visible == 1 and not v.kills_by_timeout

def test_hidden_only_kill_is_not_valid():
    # a + b*1 ... craft: mutant that passes visible (1,2)->3 but fails hidden (0,0)->0: return a + b if a else 1
    m = _mut("ReplaceBinaryOperator_Add_Sub")
    from dataclasses import replace
    m2 = replace(m, mutated_src="def f(a, b):\n    return a + b if a else 1\n", key="k2")
    v = validate_mutant(U, m2)
    assert not v.valid and v.reason == "hidden-only"

def test_equivalent_is_not_valid():
    m = _mut("ReplaceBinaryOperator_Add_Sub")
    from dataclasses import replace
    m3 = replace(m, mutated_src="def f(a, b):\n    return b + a\n", key="k3")
    assert validate_mutant(U, m3).reason == "equivalent"

def test_hang_is_valid_and_flagged():
    m = _mut("ReplaceBinaryOperator_Add_Sub")
    from dataclasses import replace
    m4 = replace(m, mutated_src="def f(a, b):\n    while True: pass\n", key="k4")
    v = validate_mutant(U, m4, per_test_timeout_s=1.0)
    assert v.valid and v.kills_by_timeout

def test_validate_many_preserves_order_and_round_trips():
    ms = [_mut("ReplaceBinaryOperator_Add_Sub"), _mut("ReplaceBinaryOperator_Add_Mul")]
    vs = validate_many(U, ms, jobs=2)
    assert [v.mutant_key for v in vs] == [m.key for m in ms]
    assert Validation.from_dict(vs[0].to_dict()) == vs[0]
```

- [ ] **Step 2: Run → fails. Implement `crucible/stream/validate.py`**

```python
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

from ..sandbox.runner import run_tests
from .mutants import Mutant
from .units import Unit


@dataclass(frozen=True)
class Validation:
    mutant_key: str
    valid: bool
    reason: str
    kills_by_timeout: bool
    n_killing_visible: int
    visible_failed: tuple[str, ...]

    def to_dict(self) -> dict:
        d = asdict(self); d["visible_failed"] = list(self.visible_failed); return d

    @classmethod
    def from_dict(cls, d: dict) -> "Validation":
        d = dict(d); d["visible_failed"] = tuple(d["visible_failed"]); return cls(**d)


def validate_mutant(unit: Unit, mutant: Mutant, *, per_test_timeout_s: float = 5.0,
                    wall_cap_s: float = 60.0) -> Validation:
    try:
        compile(mutant.mutated_src, unit.module_name, "exec")
    except SyntaxError:
        return Validation(mutant.key, False, "syntax", False, 0, ())
    rv = run_tests(unit.module_name, mutant.mutated_src, unit.visible_test_src,
                   per_test_timeout_s=per_test_timeout_s, wall_cap_s=wall_cap_s)
    if rv.infra_error is not None:
        return Validation(mutant.key, False, "infra", False, 0, ())
    if rv.killed:
        killing = rv.failed + rv.timed_out + rv.errored
        return Validation(mutant.key, True, "killed-visible", bool(rv.timed_out), len(killing), killing)
    if unit.n_hidden:
        rh = run_tests(unit.module_name, mutant.mutated_src, unit.hidden_test_src,
                       per_test_timeout_s=per_test_timeout_s, wall_cap_s=wall_cap_s)
        if rh.infra_error is not None:
            return Validation(mutant.key, False, "infra", False, 0, ())
        if rh.killed:
            return Validation(mutant.key, False, "hidden-only", bool(rh.timed_out), 0, ())
    return Validation(mutant.key, False, "equivalent", False, 0, ())


def validate_many(unit: Unit, mutants: list[Mutant], *, jobs: int = 8) -> list[Validation]:
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(lambda m: validate_mutant(unit, m), mutants))
```

- [ ] **Step 3: Run → `5 passed`. Mutation check: make `valid=True` for the `hidden-only` branch; purge; expect `test_hidden_only_kill_is_not_valid` FAILS; restore; rerun green. Commit** — `git add crucible/stream/validate.py tests/stream/test_validate.py && git commit -m "feat(stream): sandbox validation — valid means killed by the visible suite"`

---

### Task 13: Stream composition and manifest (`stream/compose.py`)

**Files:**
- Create: `crucible/stream/compose.py`, `tests/stream/test_compose.py`

**Interfaces:**
- Consumes: `Unit`, `Mutant`, `Validation`, `sha256_text`.
- Produces:
  ```python
  @dataclass(frozen=True)
  class TaskSpec:
      task_key: str; unit_id: str; family: str; class_id: str; phase: int; kind: str   # kind ∈ first|second|novel
      span: Span; kills_by_timeout: bool; n_killing_visible: int
      def to_dict(self) -> dict; @classmethod from_dict(d)
  @dataclass(frozen=True)
  class StreamManifest:
      stream_hash: str; seed: int; C: int; n_nov: int; rung: str
      unit_ids: tuple[str, ...]; tasks: tuple[TaskSpec, ...]
      classes: dict[str, tuple[str, str]]            # class_id -> (m1_key, m2_key)
      dropped: tuple[tuple[str, str], ...]; counts: dict[str, int]
      def to_dict(self) -> dict; @classmethod from_dict(d)
      def phase(self, n: int) -> list[TaskSpec]
  class NotEnoughClasses(RuntimeError)
  def class_id(unit_id: str, family: str) -> str      # f"{unit_id}|{family}"
  def compose(units: list[Unit], validated: dict[str, list[tuple[Mutant, Validation]]], *,
              seed: int, C: int, n_nov: int, rung: str = "base") -> StreamManifest
  ```
- Invariants (each pinned by a test): every class has `m1.span != m2.span`; novel units ∩ phase-1 units = ∅; phase-1 has exactly C tasks of kind `first`, phase-2 has C `second` + n_nov `novel`; stream order is seeded and reproducible; `task_key == mutant key`; `stream_hash` changes when seed, rung, or any task changes; `counts` names every exclusion reason (`hidden-only`, `equivalent`, `infra`, `syntax`, `ineligible-class`, `unit-no-valid`).

- [ ] **Step 1: Failing tests**

`tests/stream/test_compose.py`:
```python
import random, pytest
from dataclasses import replace
from crucible.stream.compose import compose, StreamManifest, TaskSpec, NotEnoughClasses, class_id
from crucible.stream.units import Unit, sha256_text
from crucible.stream.mutants import Mutant
from crucible.stream.validate import Validation

def _unit(i):
    src = f"def f{i}(a, b):\n    return a + b\n"
    return Unit(f"X/{i}", f"unit_x_{i}", f"f{i}", src, "v", "h", sha256_text(src), 1, 1, ())

def _mut(u, fam, line, tag, timeout=False):
    key = sha256_text(f"{u.unit_id}:{fam}:{line}:{tag}")
    m = Mutant(u.unit_id, key, "Op", 0, fam, ((line, 1), (line, 2)), "src", "diff")
    v = Validation(key, True, "killed-visible", timeout, 1, ("test_v0",))
    return (m, v)

def _world(n_units=6):
    units = [_unit(i) for i in range(n_units)]
    validated = {}
    for u in units:
        validated[u.unit_id] = [_mut(u, "ARITH", 2, "a"), _mut(u, "ARITH", 3, "b"), _mut(u, "CMP", 2, "c"),
                                (Mutant(u.unit_id, "eq" + u.unit_id, "Op", 0, "CMP", ((2, 1), (2, 2)), "s", "d"),
                                 Validation("eq" + u.unit_id, False, "equivalent", False, 0, ()))]
    return units, validated

def test_compose_shapes_and_invariants():
    units, validated = _world()
    m = compose(units, validated, seed=0, C=3, n_nov=2)
    p1, p2 = m.phase(1), m.phase(2)
    assert len(p1) == 3 and all(t.kind == "first" for t in p1)
    assert sum(t.kind == "second" for t in p2) == 3 and sum(t.kind == "novel" for t in p2) == 2
    novel_units = {t.unit_id for t in p2 if t.kind == "novel"}
    assert novel_units.isdisjoint({t.unit_id for t in p1})
    for cid, (k1, k2) in m.classes.items():
        t1 = next(t for t in p1 if t.task_key == k1); t2 = next(t for t in p2 if t.task_key == k2)
        assert t1.span != t2.span and t1.class_id == t2.class_id == cid
    assert m.counts["equivalent"] == 6 and "ineligible-class" in m.counts
    assert StreamManifest.from_dict(m.to_dict()) == m

def test_compose_is_deterministic_and_seed_sensitive():
    units, validated = _world()
    a = compose(units, validated, seed=0, C=3, n_nov=2)
    b = compose(units, validated, seed=0, C=3, n_nov=2)
    c = compose(units, validated, seed=1, C=3, n_nov=2)
    assert a.stream_hash == b.stream_hash and [t.task_key for t in a.tasks] == [t.task_key for t in b.tasks]
    assert a.stream_hash != c.stream_hash

def test_not_enough_classes_raises():
    units, validated = _world(3)
    with pytest.raises(NotEnoughClasses):
        compose(units, validated, seed=0, C=5, n_nov=1)

def test_prefers_non_timeout_mutants_for_m1():
    units, validated = _world(2)
    u = units[0]
    validated[u.unit_id] = [_mut(u, "ARITH", 2, "a", timeout=True), _mut(u, "ARITH", 3, "b"), _mut(u, "ARITH", 4, "c")]
    m = compose(units, validated, seed=3, C=1, n_nov=1)
    first = m.phase(1)[0]
    assert first.kills_by_timeout is False
```

- [ ] **Step 2: Run → fails. Implement `crucible/stream/compose.py`**

```python
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field

from .mutants import Mutant, Span
from .units import Unit, sha256_text
from .validate import Validation


class NotEnoughClasses(RuntimeError):
    pass


def class_id(unit_id: str, family: str) -> str:
    return f"{unit_id}|{family}"


@dataclass(frozen=True)
class TaskSpec:
    task_key: str
    unit_id: str
    family: str
    class_id: str
    phase: int
    kind: str
    span: Span
    kills_by_timeout: bool
    n_killing_visible: int

    def to_dict(self) -> dict:
        d = asdict(self); d["span"] = [list(self.span[0]), list(self.span[1])]; return d

    @classmethod
    def from_dict(cls, d: dict) -> "TaskSpec":
        d = dict(d); d["span"] = (tuple(d["span"][0]), tuple(d["span"][1])); return cls(**d)


@dataclass(frozen=True)
class StreamManifest:
    stream_hash: str
    seed: int
    C: int
    n_nov: int
    rung: str
    unit_ids: tuple[str, ...]
    tasks: tuple[TaskSpec, ...]
    classes: dict[str, tuple[str, str]]
    dropped: tuple[tuple[str, str], ...]
    counts: dict[str, int] = field(default_factory=dict)

    def phase(self, n: int) -> list[TaskSpec]:
        return [t for t in self.tasks if t.phase == n]

    def to_dict(self) -> dict:
        return {"stream_hash": self.stream_hash, "seed": self.seed, "C": self.C, "n_nov": self.n_nov,
                "rung": self.rung, "unit_ids": list(self.unit_ids), "tasks": [t.to_dict() for t in self.tasks],
                "classes": {k: list(v) for k, v in self.classes.items()},
                "dropped": [list(x) for x in self.dropped], "counts": dict(self.counts)}

    @classmethod
    def from_dict(cls, d: dict) -> "StreamManifest":
        return cls(d["stream_hash"], d["seed"], d["C"], d["n_nov"], d["rung"], tuple(d["unit_ids"]),
                   tuple(TaskSpec.from_dict(t) for t in d["tasks"]),
                   {k: (v[0], v[1]) for k, v in d["classes"].items()},
                   tuple(tuple(x) for x in d["dropped"]), dict(d["counts"]))


def _task(m: Mutant, v: Validation, phase: int, kind: str) -> TaskSpec:
    return TaskSpec(m.key, m.unit_id, m.family, class_id(m.unit_id, m.family), phase, kind, m.span,
                    v.kills_by_timeout, v.n_killing_visible)


def _pick_pair(pairs: list[tuple[Mutant, Validation]], rng: random.Random):
    pool = list(pairs)
    rng.shuffle(pool)
    pool.sort(key=lambda mv: mv[1].kills_by_timeout)        # stable: non-timeout first
    m1 = pool[0]
    for cand in pool[1:]:
        if cand[0].span != m1[0].span:
            return m1, cand
    return None


def compose(units: list[Unit], validated: dict[str, list[tuple[Mutant, Validation]]], *,
            seed: int, C: int, n_nov: int, rung: str = "base") -> StreamManifest:
    rng = random.Random(f"{seed}:compose")
    counts: dict[str, int] = {}
    dropped: list[tuple[str, str]] = []
    valid_by_unit: dict[str, list[tuple[Mutant, Validation]]] = {}
    for u in units:
        keep = []
        for m, v in validated.get(u.unit_id, []):
            if v.valid:
                keep.append((m, v))
            else:
                counts[v.reason] = counts.get(v.reason, 0) + 1
        if keep:
            valid_by_unit[u.unit_id] = keep
        else:
            dropped.append((u.unit_id, "unit-no-valid")); counts["unit-no-valid"] = counts.get("unit-no-valid", 0) + 1
    candidates = sorted(valid_by_unit)
    rng.shuffle(candidates)
    if len(candidates) < n_nov + 1:
        raise NotEnoughClasses(f"only {len(candidates)} units with valid mutants; need n_nov={n_nov} plus class units")
    novel_units, class_units = candidates[:n_nov], candidates[n_nov:]

    classes: dict[str, tuple[str, str]] = {}
    p1, p2 = [], []
    for uid in class_units:
        by_fam: dict[str, list] = {}
        for m, v in valid_by_unit[uid]:
            by_fam.setdefault(m.family, []).append((m, v))
        for fam in sorted(by_fam):
            if len(classes) >= C:
                break
            pair = _pick_pair(by_fam[fam], rng) if len({m.span for m, _ in by_fam[fam]}) >= 2 else None
            if pair is None:
                dropped.append((class_id(uid, fam), "ineligible-class")); counts["ineligible-class"] = counts.get("ineligible-class", 0) + 1
                continue
            (m1, v1), (m2, v2) = pair
            classes[class_id(uid, fam)] = (m1.key, m2.key)
            p1.append(_task(m1, v1, 1, "first")); p2.append(_task(m2, v2, 2, "second"))
    counts.setdefault("ineligible-class", 0)
    if len(classes) < C:
        raise NotEnoughClasses(f"eligible classes {len(classes)} < C={C}")
    for uid in novel_units:
        pool = list(valid_by_unit[uid]); rng.shuffle(pool); pool.sort(key=lambda mv: mv[1].kills_by_timeout)
        m, v = pool[0]
        p2.append(_task(m, v, 2, "novel"))
    random.Random(f"{seed}:phase1").shuffle(p1)
    random.Random(f"{seed}:phase2").shuffle(p2)
    tasks = tuple(p1 + p2)
    unit_ids = tuple(sorted({t.unit_id for t in tasks}))
    src_hashes = sorted(u.src_hash for u in units if u.unit_id in unit_ids)
    h = sha256_text(json.dumps({"seed": seed, "C": C, "n_nov": n_nov, "rung": rung, "units": src_hashes,
                                "tasks": [(t.task_key, t.phase, t.kind) for t in tasks]}, sort_keys=True))
    counts.update({"eligible_classes": len(classes), "valid_mutants": sum(len(v) for v in valid_by_unit.values())})
    return StreamManifest(h, seed, C, n_nov, rung, unit_ids, tasks, classes, tuple(dropped), counts)
```

- [ ] **Step 3: Run → `4 passed`. Mutation check: in `_pick_pair`, return `(m1, pool[1])` without the span check; purge; expect `test_compose_shapes_and_invariants` FAILS (spans equal) — if it passes by luck of data, the fixture has same-span pairs; adjust `_world` so ARITH has two mutants on the same line plus one on another, rerun; restore; green. Commit** — `git add crucible/stream/compose.py tests/stream/test_compose.py && git commit -m "feat(stream): compose phases/novel into a content-hashed manifest"`

---

### Task 14: Stream directory store (`stream/store.py`)

**Files:**
- Create: `crucible/stream/store.py`, `tests/stream/test_store.py`

**Interfaces:**
- Produces:
  ```python
  def stream_dir(root: Path, manifest: StreamManifest) -> Path       # root / manifest.stream_hash[:12]
  def write_stream(root: Path, manifest, units: list[Unit], mutants: dict[str, Mutant],
                   validations: list[Validation]) -> Path
  def read_manifest(d: Path) -> StreamManifest
  def read_unit(d: Path, unit_id: str) -> Unit
  def read_mutant(d: Path, key: str) -> Mutant
  def read_validations(d: Path) -> list[Validation]
  ```
- Layout: `manifest.json`; `units/<module_name>/{module.py,test_visible.py,test_hidden.py,unit.json}`; `mutants/<key>.json`; `validations.jsonl` (all, including invalid — provenance).

- [ ] **Step 1: Failing test**

`tests/stream/test_store.py`:
```python
from pathlib import Path
from crucible.stream import store
from crucible.stream.compose import StreamManifest, TaskSpec
from crucible.stream.units import Unit, sha256_text
from crucible.stream.mutants import Mutant
from crucible.stream.validate import Validation

def test_write_and_read_round_trip(tmp_path: Path):
    u = Unit("X/0", "unit_x_0", "f", "def f():\n    return 1\n", "v", "h", sha256_text("s"), 1, 1, ())
    m = Mutant("X/0", "k1", "Op", 0, "ARITH", ((1, 1), (1, 2)), "def f():\n    return 2\n", "d")
    v = Validation("k1", True, "killed-visible", False, 1, ("test_v0",))
    t = TaskSpec("k1", "X/0", "ARITH", "X/0|ARITH", 1, "first", ((1, 1), (1, 2)), False, 1)
    man = StreamManifest("abc123def456789", 0, 1, 0, "base", ("X/0",), (t,), {"X/0|ARITH": ("k1", "k1")}, (), {"x": 1})
    d = store.write_stream(tmp_path, man, [u], {"k1": m}, [v])
    assert d == tmp_path / "abc123def456"
    assert store.read_manifest(d) == man and store.read_unit(d, "X/0") == u
    assert store.read_mutant(d, "k1") == m and store.read_validations(d) == [v]
    assert (d / "units" / "unit_x_0" / "module.py").read_text() == u.module_src
```

- [ ] **Step 2: Run → fails. Implement `crucible/stream/store.py`**

```python
from __future__ import annotations

import json
from pathlib import Path

from .compose import StreamManifest
from .mutants import Mutant
from .units import Unit, module_name_for
from .validate import Validation


def stream_dir(root: Path, manifest: StreamManifest) -> Path:
    return Path(root) / manifest.stream_hash[:12]


def write_stream(root: Path, manifest: StreamManifest, units: list[Unit], mutants: dict[str, Mutant],
                 validations: list[Validation]) -> Path:
    d = stream_dir(root, manifest)
    (d / "units").mkdir(parents=True, exist_ok=True)
    (d / "mutants").mkdir(exist_ok=True)
    (d / "manifest.json").write_text(json.dumps(manifest.to_dict(), indent=1, sort_keys=True))
    for u in units:
        ud = d / "units" / u.module_name
        ud.mkdir(exist_ok=True)
        (ud / "module.py").write_text(u.module_src)
        (ud / "test_visible.py").write_text(u.visible_test_src)
        (ud / "test_hidden.py").write_text(u.hidden_test_src)
        (ud / "unit.json").write_text(json.dumps(u.to_dict(), sort_keys=True))
    for key, m in mutants.items():
        (d / "mutants" / f"{key}.json").write_text(json.dumps(m.to_dict(), sort_keys=True))
    with open(d / "validations.jsonl", "w") as fh:
        for v in validations:
            fh.write(json.dumps(v.to_dict(), sort_keys=True) + "\n")
    return d


def read_manifest(d: Path) -> StreamManifest:
    return StreamManifest.from_dict(json.loads((Path(d) / "manifest.json").read_text()))


def read_unit(d: Path, unit_id: str) -> Unit:
    return Unit.from_dict(json.loads((Path(d) / "units" / module_name_for(unit_id) / "unit.json").read_text()))


def read_mutant(d: Path, key: str) -> Mutant:
    return Mutant.from_dict(json.loads((Path(d) / "mutants" / f"{key}.json").read_text()))


def read_validations(d: Path) -> list[Validation]:
    with open(Path(d) / "validations.jsonl") as fh:
        return [Validation.from_dict(json.loads(line)) for line in fh if line.strip()]
```

- [ ] **Step 3: Run → `1 passed`. Commit** — `git add crucible/stream/store.py tests/stream/test_store.py && git commit -m "feat(stream): on-disk stream store"`

---

### Task 15: Structural pre-checks §4.8.1–3 (`stream/precheck.py`)

**Files:**
- Create: `crucible/stream/precheck.py`, `tests/stream/test_precheck.py`

**Interfaces:**
- Produces:
  ```python
  @dataclass(frozen=True)
  class Check: name: str; passed: bool; detail: str
  @dataclass(frozen=True)
  class PrecheckReport:
      checks: tuple[Check, ...]
      @property ok(self) -> bool
      def to_dict(self) -> dict
  def precheck(manifest: StreamManifest, units_by_id: dict[str, Unit]) -> PrecheckReport
  def two_proportion_band(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float, float]   # (p1-p2, 2*SE, passes?)
  def mean_band(xs: list[float], ys: list[float]) -> tuple[float, float, bool]
  ```
- Checks (names fixed; every one appears in the report, passed or not):
  1. `family-distribution-identical` — multiset of families for `first` tasks == for `second` tasks.
  2. `killing-count-band` — mean `n_killing_visible` of first vs second within 2·SE (Welch).
  3. `unit-length-identical` — for each class, the unit is the same ⇒ assert multiset of unit src lengths equal.
  4. `timeout-rate-band` — `kills_by_timeout` rate first vs second within 2·SE (two-proportion, pooled).
  5. `novel-disjoint` — novel unit ids ∩ first-task unit ids == ∅.
  6. `distinct-sites` — every class: span(m1) != span(m2).
  7. `counts-named` — `manifest.counts` contains keys `hidden-only equivalent infra syntax ineligible-class unit-no-valid eligible_classes valid_mutants` (None-vs-zero: absent is a failure; zero is fine).

- [ ] **Step 1: Failing tests**

`tests/stream/test_precheck.py`:
```python
from dataclasses import replace
from crucible.stream.precheck import precheck, two_proportion_band, mean_band
from tests.stream.test_compose import _world
from crucible.stream.compose import compose

def _full_counts(m):
    c = dict(m.counts)
    for k in ("hidden-only", "equivalent", "infra", "syntax", "ineligible-class", "unit-no-valid", "eligible_classes", "valid_mutants"):
        c.setdefault(k, 0)
    return replace(m, counts=c)

def test_precheck_passes_on_a_well_formed_stream():
    units, validated = _world(8)
    m = _full_counts(compose(units, validated, seed=0, C=4, n_nov=2))
    rep = precheck(m, {u.unit_id: u for u in units})
    assert rep.ok, [c for c in rep.checks if not c.passed]
    assert {c.name for c in rep.checks} == {"family-distribution-identical", "killing-count-band", "unit-length-identical",
                                            "timeout-rate-band", "novel-disjoint", "distinct-sites", "counts-named"}

def test_precheck_fails_when_a_second_task_shares_the_site():
    units, validated = _world(8)
    m = _full_counts(compose(units, validated, seed=0, C=4, n_nov=2))
    p1 = m.phase(1)[0]
    bad_tasks = tuple(replace(t, span=p1.span) if (t.phase == 2 and t.class_id == p1.class_id) else t for t in m.tasks)
    rep = precheck(replace(m, tasks=bad_tasks), {u.unit_id: u for u in units})
    assert not rep.ok and not next(c for c in rep.checks if c.name == "distinct-sites").passed

def test_precheck_fails_when_counts_missing():
    units, validated = _world(8)
    m = compose(units, validated, seed=0, C=4, n_nov=2)
    m2 = replace(m, counts={k: v for k, v in m.counts.items() if k != "equivalent"})
    rep = precheck(m2, {u.unit_id: u for u in units})
    assert not next(c for c in rep.checks if c.name == "counts-named").passed

def test_bands():
    d, band, ok = two_proportion_band(10, 100, 12, 100)
    assert ok and abs(d) < band
    d, band, ok = two_proportion_band(10, 100, 60, 100)
    assert not ok
    assert mean_band([1, 2, 3, 4], [1, 2, 3, 4])[2] and not mean_band([1, 1, 1, 1, 1], [9, 9, 9, 9, 9])[2]
```

- [ ] **Step 2: Run → fails. Implement `crucible/stream/precheck.py`**

```python
from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass

from .compose import StreamManifest
from .units import Unit

REQUIRED_COUNTS = ("hidden-only", "equivalent", "infra", "syntax", "ineligible-class", "unit-no-valid",
                   "eligible_classes", "valid_mutants")


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class PrecheckReport:
    checks: tuple[Check, ...]

    @property
    def ok(self) -> bool:
        return all(c.passed for c in self.checks)

    def to_dict(self) -> dict:
        return {"ok": self.ok, "checks": [asdict(c) for c in self.checks]}


def two_proportion_band(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float, bool]:
    if n1 == 0 or n2 == 0:
        return (float("nan"), float("nan"), False)
    p1, p2, p = k1 / n1, k2 / n2, (k1 + k2) / (n1 + n2)
    se = math.sqrt(max(p * (1 - p), 1e-12) * (1 / n1 + 1 / n2))
    d = p1 - p2
    return (d, 2 * se, abs(d) <= 2 * se)


def mean_band(xs: list[float], ys: list[float]) -> tuple[float, float, bool]:
    if len(xs) < 2 or len(ys) < 2:
        return (float("nan"), float("nan"), False)
    def var(v):
        mu = sum(v) / len(v); return sum((x - mu) ** 2 for x in v) / (len(v) - 1)
    d = sum(xs) / len(xs) - sum(ys) / len(ys)
    se = math.sqrt(var(xs) / len(xs) + var(ys) / len(ys))
    return (d, 2 * se, abs(d) <= 2 * se + 1e-12)


def precheck(manifest: StreamManifest, units_by_id: dict[str, Unit]) -> PrecheckReport:
    first = [t for t in manifest.tasks if t.kind == "first"]
    second = [t for t in manifest.tasks if t.kind == "second"]
    novel = [t for t in manifest.tasks if t.kind == "novel"]
    checks: list[Check] = []

    fam1, fam2 = Counter(t.family for t in first), Counter(t.family for t in second)
    checks.append(Check("family-distribution-identical", fam1 == fam2, f"{dict(fam1)} vs {dict(fam2)}"))

    d, band, ok = mean_band([t.n_killing_visible for t in first], [t.n_killing_visible for t in second])
    checks.append(Check("killing-count-band", ok, f"diff={d:.3f} band={band:.3f}"))

    len1 = Counter(len(units_by_id[t.unit_id].module_src) for t in first)
    len2 = Counter(len(units_by_id[t.unit_id].module_src) for t in second)
    checks.append(Check("unit-length-identical", len1 == len2, "multiset of unit lengths"))

    d, band, ok = two_proportion_band(sum(t.kills_by_timeout for t in first), len(first),
                                      sum(t.kills_by_timeout for t in second), len(second))
    checks.append(Check("timeout-rate-band", ok, f"diff={d:.3f} band={band:.3f}"))

    inter = {t.unit_id for t in novel} & {t.unit_id for t in first}
    checks.append(Check("novel-disjoint", not inter, f"overlap={sorted(inter)}"))

    by_key = {t.task_key: t for t in manifest.tasks}
    bad = [cid for cid, (k1, k2) in manifest.classes.items()
           if k1 not in by_key or k2 not in by_key or by_key[k1].span == by_key[k2].span]
    checks.append(Check("distinct-sites", not bad, f"same-site classes={bad[:5]}"))

    missing = [k for k in REQUIRED_COUNTS if k not in manifest.counts]
    checks.append(Check("counts-named", not missing, f"missing={missing}"))
    return PrecheckReport(tuple(checks))
```

- [ ] **Step 3: Run → `4 passed`. Also make `compose` emit every `REQUIRED_COUNTS` key (zero when unobserved): in `compose`, before building the manifest add `for k in ("hidden-only","equivalent","infra","syntax","unit-no-valid"): counts.setdefault(k, 0)`; then the `_full_counts` helper in the tests becomes a no-op — keep it anyway. Mutation check: in `precheck`, make `distinct-sites` always pass; purge; expect the second test FAILS; restore; green. Commit** — `git add crucible/stream/precheck.py tests/stream/test_precheck.py crucible/stream/compose.py && git commit -m "feat(stream): structural pre-checks 4.8.1-3"`

---

### Task 16: Build pipeline + CLI (`stream/pipeline.py`, `cli.py`)

**Files:**
- Create: `crucible/stream/pipeline.py`, `crucible/cli.py`, `tests/stream/test_pipeline.py`, `tests/test_cli.py`
- (add `crucible/stream/pipeline.py` to the file-structure list above)

**Interfaces:**
- Consumes: everything in `stream/` and `sandbox/`.
- Produces:
  ```python
  @dataclass(frozen=True)
  class BuildConfig:
      seed: int = 0; C: int = 200; n_nov: int = 50; per_family: int = 6; max_hidden: int = 100
      limit_units: int | None = None; jobs: int = 8; rung: str = "base"; sources: tuple[str, ...] = ("humaneval", "mbpp")
  def build_stream(cfg: BuildConfig, out_root: Path, *, recs: list[dict] | None = None, log=print) -> Path
      # recs=None → load from EvalPlus; returns the written stream dir
  def smoke(stream_d: Path, n: int = 30, *, log=print) -> dict   # applies n tasks' mutants, runs visible suite, asserts killed; returns counts
  # cli.py: `crucible stream build [--seed --C --n-nov --per-family --max-hidden --limit-units --jobs --rung --out]`
  #         `crucible stream precheck <dir>`   (exit 0 iff ok; prints report JSON)
  #         `crucible stream smoke <dir> [--n 30]`
  ```
- Invariants: `build_stream` is deterministic for fixed `cfg` and inputs (same `stream_hash`); `limit_units` samples units with `random.Random(f"{seed}:units")` — never "the first N"; the build log names counts of units built/dropped and mutants valid/invalid by reason.

- [ ] **Step 1: Failing tests**

`tests/stream/test_pipeline.py`:
```python
import gzip, json, pathlib
from crucible.stream.pipeline import build_stream, BuildConfig, smoke
from crucible.stream import store

FIX = pathlib.Path(__file__).resolve().parents[1] / "fixtures"
def _recs():
    out = []
    for n in ("mini_humaneval.jsonl.gz", "mini_mbpp.jsonl.gz"):
        with gzip.open(FIX / n, "rt") as fh:
            out += [json.loads(l) for l in fh]
    return out

def test_build_stream_is_deterministic_and_structural_prechecks_pass(tmp_path):
    # n_nov=0 so all three fixture units are class units (is_pos has BOOL+SDL, add2 has ARITH ⇒ C=2 is reachable).
    cfg = BuildConfig(seed=0, C=2, n_nov=0, per_family=3, max_hidden=2, jobs=2)
    d1 = build_stream(cfg, tmp_path / "a", recs=_recs(), log=lambda *a: None)
    d2 = build_stream(cfg, tmp_path / "b", recs=_recs(), log=lambda *a: None)
    m1, m2 = store.read_manifest(d1), store.read_manifest(d2)
    assert m1.stream_hash == m2.stream_hash and len(m1.tasks) == 4
    from crucible.stream.precheck import precheck
    units = {u: store.read_unit(d1, u) for u in m1.unit_ids}
    rep = precheck(m1, units)
    by = {c.name: c for c in rep.checks}
    # The statistical bands are meaningless at n=2; the structural checks must hold at any size.
    for name in ("family-distribution-identical", "novel-disjoint", "distinct-sites", "counts-named"):
        assert by[name].passed, by[name]

def test_smoke_reports_all_killed(tmp_path):
    cfg = BuildConfig(seed=0, C=2, n_nov=0, per_family=3, max_hidden=2, jobs=2)
    d = build_stream(cfg, tmp_path, recs=_recs(), log=lambda *a: None)
    res = smoke(d, n=3, log=lambda *a: None)
    assert res["ran"] == 3 and res["killed"] == 3 and res["infra"] == 0
```

`tests/test_cli.py`:
```python
import subprocess, sys
def test_cli_help_runs():
    out = subprocess.run([sys.executable, "-m", "crucible.cli", "stream", "--help"], capture_output=True, text=True)
    assert out.returncode == 0 and "build" in out.stdout and "precheck" in out.stdout
```

- [ ] **Step 2: Run → fails. Implement `crucible/stream/pipeline.py`**

```python
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from ..sandbox.runner import run_tests
from . import evalplus, store
from .build import build_units
from .compose import compose
from .families import all_operator_names
from .mutants import enumerate_specs, make_mutant, sample_specs
from .validate import validate_many


@dataclass(frozen=True)
class BuildConfig:
    seed: int = 0
    C: int = 200
    n_nov: int = 50
    per_family: int = 6
    max_hidden: int = 100
    limit_units: int | None = None
    jobs: int = 8
    rung: str = "base"
    sources: tuple[str, ...] = ("humaneval", "mbpp")


def _load_recs(cfg: BuildConfig) -> list[dict]:
    recs: list[dict] = []
    for s in cfg.sources:
        recs += evalplus.load(s)
    return recs


def _limit(recs: list[dict], cfg: BuildConfig) -> list[dict]:
    if cfg.limit_units is None or cfg.limit_units >= len(recs):
        return recs
    rng = random.Random(f"{cfg.seed}:units")
    idx = sorted(rng.sample(range(len(recs)), cfg.limit_units))
    return [recs[i] for i in idx]


def build_stream(cfg: BuildConfig, out_root: Path, *, recs: list[dict] | None = None, log=print) -> Path:
    recs = _limit(recs if recs is not None else _load_recs(cfg), cfg)
    units, dropped = build_units(recs, seed=cfg.seed, max_hidden=cfg.max_hidden, jobs=cfg.jobs)
    log(f"units built={len(units)} dropped={len(dropped)}")
    ops = all_operator_names()
    validated, mutants, validations = {}, {}, []
    for u in units:
        rng = random.Random(f"{cfg.seed}:{u.unit_id}:specs")
        specs = sample_specs(enumerate_specs(u.module_src, ops), per_family=cfg.per_family, rng=rng)
        ms = [m for m in (make_mutant(u, s) for s in specs) if m is not None]
        vs = validate_many(u, ms, jobs=cfg.jobs)
        validated[u.unit_id] = list(zip(ms, vs))
        mutants.update({m.key: m for m in ms}); validations += vs
        log(f"{u.unit_id}: specs={len(specs)} mutants={len(ms)} valid={sum(v.valid for v in vs)}")
    manifest = compose(units, validated, seed=cfg.seed, C=cfg.C, n_nov=cfg.n_nov, rung=cfg.rung)
    d = store.write_stream(out_root, manifest, units, mutants, validations)
    log(f"stream {manifest.stream_hash[:12]} tasks={len(manifest.tasks)} counts={manifest.counts} -> {d}")
    return d


def smoke(stream_d: Path, n: int = 30, *, log=print) -> dict:
    man = store.read_manifest(stream_d)
    res = {"ran": 0, "killed": 0, "not_killed": 0, "infra": 0}
    for t in man.tasks[:n]:
        u, m = store.read_unit(stream_d, t.unit_id), store.read_mutant(stream_d, t.task_key)
        r = run_tests(u.module_name, m.mutated_src, u.visible_test_src)
        res["ran"] += 1
        key = "infra" if r.infra_error else ("killed" if r.killed else "not_killed")
        res[key] += 1
        log(f"{t.task_key[:10]} {t.unit_id} {t.family} phase={t.phase} -> {key}")
    return res
```

`crucible/cli.py`:
```python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="crucible")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stream").add_subparsers(dest="scmd", required=True)
    b = s.add_parser("build")
    b.add_argument("--seed", type=int, default=0); b.add_argument("--C", type=int, default=200)
    b.add_argument("--n-nov", type=int, default=50); b.add_argument("--per-family", type=int, default=6)
    b.add_argument("--max-hidden", type=int, default=100); b.add_argument("--limit-units", type=int, default=None)
    b.add_argument("--jobs", type=int, default=8); b.add_argument("--rung", default="base")
    b.add_argument("--out", type=Path, default=Path("streams"))
    pc = s.add_parser("precheck"); pc.add_argument("dir", type=Path)
    sm = s.add_parser("smoke"); sm.add_argument("dir", type=Path); sm.add_argument("--n", type=int, default=30)
    a = p.parse_args(argv)
    if a.scmd == "build":
        from crucible.stream.pipeline import BuildConfig, build_stream
        cfg = BuildConfig(a.seed, a.C, a.n_nov, a.per_family, a.max_hidden, a.limit_units, a.jobs, a.rung)
        print(build_stream(cfg, a.out)); return 0
    if a.scmd == "precheck":
        from crucible.stream import store
        from crucible.stream.precheck import precheck
        man = store.read_manifest(a.dir)
        rep = precheck(man, {u: store.read_unit(a.dir, u) for u in man.unit_ids})
        print(json.dumps(rep.to_dict(), indent=1)); return 0 if rep.ok else 1
    if a.scmd == "smoke":
        from crucible.stream.pipeline import smoke
        print(json.dumps(smoke(a.dir, a.n))); return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Run → `3 passed`. Commit** — `git add crucible/stream/pipeline.py crucible/cli.py tests/stream/test_pipeline.py tests/test_cli.py && git commit -m "feat: stream build pipeline and CLI"`

- [ ] **Step 4: Exit-criterion run (real data, smoke scale).** Build twice with a 40-unit seeded subset and compare hashes; precheck; smoke 30:
```bash
.venv/bin/crucible stream build --seed 0 --C 20 --n-nov 5 --limit-units 40 --jobs 8 --out streams/smoke-a
.venv/bin/crucible stream build --seed 0 --C 20 --n-nov 5 --limit-units 40 --jobs 8 --out streams/smoke-b
ls streams/smoke-a streams/smoke-b          # same 12-char dir name ⇒ same stream_hash
.venv/bin/crucible stream precheck streams/smoke-a/*/
.venv/bin/crucible stream smoke streams/smoke-a/*/ --n 30
```
Expected: identical hashes; precheck `ok: true`; smoke `{"ran": 30, "killed": 30, "not_killed": 0, "infra": 0}`. If the hashes differ, diff the two `validations.jsonl` — a mutant flipping between `killed-visible` (by timeout) and not is the flake class; record it in CARRIED-DEBT and, if it recurs, raise `per_test_timeout_s` for validation only to 10 s (record the amendment). If `NotEnoughClasses` is raised with 40 units, raise `--limit-units` to 80 — this also tells us the eligible-class yield per unit for sizing the full build (record the number).

- [ ] **Step 5: Full build (background, OS-detached — it takes tens of minutes).**
```bash
mkdir -p runs && setsid nohup bash -c '.venv/bin/crucible stream build --seed 0 --C 200 --n-nov 50 --jobs 12 --out streams/full > runs/stream-build.log 2>&1; touch runs/stream-build.DONE' > /dev/null 2>&1 &
echo $! > runs/stream-build.pid
```
Then watch `runs/stream-build.log`; when `.DONE` exists run `precheck` and `smoke --n 30` on the produced dir, and record `stream_hash`, `counts`, wall time, and the eligible-class yield in `docs/CARRIED-DEBT.md` (S1 → Settled). Commit the manifest? **No** — `streams/` is git-ignored (size); record the hash + the exact command in CARRIED-DEBT; the manifest is reproducible by construction (that is what Step 4 verified).

---

### Task 17: Serving + LoRA environment spike, served-identity assertion (`proposer/identity.py`)

This task is exploratory with a **timebox of one working day** for the torch/vLLM part. Its code deliverable is small (`identity.py`); its main deliverable is `docs/findings/S1-serving.md` and the fallback decisions it records. It is independent of Tasks 5–16 and may run in parallel with them.

**Files:**
- Create: `crucible/proposer/identity.py`, `tests/proposer/test_identity.py`, `docs/findings/S1-serving.md`, `scripts/lora_attach_smoke.py`
- Modify: `pyproject.toml` (add optional-dependency group `serve` = torch/vllm/transformers/peft as resolved), `THIRD_PARTY.md` (rows for torch BSD-3, vLLM Apache-2.0 or llama.cpp MIT, transformers Apache-2.0, peft Apache-2.0, bitsandbytes MIT, the model licenses), `docs/CARRIED-DEBT.md`

**Interfaces:**
- Produces:
  ```python
  @dataclass(frozen=True)
  class ServedIdentity: kind: str; model: str; extra: dict    # kind ∈ {"vllm","llamacpp"}
  class IdentityMismatch(RuntimeError)
  def probe(base_url: str, timeout_s: float = 5.0) -> ServedIdentity
      # vLLM: GET {base}/v1/models → data[0].id ; llama.cpp: GET {base}/props → default_generation_settings.model (or "model_path")
  def assert_identity(base_url: str, expected_model: str) -> ServedIdentity   # raises IdentityMismatch unless expected_model == model or is a suffix of it
  ```

- [ ] **Step 1: Failing tests for `identity.py` (fake servers, no GPU)**

`tests/proposer/test_identity.py`:
```python
import json, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import pytest
from crucible.proposer.identity import probe, assert_identity, IdentityMismatch

def _serve(routes):
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            body = routes.get(self.path)
            self.send_response(200 if body is not None else 404); self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(json.dumps(body or {}).encode())
        def log_message(self, *a): pass
    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}"

def test_probe_vllm():
    srv, url = _serve({"/v1/models": {"data": [{"id": "Qwen/Qwen3.5-2B"}]}})
    ident = probe(url)
    assert ident.kind == "vllm" and ident.model == "Qwen/Qwen3.5-2B"
    assert assert_identity(url, "Qwen/Qwen3.5-2B").model == "Qwen/Qwen3.5-2B"
    with pytest.raises(IdentityMismatch):
        assert_identity(url, "Qwen/Qwen3.5-9B")
    srv.shutdown()

def test_probe_llamacpp():
    srv, url = _serve({"/props": {"model_path": "/models/Qwen3.5-2B-Q6_K.gguf", "default_generation_settings": {"model": "/models/Qwen3.5-2B-Q6_K.gguf"}}})
    ident = probe(url)
    assert ident.kind == "llamacpp" and ident.model.endswith("Qwen3.5-2B-Q6_K.gguf")
    assert assert_identity(url, "Qwen3.5-2B-Q6_K.gguf").kind == "llamacpp"
    srv.shutdown()

def test_probe_unknown_server_raises():
    srv, url = _serve({})
    with pytest.raises(IdentityMismatch):
        probe(url)
    srv.shutdown()
```

- [ ] **Step 2: Run → fails. Implement `crucible/proposer/identity.py`**

```python
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass


class IdentityMismatch(RuntimeError):
    pass


@dataclass(frozen=True)
class ServedIdentity:
    kind: str
    model: str
    extra: dict


def _get(url: str, timeout_s: float):
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as r:
            return json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, OSError):
        return None


def probe(base_url: str, timeout_s: float = 5.0) -> ServedIdentity:
    base = base_url.rstrip("/")
    # llama.cpp ALSO serves /v1/models (OpenAI-compatible), so its own /props must be checked first.
    p = _get(f"{base}/props", timeout_s)
    if p and (p.get("model_path") or p.get("default_generation_settings", {}).get("model")):
        model = p.get("model_path") or p["default_generation_settings"]["model"]
        return ServedIdentity("llamacpp", model, {k: p[k] for k in ("total_slots",) if k in p})
    v = _get(f"{base}/v1/models", timeout_s)
    if v and v.get("data"):
        return ServedIdentity("vllm", v["data"][0]["id"], {"n_models": len(v["data"])})
    raise IdentityMismatch(f"no recognisable server at {base_url}")


def assert_identity(base_url: str, expected_model: str) -> ServedIdentity:
    ident = probe(base_url)
    if not (ident.model == expected_model or ident.model.endswith(expected_model)):
        raise IdentityMismatch(f"served {ident.model!r} at {base_url}, expected {expected_model!r}")
    return ident
```

- [ ] **Step 3: Run → `3 passed`. Commit** — `git add crucible/proposer/identity.py tests/proposer/test_identity.py && git commit -m "feat(proposer): served-model identity probe for vLLM and llama.cpp"`

- [ ] **Step 4: torch for sm_120 (timebox starts).** Stop Ollama first (`systemctl --user stop ollama 2>/dev/null || pkill -f 'ollama serve'`; verify `nvidia-smi` shows < 1 GB used). Then, in the project venv:
```bash
# Pick the current CUDA ≥12.8 wheel index from https://pytorch.org/get-started/locally/ (cu128 or newer). Example:
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_capability()); import torch as t; a=t.randn(2048,2048,device='cuda'); print((a@a).sum().item())"
```
Expected: `True`, capability `(12, 0)`, a finite number. If `is_available()` is False or the matmul errors with "no kernel image": try the next-newer CUDA index (cu129/cu130) once; if still failing, record and fall back to llama.cpp for serving (LoRA *training* then needs a CPU-offloaded torch or a rental — record as S3 risk).

- [ ] **Step 5: vLLM serve attempt (same timebox).**
```bash
uv pip install --python .venv/bin/python vllm
VLLM_ALLOW_RUNTIME_LORA_UPDATING=True .venv/bin/vllm serve Qwen/Qwen3.5-2B --max-model-len 8192 --gpu-memory-utilization 0.45 --enable-lora --max-lora-rank 32 --port 8001 > runs/vllm.log 2>&1 &
sleep 90; .venv/bin/python -c "from crucible.proposer.identity import assert_identity; print(assert_identity('http://127.0.0.1:8001','Qwen/Qwen3.5-2B'))"
curl -s http://127.0.0.1:8001/v1/completions -H 'Content-Type: application/json' -d '{"model":"Qwen/Qwen3.5-2B","prompt":"def add(a, b):\n    return","max_tokens":8,"n":2,"logprobs":1,"temperature":0.7,"seed":1}' | head -c 800
```
Expected: identity OK; two choices each with `logprobs`. Record: vLLM version, startup time, VRAM used (`nvidia-smi`), tokens/s on a 256-token completion (time it). If install or serve fails within the timebox ⇒ **fallback**: `cd ~/llama.cpp && git pull && cmake -B build -DGGML_CUDA=ON && cmake --build build -j16 --target llama-server`; download `unsloth/Qwen3.5-2B-GGUF` Q6_K (text-only; do **not** download the `mmproj-*` file); run `./build/bin/llama-server -m <gguf> --port 8001 -np 4 -c 8192 --lora-init-without-apply` and verify `GET /props`, `n_probs` in a completion, and `GET /lora-adapters`. Record which server is the Phase-A server in `docs/findings/S1-serving.md` **and** as the `server.kind` lens value.

- [ ] **Step 6: LoRA-attach smoke on Qwen3.5-2B (decides the proposer fallback).**
Write `scripts/lora_attach_smoke.py`:
```python
"""Can PEFT attach a LoRA to Qwen3.5-2B's language model, run fwd/bwd, save, and can the server load it?"""
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import LoraConfig, get_peft_model

name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.5-2B"
cfg = AutoConfig.from_pretrained(name)
print("config class:", type(cfg).__name__, "| architectures:", getattr(cfg, "architectures", None))
try:
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, device_map="cuda")
except Exception as e:
    print("AutoModelForCausalLM failed:", e); from transformers import AutoModelForImageTextToText
    model = AutoModelForImageTextToText.from_pretrained(name, torch_dtype=torch.bfloat16, device_map="cuda")
lin = sorted({n.split(".")[-1] for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and "vis" not in n.lower()})
print("linear leaf names:", lin)
targets = [t for t in lin if any(k in t for k in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "in_proj", "out_proj"))] or lin[:4]
print("LoRA targets:", targets)
pm = get_peft_model(model, LoraConfig(r=16, lora_alpha=16, target_modules=targets, task_type="CAUSAL_LM"))
pm.print_trainable_parameters()
tok = AutoTokenizer.from_pretrained(name)
batch = tok(["def add(a, b):\n    return a + b\n"] * 2, return_tensors="pt").to("cuda")
out = pm(**batch, labels=batch["input_ids"]); out.loss.backward()
print("loss:", float(out.loss), "| peak VRAM GB:", torch.cuda.max_memory_allocated() / 1e9)
pm.save_pretrained("runs/lora-smoke"); print("saved runs/lora-smoke")
```
Run: `uv pip install --python .venv/bin/python transformers peft accelerate bitsandbytes && .venv/bin/python scripts/lora_attach_smoke.py Qwen/Qwen3.5-2B`.
Then load the adapter into the server (vLLM: `curl -X POST :8001/v1/load_lora_adapter -d '{"lora_name":"smoke","lora_path":"runs/lora-smoke"}'`; llama.cpp needs a GGUF-converted adapter via `convert_lora_to_gguf.py` — record if that conversion works for Qwen3.5's architecture) and sample once with the adapter active.
**Decision rule (record it):** if attach + fwd/bwd + save + server-load all succeed ⇒ proposer = Qwen3.5-2B. If any step fails and cannot be fixed within the timebox ⇒ run the same script with `Qwen/Qwen2.5-Coder-1.5B-Instruct`; proposer for all small arms = Qwen2.5-Coder-1.5B-Instruct (spec §2 fallback), and write that in `docs/findings/S1-serving.md` + CARRIED-DEBT.

- [ ] **Step 7: Write `docs/findings/S1-serving.md`** with: torch version/index used and result; server chosen (vLLM version or llama.cpp commit) with startup time, VRAM, tok/s at 256 tokens for Qwen3.5-2B and the baseline Qwen3.5-9B Q6_K (or Q4_K_M if Q6 does not leave ≥ 4 GB free); LoRA attach outcome and the proposer decision; exact commands; measured free VRAM on idle desktop. Add THIRD_PARTY rows (verify each license with `gh api repos/<owner>/<repo> --jq .license.spdx_id` or the HF API and paste the output). Commit:
```bash
git add docs/findings/S1-serving.md scripts/lora_attach_smoke.py THIRD_PARTY.md pyproject.toml docs/CARRIED-DEBT.md && git commit -m "docs: S1 serving/LoRA findings and proposer decision"
```

---

## S1 exit criteria (from spec §10) — all must be true before S2's plan is written

- [ ] `uv run pytest` green, every load-bearing test mutation-checked as its task specifies.
- [ ] `crucible stream build` run twice on the same 40-unit seeded subset produced the **same `stream_hash`**.
- [ ] `crucible stream precheck` on the full build reports `ok: true` (family-identical, killing-count band, unit-length identical, timeout-rate band, novel-disjoint, distinct-sites, counts-named).
- [ ] `crucible stream smoke --n 30` on the full build: 30 ran, 30 killed, 0 infra.
- [ ] A server (vLLM or llama.cpp) serves Qwen3.5-2B with n-best + logprobs, identity-asserted; findings recorded.
- [ ] The LoRA-attach decision is recorded (proposer = Qwen3.5-2B or fallback Qwen2.5-Coder-1.5B-Instruct).
- [ ] `docs/CARRIED-DEBT.md` S1 section filled (settled / deferred-with-rulings / process lessons), including the full-build `stream_hash`, counts, eligible-class yield, and wall time.

## Plan self-review (done by the author before handoff)

- **Spec coverage:** §4.1 units (T5–T8), §4.2 mutants + families + SDL (T9–T11), validity = visible kill (T12), §4.3 classes/phases/novel (T13), §4.5 sandbox (T2–T3), §4.6 budget (T4), §4.8.1–3 (T15), §8 None-vs-zero/infra/keys/round-trips (T3, T4, T8, T12, T13, T15), §9 package layout (all), §10 S1 exit (above), §3 identity assertion (T17). §4.7 codec probe → S2 (amendment A3). §4.8.4 pilot and §4.8.5 live control → S2/S4 by design.
- **Placeholders:** none — every step has runnable content or an exact command; T17's exploratory steps carry explicit decision rules and what to record.
- **Type consistency:** `TestReport` fields (`passed/failed/timed_out/errored/wall_s/infra_error`) used identically in T3, T4, T8, T12, T16; `Mutant.key == TaskSpec.task_key` (T11/T13/T14/T16); `Validation.reason` vocabulary (`killed-visible|hidden-only|equivalent|infra|syntax`) consistent between T12, T13 (`counts`), T15 (`REQUIRED_COUNTS`); `Span` tuple shape identical in T11/T13; `module_name_for` used for unit dirs in T14 and imports in T7; `run_tests(module_name, module_src, test_src, subset=..)` signature identical everywhere.
- **Known soft spot called out for the executor:** T11's SDL position count in `test_enumerate_add_sub_and_lt_gte` was corrected in-line to 2; T15's `counts-named` relies on the `compose` tweak added in T15 Step 3 — do both.

