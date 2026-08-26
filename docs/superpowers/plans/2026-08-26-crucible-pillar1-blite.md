# Pillar-1 B-lite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run the pre-registered Pillar-1 spike: a ~120M latent execution predictor (B-lite) vs the microsoft/codeexecutor 125M token-space control, gated on paired DeLong AUROC at 2·SE over shared held-out outcomes.

**Architecture:** New subpackage `crucible/latent/`: sensorium-backed harvest runner → 1.5B-generated corpus with function-level splits and floors → fixed-vocab state serialization → B-lite (frozen jina code encoder + trained state encoder + causal latent predictor + grounded head, LeWM two-term objective) and the control fine-tune, sharing one data loader and early-stop protocol → a pure-numpy paired-DeLong evaluator that reads the test split exactly once.

**Tech Stack:** Python 3.12 (`.venv/bin/python`), torch + transformers (present), numpy (present; NO scipy/sklearn — DeLong is implemented here), sensorium (editable install from `~/workspace/sensorium`, Task 1), vLLM-served 1.5B for generation only.

**Spec:** `docs/superpowers/specs/2026-08-25-crucible-pillar1-blite-prereg.md` — read it first; §6–§8 freeze at `prereg-lock-blite` (Task 10). Licensing rules from `docs/research/05-latent-predictors-world-models.md` §Risks are binding: LeWM (MIT) and `klindtlab/lejepa-identifiability` (MIT) are liftable; **`lejepa`, TD-JEPA, I-JEPA, V-JEPA v1 (all CC-BY-NC), TRACED (unlicensed) must never be cloned, vendored, or copied from**.

## Global Constraints

- PUBLIC repo: secret-scan every staged diff before ANY push; verify origin sync after ANY push.
- Full pytest under `systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q` — so every test-time model must be TINY (d≤32, ≤2 layers); no test downloads weights (inject model factories); no test touches the GPU.
- Mutation restores from `cp` backup, never `git checkout`. Determinism everywhere: seeded, no wall-clock in library code, ranking ties broken explicitly.
- None-vs-zero: unmeasured is None + counted, never a fake number. Truncations, rejections, and dropped samples are counted and reported, never silent.
- The test split is read by NOTHING until the Task-11 gate. Loaders take an explicit `split` argument; nothing defaults to "test".
- HF downloads (jina, codeexecutor) happen in ops tasks only, revision-pinned; digests recorded in LOCK-BLITE.
- Chosen (non-derived) numbers live in ONE module `crucible/latent/config.py` as named constants with a comment citing spec §4/§5 — the lock records the file verbatim.
- Box gotchas: `.venv/bin` on PATH for vllm; detached runs >30 min via setsid + pid + .DONE marker + Monitor; exit 144 = harness kill.

---

### Task 1: sensorium install + harvest runner

**Files:**
- Create: `crucible/latent/__init__.py`, `crucible/latent/harvest.py`, `crucible/latent/config.py`
- Test: `tests/latent/__init__.py`, `tests/latent/test_harvest.py`

**Interfaces (produced):**
- `config.py` first constants: `EXEC_TIMEOUT_S = 3.0`, `EXEC_RLIMIT_AS_MB = 512`, `MAX_SNAPSHOTS = 32` (chosen, spec §4).
- `harvest.py`:
```python
@dataclass(frozen=True)
class HarvestResult:
    outcome: str                     # "return" | "exception:<TypeName>" | "timeout"
    return_repr: str | None          # repr of the return value, None unless outcome=="return"
    snapshots: tuple[Snapshot, ...]  # per-line locals states, in execution order, truncated to MAX_SNAPSHOTS
    truncated: bool                  # snapshots dropped beyond the cap OR sensorium marked truncation
    deterministic: bool              # two independent recorded runs agreed (outcome + state-sequence hash)

@dataclass(frozen=True)
class Snapshot:
    line: int
    locals: tuple[tuple[str, str, str], ...]   # (name, type_name, value_repr≤64ch) sorted by name

def harvest(function_src: str, args_literal: str, workdir: Path) -> HarvestResult
```
Mechanics (the implementer reads `~/workspace/sensorium/src/sensorium/` — cli.py, store/, record/ — before writing): write a runner script into `workdir` that defines the function and calls it with `ast.literal_eval(args_literal)`, printing nothing; execute it TWICE as `sensorium run --focus <module:f> -- <script>` subprocesses (each with `resource` rlimits set via a preexec wrapper and `timeout=EXEC_TIMEOUT_S`); read each run's SQLite store via sensorium's own store module for focused line events + captured locals + exception/return; `deterministic` = both runs agree on `(outcome, hash of the snapshot sequence)`; sensorium-marked truncated captures set `truncated=True` (never silently kept). Timeout of the subprocess → outcome "timeout", snapshots may be partial, `truncated=True`.

- [ ] **Step 0:** `.venv/bin/pip install -e /home/brice/workspace/sensorium` and a pin test `test_sensorium_importable`: `import sensorium.store` succeeds and `shutil.which("sensorium")` under `.venv/bin` is not None.
- [ ] **Step 1: failing tests** (no GPU, tiny functions, tmp workdirs):
```python
def test_harvest_clean_return(tmp_path):
    r = harvest("def f(a, b):\n    c = a + b\n    return c\n", "(2, 3)", tmp_path)
    assert r.outcome == "return" and r.return_repr == "5" and r.deterministic
    assert any("c" in [n for n, _, _ in s.locals] for s in r.snapshots)   # locals really captured

def test_harvest_exception_names_the_type(tmp_path):
    r = harvest("def f(a):\n    return a[10]\n", "([1],)", tmp_path)
    assert r.outcome == "exception:IndexError" and r.return_repr is None

def test_harvest_timeout_is_marked_not_hung(tmp_path):
    r = harvest("def f():\n    while True:\n        pass\n", "()", tmp_path)
    assert r.outcome == "timeout" and r.truncated

def test_harvest_nondeterminism_is_detected(tmp_path):
    r = harvest("import random\ndef f():\n    return random.random()\n", "()", tmp_path)
    assert r.deterministic is False
```
(The nondeterminism test may need the validator's no-import rule relaxed at HARVEST level — harvest executes what it is given; the no-import rule lives in Task 2's validator. Keep harvest permissive.)
- [ ] **Step 2: RED** (ModuleNotFoundError), **Step 3: implement**, **Step 4: GREEN** (`tests/latent/test_harvest.py`; wrap in the 4G scope — sensorium subprocesses are small), **Step 5: commit** `feat: sensorium harvest runner for the B-lite corpus (prereg §4/§5.1)`.

### Task 2: generator + validator (`crucible/latent/gen.py`)

**Interfaces:**
- Consumes: the house `VLLMProposer` pattern (see `crucible/proposer/client.py`) via an injected `proposer` with `.generate(prompt, n, seed, ...)`; Task 1's `harvest`.
- Produces:
```python
GEN_PROMPT: str        # chat prompt: ONE self-contained def f(...) ≤ 30 lines, no imports,
                       # plus a line `INPUTS: [<tuple literal>, ...]` with 3-5 argument tuples
def parse_candidate(text: str) -> tuple[str, list[str]] | None    # (function_src, args_literals) or None
def validate(function_src: str) -> str | None                     # None if OK else the rejection reason
def generate_corpus(proposer, target_functions: int, out_dir: Path, *, seed: int, log=print) -> dict
```
`validate` (ast-based, each rule mutation-testable): parses; exactly one top-level FunctionDef named `f`; REJECTS Import/ImportFrom, Exec/Eval/compile/open/`__`-attributes, Global/Nonlocal; node count ≤ 400 (`config.MAX_AST_NODES = 400`, chosen). `generate_corpus` loops: generate n=8 per call → parse → validate → for each accepted (function, input): `harvest` twice-in-one (harvest already runs twice) → keep only `deterministic and not truncated` samples with outcome in {return, exception:*} or timeout; write one JSON line per SAMPLE to `out_dir/samples.jsonl` `{fn_id (sha256/16 of src), function_src, args, outcome, return_repr, snapshots, }` and per-function record to `functions.jsonl`. **Balance guard (spec §4):** once ≥ 1000
samples are accepted, if the running binary balance exceeds `config.SKEW_LIMIT`, further
MAJORITY-class samples are rejected (functions still accepted; each drop counted in
`balance_rejected`) until balance recovers. Returns and also writes `gen_stats.json`: candidates, parse-fail, validate-fail (by reason), nondet-rejected, truncated-rejected, balance_rejected, accepted-functions, accepted-samples — every drop COUNTED (None-vs-zero).

- [ ] Tests with a `FakeProposer` (scripted texts, house pattern from tests/run) and monkeypatched `harvest` (no subprocesses): parse round-trip; each validator rule rejects its construct (one test per rule — these are the mutation pins); stats add up exactly (accepted + every rejection bucket == candidates processed); fn_id stability; the balance guard drops a majority sample and counts it (monkeypatched threshold — mutation pin).
- [ ] RED → implement → GREEN (`tests/latent/test_gen.py`) → commit `feat: corpus generator + AST validator (prereg §4)`.

### Task 3: corpus assembly, splits, floors (`crucible/latent/corpus.py`)

**Interfaces:**
```python
# config.py additions (chosen, spec §4): TARGET_FUNCTIONS = 5000; FLOOR_FUNCTIONS = 3000;
# NONDET_REJECT_KILL = 0.40; SPLIT_SEED = 0; SPLIT_FRACTIONS = (0.8, 0.1, 0.1); SKEW_LIMIT = 0.80
def assign_split(fn_id: str, seed: int) -> str        # "train"|"val"|"test" — pure hash of (seed, fn_id):
                                                      # sha256, first 8 bytes as int, < 0.8 / < 0.9 / else
def build_manifest(corpus_dir: Path) -> dict          # counts, class balance (binary + multiclass),
                                                      # split sizes BY FUNCTION AND BY SAMPLE, floors verdicts,
                                                      # sha256 of samples.jsonl — written to manifest.json
@dataclass(frozen=True)
class Sample:                                          # one (function, input) measurement
    fn_id: str; function_src: str; args: str
    outcome: str; return_repr: str | None
    snapshots: tuple                                   # Task 1's Snapshot tuples, deserialized
def load_split(corpus_dir: Path, split: str) -> list[Sample]   # explicit split arg, no default
```
Floors evaluated in `build_manifest` and stamped as `"floor_functions": "PASS"|"FAIL"`, `"nondet_rate": float, "nondet_kill": "PASS"|"FAIL"`, `"balance": ..., "skew_ok": bool` — the ops task READS these verdicts, it never recomputes them ad hoc.

- [ ] Tests: `assign_split` is deterministic, function-level (all samples of one fn share a split — construct two samples same fn_id), fractions within ±3pp on 10k synthetic ids (seeded), and CHANGES with seed (mutation pin: dropping seed from the hash); `build_manifest` on a tiny synthetic corpus: exact counts, floors verdicts flip when constants are monkeypatched tighter (mutation pin for each floor); `load_split("test")` returns only test functions; no default-split (calling without split is a TypeError — signature pin).
- [ ] RED → implement → GREEN → commit `feat: corpus manifest, hash splits, floors (prereg §4)`.

### Task 4: state serialization (`crucible/latent/state.py`)

**Interfaces:**
```python
# Fixed, corpus-independent vocabulary (leakage-proof): 256 byte tokens + specials:
# PAD=256, BOS=257, EOS=258, KEY=259, TYPE=260, VAL=261, LINE_BASE=262 (+ line bucket 0..63)
VOCAB_SIZE = 262 + 64
def encode_snapshot(s: Snapshot) -> list[int]         # [LINE_BASE+min(line,63)] then per local (name-sorted):
                                                      # KEY + utf8 bytes of name + TYPE + bytes of type_name
                                                      # + VAL + bytes of value_repr[:24]; cap 128 tokens (config)
def encode_input(args_literal: str) -> list[int]      # BOS + bytes of the literal[:96] + EOS
def encode_state_sequence(snaps, max_snapshots) -> list[list[int]]   # truncation COUNTED by caller
```
- [ ] Tests: deterministic; name-sorted order (mutation pin: two locals, assert byte order); the 24-char value truncation (pin); vocab bounds (every id < VOCAB_SIZE); cap honored.
- [ ] RED → implement → GREEN → commit `feat: fixed-vocab state serialization (prereg §5.1)`.

### Task 5: B-lite model + losses (`crucible/latent/model.py`)

**Interfaces:**
```python
# config.py additions (chosen, recorded at lock): D_MODEL=768; STATE_ENC_LAYERS=4; STATE_ENC_D=512
# (projected to 768); PRED_LAYERS=12; PRED_HEADS=12; LAMBDA_ISO=0.1; N_OUTCOME_CLASSES=3
class StateEncoder(nn.Module):     # token ids -> 768-d snapshot embedding (mean-pool + proj), ~20M
class LatentPredictor(nn.Module):  # causal transformer over [z_code, z_input, z_s1..z_st] -> per-step
                                   # predicted next-state embeddings + final hidden
class GroundedHead(nn.Module):     # final hidden -> binary logit + 3-class logits (return/exception/timeout)
class BLite(nn.Module):            # bundles the three; code_embed passed IN (frozen encoder lives outside)
def prediction_loss(pred, target) -> Tensor           # 1 - cosine, masked mean over valid steps
def isotropy_loss(z: Tensor) -> Tensor                # LeWM-style: batch embeddings -> penalize anisotropy:
                                                      # mean-center, cov C; loss = ||C - (tr(C)/d) I||_F^2 / d
def blite_loss(...) -> tuple[Tensor, dict]            # pred + LAMBDA_ISO*iso + grounded CE (binary+aux); dict of parts
```
The frozen jina encoder is loaded ONLY in ops/training code (`transformers`, revision-pinned, `requires_grad_(False)`); model.py never downloads.
- [ ] Tests (CPU, tiny dims via constructor args): shapes; `prediction_loss` is 0 for identical vectors and >0 for orthogonal (mutation pin: cosine sign); `isotropy_loss` is ~0 for an isotropic Gaussian batch and large for a collapsed (rank-1) batch (THE collapse pin — kills a mutant that returns 0); grounded gradients flow to StateEncoder+Predictor but a param registered as frozen stays frozen (simulate with a frozen linear stand-in for the code encoder: after `loss.backward()`, its `.grad` is None — the frozen-encoder pin); causal masking (perturbing a LATER snapshot does not change an EARLIER step's prediction — mutation pin for the mask).
- [ ] RED → implement → GREEN → commit `feat: B-lite model — LeWM two-term objective + grounded head (prereg §3/§5.2)`.

### Task 6: training harness (`crucible/latent/train.py`)

**Interfaces:**
```python
# config.py: LR=3e-4; BATCH=64; MAX_STEPS=20000; EVAL_EVERY=500; PATIENCE=5; TRAIN_SEED=0
def train_blite(corpus_dir, out_dir, *, code_embedder, device, config_overrides=None) -> dict
```
Loop: seeded; bf16 on cuda / fp32 on cpu; AdamW; every EVAL_EVERY steps compute VAL grounded AUROC (val split only — assert split != "test" at the loader call, a literal guard) + collapse probes (per-dim std mean, effective rank via singular values, val AUROC) appended to `out_dir/probes.jsonl`; early stop on best val AUROC with PATIENCE; save best checkpoint + `train_summary.json` (steps, best_val_auroc, stopped_reason, wall_s). NaN/inf loss → raise (infra, per spec §6 CONFOUNDED).
- [ ] Test: tiny synthetic corpus (20 samples, random-but-seeded token data), tiny dims, device cpu, MAX_STEPS=30 via overrides → returns; loss at step 30 < loss at step 1 (learns); probes.jsonl has ≥1 line with the three probe keys; a run with `split="test"` anywhere raises (grep-proof: pass a corpus whose loader asserts).
- [ ] RED → implement → GREEN → commit `feat: B-lite training harness with collapse probes + val-only early stop (prereg §5.2/§5.5)`.

### Task 7: control harness (`crucible/latent/control.py`)

**Interfaces:**
```python
# config.py: CTRL_LR=2e-5; CTRL_MAX_EPOCHS=5; CTRL_MAXLEN=512
def render_control_input(function_src, args_literal) -> str    # function_src + "\nINPUT: " + args_literal
def train_control(corpus_dir, out_dir, *, model_factory, tokenizer, device) -> dict
```
`model_factory` injects the model (ops passes codeexecutor + a fresh classification head; tests pass a 2-layer stub with the same interface). Same loader, same val-only early-stop rule (per-epoch eval), same summary/probe conventions (probes = val AUROC only).
- [ ] Tests: tiny stub model learns a separable synthetic task (val AUROC > 0.9 after 2 epochs — proves the loop trains and evaluates); early stop honors patience (stub whose val AUROC is rigged to decline → stops before CTRL_MAX_EPOCHS, mutation pin); `render_control_input` deterministic + truncation at CTRL_MAXLEN tokens happens in the loader (pin).
- [ ] RED → implement → GREEN → commit `feat: token-space control fine-tune harness (prereg §3)`.

### Task 8: evaluation (`crucible/latent/eval.py`) — mutation-critical

**Interfaces:**
```python
def delong_paired(y: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> dict
    # {"auroc1","auroc2","diff","se_diff","z"} — DeLong 1988 / Sun-Xu fast structural components,
    # pure numpy, midrank ties handled
def bootstrap_diff_ci(y, s1, s2, n=10000, seed=0) -> tuple[float, float]
def ece(y, p, bins=10) -> float
def fit_static_floor(train_samples, *, seed=0) -> "callable"
    # tiny seeded torch logistic on [len(src), ast node count, n_args] — the meaningful floor
    # (majority-class scores are constant -> AUROC 0.5 by construction, reported as the trivial floor)
def evaluate_gate(corpus_dir, blite_scores_path, ctrl_scores_path, floors_scores_path, out_path) -> dict
    # reads the TEST split ONCE; computes P1 (diff >= 2*se_diff), P2 floor comparisons (each arm vs static floor by 2·SE,
    # paired), P3 extras (multiclass breakdown, ECE, both arms);
    # writes gate_report.json; REFUSES to run twice (out_path exists -> raise, the one-read pin)
```
- [ ] Tests: `fit_static_floor` learns a separable synthetic (AUROC > 0.9) and is deterministic across two fits; `delong_paired` against a hand-computed 6-item case (y=[1,1,1,0,0,0], s1 perfectly separating → auroc1==1.0; s2 with one inversion → auroc2==8/9; assert both exact and se_diff>0); identical scores → diff==0 and z==0 (mutation pin: unpaired-SE mutant gives wrong se on correlated scores — construct s2=s1 and assert se_diff==0 exactly, which ONLY holds for the paired form); tie handling via midranks (case with tied scores, assert auroc == hand value); `ece` on a perfectly calibrated synthetic → ~0, on inverted → large; `evaluate_gate` one-read pin (second call raises), and P1 verdict flips when scores are degraded (pin the >= 2*se comparison direction).
- [ ] RED → implement → GREEN → commit `feat: paired DeLong gate evaluator, one-read enforced (prereg §5.4/§6)`.

### Task 9: mutations + full suite

cp-backup harness; every mutation must print killed (templates — adapt to real code, verify with cmp + git diff, restore):
```
isotropy_loss -> return zeros            vs test_isotropy (collapse pin)
causal mask dropped                      vs the causality test
assign_split seed dropped                vs the seed-sensitivity test
validator Import rule deleted            vs its rule test
delong paired-SE -> unpaired formula     vs the s2==s1 se_diff==0 pin
evaluate_gate one-read guard deleted     vs the second-call-raises pin
floor verdict comparison flipped         vs the floors test
```
Then the full suite under the 4G scope (exit 0), `git status --porcelain` clean. Controller pushes after review.

### Task 10 (ops, controller): corpus run + LOCK-BLITE

- [ ] Serve the 1.5B (PATH prefix, existing serve_model.sh). Detached `generate_corpus` driver run (setsid + pid + DONE + Monitor; ~2–4 h). On DONE: `build_manifest` → floors verdicts read off manifest.json. FLOOR FAIL or NONDET_KILL FAIL → STOP per spec §8, report to Brice. Teardown the server (corpus done, GPU freed for training).
- [ ] Download + pin: jina (revision + sha256 of weights) and codeexecutor (same); record.
- [ ] Control pre-lock check (spec §8): fine-tune control, VAL must beat the floors — else fix harness before lock.
- [ ] `docs/LOCK-BLITE.md`: spec text final; manifest hash + counts + balance + nondet rate; split seed + assignment hash; model digests; `crucible/latent/config.py` verbatim; DeLong method; commit, tag `prereg-lock-blite`, push, verify sync + tag on origin.

### Task 11 (ops, controller): train, evaluate once, verdict

- [ ] Train B-lite (detached + Monitor; probes watched — smooth-loss collapse is a RESULT, NaN is infra→archive+rerun). Train the control POST-LOCK under the locked config — Task 10's pre-lock control run is a harness check only and its checkpoint is DISCARDED (stated in the findings; keeps both gate arms trained after the lock, symmetrically).
- [ ] Score BOTH arms + floors on the test split via their checkpoints (score files), then `evaluate_gate` — the single test read. GO_P1 / NO-GO / CONFOUNDED exactly per spec §6–§7.
- [ ] `docs/findings/GATE-P1.md` (run ledger, P1 table with AUROCs + diff ± 2·SE, P2 floors, P3 calibration + collapse-probe trajectory summary, verdict, GPU accounting). Teardown, secret-scan, push, sync. Memory + handoff updated; report to Brice with the verdict and §7's prescribed next step.
