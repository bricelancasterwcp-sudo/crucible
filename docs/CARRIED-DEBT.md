# CARRIED-DEBT

Appended at every slice merge: what the slice settled → deferred, with rulings → process lessons. Resolved items are struck through, never deleted.

## S1 (in progress)
### Settled
- (fill at merge)
- **Served-model identity assertion shipped** (Task 17). `crucible/proposer/identity.py`
  probes a server and asserts *which* model it serves before any run trusts its completions
  (`probe` / `assert_identity` / `IdentityMismatch`). Ruling **R6**: llama.cpp also serves an
  OpenAI-compatible `/v1/models`, so `probe` checks llama.cpp's `/props` first and only falls
  back to `/v1/models` (else llama.cpp is misread as vLLM). Suffix match lets a bare gguf
  filename match a full served path. Mutation-checked (reorder / accept-any / drop-suffix all
  break a test). Fake-server tests only — no GPU.
### Deferred, with rulings
- ~~**Live serving/LoRA run deferred to the operator** (ruling **R-T17-1**, Task 17).~~ **RESOLVED 2026-08-23 — live run executed on this box; both spec §10 serving exits CLOSED.** vLLM 0.27.1 serves `Qwen/Qwen3.5-2B` (identity-asserted via `probe`/`assert_identity`), n-best + logprobs confirmed, ~138 tok/s @ 256 tokens, 9.58 GiB VRAM; the LoRA smoke attached a rank-16 adapter (16.8 M params, loss 0.697, 4.54 GiB), generated with it active, and loaded it server-side (`/v1/load_lora_adapter` -> HTTP 200). **Proposer = Qwen3.5-2B** (fallback 1.5B not needed). Full record in `docs/findings/S1-serving.md §7`. Original deferral text: The
  *headless* half of the S1 serving spike is delivered: identity code + tests, the runnable
  `scripts/lora_attach_smoke.py`, the `serve` optional-deps group, THIRD_PARTY license rows,
  and the full decision framework/constraints/commands in `docs/findings/S1-serving.md`. The
  *live* half — standing up vLLM (or llama.cpp) on the GPU, downloading multi-GB weights, and
  running the LoRA-attach smoke against a real model — is a heavy, multi-hour, GPU-committing
  operation and is **not** run in-session. Turnkey steps + the exact blanks to fill are in
  S1-serving.md §7 (PENDING LIVE RUN). Until that run: the two spec §10 exit criteria
  "server serves Qwen3.5-2B, identity-asserted" and "LoRA-attach decision recorded" stay open;
  the proposer is provisionally Qwen3.5-2B with Qwen2.5-Coder-1.5B-Instruct (Apache-2.0) as the
  §2 fallback if attach fails.
- ~~**vLLM on Blackwell sm_120 is the live run's main risk** (S3 risk, Task 17).~~ **RETIRED 2026-08-23 — sm_120 was NOT the blocker.** `torch 2.11.0+cu128` (and vLLM's `torch 2.13.0+cu130`) both ship working sm_120 kernels; a CUDA matmul is finite and vLLM serves. The real gotcha was vLLM's FlashInfer sampler JIT-compiling a kernel that needs `ninja`+`nvcc` (absent here) — fixed with `pip install ninja` + `VLLM_USE_FLASHINFER_SAMPLER=0` (native sampling; n-best/logprobs unaffected). Also: `Qwen3.5-2B` is a Qwen3-VL base served text-only; port 8001 was taken, used 8010. Original risk text: This box is an
  RTX 5080, compute cap 12.0 (sm_120), driver 595.84, no `nvcc` on PATH; prebuilt CUDA wheels
  historically lag a new arch. If the stable `torch`/`vllm` wheel lacks sm_120 kernels
  ("no kernel image"), the risk-managed path is next-newer CUDA index → nightly → source-built
  llama.cpp (MIT, already checked out at `~/llama.cpp`). A source llama.cpp/vLLM build needs the
  CUDA toolkit installed. LoRA *training* on sm_120 inherits the same wheel risk — record if it
  forces a CPU-offload or a rental.
- **Sandbox isolation is Python-level, not OS-level** (ruling R-T2-3, Task 2). `crucible/sandbox/exec.py` blocks outbound sockets with a `sitecustomize.py` shim: it stops *accidental* network use by generated single-function code, which is the failure mode S1 has, but it is not an adversary barrier -- a unit that shells out to `curl` or calls `connect` through `ctypes` still reaches the network. No S1 code path produces such a unit (units are single functions the proposer writes, run by pytest). OS-level isolation (network namespace / bubblewrap) is deferred past S1. The same ruling covers the escaped-`setsid` grandchild: file-backed capture means it can no longer stall the wall cap, but reaping it would need a cgroup or PID namespace.
### Process lessons
- (fill at merge)

## S2 (in progress)
### Settled
- **Search + arms machinery built and reviewed clean** (plan Tasks 1–15 + docstring fix, branch
  `s2-search-arms`). Proposer adapters (prompt/codec/client), REx Thompson search (`search/rex.py`,
  `node.py`, `loop.py`), constant value-fn v0, driver/records/lens/pilot/landing-check, CLI
  `arm pilot|run`. Every task passed the two-stage SDD review; whole-branch review = ready to merge.
  Ruling **R-S2-T7-1**: REx posterior was inert (refinement ignored it); fixed so the scheduler's
  Beta posterior actually drives which node is refined (verified 0.730 vs 0.555).

### Deferred, with rulings — BLOCKING
- **Ceiling pilot BLOCKED at the §4.7 codec-landing gate** (plan Task 16, operational run,
  2026-08-23). The pilot runs A_noMem = the **Qwen3.5-2B** small-arm proposer; on the real
  450-task stream at the **pinned** sampler (§3, `max_new_tokens 1024`), the 2B lands **0.767**
  and the §2 alternative `Qwen2.5-Coder-1.5B-Instruct` lands **0.80** — **both fail the ≥0.95
  gate.** Diagnosis: failures are decoding artifacts (truncation-dominated — the codec re-emits
  the module *and* the whole visible test harness, overflowing 1024 tokens → unclosed fence;
  plus ~6% empty completions and, on the 2B, fragment/test-echo). At `max_new_tokens 2048` the
  1.5B rises to **0.92** (frozen) / **~0.94** (with unclosed-fence salvage), so the fix is a
  concrete **codec fix** (§4.7's other sanctioned remedy, since the fallback also fails): (A)
  raise the pinned `max_new_tokens` to 2048 [§3 amendment, recommended minimum], (B) salvage
  unclosed fences in `extract_module` [§4.4 code], (C) trim the codec so the model stops
  reproducing the visible test file [§4.4 redesign]. **All three amend pinned/frozen
  pre-registration → operator decision.** The 2B must be re-probed under the amended config
  before it is kept or swapped for the 1.5B (its failures were more test-echo than truncation).
  **p0 unmeasured; too-easy verdict N/A until the gate is cleared.** Full record:
  `docs/findings/S2-ceiling-pilot.md`. Pilot does NOT run until Brice picks a remedy and approves
  the amendment.
- **Baseline "big" proposer not yet landing-probed** (S3/S4). §2 `Qwen3.5-9B` (fallback
  `Qwen2.5-Coder-14B-Instruct Q4_K_M`) gates B_naive/big, not the pilot; must clear §4.7 before
  those arms run.

### Process lessons
- **The landing pre-check earned its place.** §4.7 caught that the pre-registered proposer would
  have run the whole experiment at 77% parseable — measurements dominated by parse failures, not
  reasoning — before a single arm ran. It also caught that the "fix" is a pinned-value amendment,
  not a silent retune. Exactly the confound §4.7/§11 was written to stop.
- **UPDATE (2026-08-23) — remedy investigated; clean fix identified, awaiting proposer greenlight.**
  Amendment **A1** applied: pinned `max_new_tokens` 1024→2048 (single source `client.MAX_NEW_TOKENS`,
  guard-tested; spec §3). Effect 2B 0.767→0.867, 1.5B 0.80→0.92. **2B ruled out** — its residual
  is repetition-degeneration (`no-fence`, salvage rescues 0/5), and decoding penalties backfire
  (`repetition_penalty=1.1`→0.233 all-empty from prompt-penalisation; `frequency_penalty=0.3`→0.633
  with 8 syntax errors from code distortion). **Clean fix = the §2 1.5B coder served in CHAT mode:**
  the 1.5B's residual was ~6% empties = an *instruct* model served raw (no chat template); via
  `/v1/chat/completions` it lands **20/20** (raw 16/20). Recommended config: proposer
  `Qwen2.5-Coder-1.5B-Instruct`, chat-served, `max_new_tokens 2048` — no codec/salvage/penalty
  change. Costs (why it's Brice's call, not a silent swap): switches the pre-registered *primary*
  proposer (2B→1.5B); needs a chat-completions path in `client.py` (logprobs shape differs — TDD +
  review); S3 must re-verify LoRA-attach on the 1.5B before A_full (spec calls it LoRA-safe, but
  unverified). Bigger-model alt = Apache **7B** coder (3B is `license:other`, excluded). Full record:
  `docs/findings/S2-ceiling-pilot.md §7`.
- **RESOLVED (2026-08-23) — gate cleared, pilot ran, p0 recorded.** Amendment **A2** (Brice-approved):
  small-arm proposer → `Qwen2.5-Coder-1.5B-Instruct` **chat-served** (client chat path added +
  tested + verified live; §2). §4.7 landing pre-check chat-served = **1.00 (30/30), PASSES**.
  Ceiling pilot (A_noMem, 1.5B chat, 30 phase-1 tasks, K=8): **p0 = 0.767, too_easy = true**
  (p0 > 0.70). A_noMem (search, no memory) already solves 23/30 → the base stream is too easy for
  the memory experiment; the pilot did its job catching an undiscriminating ceiling before the run.
  Records: `runs/pilot-a2-15bc/A_noMem/`. Full record: `docs/findings/S2-ceiling-pilot.md §7–§8`.
- ~~**Deferred — hardening ladder is NOT implemented**~~ **RESOLVED 2026-08-24 (S2.5):** rung (i)
  stack-2 built (branch `s2.5-stack2`), stream `1158e92f40ad`, re-pilot **p0 = 0.267 ≤ 0.70** —
  rung FIXED at `stack2`. See `docs/findings/S2.5-stack2.md` and the S2.5 section below.
- **Original text (kept):** hardening ladder was NOT implemented (spec §4.8.4 / §10 S2 "apply the hardening
  ladder if needed"). The pilot's `too_easy` verdict prescribes `FIRST_HARDENING_RUNG` = "stack two
  mutations per unit", but `rung` is only a label in the stream hash — the mutation engine
  (`mutants.py`) injects ONE mutation per task and even prefers *distinct* spans over stacking.
  Reaching a discriminating stream (p0 ≤ 0.70) therefore needs **new build work**: a stack-2
  mutation mode + rebuild + re-pilot. Scope decision (build it now as an S2.5 / fold into S4) is
  Brice's. Until then the base rung stays measured-too-easy; the gating A_full-vs-A_noMem run must
  wait for a hardened stream.


## S2.5 (rung-1 stack-2 hardening) — 2026-08-24
### Settled
- **Two-site mutant engine + rung dispatch built and reviewed clean** (branch `s2.5-stack2`,
  a155890..HEAD; 7 SDD tasks + final whole-branch review + fix wave; 301 tests green capped (299 at fix-wave + 2 SERVE-entry tests)).
  `Mutant.components`, `stack.py` (span-matched composition — the wrong-site trap is closed by
  exact-span re-selection; span-partition pairing gives disjoint site-sets by construction),
  compose site-set classes + `TaskSpec.span2` + post-walk short-stream raise, precheck
  `distinct-sites` site-sets + `two-site-at-stack2` + derived `REQUIRED_COUNTS`, CLI
  `--rung stack2 --pairs-per-family`. Rung-0 proven byte-identical (pinned fixture hash + full
  `dd5912cddedc` rebuild guard).
- **Ceiling cleared per pre-reg §4.8.4:** rung-1 stream `1158e92f40ad` (200 classes, all
  prechecks, smoke 30/30, landing 1.00) → **p0 = 0.267, too_easy=false, rung fixed at stack2.**
- Rulings of record: R-S25-1 (components AND composite each visible-killed), R-S25-2 (rung (i)
  only), CONST-early exclusion (structural: NumberReplacer double-position spans; capacity
  re-measured 238→207 eligible ≥ C=200), stack-apply census = all compose_pair Nones,
  inclusive-end span overlap (conservative).

### Deferred, with rulings
- **`_is_eligible` anchoring mutation survives the fixtures** (final review MINOR-5, parked):
  a head-anchored weakening would undercount `eligible_classes` at rung 1 — conservative
  direction only (early NotEnoughClasses, never a silent pass); killing it needs fixture
  reordering that churns the C-walk assertions.
- **Rung-1 stream hash is sensitive to `kills_by_timeout` flake reordering pair selection**
  (T7 note): the in-suite double-build determinism test is the tripwire; re-check at
  `prereg-lock-a` time.
- **hidden-only/equivalent boundary verdict flake** (count-level, one mutant, hash-neutral) —
  observed on the rung-0 guard rebuild; findings §4.
- Cosmetic minors from task reviews logged in the SDD ledger
  (`.superpowers/sdd/2026-08-23-crucible-s2.5-stack2/progress.md`, kept).

### Process lessons
- The brief's own mutation checks failed to kill twice (Task 3, Task 4) and both times the
  implementer chased the surviving mutant to a REAL property (NumberReplacer same-span
  positions; span-shuffle locality confound) instead of accepting green — the checks exist to
  be chased, not passed.
- Build ops: `--jobs 8` OOMs `build_units` at any tried cap; the proven recipe is jobs 5-6
  under `MemoryMax=12G` with disk `TMPDIR` (findings §4).
