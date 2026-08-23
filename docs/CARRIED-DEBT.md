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
