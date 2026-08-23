# S1 findings — serving + LoRA environment spike

Task 17. Companion to `crucible/proposer/identity.py` (served-model identity assertion) and
`scripts/lora_attach_smoke.py` (LoRA-attach smoke). Date started: 2026-08-23.

## Scope decision (read first)

This spike has two halves with different completion bars.

- **Delivered now (headless, no GPU):** the identity-assertion code + its mutation-checked
  tests, the runnable LoRA-attach smoke script, the `serve` optional-dependency group, the
  THIRD_PARTY license rows, and everything about the serve/LoRA decision that is knowable
  *without* committing the GPU — the decision framework, this box's constraints, the two
  candidate serve commands, and the identity endpoints.
- **DEFERRED to the operator (Brice):** the *live* serving/LoRA run — standing up vLLM (or
  llama.cpp) on the GPU, downloading multi-GB weights, and running the LoRA-attach smoke
  against a real model. That is a heavy, GPU-committing, multi-hour operation; it is not run
  in this session. The exact turnkey steps and the numbers to record are in
  **[§7 LIVE RUN — DONE](#7-live-run--done-2026-08-23)**. Live run executed 2026-08-23 — §7 carries the measured results (no fabricated numbers).

## 1. This box (measured 2026-08-23, idle desktop)

| Fact | Value | How |
|---|---|---|
| GPU | NVIDIA GeForce RTX 5080 | `nvidia-smi` |
| VRAM total | 16303 MiB (~15.9 GiB) | `nvidia-smi` |
| VRAM free (idle) | 15217 MiB (~14.86 GiB) | `nvidia-smi`, desktop up |
| Compute capability | **12.0 → sm_120 (Blackwell)** | `nvidia-smi --query-gpu=compute_cap` |
| Driver | 595.84 | `nvidia-smi` |
| CUDA toolkit (`nvcc`) | **not on PATH** | `nvcc --version` → not found |
| `~/llama.cpp` checkout | present, commit `4988f6e` | `git -C ~/llama.cpp log -1` |
| Ollama | a system `ollama serve` (PID seen) is running, ~0.6 GiB idle | `pgrep -af 'ollama serve'` |

Consequences that shape the plan:

- **sm_120 is new.** Prebuilt CUDA wheels historically lag a fresh arch by weeks-to-months.
  A stable-channel `torch`/`vllm` wheel may not ship sm_120 kernels yet; the symptom is
  `CUDA error: no kernel image is available for execution on the device` at first matmul.
  This is the single biggest risk in the spike and the reason llama.cpp is kept as a
  first-class fallback rather than an afterthought.
- **No `nvcc` on PATH.** A *source* build of vLLM or of llama.cpp with `-DGGML_CUDA=ON`
  needs the CUDA toolkit. The prebuilt `torch` wheel bundles its own CUDA runtime and does
  **not** need `nvcc`; a from-source path does. If the toolkit must be installed, record it.
- **~15 GiB free.** Qwen3.5-2B in bf16 (~4.5 GiB weights) + a 8k KV cache fits with room for
  LoRA. The Qwen3.5-9B baseline must be quantized (Q6_K ≈ 7.5 GiB, or Q4_K_M ≈ 5.5 GiB if
  Q6 does not leave ≥ 4 GiB free). Stop `ollama serve` first so its VRAM is reclaimed.

## 2. Models (licenses verified 2026-08-23 — see THIRD_PARTY.md)

| Role | Model | License |
|---|---|---|
| Proposer (primary) | `Qwen/Qwen3.5-2B` | Apache-2.0 |
| Proposer (fallback, spec §2) | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | Apache-2.0 |
| Baseline (tok/s compare) | `Qwen/Qwen3.5-9B` (Q6_K/Q4_K_M GGUF) | Apache-2.0 |

`Qwen/Qwen2.5-Coder-3B-Instruct` is **`other`** (Qwen research license), not Apache — excluded.
Only MIT/Apache/BSD may enter the tree, so the fallback proposer is the 1.5B, not the 3B.

## 3. Identity-assertion protocol (shipped this task)

Before any run reads a completion we assert *which* model the server actually serves.
`crucible/proposer/identity.py`:

- `probe(base_url)` → `ServedIdentity(kind, model, extra)`, `kind ∈ {"vllm","llamacpp"}`.
- `assert_identity(base_url, expected_model)` raises `IdentityMismatch` unless
  `expected_model == model` **or** is a suffix of it (so a bare `…-Q6_K.gguf` matches a full
  `/models/…-Q6_K.gguf` path).

**Ordering matters (ruling R6):** llama.cpp *also* serves an OpenAI-compatible `/v1/models`,
so a `/v1/models`-first probe would misclassify llama.cpp as vLLM. `probe` therefore checks
llama.cpp's own `/props` **first** and only falls back to `/v1/models`. The test's fake
llama.cpp server exposes *both* routes so this ordering is actually exercised (a mutation that
reorders the checks fails `test_probe_llamacpp`).

Identity endpoints:

| Server | Endpoint | Identity field |
|---|---|---|
| vLLM | `GET {base}/v1/models` | `data[0].id` (= `--served-model-name`) |
| llama.cpp | `GET {base}/props` | `model_path` / `default_generation_settings.model` |

## 4. vLLM vs llama.cpp — decision framework

Primary = **vLLM** (best n-best + logprobs + runtime-LoRA ergonomics for the proposer loop).
Fallback = **llama.cpp** (MIT, already checked out; robust on new arches via its own CUDA
build; serves the same OpenAI-compatible surface).

Pick vLLM when, on the live run, **all** hold:
1. `torch.cuda.is_available()` is `True` and a `2048×2048` matmul returns a finite number on
   `device='cuda'` (no "no kernel image" error) — i.e. the wheel has sm_120 kernels;
2. `vllm serve` reaches "Application startup complete" inside the timebox;
3. `assert_identity(...)` passes and a completion returns `n>1` choices each with `logprobs`.

Fall back to llama.cpp when any of the above fails and cannot be fixed within the one-day
timebox. The failure that most likely forces the fallback is (1): sm_120 kernels missing from
the stable wheel. Mitigations, in order: try the next-newer CUDA index (`cu128 → cu129/cu130`)
once; then a nightly `torch`/`vllm`; then llama.cpp. Record which rung worked.

**`server.kind` lens value** = whichever server comes up = the `kind` that `probe` returns
(`"vllm"` or `"llamacpp"`). Record it here and wherever the run stamps provenance.

## 5. Candidate serve commands

Stop Ollama first so its VRAM is freed, and confirm the GPU is idle:

```bash
systemctl --user stop ollama 2>/dev/null || pkill -f 'ollama serve'
nvidia-smi --query-gpu=memory.used --format=csv   # expect < ~1 GiB before serving
```

**A) vLLM (primary).** Pick the torch CUDA index for sm_120 from pytorch.org first:

```bash
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_capability()); a=torch.randn(2048,2048,device='cuda'); print((a@a).sum().item())"
# expect: True, capability (12, 0), a finite number
uv pip install --python .venv/bin/python vllm
VLLM_ALLOW_RUNTIME_LORA_UPDATING=True .venv/bin/vllm serve Qwen/Qwen3.5-2B \
    --served-model-name Qwen/Qwen3.5-2B \
    --max-model-len 8192 --gpu-memory-utilization 0.45 \
    --enable-lora --max-lora-rank 32 --port 8001 > runs/vllm.log 2>&1 &
sleep 90
.venv/bin/python -c "from crucible.proposer.identity import assert_identity; print(assert_identity('http://127.0.0.1:8001','Qwen/Qwen3.5-2B'))"
curl -s http://127.0.0.1:8001/v1/completions -H 'Content-Type: application/json' \
    -d '{"model":"Qwen/Qwen3.5-2B","prompt":"def add(a, b):\n    return","max_tokens":8,"n":2,"logprobs":1,"temperature":0.7,"seed":1}' | head -c 800
```

**B) llama.cpp (fallback).** Needs the CUDA toolkit (`nvcc` — install if missing):

```bash
cd ~/llama.cpp && git pull && cmake -B build -DGGML_CUDA=ON && cmake --build build -j16 --target llama-server
# download unsloth/Qwen3.5-2B-GGUF Q6_K (TEXT ONLY — do NOT download the mmproj-* file)
./build/bin/llama-server -m <path-to-Q6_K.gguf> --port 8001 -np 4 -c 8192 --lora-init-without-apply > runs/llama.log 2>&1 &
sleep 20
curl -s http://127.0.0.1:8001/props | head -c 400          # identity source
curl -s http://127.0.0.1:8001/lora-adapters                 # LoRA surface
.venv/bin/python -c "from crucible.proposer.identity import assert_identity; print(assert_identity('http://127.0.0.1:8001','Qwen3.5-2B-Q6_K.gguf'))"
```

## 6. LoRA-attach smoke

`scripts/lora_attach_smoke.py` (heavy imports guarded inside functions — it imports and
`--help`s with none of torch/transformers/peft installed):

```bash
uv pip install --python .venv/bin/python -e '.[serve]'    # torch (correct CUDA index) + vllm + transformers + peft + accelerate
.venv/bin/python scripts/lora_attach_smoke.py --model Qwen/Qwen3.5-2B \
    --base-url http://127.0.0.1:8001 --served-model Qwen/Qwen3.5-2B --load-into-server
```

It attaches a LoRA to the attention+MLP projections, runs one fwd/bwd, **generates once with
the adapter active** (proving it loads), saves to `runs/lora-smoke`, asserts the server's
identity, and (vLLM) POSTs the adapter via `/v1/load_lora_adapter`. For **llama.cpp** the
adapter must first be converted with `convert_lora_to_gguf.py` and passed at launch via
`--lora-adapters` — record whether that conversion works for Qwen3.5's architecture.

**Decision rule (record the outcome in §7 and CARRIED-DEBT):** attach + fwd/bwd + generate +
save + server-load all succeed ⇒ proposer = **Qwen3.5-2B**. Any step fails unrecoverably in
the timebox ⇒ re-run the smoke with `Qwen/Qwen2.5-Coder-1.5B-Instruct`; proposer for all
small arms = **Qwen2.5-Coder-1.5B-Instruct** (spec §2 fallback).

## 7. LIVE RUN — DONE 2026-08-23

Executed §5→§6 on this box (RTX 5080, sm_120). The vLLM path succeeded on the **first** CUDA
rung (cu128) — no cu129/cu130/nightly/llama.cpp fallback needed. Every value below is measured.

**Record:**

| Measurement | Value |
|---|---|
| torch version / CUDA index that worked | `torch 2.11.0+cu128` verified sm_120 first; **vllm 0.27.1 then upgraded it to `torch 2.13.0+cu130`** (also sm_120-good). cu128 worked on rung 1 — no fallback ladder used. |
| `torch.cuda.is_available()` / capability | `True` / `(12, 0)` — a 2048x2048 CUDA matmul returned finite on **both** torch builds |
| Server that came up on sm_120 + version | **vLLM 0.27.1** (V1 engine) — with the ninja/FlashInfer workaround below |
| Server startup time (s) | ~145 s (weights 3.2 s + profiling/warmup ~32 s + KV-cache init and CUDA-graph capture) |
| VRAM used while serving Qwen3.5-2B (GiB) | **9.58 GiB** at `--gpu-memory-utilization 0.6` (weights alone 4.32 GiB, bf16); ~6 GiB free |
| tok/s @ 256-token completion, Qwen3.5-2B | **138.5 tok/s** (single request, temp 0.7, 256/256 tokens) |
| tok/s @ 256-token completion, Qwen3.5-9B | `n/a this run` — 9B bf16 ~18 GiB > 16 GiB card; needs a quantized load (GGUF via llama.cpp — blocked on `nvcc`; or a vLLM AWQ/GPTQ). Deferred — **not an S1 exit criterion**. |
| n-best + logprobs present in a completion? | **Yes** — an `n=2` request returned 2 choices, each with `logprobs.token_logprobs` populated |
| LoRA attach + fwd/bwd + save OK? | **Yes** — rank-16 LoRA on 12 modules (attn q/k/v/o + MLP up/gate/down + Qwen3.5 `in_proj_*`), **16.82 M** trainable params (0.886 %), fwd/bwd loss 0.697, peak **4.54 GiB**, saved to `runs/lora-smoke/` |
| Adapter-active generation sample | local (peft): `def add(a, b):\n    return a + b\ndef subtract(a, b):\n   ` -- served (vLLM `model=smoke`): ` a + b\n\ndef subtract(a, b):\n   ` |
| Server-side adapter load OK? | **Yes** -- `POST /v1/load_lora_adapter {lora_name:"smoke", lora_path:runs/lora-smoke}` -> HTTP 200; `/v1/models` then lists `["Qwen/Qwen3.5-2B","smoke"]`. Runtime-LoRA (sleep/consolidation) path proven. |
| **Proposer decision** | **Qwen3.5-2B** — §6 rule: attach + fwd/bwd + generate + save + server-load all passed; the Qwen2.5-Coder-1.5B fallback is not needed |
| Measured free VRAM, idle desktop | 15207 MiB (~14.85 GiB) free / 636 MiB used — cross-checks §1 |

**Gotchas recorded (sm_120 + VLM base):**

- `Qwen/Qwen3.5-2B` is a **vision-language base** (`Qwen3_5ForConditionalGeneration`, i.e. Qwen3-VL
  with a video processor), not a plain text LM. transformers 5.15.1 and vLLM 0.27.1 both load it
  natively for **text-only** completions (the vision tower loads but is unused for text). Budget
  VRAM for the whole model.
- vLLM's default **FlashInfer sampler JIT-compiles a top-k/top-p CUDA kernel at engine init**, which
  needs `ninja` and (for the compile) `nvcc`. This box has neither by default, so engine-core init
  crashes in `_initialize_kv_caches` -> `_dummy_sampler_run` with `No such file or directory:
  'ninja'`. **Fix: `pip install ninja` and launch with `VLLM_USE_FLASHINFER_SAMPLER=0`** (native
  torch sampling; n-best and logprobs are unaffected). Without a CUDA toolkit, disabling the JIT
  sampler is the reliable path.
- Port 8001 (the §5 example) was already bound on this box; served on **8010** instead.

**S1 exit criteria this run closes (spec §10):**
- [x] A server (vLLM 0.27.1) serves Qwen3.5-2B with n-best + logprobs, identity-asserted.
- [x] The LoRA-attach decision is recorded -- **proposer = Qwen3.5-2B**.

Remaining (NOT an S1 exit): Qwen3.5-9B baseline tok/s needs a quantized load to fit the 16 GiB card.
