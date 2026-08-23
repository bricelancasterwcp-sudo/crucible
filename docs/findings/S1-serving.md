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
  **[§7 PENDING LIVE RUN](#7-pending-live-run-brice)**. No benchmark numbers in this doc are
  invented — live-run cells are left blank (`____`) for the operator to fill.

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

## 7. PENDING LIVE RUN (Brice)

Not run in this session (GPU-committing, multi-hour, multi-GB download). Execute §5→§6, then
fill every `____` below. Do not delete a blank you could not measure — mark it `n/a` + why.

**Steps, in order:**
1. Stop Ollama; confirm `< ~1 GiB` VRAM used.
2. Install `torch` for sm_120 (§5A); run the cuda self-check. If "no kernel image": try
   `cu129`/`cu130` once, then nightly, then go to llama.cpp (§5B). Record which rung worked.
3. Bring up the server; `assert_identity` must pass; a completion must return `n>1` choices
   each with `logprobs` (vLLM) or `n_probs` (llama.cpp).
4. Run the LoRA smoke (§6) on Qwen3.5-2B; load the adapter into the server; sample once with
   it active. Apply the §6 decision rule.
5. Commit the filled doc + CARRIED-DEBT.

**Record (no fabrication — blanks until measured):**

| Measurement | Value |
|---|---|
| torch version / CUDA index that worked | `____` |
| `torch.cuda.is_available()` / capability | `____` / `____` |
| Server that came up on sm_120 (`vllm` \| `llamacpp`) + version/commit | `____` |
| Server startup time (s) | `____` |
| VRAM used while serving Qwen3.5-2B (GiB) | `____` |
| tok/s @ 256-token completion, Qwen3.5-2B | `____` |
| tok/s @ 256-token completion, Qwen3.5-9B (Q6_K or Q4_K_M — say which) | `____` |
| n-best + logprobs present in a completion? | `____` |
| LoRA attach + fwd/bwd + save OK? | `____` |
| Adapter-active generation sample | `____` |
| Server-side adapter load OK? (vLLM POST / llama.cpp GGUF-convert) | `____` |
| **Proposer decision** (Qwen3.5-2B \| Qwen2.5-Coder-1.5B-Instruct) | `____` |
| Measured free VRAM, idle desktop (cross-check §1) | `____` |

**S1 exit criteria this run closes (spec §10):**
- [ ] A server (vLLM or llama.cpp) serves Qwen3.5-2B with n-best + logprobs, identity-asserted.
- [ ] The LoRA-attach decision is recorded (proposer = Qwen3.5-2B or fallback 1.5B).
