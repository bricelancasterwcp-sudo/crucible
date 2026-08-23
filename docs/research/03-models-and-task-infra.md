# 03 — Models and Task Infrastructure

**Survey date:** 2026-08-23
**Scope:** proposer models, baseline models, local inference stack, LoRA tooling, Python mutation tools, seed corpora.
**Target hardware:** 1× RTX 5080, 16303 MiB VRAM (measured), driver 595.84, 29 GB system RAM.

**Licensing rule applied throughout.** Apache-2.0 / MIT / BSD = **adoptable**. GPL / AGPL / LGPL = **copyleft**, reference or isolated-process use only. CC-BY-NC, "research only", and custom model licenses = **FLAG** with the actual terms transcribed. Every license claim below is traceable to a command in the Evidence log; nothing is asserted from memory.

---

## (a) Proposer candidates — small open-weight code LLMs (≤3B, with 0.5B and 4B edges)

The headline finding is that **the 2026 landscape changed under two of the assumptions in the brief**, and both changes matter:

1. **There is no 2026 code-specialised Qwen at or below 14B.** `curl` against the Qwen HF org filtered to `Coder` returns nothing newer than the Qwen2.5-Coder line (2024/2025) except `Qwen3-Coder-30B-A3B`, `Qwen3-Coder-Next` (80B-A3B) and `Qwen3-Coder-480B-A35B`. The small-coder niche was not refreshed; instead the *general* small models absorbed coding ability.
2. **The Qwen3.5 generation (2026-03) shipped a real small family** — 0.8B / 2B / 4B / 9B dense, all Apache-2.0, all with Base checkpoints — and these are the strongest permissive sub-4B options now available.

### Table

| Name | Source URL | License (verified how) | Last modified | Key facts | Verdict |
|---|---|---|---|---|---|
| **Qwen3.5-2B** | `huggingface.co/Qwen/Qwen3.5-2B` | **Apache-2.0** — HF API `.cardData.license` = `apache-2.0`; card front-matter `license: apache-2.0` | 2026-03-02 | 2.27 B params (safetensors total). LM = 2 B, 24 layers, hidden 2048. Hybrid **Gated DeltaNet + Gated Attention**. 262 144 ctx native. Vision encoder present (`image-text-to-text`). Base ckpt `Qwen3.5-2B-Base` (Apache-2.0, 2026-04-23). GGUF: `unsloth/Qwen3.5-2B-GGUF` Q4_K_M **1.28 GB** / Q6_K 1.57 GB. Card carries **no** LiveCodeBench/HumanEval row for 2B (verified by parsing every `<tr>`); only MMLU-Pro 55.3, SuperGPQA 30.4, GPQA 51.6. | **ADOPT** (primary proposer) |
| **Qwen3.5-4B** | `huggingface.co/Qwen/Qwen3.5-4B` | **Apache-2.0** — same two sources | 2026-03-02 | 4.66 B params total (LM 4 B, 32 layers, hidden 2560). Same GDN hybrid, 262 K ctx extensible to 1.01 M. **LiveCodeBench v6 = 55.8** (stated on card; same row: Qwen3.5-9B 65.6, Qwen3-30B-A3B-Thinking 66.0, GPT-OSS-20B 74.6). GGUF Q4_K_M **2.74 GB** / Q6_K 3.53 GB. | **ADOPT** (the 4B edge; use if 2B underperforms) |
| **Qwen3.5-0.8B** | `huggingface.co/Qwen/Qwen3.5-0.8B` | **Apache-2.0** — HF API | 2026-03-02 | 0.87 B params. MMLU-Pro 29.7 / GPQA 11.9 — a real cliff below 2B. Useful only as a "does the harness work at all" smoke model. | REFERENCE |
| **Qwen2.5-Coder-1.5B-Instruct** | `huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct` | **Apache-2.0** — HF API `.cardData.license` = `apache-2.0` | 2025-01-12 | 1.54 B params, 32 768 ctx. The only **code-specialised** Apache-2.0 model in the size band. Card states context length only — **no benchmark table on the card** (verified by grep). First-party GGUF `Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF` q4_k_m 1.12 GB / q6_k 1.46 GB. Already pulled on this box (`qwen2.5-coder:1.5b-instruct-q8_0`). | **ADOPT** (code-specialist arm of the comparison) |
| **Qwen2.5-Coder-0.5B-Instruct** | `huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct` | **Apache-2.0** — HF API | 2024-11-18 | 0.5 B. Already on box. Smoke-test tier. | REFERENCE |
| ⚠️ **Qwen2.5-Coder-3B-Instruct** | `huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct` | ⚠️ **`other` / `qwen-research`** — HF API returns `license: "other", license_name: "qwen-research"`; LICENSE file fetched and read | 2025-01-12 | **NON-COMMERCIAL ONLY.** See flag box below. This is the single most likely trap in the whole size band — the 0.5B, 1.5B, 7B and 14B siblings are Apache-2.0 and the 3B is not. | **SKIP** (research-only; excludes the ordinary path) |
| **Ministral-3-3B-Instruct-2512** | `huggingface.co/mistralai/Ministral-3-3B-Instruct-2512` | **Apache-2.0** — HF API `tags[]` contains `license:apache-2.0` | 2026-07-15 | 3.85 B params. Dense. Also `Ministral-3-3B-Reasoning-2512` (4.25 B, Apache-2.0). Not gated. 371 k downloads. Mistral returned to Apache-2.0 for this tier. | **ADOPT** (best non-Qwen 3B alternative) |
| **SmolLM3-3B** | `huggingface.co/HuggingFaceTB/SmolLM3-3B` | **Apache-2.0** — HF API + card front-matter | 2025-09-10 | 3.08 B params, 65 536 ctx (YaRN to 128 K/256 K). Fully-open training recipe. Card states **LiveCodeBench v4 = 15.2** vs Qwen3-4B 24.9 and Qwen3-1.7B 15.0 — i.e. **materially weaker at code** than the Qwen small models. Value is the open data/recipe, not the score. | REFERENCE |
| **Granite 4.1-3b** | `huggingface.co/ibm-granite/granite-4.1-3b` | **Apache-2.0** — HF API `tags[]` `license:apache-2.0` | 2026-05-04 | 3.40 B params. `GraniteMoeHybridForCausalLM`. IBM ships first-party GGUF for the 4.0 micro line. Clean provenance/indemnity story if that ever matters. | PORT (secondary) |
| **Granite 4.0-micro** | `huggingface.co/ibm-granite/granite-4.0-micro` | **Apache-2.0** — HF API | 2025-11-03 | 3.40 B, hybrid MoE. First-party GGUF `ibm-granite/granite-4.0-micro-GGUF`. | REFERENCE |
| **Granite 4.0-1b-base** | `huggingface.co/ibm-granite/granite-4.0-1b-base` | **Apache-2.0** — HF API | 2026-06-12 | 1.63 B. A *base* checkpoint at 1.6B is rare and useful if we want to own the instruct-tuning. | REFERENCE |
| **Gemma 4 E2B-it** | `huggingface.co/google/gemma-4-E2B-it` | ⚠️→✅ **Apache-2.0** — HF API `tags[]` contains `license:apache-2.0`, and `gated: false` | 2026-07-20 | **Google dropped the custom Gemma Terms for Gemma 4.** Verified across E2B / E4B / 12B / 26B-A4B / 31B — all `license:apache-2.0`, all **ungated** (Gemma 3 and CodeGemma remain `license: gemma` + `gated: "manual"`, confirmed by the same command). E2B = 5.12 B stored params with ~2 B effective (MatFormer-style); multimodal `any-to-any`. | PORT — verify the effective-param story before trusting the "2B" label |
| ⚠️ **StarCoder2-3b** | `huggingface.co/bigcode/starcoder2-3b` | ⚠️ **`bigcode-openrail-m`** — HF API `.cardData.license` | 2024-03-04 | Behaviourally obsolete (2024) *and* license-encumbered. See flag box. | **SKIP** |
| ⚠️ **DeepSeek-Coder-1.3b-instruct** | `huggingface.co/deepseek-ai/deepseek-coder-1.3b-instruct` | ⚠️ **`other` / `deepseek`** — HF API `license: "other", license_name: "deepseek", license_link: "LICENSE"`; LICENSE fetched and read | 2024-03-07 | Custom OpenRAIL-derived license. Commercial use **is** permitted (card §4: "DeepSeek Coder supports commercial use") but Attachment A use-restrictions must propagate to every derivative. 2024 vintage. | REFERENCE only |
| ⚠️ **CodeGemma-2b** | `huggingface.co/google/codegemma-2b` | ⚠️ **`gemma`** — HF API, `license_link: https://ai.google.dev/gemma/terms`, `gated: "manual"` | 2024-08-07 | Gated behind manual approval; custom Google terms with a use-policy that binds derivatives. Superseded by Gemma 4 (Apache-2.0). | **SKIP** |
| **Phi-4-mini-instruct** | `huggingface.co/microsoft/Phi-4-mini-instruct` | **MIT** — HF API `.cardData.license` = `mit` | 2025-12-10 | ~3.8 B. MIT is the cleanest license in this table. Not code-specialised. | PORT (license-clean fallback) |
| **Yi-Coder-1.5B-Chat** | `huggingface.co/01-ai/Yi-Coder-1.5B-Chat` | **Apache-2.0** — HF API `tags[]` + `.cardData.license` | 2024-09-06 | 1.48 B, code-specialised, 128 K ctx. 2024 vintage, **623 lifetime downloads** — effectively abandoned. | REFERENCE |
| **Mellum-4b-base** (JetBrains) | `huggingface.co/JetBrains/Mellum-4b-base` | **Apache-2.0** — HF API | 2025-05-07 | 4.02 B, code-completion-focused. `Mellum-4b-sft-python` is a Python-specialised SFT (Apache-2.0, 4.02 B) — interesting given our domain, but 55 downloads and no 2026 update. | REFERENCE |
| ⚠️ **IQuest-Coder-V1-7B-Instruct** | `huggingface.co/IQuestLab/IQuest-Coder-V1-7B-Instruct` | ⚠️ **`other`** — HF API `tags[]` `license:other` | 2026-03-04 | 7.6 B. New 2026 code family (also 40B). Custom license not yet transcribed; 121 downloads. Named in ExLlamaV3's arch list, so it has some traction. | **SKIP** for now (unverified custom license, negligible adoption) |

### ⚠️ License flag boxes (actual terms, transcribed from the fetched files)

**Qwen RESEARCH LICENSE (applies to `Qwen2.5-Coder-3B-Instruct` only within our band).** Fetched from `https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct/raw/main/LICENSE`. §1(i): *"'Non-Commercial' shall mean for research or evaluation purposes only."* §2(a) grants rights *"FOR NON-COMMERCIAL PURPOSES ONLY"*; §2(b): *"If you are commercially using the Materials, you shall request a license from us."*
- **Commercial use: NO** (requires a separate license from Alibaba Cloud).
- **Fine-tuning: YES** — §2(a) permits "create derivative works of, and make modifications to the Materials", within the non-commercial limit.
- **Redistribution of derivatives: YES**, under §3, but you must ship a copy of the agreement, mark modified files, and carry the notice *"Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c) Alibaba Cloud."*
- **Extra obligation:** §4(b) — if you use it to train or improve a distributed model, you must **prominently display "Built with Qwen" or "Improved using Qwen"**. That clause would attach to any crucible LoRA consolidated on top of it.
- **Net:** fine for a pure research spike, but it silently forecloses the ordinary path. Since Qwen3.5-2B is Apache-2.0 and newer, there is no reason to take the risk.

**BigCode OpenRAIL-M v1 (StarCoder2).** The model card points at `huggingface.co/spaces/bigcode/bigcode-model-license-agreement`; fetched and grepped. The agreement's own preamble states it permits *"commercial versions of it. Use restrictions are included to prevent misuse of the Model."*
- **Commercial use: YES.** **Fine-tuning/redistribution: YES.**
- **But:** the use-based restrictions are a **viral obligation** — they must be reproduced in every downstream license governing the model or its derivatives. That makes it non-Apache-equivalent and it contaminates any artifact we publish. Combined with StarCoder2 being a 2024 model, **SKIP**.

**DeepSeek model license.** Fetched `.../deepseek-coder-6.7b-instruct/raw/main/LICENSE`. OpenRAIL-derived, Sections I–IV plus Attachment A ("Use Restrictions").
- **Commercial use: YES** — model card §4 states *"DeepSeek Coder supports commercial use."*
- **Fine-tuning: YES** — §5 explicitly lists "finetuning, updating, running, training, evaluating and/or reparametrizing".
- **Redistribution: YES**, but §III(a) requires the Attachment A use-restrictions be carried as *"an enforceable provision … in any type of legal agreement … governing the use and/or distribution of the Model or Derivatives"*. §7 also reserves DeepSeek's right to *"restrict (remotely or otherwise) usage of the Model in violation of this License."*
- **Net:** usable, but viral use-restrictions. Reference only.

**Gemma terms (Gemma ≤3, CodeGemma).** `google/codegemma-2b` and `google/gemma-3-4b-it` both return `license: "gemma"` with `gated: "manual"` — you must click through and be approved before download, and the Gemma Prohibited Use Policy binds derivatives. **Superseded:** every Gemma 4 checkpoint checked returns `license:apache-2.0` and `gated: false`. If you want a Google model, use Gemma 4.

---

## (b) Baseline candidates — 7–14B, permissive, fits 16 GB at Q4–Q6

The baseline must be a *bigger frozen model with the same verification budget and no memory*. It must therefore fit alongside a running pytest workload, so headroom matters more than peak quality.

| Name | Source URL | License (verified how) | Last modified | Key facts | Verdict |
|---|---|---|---|---|---|
| **Qwen2.5-Coder-14B-Instruct** | `huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct` | **Apache-2.0** — HF API `.cardData.license` | 2025-01-12 | The strongest permissive *code-specialised* model that fits. First-party GGUF: **q4_k_m 8.99 GB**, q5_k_m 10.51 GB. Leaves ~5 GB for KV + desktop at Q4. **Already on this box** as `qwen2.5-coder:14b-instruct-q4_K_M` (9.0 GB). 2.28 M downloads. | **ADOPT** (primary baseline) |
| **Qwen3.5-9B** | `huggingface.co/Qwen/Qwen3.5-9B` | **Apache-2.0** — HF API + card | 2026-03-02 | 9.65 B params. **LiveCodeBench v6 = 65.6** (card) — beats Qwen3-30B-A3B-Thinking's 66.0 to within noise at a third the size, and beats Qwen3.5-4B's 55.8 by ~10 pts. GGUF Q4_K_M **5.68 GB** / Q6_K **7.46 GB** — Q6_K fits comfortably. Same architecture family as the proposer, which makes the comparison cleaner. | **ADOPT** (best same-family baseline) |
| **Qwen2.5-Coder-7B-Instruct** | `huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct` | **Apache-2.0** — HF API | 2025-01-12 | 2.41 M downloads. Already on box at q8_0 (8.1 GB). Known-good on this hardware: robigo measured **56 KiB/token** KV cost (28 layers × 4 kv-heads × 128) — the cheapest KV in its class by 14× vs the worst. | **ADOPT** (low-risk baseline; best VRAM economics) |
| **Ministral-3-14B-Instruct-2512** | `huggingface.co/mistralai/Ministral-3-14B-Instruct-2512` | **Apache-2.0** — HF API `tags[]` | 2026-07-15 | 13.95 B, 2026 vintage, ungated, 165 k downloads. The most current permissive 14B dense. | **ADOPT** (modern alternative) |
| **Ministral-3-8B-Instruct-2512** | `huggingface.co/mistralai/Ministral-3-8B-Instruct-2512` | **Apache-2.0** — HF API | 2026-07-15 | 8.92 B. | PORT |
| **Granite 4.1-8b** | `huggingface.co/ibm-granite/granite-4.1-8b` | **Apache-2.0** — HF API | 2026-05-04 | 8.79 B, 2.44 M downloads. ⚠️ robigo measured granite-code-8b landing **≈0 %** of SEARCH/REPLACE edits where qwen2.5-coder-7b landed ≥90 %. That was the older `granite-code` line, not 4.1 — but it is a standing reason to measure codec landing before trusting any Granite here. | PORT (measure the codec first) |
| **Gemma 4-12B-it** | `huggingface.co/google/gemma-4-12B-it` | **Apache-2.0** — HF API `tags[]`, `gated: false` | 2026-07-20 | 11.96 B, multimodal `any-to-any`. ⚠️ robigo measured `codegemma:7b` at **448 KiB/token** KV and `gemma2:9b` at **336 KiB/token** — 6–8× worse than qwen2.5-coder-7b. Gemma 4 geometry must be re-measured, but the family has a history of expensive KV that eats a 16 GB card. | PORT (measure KV/token first) |
| **Devstral-Small-2-24B-Instruct-2512** | `huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512` | **Apache-2.0** — HF API | 2026-07-15 | 24 B — code-agent specialised. Above the stated 7–14B band and tight at Q4 on 16 GB (~13–14 GB weights), leaving no room for pytest. | REFERENCE (out of band) |
| **Qwen3-Coder-30B-A3B-Instruct** | `huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct` | **Apache-2.0** — HF API | 2025-12-03 | MoE, 3 B active. Q4 ≈ 17–18 GB — does **not** fit 16 GB without offload. | **SKIP** (does not fit) |
| **Qwen3-Coder-Next** | `huggingface.co/Qwen/Qwen3-Coder-Next` | **Apache-2.0** — HF API + card `license_link` | 2026-02-04 | **79.67 B total / 3 B activated** (card: "80B in total and 3B activated"), 262 144 ctx. Card's own serving examples assume **tensor parallel across 4 GPUs**. Q4 ≈ 40 GB. | **SKIP** (does not fit) |
| **Qwen3.8-27B** | `huggingface.co/Qwen/Qwen3.8-27B` | **Apache-2.0** — HF API | 2026-08-14 | Newest Qwen dense. Already on this box at UD-Q3_K_XL (14 GB) — but that is a Q3 quant filling almost the whole card, with no room for a test-runner alongside. | **SKIP** for baseline (no headroom) |

**VRAM arithmetic.** The card reports 16303 MiB total, and robigo measured only **14558 MiB actually free** on an idle desktop — the compositor holds the rest. Budget against 14.5 GB, not 16 GB. Weights + KV + a pytest subprocess must all fit. Qwen3.5-9B at Q6_K (7.46 GB) or Qwen2.5-Coder-14B at Q4_K_M (8.99 GB) both leave workable headroom; anything ≥ 12 GB of weights does not.

---

## (c) Local inference stack — fast batched proposal sampling

Requirements, in priority order: **n-best sampling**, **logprobs** (for the value function and for structural uncertainty), **LoRA hot-swap** (for sleep consolidation without a server restart), and batch throughput on one consumer GPU.

| Name | Source URL | License (verified how) | Last push | LoRA | Logprobs | n-best | Verdict |
|---|---|---|---|---|---|---|---|
| **vLLM** | `github.com/vllm-project/vllm` | **Apache-2.0** — `gh api repos/vllm-project/vllm --jq .license.spdx_id` | 2026-08-23 | ✅ **runtime hot-swap** | ✅ | ✅ | **ADOPT** (primary) |
| **SGLang** | `github.com/sgl-project/sglang` | **Apache-2.0** — `gh api` + PyPI classifier `License :: OSI Approved :: Apache Software License` | 2026-08-23 | ✅ multi-LoRA batching | ✅ | ✅ | **ADOPT** (alternate) |
| **llama.cpp** | `github.com/ggml-org/llama.cpp` | **MIT** — `gh api` | 2026-08-23 | ✅ **hot-swap via REST** | ✅ `n_probs` / `top_logprobs` | ✅ (slots) | **ADOPT** (fallback / already installed) |
| **Ollama** | `github.com/ollama/ollama` | **MIT** — `gh api` | 2026-08-22 | ⚠️ Modelfile-only | partial | limited | REFERENCE (already the box's default) |
| **LM Studio** | `github.com/lmstudio-ai/lms`, `.../lmstudio-js` | **MIT** on the CLI and JS SDK — `gh api`. ⚠️ **The LM Studio desktop app itself is closed-source**; only the tooling is MIT. | 2026-08-18 / 2026-08-21 | ⚠️ | ✅ | ⚠️ | **SKIP** (GUI-first, non-OSS core, unattended runs need a daemon) |
| **TGI** | `github.com/huggingface/text-generation-inference` | **Apache-2.0** — `gh api` | **2026-03-21** | ✅ | ✅ | ✅ | **SKIP** — 5 months stale vs vLLM's same-day pushes; HF's own effort moved on |
| **ExLlamaV3** | `github.com/turboderp-org/exllamav3` | **MIT** — `gh api` | 2026-08-22 | ✅ (README "LoRA support") | ✅ | ✅ dynamic batching | PORT — best VRAM efficiency; needs TabbyAPI for an OpenAI API |
| **ExLlamaV2** | `github.com/turboderp-org/exllamav2` | **MIT** — `gh api` | **2026-03-04** | ✅ | ✅ | ✅ | **SKIP** (superseded by v3) |
| **LoRAX** | `github.com/predibase/lorax` | **Apache-2.0** — `gh api` | **2026-05-28** | ✅ 1000s of adapters | ✅ | ✅ | REFERENCE — the multi-LoRA idea is right, but 3 months stale and TGI-derived |
| **LMDeploy** | `github.com/InternLM/lmdeploy` | **Apache-2.0** — `gh api` | 2026-08-21 | ✅ | ✅ | ✅ | REFERENCE |
| **TensorRT-LLM** | `github.com/NVIDIA/TensorRT-LLM` | ⚠️ `gh api` returns **`NOASSERTION`**; LICENSE file fetched — *"This project is licensed under the Apache 2.0 license"* with bundled third-party code under other licenses. Effectively **Apache-2.0 + mixed third-party**. | 2026-08-23 | ✅ | ✅ | ✅ | **SKIP** (engine-build friction not worth it for a spike) |
| **MLC-LLM** | `github.com/mlc-ai/mlc-llm` | **Apache-2.0** — `gh api` | 2026-08-17 | ⚠️ | ⚠️ | ✅ | REFERENCE |

### Verified capability details

**vLLM 0.27.1** (latest release 2026-08-11).
- **Model support:** `vllm/model_executor/models/` contains **`qwen3_5.py`** and `qwen3_5_mtp.py` — Qwen3.5 is natively supported, including the multi-token-prediction head.
- **n-best + logprobs:** `vllm/sampling_params.py` exposes `n`, `logprobs`, and `prompt_logprobs` as first-class fields. `prompt_logprobs` in particular gives us scoring of a candidate patch without regenerating it.
- **LoRA hot-swap:** `docs/features/lora.md` documents a `/v1/load_lora_adapter` POST endpoint gated on `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`, plus a `LoRAResolver` plugin that resolves an unseen adapter name per-request from local disk or S3. This is exactly the sleep-consolidation loop we need: train a LoRA, POST it, keep serving.
- **LoRA internals present:** `vllm/lora/` ships `model_manager.py`, `worker_manager.py`, `peft_helper.py`, `punica_wrapper/` — i.e. real batched multi-LoRA, not a merge-and-reload hack.

**SGLang 0.5.18.** `python/sglang/srt/models/` contains **`qwen3_5.py`, `qwen3_5_text.py`, `qwen3_5_mtp.py`** — note the dedicated **text-only** variant, which lets us skip loading the Qwen3.5 vision tower entirely. `python/sglang/srt/lora/` is a substantial subsystem (`lora_manager.py`, `lora_registry.py`, `lora_drainer.py`, `eviction_policy.py`, `lora_overlap_loader.py`) and the README advertises *"multi-LoRA batching"* alongside RadixAttention prefix caching. **RadixAttention is a genuine fit for tree search**: sibling candidate repairs share a long prefix, and prefix caching is exactly the optimisation that makes n-best tree expansion affordable.

**llama.cpp** (local checkout `~/llama.cpp` @ `4988f6e`, 2026-06-13).
- **Qwen3.5 supported:** `src/models/` contains **`qwen35.cpp`** and `qwen35moe.cpp` (alongside `qwen3next.cpp`) — the Gated DeltaNet hybrid is implemented.
- **LoRA hot-swap:** `common/arg.cpp` exposes `--lora`, `--lora-scaled`, `--lora-init-without-apply`. The server README documents **`GET /lora-adapters`** and **`POST /lora-adapters`** to set adapter scales at runtime, and a per-request `lora` field that overrides the global scale. Setting scale 0/1 per request means we can A/B the consolidated LoRA against the frozen base *within a single batch*.
- **Logprobs:** server README documents `n_probs` and `top_logprobs`.
- ⚠️ The local checkout is ~2 months old; rebuild before use.

**Blackwell / sm_120 risk.** The RTX 5080 is `sm_120`. vLLM requires **CUDA 12.8+** for Blackwell, and as of early 2026 prebuilt `xformers` wheels on PyPI only covered up to `sm_89` (Ada), which is a known source of install failures on RTX 50-series. There is an open vLLM issue (#31085) requesting SM120 support for native NVFP4 MoE kernels. **Treat the vLLM install as a spike task with a llama.cpp fallback**, not as a given.

### What this box already runs (read-only skim)

- **GPU:** `nvidia-smi` → NVIDIA GeForce RTX 5080, 16303 MiB, driver 595.84.
- **Ollama 0.32.13** is the incumbent server, with a large local model set already pulled: `qwen2.5-coder` at 0.5b/1.5b/7b-q8_0/14b-q4_K_M, `qwen3.8:27b`, `deepseek-coder-v2:16b`, `granite-code:8b-q8_0`, `codegemma:7b-q8_0`, plus project-specific VTT models.
- **Model format in use: GGUF**, served through Ollama, with llama.cpp checked out at `~/llama.cpp` for cases where GGUF geometry must be read directly.
- **`assay`** (`/home/brice/workspace/assay/README.md`): stdlib-only, zero-runtime-dependency capability prober. Speaks **`--backend ollama|openai`**. Its three findings are directly load-bearing for crucible: (1) silent front-truncation of oversized prompts, detected by a canary instruction at the prompt head; (2) the "stats-free-200" ceiling class — HTTP 200 with plausible text but **no token counts**, treated as a `ContractViolation`; (3) **the codec landing split** — granite-code:8b landed ≈0 % of SEARCH/REPLACE edits where qwen2.5-coder:7b landed ≥90 % on identical prompts. Fifteen models are already profiled on this exact hardware (tier-enthusiast-2026-08).
- **`robigo`** (`/home/brice/workspace/robigo/README.md`): a VRAM-budgeted coding agent that repairs **one failing test at a time** — structurally the closest existing thing to crucible's inner loop. Backends: **Ollama** (geometry over HTTP) and **llama.cpp** (`--gguf` path, because `llama-server` does not expose KV head counts over HTTP). Contributes measured KV-cost-per-token figures (qwen2.5-coder-7b **56 KiB/tok** … codegemma-7b **448 KiB/tok**, a 14× spread at comparable parameter counts), the 14558 MiB-free measurement, a fixed 5-rung context degradation ladder chosen *by measurement not arithmetic*, and per-run recording to `.robigo/runs/<id>/` with exact prompt, reply and test output.

**Implication for crucible:** the box is a **GGUF/Ollama** shop today, and there is **no `torch` in the system Python** (`import torch` → `ModuleNotFoundError`), so vLLM, PEFT and TRL are all greenfield installs. `assay` gives us a ready-made instrument for validating any new endpoint before trusting it, and `robigo` already solved the context-budget and run-recording problems.

---

## (d) LoRA fine-tuning tooling for sleep consolidation (must fit 16 GB for a 1.5–3B model)

| Name | Source URL | License (verified how) | Last push | QLoRA | Grad ckpt | Own JSONL | Verdict |
|---|---|---|---|---|---|---|---|
| **PEFT** | `github.com/huggingface/peft` | **Apache-2.0** — `gh api` + PyPI classifier | 2026-08-22 | ✅ | ✅ | n/a (library) | **ADOPT** (core) |
| **TRL** | `github.com/huggingface/trl` | **Apache-2.0** — `gh api .license.spdx_id` | 2026-08-23 | ✅ | ✅ | ✅ | **ADOPT** (trainer) |
| **bitsandbytes** | `github.com/bitsandbytes-foundation/bitsandbytes` | **MIT** — `gh api` | 2026-08-19 | ✅ (the 4-bit kernel) | n/a | n/a | **ADOPT** (dependency) |
| ⚠️ **Unsloth** | `github.com/unslothai/unsloth` | ⚠️ **DUAL Apache-2.0 + AGPL-3.0** — repo has **both** `LICENSE` (Apache 2.0, verified by fetch) **and** `COPYING` (**GNU AFFERO GPL v3**, verified by fetch). README §License: *"Unsloth uses a dual-licensing model of Apache 2.0 and AGPL-3.0. The core Unsloth package remains licensed under Apache 2.0, while certain optional components, such as the Unsloth Studio UI are licensed under … AGPL-3.0."* | 2026-08-23 | ✅ | ✅ | ✅ | **ADOPT — core only.** Do **not** import, vendor or ship `unsloth/studio` or anything under `COPYING`. |
| **Axolotl** | `github.com/axolotl-ai-cloud/axolotl` | **Apache-2.0** — `gh api` (note: `OpenAccess-AI-Collective/axolotl` redirects here) | 2026-08-21 | ✅ | ✅ | ✅ YAML+JSONL | PORT |
| **LLaMA-Factory** | `github.com/hiyouga/LLaMA-Factory` → `hiyouga/LlamaFactory` | **Apache-2.0** — `gh api`; PyPI `llamafactory` 0.9.5 classifier `License :: OSI Approved :: Apache Software License` | 2026-08-20 | ✅ | ✅ | ✅ | PORT |
| **torchtune** | `github.com/meta-pytorch/torchtune` (`pytorch/torchtune` redirects here) | **BSD-3-Clause** — `gh api` + PyPI `.info.license` = full BSD-3 text, "Copyright 2024 Meta" | 2026-08-22 | ✅ | ✅ | ✅ | REFERENCE — ⚠️ `recipes/configs/` tops out at **`qwen3`**; **no Qwen3.5 config exists** |

### Verified capability details

- **PEFT 0.20.0** exports `LoraConfig`, `get_peft_model`, and **`prepare_model_for_kbit_training`** (verified by decoding `src/peft/__init__.py`) — the standard QLoRA entry point. `src/peft/utils/` also contains **`hotswap.py`**, a first-class adapter hot-swap utility, and `loftq_utils.py` / `quantization_utils.py`. PEFT + bitsandbytes is the shortest path from "verified episodes JSONL" to "a LoRA vLLM can load over HTTP".
- **TRL 1.10.0** ships `sft_trainer.py`, `dpo_trainer.py`, and **`grpo_trainer.py`**. `SFTTrainer` handles our JSONL consolidation directly. **GRPO is worth flagging beyond the brief**: crucible already has a verifier (the test suite) that emits a scalar reward per candidate repair, which is precisely GRPO's input contract — so the same library covers both the SFT consolidation step *and* a possible RL-on-verified-outcomes extension.
- **Unsloth 2026.8.19** (PyPI) claims Qwen3.8, Gemma 4 and DeepSeek-V4 support in its repo description — the most current model coverage of any tool here, and the best 16 GB memory profile. The dual-license split is the catch: the **core package is Apache-2.0** and safe, but `COPYING` really is AGPL-3.0 and it covers the Studio UI. Since we only need the training API, this is manageable — but it must be an explicit rule in the repo, not a hope.
- **VRAM feasibility.** A 2–3 B model under QLoRA (4-bit base + bf16 adapters + gradient checkpointing + paged AdamW) is roughly 1.5–2.5 GB of frozen weights plus optimiser/activation overhead — comfortably inside 14.5 GB with the inference server stopped. **The real constraint is time-sharing, not capacity**: consolidation and serving cannot both hold the card. Design the sleep phase as an explicit stop-serve → train → reload-adapter cycle.

---

## (e) Python mutation tools — the bug-injection stream

Our contract is narrow and it is **not** what these tools are built for: we need *"generate mutants for file X with operators Y"* returning **(mutated source, location, operator name)**, called from Python, with **no** CLI, no config file, no session DB, and no test execution — because crucible applies the mutant and runs the tests itself. Most mutation-testing tools bury generation inside a run loop. The survey verified the API by installing and executing the candidates, not by reading docs.

| Name | Source URL | License (verified how) | Last push | Key facts | Verdict |
|---|---|---|---|---|---|
| **cosmic-ray** | `github.com/sixty-north/cosmic-ray` | **MIT** — `gh api .license.spdx_id` = `MIT`; PyPI `.info.license` = full MIT text | 2026-08-09 (v8.7.0) | Pure `mutate_code(code, operator, occurrence) -> str \| None`. **213 operators.** Installed and run end-to-end: returned mutated source + `((line,col),(line,col))` span + operator name, with no CLI/DB/config/test-run touched. 653★ | **ADOPT** (primary) |
| **poodle** | `github.com/WiredNerd/poodle` | **MIT** — `gh api`; PyPI full MIT text | 2026-04-05 (v1.3.4) | `create_mutations(ast, file_lines) -> list[FileMutation(mutator_name, lineno, col_offset, end_lineno, end_col_offset, text)]` — span **plus replacement text**, the ideal shape. Only 5★ (bus factor) | **ADOPT** (secondary; vendor, don't depend) |
| **pytest-gremlins** | `github.com/mikelane/pytest-gremlins` | **MIT** — `gh api`; PyPI agrees | 2026-08-17 (v1.9.0) | Cleanest operator primitive: `GremlinOperator` Protocol with `can_mutate(node)` / `mutate(node) -> list[ast.AST]`, importing only `ast`/`copy`. Its file-walker only implements `visit_Compare` | **PORT** (operator interface) |
| **mutpy** | `github.com/mutpy/mutpy` | **Apache-2.0** — API said `NOASSERTION`; LICENSE file fetched = "Licensed under the Apache License, Version 2.0", © Konrad Hałas | 2024-04-23 | Richest operator set: 20 standard + 7 experimental **including statement deletion (SDL)**. `MutationOperator.mutate(...)` yields `(Mutation, new_node)`. ⚠️ PyPI frozen at 0.6.1 (**2019**); `ast.Num`/`ast.Str` break on Py3.12+ | **PORT** (operators only) |
| **mutmut** | `github.com/boxed/mutmut` | **BSD-3-Clause** — `gh api`; LICENSE "Copyright (c) 2016, Anders Hovmöller"; PyPI agrees | 2026-08-17 (v3.7.0) | Most active, most starred (1399★), libcst-based. But `create_mutations()` **raised `FileNotFoundError` until a `setup.cfg` was added** — coupled to global `Config.get()`. Returns nodes with **no operator name and no position** | **REFERENCE** |
| **mutatest** | `github.com/EvanKepner/mutatest` | **MIT** — `gh api` | 2023-02-17 (PyPI 2022) | Best API *design*: `Genome.targets -> Set[LocIndex]`, `.mutate(idx, op) -> Mutant`. Confirmed still runs on Py3.14. ⚠️ `Mutant.mutant_code` is a **`code` object, not source** | **REFERENCE** (API design) |
| ⚠️ **universalmutator** | `github.com/agroce/universalmutator` | ⚠️ **CONFLICTED** — LICENSE file = Apache-2.0 text; `setup.py` = `license='MIT'`; PyPI = MIT. Both permissive, but the metadata disagrees with itself | 2026-05-20 (v1.14.1) | Regex `.rules` text substitution, not AST. All logic inside `genmutants.main()` behind argparse — **no importable API**, no reliable column info | **SKIP** |
| ⚠️ **mutahunter** | `github.com/codeintegrity-ai/mutahunter` | ⚠️ **AGPL-3.0** — `gh api .license.spdx_id` = `AGPL-3.0`; LICENSE header "GNU AFFERO GENERAL PUBLIC LICENSE Version 3" | 2025-04-17 | **Copyleft.** Importing it would relicense crucible, and AGPL's network clause reaches past distribution. Also LLM-driven, therefore non-deterministic — wrong for a controlled bug-injection stream | **SKIP (copyleft)** |
| **typemut** | `github.com/nkhitrov/typemut` | **MIT** — LICENSE "MIT License © 2026 Nick Khitrov" | 2026-06-17 | Mutates **type annotations only**; shells `test_command` itself and keeps a session DB | **SKIP** (wrong scope) |
| **LibCST** (substrate) | `github.com/Instagram/LibCST` | **MIT** — API `NOASSERTION`; LICENSE file "All contributions towards LibCST are MIT licensed" (+PSF dual on ~6 parser files) | 2026-08-11 (v1.9.0) | Not a mutation tool. Lossless round-trip; a `PositionProvider` visitor yielded `(line, col, op-type)` in ~15 lines | **ADOPT** (fallback substrate) |
| **parso** (substrate) | `pypi.org/project/parso` | **MIT** — PyPI `.info.license` = `MIT` | v0.8.7 @ 2026-05-01 | cosmic-ray's parser; also lossless round-trip | ADOPT (transitive) |

### Verified API details

**cosmic-ray — contract proven by execution.** `cosmic_ray.mutating.mutate_code(code: str, operator, occurrence: int) -> str | None` is pure string-in/string-out. Enumerate with `cosmic_ray.ast.get_ast(src)` + `ast_nodes(tree)` + `operator.mutation_positions(node)`, which yields `((start_line,start_col),(end_line,end_col))`. The `Operator` ABC is just two methods (`mutation_positions`, `mutate`). Operator classes import directly (`from cosmic_ray.operators.binary_operator_replacement import ReplaceBinaryOperator_Add_Sub`), so the stevedore plugin registry can be bypassed entirely — or use `cosmic_ray.plugins.operator_names()`, which returned **213** names. **Live check:** `ReplaceBinaryOperator_Add_Sub` reported positions `[((3,17),(3,18))]` and `mutate_code(src, op, 0)` returned source with `return a - b`; `ReplaceComparisonOperator_Lt_GtE` produced `if a >= b`.
**Operators:** binary (Add/Sub/Mul/Div/FloorDiv/Mod/Pow/RShift/LShift/BitOr/BitAnd/BitXor as a full permutation cross-product), comparison (`==`,`!=`,`<`,`<=`,`>`,`>=`,`is`,`is not` cross-product), unary (`+`,`-`,`~`,`not`, plus deletion), boolean (`AddNot`, True↔False, and↔or), break↔continue, `ExceptionReplacer`, `NumberReplacer`, `RemoveDecorator`, `ZeroIterationForLoop`, `VariableReplacer`, `VariableInserter`. **No statement/line deletion.**
**Coupling:** the SQLite `WorkDB` and `run_tests` appear only in `commands/init.py` and `mutate_and_test()`; the generation primitives touch neither.
⚠️ **Gotcha:** `Operator.mutate()` mutates the parso node **in place**, so re-parse per mutant — `mutate_code` already does.

**poodle — best output shape.** `FileMutation` carries a span *and* the **replacement snippet**, so splicing leaves the rest of the file byte-identical. Mutators: Comparison, Number, String, Keyword, BinaryOperation, AugAssign, UnaryOperation, Decorator, DictArrayCall, FunctionCall, LambdaReturn, Return. `Mutator.__init__` wants a `PoodleConfig`, but only two `self.config` reads exist across all five mutator modules — a stub with `mutator_opts={}` suffices.

**mutmut — reject despite being the most popular.** Three blockers confirmed by running it: (1) `create_mutations()` raised `FileNotFoundError: Could not figure out where the code to mutate is` because `Config.get()` reads `pyproject.toml`/`setup.cfg` from **cwd**; (2) `Mutation` carries **no operator name and no line/col**; (3) it returns mutated *nodes*, and mutmut's actual model merges **all** mutants into one trampolined file selected at runtime by env var — the opposite of one-mutant-per-file. These are internal symbols with no stability guarantee.

**mutpy — harvest the operators, not the package.** `MutationOperator.mutate(...)` is a generator yielding `(Mutation, new_node)` with `.operator` and `.node.lineno/.col_offset`; sets are importable frozensets (`standard_operators`, `experimental_operators`); the test runner is a separate `MutationController`. Its operator catalogue is the broadest surveyed: arithmetic replace + deletion, assignment, relational, logical connector/operator replace + delete, conditional insert/delete, constant replacement, slice-index removal, break/continue swap, exception-handler deletion, exception swallowing, decorator deletion, inheritance/super ops, **statement deletion (SDL)**, loop zero/one/reverse iteration.

**mutatest — right shape, wrong payload.** `LocIndex(ast_class, lineno, col_offset, op_type, end_lineno, end_col_offset)` is precisely crucible's tuple, and discovery works on Py3.14. But `Genome.mutate()` returns a `Mutant` whose `mutant_code` is a **compiled `code` object** destined for `__pycache__` (`type(m.mutant_code) is code`, verified). Getting source back requires bypassing it with `MutateAST(...).visit(...)` + `ast.unparse`, which reformats the whole file. 12 operator categories, **no string or statement-deletion operators.** Unmaintained since 2023-02.

⚠️ **`__pycache__` interaction — a known trap on this box.** A same-length mutation restored within the same second lets CPython run the **mutant** from stale bytecode after the restore. Any mutate→test→restore loop must purge `__pycache__` and set `PYTHONDONTWRITEBYTECODE=1`. This is not hypothetical; it has already bitten a prior project here.

### Recommendation for (e)

**cosmic-ray is the primary mutant generator.** It is the only tool where the exact required contract was demonstrated by execution — mutated source, `(line,col)` spans, 213 operator labels — with zero coupling to its CLI, work-DB, config, or runner. MIT, actively released. Import the operator classes directly and skip even the entry-point lookup.
**Gap:** cosmic-ray has no statement-deletion operator. Port MutPy's `StatementDeletion` (Apache-2.0; rewrite `ast.Num`/`ast.Str` → `ast.Constant` for Py3.12+) against cosmic-ray's two-method `Operator` ABC so it drops into the same pipeline.
**Fallback:** a ~200-line mutator on **LibCST (MIT)**, whose lossless round-trip guarantees a mutant differs from the original *only* at the mutated span — unlike `ast.unparse`. Model the output type on poodle's `FileMutation` and the operator interface on pytest-gremlins' `GremlinOperator` Protocol; both are MIT and small enough to vendor outright.

## (f) Seed corpora with executable tests

Our unit of task is **"small pure-Python module + a passing test suite that runs offline"**. That constraint is more discriminating than it looks: it eliminates almost the entire SWE-bench family, which is Docker-per-instance by construction. Licenses were verified per-dataset via the HF API, which turned up two datasets whose card carries **no license at all** even though their harness is MIT.

| Name | Source URL | License (verified how) | Last push/modified | Size | Tests offline? | Verdict |
|---|---|---|---|---|---|---|
| **QuixBugs** | `github.com/jkoppel/QuixBugs` | **MIT** — `gh api .../license` → spdx `MIT`; LICENSE text read ("Copyright 2017-2019 James Koppel"); plus `legal_notes.txt` provenance | 2022-08-29 | 40 programs / 278 pytest cases | ✅ **YES — verified by running it** | **ADOPT** (held-out set) |
| **HumanEvalPack / HumanEvalFix** | `hf.co/datasets/bigcode/humanevalpack` | **MIT** — HF API `cardData.license` = `"mit"`; tag `license:mit` | 2025-08-19 | 164 Python (×6 langs) | ✅ YES — self-contained asserts, stdlib only | **ADOPT** (primary) |
| **EvalPlus HumanEval+** | `hf.co/datasets/evalplus/humanevalplus` | **Apache-2.0** — HF API `"apache-2.0"`; upstream `openai/human-eval` = **MIT** via `gh api` | 2024-05-01 | 164 | ✅ YES after one cached download (load verified) | **ADOPT** (primary) |
| **EvalPlus MBPP+** | `hf.co/datasets/evalplus/mbppplus` | **Apache-2.0** — HF API; upstream MBPP in `google-research` = **Apache-2.0** via `gh api` | 2024-04-17 | 378 | ✅ YES, needs numpy | **ADOPT** (primary) |
| **BigCodeBench** | `hf.co/datasets/bigcode/bigcodebench` | **Apache-2.0** — HF API; repo `bigcode-project/bigcodebench` Apache-2.0 via `gh api` | 2026-01-03 (repo) | 1140 × 5 versions | ⚠️ Partly — `--execution local` exists but default path is remote; ~140 third-party libs | **PORT** (second held-out tier) |
| **SWE-smith** | `github.com/SWE-bench/SWE-smith` | **MIT** — `gh api .../license` → MIT; HF card `"mit"` | 2026-08-17 | 59 136 instances | ❌ NO — README: *"requires Docker to create execution environments … do not plan on supporting Windows or MacOS"* | **REFERENCE** (design input) |
| **SWE-bench Lite / Verified** | `hf.co/datasets/princeton-nlp/SWE-bench_{Lite,Verified}` | ⚠️ **Harness MIT** (`gh api` repo); **dataset card license = `null`** (HF API; `cardData` keys are only `configs`/`dataset_info`) | 2025-03-03 / 2025-02-18 | 323 / 500 | ❌ NO — README requires Docker, **120 GB disk, 16 GB RAM, 8 cores** | **SKIP** as task stream |
| **SWE-bench Multilingual** | `hf.co/datasets/SWE-bench/SWE-bench_Multilingual` | **MIT** — HF API `"mit"` | 2026-08-17 | 300 | ❌ NO — Docker | SKIP |
| **SWE-bench Multimodal** | `hf.co/datasets/SWE-bench/SWE-bench_Multimodal` | ⚠️ **`null`** on card — HF API | 2026-08-18 | 480 test / 100 dev | ❌ NO — Docker, JS/web | SKIP |
| **SWE-Gym** | `github.com/SWE-Gym/SWE-Gym` | ⚠️ **MISMATCH** — repo **Apache-2.0** (`gh api`) vs **HF card `"mit"`**. Resolve before redistributing | 2025-07-29 | ~2.4 k | ❌ NO — Docker | REFERENCE |
| **R2E-Gym** | `github.com/R2E-Gym/R2E-Gym` | **Apache-2.0** — `gh api`; HF card `"apache-2.0"` | 2025-07-13 | 4 578 | ❌ NO — README: docker image **300–500 MB per instance** | REFERENCE |
| **SWE-rebench** | `hf.co/datasets/nebius/SWE-rebench` | ⚠️ **CC-BY-4.0** — HF API | 2025-12-23 | 21 336 / 6 542 filtered | ❌ NO — Docker | REFERENCE |
| **SWE-Perf** | `hf.co/datasets/SWE-Perf/SWE-Perf` | **Apache-2.0** — HF API | 2025-08-05 | 140 | ❌ NO — Docker; measures perf not correctness | SKIP |
| ⚠️ **BugsInPy** | `github.com/soarsmu/BugsInPy` | 🚫 **NO LICENSE AT ALL** — `gh api .../license` → **404**; `gh api repos/.../contents` shows no LICENSE file; code search for LICENSE → **0 hits**; `.license.spdx_id` = `null` | 2026-02-10 | 17 projects (493 bugs claimed) | ❌ NO — Docker + per-bug `pip install` (network) | 🚫 **SKIP — legal blocker** |
| **RepoBench** | `github.com/Leolty/RepoBench` | ⚠️ **CC-BY-4.0** — `gh api` spdx; HF card `"cc"` (unversioned) | 2024-08-16 | — | n/a — completion task, **no test suites** | SKIP |
| **LiveCodeBench** | `github.com/LiveCodeBench/LiveCodeBench` | ⚠️ repo **MIT** (`gh api`); **dataset card `license: cc` — bare, no version.** "cc" alone is not a license | 2025-07-16 | <1 k | ⚠️ Partly — stdin/stdout, not pytest | SKIP |
| ⚠️ **PyBugHive** | paper @ `publicatio.bibl.u-szeged.hu` | 🚫 **CC BY-NC-ND 4.0** — paper text | 2024-08-23 | — | Docker offered | 🚫 **SKIP** — NonCommercial **and** NoDerivatives; ND alone blocks redistributing a mutated derivative |
| ⚠️ **SWE-bench Pro** | `labs.scale.com` | 🚫 Public set **deliberately built on GPL/copyleft repos** to reduce contamination risk | 2026 | 1 865 | ❌ NO — Docker | 🚫 **SKIP — copyleft by design** |

### Detail

**QuixBugs — the ideal held-out set, and the only corpus verified by actually running it.** A unit is `python_programs/X.py` (one ~15-line stdlib-only algorithm) + `python_testcases/test_X.py` (parametrized pytest) + `correct_python_programs/X.py` as oracle. **Zero surgery** — it is already a pytest repo with a root `conftest.py` and a `--correct` flag. Measured: correct side `276 passed, 2 skipped in 0.58s`; buggy side `187 failed, 89 passed, 2 skipped in 87.67s`.
🔴 **Operational finding that changes crucible's harness design:** the buggy run **hung indefinitely** and had to be killed after >120 s. Several defective variants infinite-loop. **Per-test timeouts (`pytest-timeout`) or subprocess isolation are mandatory**, not optional, the moment you inject mutations — and this generalises: a mutation that turns `i += 1` into `i -= 1` inside a `while` is a *very* common mutant and it does not fail, it hangs. Budget for wall-clock kill, not just assertion failure.

**HumanEvalPack / HumanEvalFix — the closest existing thing to crucible's own task shape.** Row fields: `declaration`, `canonical_solution`, **`buggy_solution`**, `test`, **`bug_type`**, **`failure_symptoms`**, `entry_point`, `test_setup`. The `test` field is a bare `def check(fn): assert ...` block — concatenate `declaration + solution + test + check(entry_point)` and it runs on stdlib Python. Uniquely valuable: it ships a **human-authored bug taxonomy**, which lets us check that our mutation operators produce a *realistic* bug distribution rather than an artificial one. That is a direct validity check on the whole "mutation-injected bugs" premise.

**EvalPlus** — verified loadable: a one-time fetch of `HumanEvalPlus.jsonl.gz` / `MbppPlus.jsonl.gz` from GitHub releases, cached, offline thereafter. Keys: `prompt, canonical_solution, entry_point, test, contract, base_input, plus_input, atol`. ⚠️ **Porting cost is real:** the "tests" are **input vectors + a canonical-solution oracle**, not a pytest file — we must generate asserts by running the canonical solution over `base_input + plus_input`. Mechanical, but it is a build step, not a download.

**BigCodeBench** — the only corpus here with genuine `unittest.TestCase` suites (`import unittest; from unittest.mock import patch; class TestCases(unittest.TestCase)`) and realistic third-party imports, which makes it the best proxy for "real module + real test suite". Costs: ~140 third-party libs across tasks (pandas, sklearn, requests, flask) so offline needs a pre-baked wheelhouse, and the default eval path is **remote** (gradio/e2b with an `E2B_API_KEY`) with `--execution local` present but not the maintainers' preferred route.

**SWE-smith — read it, don't run it.** Conceptually crucible's twin: *"turn any GitHub repo into a gym, create unlimited tasks, keep tasks that break 1+ unit tests."* MIT, 59 k instances. Its **bug-synthesis operators** and its **"keep tasks that break ≥1 test" filter** are exactly the design decisions we face, and are worth mining. But it is Docker-per-repo by construction and Ubuntu-only.

**BugsInPy — the Defects4J-for-Python everyone cites, and it is legally unusable.** It has **no license file whatsoever** (three independent checks agree: `/license` → 404, contents listing has no LICENSE, code search → 0 hits). Default copyright is all-rights-reserved, which forecloses redistributing any derived task set. Separately, `bugsinpy-compile` pip-installs each project's requirements at checkout, so it is not offline either. Double blocker — **do not build on it**.

### Recommendation for (f)

Build the **training task stream** from **HumanEvalPack (MIT, 164)** + **EvalPlus HumanEval+/MBPP+ (Apache-2.0, 542)**: ~700 seed units, all single-file pure Python with self-contained assertions, all offline under plain pytest, all cleanly redistributable. That is enough to generate thousands of injected mutants.
Keep **QuixBugs (MIT, 40)** as the **held-out generalization set** — small, independent-origin (2011–13 Quixey interview problems, **pre-LLM**, so low contamination risk), and it ships *human-written* one-line defects we did not generate. It therefore measures transfer rather than mutation-operator overfit, which is precisely the confound our design is most exposed to.
Promote **BigCodeBench (Apache-2.0, 1140)** to a second held-out tier once a vendored wheelhouse exists.
🚫 **Hard exclusions:** BugsInPy (no license), PyBugHive (CC BY-NC-ND), SWE-bench Pro (GPL by design). ⚠️ **Resolve before redistributing:** SWE-bench dataset cards carry no license despite an MIT harness; SWE-Gym's repo (Apache-2.0) contradicts its HF card (MIT); LiveCodeBench's card says bare `"cc"`, which is not a license.

---

## Recommended stack

| Role | Choice | Reason |
|---|---|---|
| **Proposer** | **Qwen3.5-2B** (Apache-2.0), Qwen2.5-Coder-1.5B as the code-specialist arm | Only 2026-current, permissive, sub-3B family with Base checkpoints; Q4_K_M is 1.28 GB, leaving the card free for search + tests. |
| **Baseline** | **Qwen3.5-9B @ Q6_K** (7.46 GB), cross-check vs **Qwen2.5-Coder-14B @ Q4_K_M** (8.99 GB) | Same architecture family as the proposer isolates *scale* as the variable; LiveCodeBench v6 65.6 vs 4B's 55.8. Both already fit with pytest headroom. |
| **Inference server** | **vLLM** (Apache-2.0), **llama.cpp** (MIT) as fallback | vLLM has `n` + `logprobs` + `prompt_logprobs` and `/v1/load_lora_adapter` runtime hot-swap — all three requirements in one server. llama.cpp is already on the box, supports `qwen35`, and hot-swaps LoRA via `POST /lora-adapters`. |
| **LoRA tool** | **PEFT + TRL + bitsandbytes** (Apache-2.0 / Apache-2.0 / MIT) | `prepare_model_for_kbit_training` + `SFTTrainer` is the shortest verified path from episode JSONL to a servable adapter; TRL's `GRPOTrainer` also covers the RL-on-verified-outcomes extension. Unsloth core is Apache-2.0 and faster, but has **no `models/qwen3_5.py`** — its Qwen3.5 code sits under the **AGPL** `studio/` tree. |
| **Mutation tool** | **cosmic-ray** (MIT), + MutPy's `StatementDeletion` ported | Only tool where `mutate_code(src, op, occurrence)` → (source, span, operator) was proven by execution with zero CLI/DB/config coupling. 213 operators. |
| **Seed corpus** | **HumanEvalPack + EvalPlus** (train, ~700 units); **QuixBugs** (held out) | Offline, no Docker, redistributable, single-file pure Python. QuixBugs is pre-LLM and human-authored, so it measures transfer rather than operator overfit. |

## Gaps and risks

1. 🔴 **Blackwell install risk is the #1 schedule threat.** The 5080 is `sm_120`; vLLM needs CUDA 12.8+, prebuilt `xformers` wheels historically topped out at `sm_89`, and vLLM issue #31085 shows SM120 kernel coverage is still being filled in. **There is no `torch` in this box's Python at all** — vLLM/PEFT/TRL are a greenfield install on a GPU generation with known wheel gaps. Timebox the vLLM spike and keep llama.cpp (already building here) as the committed fallback.
2. 🔴 **Infinite-loop mutants are a correctness requirement, not an edge case.** Measured on QuixBugs: the buggy suite hung >120 s and needed `--timeout=5` to complete. Loop-counter mutations *hang* rather than fail. Every test execution in the value function must be wall-clock bounded and subprocess-isolated, or the tree search deadlocks.
3. 🔴 **`__pycache__` can execute the mutant after you restore the original.** A same-length mutation restored within the same second leaves CPython running stale bytecode. Purge `__pycache__` and set `PYTHONDONTWRITEBYTECODE=1` in the harness. Already observed on this box in a prior project.
4. 🟠 **Qwen3.5 is multimodal and architecturally novel.** Every Qwen3.5 checkpoint is `image-text-to-text` with a vision encoder, and the LM is a **Gated DeltaNet + Gated Attention hybrid**, not a plain transformer. Two consequences: (a) load the text-only path (SGLang ships `qwen3_5_text.py`; GGUF splits the tower into a separate `mmproj-*.gguf`) or you waste VRAM on a tower we never use; (b) **PEFT LoRA target-module selection is unverified for GDN layers** — `target_modules` names differ from the usual `q_proj/k_proj/v_proj/o_proj`. Verify adapter attachment on Qwen3.5-2B before committing, and keep Qwen2.5-Coder-1.5B (plain `Qwen2ForCausalLM`) as the LoRA-safe fallback.
5. 🟠 **Mutation-injected bugs may not resemble real bugs.** The whole "second exposure" design rests on the injected distribution being meaningful. HumanEvalFix's `bug_type` / `failure_symptoms` fields are the cheapest available validity check — compare our operator distribution against their human-authored taxonomy before trusting any transfer claim.
6. 🟠 **cosmic-ray has no statement-deletion operator**, which is one of the more realistic human bug classes. Porting MutPy's SDL requires rewriting `ast.Num`/`ast.Str` → `ast.Constant` for Py3.12+.
7. 🟠 **EvalPlus is not pytest-shaped.** Its tests are input vectors plus a canonical-solution oracle; generating assert files is a real build step. Budget for it rather than assuming a download.
8. 🟡 **Codec landing is per-model and must be measured, not assumed.** `assay` already measured granite-code:8b landing ≈0 % of SEARCH/REPLACE edits where qwen2.5-coder:7b landed ≥90 %. Whatever patch format the proposer emits, profile it with `assay` **before** interpreting any repair-rate number — a 0 % result may be a codec failure, not a reasoning failure.
9. 🟡 **VRAM is 14.5 GB, not 16 GB** (measured free on an idle desktop), and KV cost per token varies **14×** across models of similar size. Consolidation and serving cannot co-reside; design sleep as an explicit stop-serve → train → reload-adapter cycle.
10. 🟡 **Licensing landmines confirmed in-band:** `Qwen2.5-Coder-3B` is non-commercial (its 1.5B/7B/14B siblings are not); unsloth ships an AGPL `COPYING` alongside its Apache `LICENSE`; BugsInPy has no license; SWE-bench dataset cards are license-less; SWE-Gym's repo and HF card disagree.
11. 🟡 **`assay` and `robigo` are prior art we should reuse, not re-derive** — context budgeting, the degradation ladder, per-run recording of exact prompt/reply/test-output, and endpoint validation are all already built and measured on this exact hardware.

## Evidence log

Every license claim above traces to one of these. Commands are terse; outputs are the load-bearing fields only.

### Environment
```
nvidia-smi --query-gpu=name,memory.total,driver_version → NVIDIA GeForce RTX 5080, 16303 MiB, 595.84
ollama --version → 0.32.13
ollama list → qwen2.5-coder{0.5b,1.5b,1.5b-q8_0,7b-q8_0,14b-q4_K_M}, qwen3.8:27b, deepseek-coder-v2:16b,
              granite-code:8b-q8_0, codegemma:7b-q8_0, hf.co/unsloth/Qwen3.8-27B-GGUF:UD-Q3_K_XL
python3 -c "import torch" → ModuleNotFoundError: No module named 'torch'   ← no torch on the box
git -C ~/llama.cpp log -1 → 4988f6e 2026-06-13
```

### (a)/(b) Models — HF API
```
curl .../api/models/Qwen/Qwen2.5-Coder-1.5B-Instruct  → license apache-2.0, 2025-01-12, dl 861059
curl .../api/models/Qwen/Qwen2.5-Coder-3B-Instruct    → license "other", license_name "qwen-research"   ⚠️
curl .../api/models/Qwen/Qwen2.5-Coder-0.5B-Instruct  → apache-2.0 | -7B-Instruct → apache-2.0 | -14B-Instruct → apache-2.0
curl .../Qwen2.5-Coder-3B-Instruct/raw/main/LICENSE → "Qwen RESEARCH LICENSE"; §1(i) Non-Commercial = "research or
     evaluation purposes only"; §2(a) "FOR NON-COMMERCIAL PURPOSES ONLY"; §2(b) commercial ⇒ request a license;
     §4(b) must display "Built with Qwen"
curl .../api/models?author=Qwen&sort=lastModified → Qwen3.8-27B (2026-08-14), Qwen3.8-2.4T-A95B (2026-08-12),
     Qwen3.6-27B / Qwen3.6-35B-A3B / Qwen3.5-{27B,122B-A10B,397B-A17B} (2026-04-24),
     Qwen3.5-{0.8B,2B,4B,9B}-Base (2026-04-23), Qwen3.5-{0.8B,2B,4B,9B} (2026-03-02), Qwen3-Coder-Next-GGUF (2026-02-04)
curl .../api/models/Qwen/Qwen3.5-{0.8B,2B,4B,9B,27B} → all apache-2.0, gated:false
curl .../api/models/Qwen/Qwen3.5-{0.8B,2B,4B,9B}-Base → all apache-2.0
curl .../api/models/Qwen/{Qwen3.6-27B,Qwen3.6-35B-A3B,Qwen3.8-27B,Qwen3-Coder-Next-GGUF,Qwen3-Coder-30B-A3B-Instruct}
     → all apache-2.0
curl .../api/models?author=Qwen&search=Coder → newest official coder = Qwen3-Coder-Next (2026-02); NO coder ≤14B in 2026
safetensors.total: Qwen3.5-0.8B 873438784 | 2B 2274069824 | 4B 4659865088 | 9B 9653104368 | Qwen3.6-27B 27781427952
     | Qwen2.5-Coder-1.5B-Instruct 1543714304 | SmolLM3-3B 3075098624 | granite-4.0-micro 3402836480
     | Qwen3-Coder-Next 79674391296
config.architectures: Qwen3.5-* → Qwen3_5ForConditionalGeneration | Qwen2.5-Coder → Qwen2ForCausalLM
     | SmolLM3 → SmolLM3ForCausalLM | granite-4.0-micro → GraniteMoeHybridForCausalLM
curl .../Qwen3.5-4B/raw/main/README.md → license apache-2.0; "Number of Parameters: 4B"; hidden 2560; 32 layers;
     "Hidden Layout: 8 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))"; ctx 262144 → 1010000;
     LiveCodeBench v6 row: GPT-OSS-120B 82.7 | GPT-OSS-20B 74.6 | Qwen3-Next-80B-A3B-Thinking 68.7
       | Qwen3-30BA3B-Thinking-2507 66.0 | Qwen3.5-9B 65.6 | Qwen3.5-4B 55.8
curl .../Qwen3.5-2B/raw/main/README.md → "Number of Parameters: 2B"; 24 layers; hidden 2048; ctx 262144;
     parsed every <tr>: NO LiveCodeBench/HumanEval row for 2B. MMLU-Pro 55.3 | SuperGPQA 30.4 | GPQA 51.6
curl .../Qwen2.5-Coder-1.5B-Instruct/raw/main/README.md | grep bench → only "Context Length: Full 32,768 tokens"
curl .../api/models/HuggingFaceTB/SmolLM3-3B → apache-2.0, 2025-09-10; card: LiveCodeBench v4 SmolLM3 15.2 |
     Qwen2.5-3B 10.5 | Llama3.1-3B 3.4 | Qwen3-1.7B 15.0 | Qwen3-4B 24.9
curl .../api/models/ibm-granite/{granite-3.3-2b-instruct,granite-4.0-micro,granite-4.0-h-micro} → all apache-2.0
curl .../api/models/ibm-granite/{granite-4.1-3b,granite-4.1-8b,granite-4.0-1b-base,granite-swash-2b,
     granite-swash-3b-a600m} → all license:apache-2.0; params 3402836480 / 8791592960 / 1631750144 / …
curl .../api/models/bigcode/starcoder2-{3b,7b} → license bigcode-openrail-m   ⚠️
curl .../bigcode-model-license-agreement (Space README/app.py) → "commercial versions of it. Use restrictions
     are included to prevent misuse of the Model."   ⇒ commercial YES, use-restrictions viral
curl .../api/models/deepseek-ai/deepseek-coder-{1.3b,6.7b}-instruct → license "other", license_name "deepseek"  ⚠️
curl .../deepseek-coder-6.7b-instruct/raw/main/LICENSE → OpenRAIL-derived; §III(a) use-based restrictions MUST be
     an enforceable provision in downstream agreements; §5 covers finetuning; §7 DeepSeek may "restrict (remotely
     or otherwise) usage"; Attachment A = Use Restrictions
curl .../deepseek-coder-6.7b-instruct/raw/main/README.md:54 → "DeepSeek Coder supports commercial use."
curl .../api/models/google/codegemma-2b → license gemma, gated "manual", license_link ai.google.dev/gemma/terms ⚠️
curl .../api/models/google/gemma-3-4b-it → license gemma, gated "manual"   |   gemma-3-270m → gemma, gated manual
curl .../api/models?author=google&sort=lastModified → gemma-4-{E2B,E4B,12B,26B-A4B,31B}-it (2026-07-20) + QAT variants
curl .../api/models/google/gemma-4-{E2B,E4B,12B,26B-A4B,31B}-it → tags contain license:apache-2.0, gated:false  ← LICENSE CHANGE
     params: E2B 5123178051 | E4B 7996156490 | 12B 11959730224 | 26B-A4B 25805936206 | 31B 31273088876
curl .../api/models/microsoft/Phi-4-mini-instruct → license mit, 2025-12-10
curl .../api/models/01-ai/Yi-Coder-{1.5B,9B}-Chat → apache-2.0, 2024-09; dl 623 / 9688
curl .../api/models/JetBrains/Mellum-4b-{base,sft-python} → apache-2.0, 4019248128 params
curl .../api/models/IQuestLab/IQuest-Coder-V1-7B-Instruct → license:other, 2026-03-04, dl 121   ⚠️
curl .../api/models?author=mistralai&sort=lastModified → Ministral-3-{3B,8B,14B}-{Instruct,Reasoning}-2512,
     Devstral-Small-2-24B-Instruct-2512 (all 2026-07-15)
curl .../api/models/mistralai/Ministral-3-{3B,8B,14B}-Instruct-2512 → all license:apache-2.0, gated:false;
     params 3849090048 / 8918026716 / 13945032240 | Ministral-3-3B-Reasoning-2512 4251743232 apache-2.0
curl .../api/models/mistralai/Devstral-Small-2-24B-Instruct-2512 → apache-2.0, 24011361840
GGUF sizes (api/models/<repo>/tree/main): unsloth/Qwen3.5-2B-GGUF Q4_K_M 1.28GB / Q5_K_M 1.44 / Q6_K 1.57 / BF16 3.78
     ; Qwen3.5-4B-GGUF Q4_K_M 2.74 / Q6_K 3.53 (+ separate mmproj-{BF16,F16,F32}.gguf ⇒ text-only load possible)
     ; Qwen3.5-9B-GGUF Q4_K_M 5.68 / Q6_K 7.46
     ; Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF q4_k_m 1.12 / q6_k 1.46
     ; Qwen/Qwen2.5-Coder-14B-Instruct-GGUF q4_k_m 8.99 / q5_k_m 10.51
```

### (c) Inference — gh api + source tree
```
gh api repos/vllm-project/vllm   → Apache-2.0, 89741★, pushed 2026-08-23 ; releases/latest v0.27.1 (2026-08-11)
gh api repos/sgl-project/sglang  → Apache-2.0, 32278★, pushed 2026-08-23
gh api repos/ggml-org/llama.cpp  → MIT, 125211★, pushed 2026-08-23
gh api repos/huggingface/text-generation-inference → Apache-2.0, 10887★, pushed 2026-03-21   (stale)
gh api repos/turboderp-org/exllamav3 → MIT, 1160★, 2026-08-22  |  exllamav2 → MIT, 4611★, 2026-03-04 (stale)
gh api repos/ollama/ollama → MIT, 179224★, 2026-08-22
gh api repos/lmstudio-ai/lms → MIT, 5216★  |  lmstudio-ai/lmstudio-js → MIT   (desktop app itself closed-source)
gh api repos/predibase/lorax → Apache-2.0, 3827★, 2026-05-28  |  InternLM/lmdeploy → Apache-2.0, 8014★, 2026-08-21
gh api repos/mlc-ai/mlc-llm → Apache-2.0, 23081★, 2026-08-17
gh api repos/NVIDIA/TensorRT-LLM → NOASSERTION, 14451★ ; contents/LICENSE → "This project is licensed under the
     Apache 2.0 license" + bundled third-party code under other licenses   ⚠️
gh api repos/vllm-project/vllm/contents/vllm/model_executor/models → qwen3_5.py, qwen3_5_mtp.py present
gh api repos/vllm-project/vllm/contents/vllm/lora → model_manager.py, worker_manager.py, peft_helper.py, punica_wrapper/
gh api .../vllm/sampling_params.py → n:int=1 (L229), logprobs (L283), prompt_logprobs (L291)
gh api .../docs/features/lora.md → "--enable-lora", max_loras/max_lora_rank/max_cpu_loras,
     "## Dynamically serving LoRA Adapters", VLLM_ALLOW_RUNTIME_LORA_UPDATING=True, POST /v1/load_lora_adapter, LoRAResolver
gh api repos/sgl-project/sglang/contents/python/sglang/srt/models → qwen3_5.py, qwen3_5_text.py, qwen3_5_mtp.py
gh api .../srt/lora → lora_manager.py, lora_registry.py, lora_drainer.py, eviction_policy.py, lora_overlap_loader.py
sglang README → "RadixAttention for prefix caching … continuous batching … multi-LoRA batching"
gh api repos/turboderp-org/exllamav3/readme → "LoRA support"; arch list incl. "Qwen 3.5 (Qwen3_5ForConditionalGeneration)",
     "Gemma 4 … (E2B/E4B currently not supported)", "SmolLM (SmolLM3ForCausalLM)"
ls ~/llama.cpp/src/models | grep qwen → qwen35.cpp, qwen35moe.cpp, qwen3next.cpp, qwen3vl.cpp …
grep -o -- '--lora[a-z-]*' ~/llama.cpp/common/arg.cpp → --lora, --lora-init-without-apply, --lora-scaled
~/llama.cpp/tools/server/README.md → "GET /lora-adapters", "POST /lora-adapters", per-request `lora` field, n_probs, top_logprobs
gh api repos/huggingface/transformers → Apache-2.0, pushed 2026-08-23; releases/latest v5.15.1 (2026-08-19);
     contents/src/transformers/models → qwen3_5, qwen3_5_moe present
WebSearch "vLLM RTX 5080 Blackwell sm_120" → vLLM needs CUDA 12.8+ for Blackwell; prebuilt xformers wheels on PyPI
     covered only up to sm_89 (Ada) in early 2026; vLLM issue #31085 requests SM120 NVFP4 MoE kernels
```

### (d) LoRA tooling
```
gh api repos/huggingface/peft → Apache-2.0, 21580★, 2026-08-22 ; PyPI peft 0.20.0, classifier "OSI Approved :: Apache"
gh api repos/huggingface/trl  → Apache-2.0, 19133★, 2026-08-23 ; PyPI trl 1.10.0
gh api repos/bitsandbytes-foundation/bitsandbytes → MIT, 8430★, 2026-08-19 ; PyPI 0.50.1
gh api repos/unslothai/unsloth → Apache-2.0 (spdx), 74432★, 2026-08-23 ; PyPI unsloth 2026.8.19
gh api .../unsloth/contents → BOTH "LICENSE" and "COPYING" present
gh api .../unsloth/contents/LICENSE → "Apache License Version 2.0"
gh api .../unsloth/contents/COPYING → "GNU AFFERO GENERAL PUBLIC LICENSE Version 3"    ⚠️
unsloth README §License → "dual-licensing model of Apache 2.0 and AGPL-3.0. The core Unsloth package remains
     licensed under Apache 2.0, while certain optional components, such as the Unsloth Studio UI … AGPL-3.0."
gh api .../unsloth/contents/unsloth/models → qwen2.py, qwen3.py, qwen3_moe.py … NO qwen3_5.py
gh api search/code q='qwen3_5 repo:unslothai/unsloth' → 20 hits, paths are studio/backend/* (AGPL tree), tests, CI  ⚠️
gh api repos/axolotl-ai-cloud/axolotl → Apache-2.0, 12389★, 2026-08-21 (OpenAccess-AI-Collective/axolotl redirects here)
gh api repos/hiyouga/LLaMA-Factory → Apache-2.0, 74292★, 2026-08-20 (redirects hiyouga/LlamaFactory);
     PyPI llamafactory 0.9.5, classifier "OSI Approved :: Apache Software License"
gh api repos/meta-pytorch/torchtune → BSD-3-Clause, 5801★, 2026-08-22 (pytorch/torchtune redirects here);
     PyPI torchtune 0.6.1 license = full BSD-3 text "Copyright 2024 Meta"
gh api .../torchtune/contents/recipes/configs → gemma, gemma2, llama2..llama4, mistral, phi3, phi4, qwen2, qwen2_5,
     qwen3  ← NO qwen3_5 config
gh api .../peft/contents/src/peft/utils → hotswap.py, loftq_utils.py, quantization_utils.py, merge_utils.py
gh api .../peft/contents/src/peft/__init__.py → exports LoraConfig, get_peft_model, prepare_model_for_kbit_training
gh api .../trl/contents/trl/trainer → sft_trainer.py, dpo_trainer.py, grpo_trainer.py
```

### (e) Mutation tools
```
gh search repos "mutation testing" --language python --sort stars --limit 20 → mutmut 1399★, cosmic-ray 653★,
     mutpy 367★, mutahunter 299★, mutatest 101★, pytest-gremlins 48★, typemut 20★
gh api repos/boxed/mutmut            → BSD-3-Clause, 1399★, 2026-08-17, archived:false
gh api repos/sixty-north/cosmic-ray  → MIT, 653★, 2026-08-09, archived:false
gh api repos/mutpy/mutpy             → NOASSERTION, 367★, 2024-04-23
gh api .../mutpy/contents/LICENSE    → "Copyright 2011 Konrad Hałas / Licensed under the Apache License, Version 2.0"  ⇒ Apache-2.0
gh api repos/agroce/universalmutator  → NOASSERTION, 162★, 2026-05-20
gh api .../universalmutator/contents/LICENSE → "Licensed under the Apache License, Version 2.0"
gh api .../universalmutator/contents/setup.py | grep license → license='MIT'                    ⚠️ CONFLICT
gh api repos/codeintegrity-ai/mutahunter → AGPL-3.0, 299★, 2025-04-17; LICENSE header
     "GNU AFFERO GENERAL PUBLIC LICENSE Version 3"                                              ⚠️ COPYLEFT
gh api repos/WiredNerd/poodle          → MIT, 5★, 2026-04-05
gh api repos/mikelane/pytest-gremlins  → MIT, 48★, 2026-08-17
gh api repos/nkhitrov/typemut          → MIT, 20★, 2026-06-17
gh api repos/EvanKepner/mutatest       → MIT, 101★, 2023-02-17
gh api repos/Instagram/LibCST          → NOASSERTION, 2026-08-11; LICENSE "All contributions towards LibCST are
     MIT licensed" (+PSF dual on ~6 parser files)  ⇒ MIT
PyPI: mutmut 3.7.0/BSD-3 @2026-07-31 | cosmic-ray 8.7.0/MIT @2026-08-09 | mutpy 0.6.1 @2019-11-17 |
     poodle 1.3.4/MIT @2026-04-05 | universalmutator 1.14.1/MIT | mutatest 3.1.0/MIT @2022-02-20 |
     libcst 1.9.0 @2026-07-29 | parso 0.8.7/MIT @2026-05-01 | pytest-gremlins 1.9.0/MIT @2026-07-01
gh api .../cosmic_ray/operators/operator.py → Operator(ABC): mutation_positions(node), mutate(node,index), arguments(), examples()
gh api .../cosmic_ray/mutating.py → mutate_code(code, operator, occurrence); mutate_and_test() is the ONLY test-coupled fn
gh api .../cosmic_ray/commands/init.py → WorkDB used only in init(); MutationSpec(module_path, operator_name,
     operator_args, occurrence, start_pos, end_pos, definition_name)
gh api .../cosmic_ray/operators/provider.py → binary+comparison+unary cross-products, AddNot, True↔False, and↔or,
     break↔continue, ExceptionReplacer, NumberReplacer, RemoveDecorator, ZeroIterationForLoop, VariableReplacer/Inserter
LIVE: pip install cosmic-ray + script → ReplaceBinaryOperator_Add_Sub positions [((3,17),(3,18))];
     mutate_code(src,op,0) → "return a - b"; ReplaceComparisonOperator_Lt_GtE → "if a >= b";
     len(plugins.operator_names()) = 213                                            ⇒ CONTRACT PROVEN
LIVE: pip install mutmut; create_mutations("f.py", src) → FileNotFoundError "Could not figure out where the code
     to mutate is" (Config.get() reads cwd); after adding setup.cfg → 3 mutations with fields
     ['original_node','mutated_node','contained_by_top_level_function']   ⇒ NO operator name, NO position
LIVE: pip install mutatest on Py3.14 → Genome.targets OK: LocIndex(ast_class='BinOp', lineno=3, col_offset=15,
     op_type=ast.Add, …); type(Mutant.mutant_code) is <class 'code'>      ⇒ compiled, not source
LIVE: pip install libcst + PositionProvider visitor → found 3 [(2,9,'LessThan'),(3,17,'Add'),(4,13,'Subtract')]
gh api .../poodle/data_types/data.py → FileMutation(mutator_name, lineno, col_offset, end_lineno, end_col_offset, text);
     only 2 self.config reads across all 5 mutator modules
gh api .../pytest_gremlins/operators/protocol.py → GremlinOperator Protocol: can_mutate(node)->bool, mutate(node)->list[ast.AST]
gh api .../mutpy/operators/base.py → MutationOperator.mutate(...) yields (Mutation(operator,node,visitor), new_node);
     misc.py uses ast.Num/ast.Str (removed in Py3.12)
gh api .../universalmutator/genmutants.py → all logic inside main() (argparse)   ⇒ no importable API
```

### (f) Seed corpora
```
gh api repos/jkoppel/QuixBugs        → MIT, 145★, 2022-08-29; contents/LICENSE = MIT text "Copyright 2017-2019
     James Koppel"; legal_notes.txt = Quixey defunct, owner gave blessing
gh api repos/evalplus/evalplus       → Apache-2.0, 1802★, 2025-10-02
gh api repos/openai/human-eval       → MIT, 3350★            (HumanEval origin)
gh api repos/google-research/google-research → Apache-2.0     (MBPP origin)
gh api repos/bigcode-project/bigcodebench → Apache-2.0, 519★, 2026-01-03
gh api repos/princeton-nlp/SWE-bench → MIT, 5689★, 2026-08-18   (harness only)
gh api repos/SWE-bench/SWE-smith     → MIT, 748★, 2026-08-17
gh api repos/SWE-Gym/SWE-Gym         → Apache-2.0, 723★       ⚠️ vs HF card "mit" — MISMATCH
gh api repos/R2E-Gym/R2E-Gym         → Apache-2.0, 323★, 2025-07-13
gh api repos/Leolty/RepoBench        → CC-BY-4.0, 212★, 2024-08-16
gh api repos/LiveCodeBench/LiveCodeBench → MIT, 934★, 2025-07-16
gh api repos/soarsmu/BugsInPy        → license:null, 151★, 2026-02-10
gh api repos/soarsmu/BugsInPy/license → 404 Not Found
gh api repos/soarsmu/BugsInPy/contents → no LICENSE file (.gitignore, Dockerfile, README, framework, projects, …)
gh api search/code q=repo:soarsmu/BugsInPy+filename:LICENSE → total_count 0     ⇒ NO LICENSE, 3 independent checks
curl HF api evalplus/humanevalplus → {"license":"apache-2.0","lastModified":"2024-05-01"}
curl HF api evalplus/mbppplus      → {"license":"apache-2.0","lastModified":"2024-04-17"}
curl HF api bigcode/humanevalpack  → {"license":"mit"}, tag license:mit, lastModified 2025-08-19
curl HF api bigcode/bigcodebench   → {"license":"apache-2.0"}
curl HF api princeton-nlp/SWE-bench_Lite     → {"license":null}; cardData keys = ["configs","dataset_info"]   ⚠️
curl HF api princeton-nlp/SWE-bench_Verified → {"license":null}
curl HF api SWE-bench/SWE-bench_Multilingual → {"license":"mit"}  | _Multimodal → {"license":null}   ⚠️
curl HF api SWE-Gym/SWE-Gym → {"license":"mit"}   ⚠️ repo says Apache-2.0
curl HF api R2E-Gym/R2E-Gym-Subset → {"license":"apache-2.0"} | nebius/SWE-rebench → {"license":"cc-by-4.0"}
curl HF api SWE-Perf/SWE-Perf → {"license":"apache-2.0"} | tianyang/repobench_python_v1.1 → {"license":"cc"}
curl HF raw livecodebench/code_generation_lite/README.md → front-matter "license: cc" — bare, no version   ⚠️
datasets-server/size: humanevalplus 164 | mbppplus 378 | humanevalpack python 164 | bigcodebench 1140 |
     SWE-bench_Verified 500 | _Lite 323 | _Multilingual 300 | _Multimodal 480+100 | SWE-smith 59136 |
     R2E-Gym-Subset 4578 | SWE-rebench 21336/6542 | SWE-Perf 140
datasets-server/rows bigcode/humanevalpack → keys incl. buggy_solution, canonical_solution, test, bug_type,
     failure_symptoms; test = "def check(has_close_elements): assert …"
datasets-server/rows bigcode/bigcodebench → test = "import unittest; from unittest.mock import patch;
     class TestCases(unittest.TestCase)"; per-row `libs` field
README princeton-nlp/SWE-bench → "SWE-bench uses Docker for reproducible evaluations"; "at least 120GB free
     storage, 16GB RAM, 8 CPU cores"
README SWE-bench/SWE-smith → "requires Docker to create execution environments … do *not* plan on supporting
     Windows or MacOS"
README R2E-Gym → "Each executable gym instance has a docker image ~300MB-500Mb"
README bigcodebench → "--execution [e2b|gradio|local]"; default remote; E2B_API_KEY
README BugsInPy → `docker build -t bugsinpy .`; bugsinpy-compile pip-installs per-project requirements (network)
LIVE: git clone --depth 1 QuixBugs; pytest python_testcases -q --correct → 276 passed, 2 skipped in 0.58s
     ⇒ OFFLINE, NO DOCKER, pytest only
LIVE: pytest python_testcases -q (buggy) → HUNG indefinitely, killed after >120s        🔴
LIVE: pytest python_testcases -q --timeout=5 (buggy) → 187 failed, 89 passed, 2 skipped in 87.67s
     ⇒ per-test timeouts MANDATORY
LIVE: python -c "evalplus.data.get_human_eval_plus()" → downloads HumanEvalPlus.jsonl.gz v0.1.10 +
     MbppPlus.jsonl.gz v0.2.0, then offline; 164 / 378; keys [atol, base_input, canonical_solution, contract,
     entry_point, plus_input, prompt, task_id, test]
WebSearch "SWE-bench Pro" → public/held-out subsets use "strong copyleft licenses (e.g., GPL) to reduce
     contamination risk"; 1865 tasks / 41 repos                                          🚫
WebSearch PyBugHive → "CC Attribution-NonCommercial-NoDerivatives 4.0"                   🚫
WebSearch 2026 newcomers → SWE-EVO (arXiv 2512.18470), PBT-Bench (2605.15229), UTBoost, EvoOtter;
     OpenAI stopped reporting SWE-bench Verified early 2026 (contamination)
gh api repos/{SWE-bench/SWE-bench-multimodal, facebookresearch/swe-perf, nus-apr/GitBug-Python,
     gitbugactions/gitbug-python} → 404 (do not exist under those names; GitBug-Java only)
```
