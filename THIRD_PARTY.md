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
| torch | github.com/pytorch/pytorch | BSD-3-Clause | LICENSE file fetched (`gh api ...` said NOASSERTION; the file is the 3-clause BSD text: "Redistribution... 1./2./3. Neither the names") | 2026-08-23 | `serve` extra: LoRA-attach smoke (fwd/bwd, generate) |
| vLLM | github.com/vllm-project/vllm | Apache-2.0 | `gh api repos/vllm-project/vllm --jq .license.spdx_id` | 2026-08-23 | `serve` extra: primary serving engine (OpenAI-compatible, n-best + logprobs, runtime LoRA) |
| llama.cpp | github.com/ggml-org/llama.cpp | MIT | `gh api repos/ggml-org/llama.cpp --jq .license.spdx_id` (also `ggerganov/llama.cpp` → MIT) | 2026-08-23 | serving fallback if vLLM lacks sm_120 wheels; `/props` identity + GGUF Q6_K baseline |
| transformers | github.com/huggingface/transformers | Apache-2.0 | `gh api repos/huggingface/transformers --jq .license.spdx_id` | 2026-08-23 | `serve` extra: model + tokenizer load in the LoRA smoke |
| peft | github.com/huggingface/peft | Apache-2.0 | `gh api repos/huggingface/peft --jq .license.spdx_id` | 2026-08-23 | `serve` extra: LoRA adapter attach/save |
| accelerate | github.com/huggingface/accelerate | Apache-2.0 | `gh api repos/huggingface/accelerate --jq .license.spdx_id` | 2026-08-23 | `serve` extra: device_map/dtype placement in the smoke |
| bitsandbytes (optional) | github.com/bitsandbytes-foundation/bitsandbytes | MIT | `gh api repos/bitsandbytes-foundation/bitsandbytes --jq .license.spdx_id` (also `TimDettmers/bitsandbytes` → MIT) | 2026-08-23 | optional LoRA-train quantization QoL; not in the `serve` extra by default |
| Qwen3.5-2B (model) | huggingface.co/Qwen/Qwen3.5-2B | Apache-2.0 | HF API `cardData.license` on `Qwen/Qwen3.5-2B` | 2026-08-23 | proposer model (weights) |
| Qwen3.5-9B (model) | huggingface.co/Qwen/Qwen3.5-9B | Apache-2.0 | HF API `cardData.license` on `Qwen/Qwen3.5-9B` | 2026-08-23 | baseline model for tok/s comparison (weights) |
| Qwen2.5-Coder-1.5B-Instruct (model) | huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct | Apache-2.0 | HF API `cardData.license` | 2026-08-23 | fallback proposer if Qwen3.5-2B LoRA attach fails (spec §2). NB: Qwen2.5-Coder-3B is `other` (Qwen license) — do not use |
| REx | github.com/haotang1995/REx | MIT | `gh api repos/haotang1995/REx --jq .license.spdx_id` | 2026-08-23 | Thompson-sampling scheduler ported (Beta bandit) to crucible/search/rex.py |
| Qwen2.5-Coder-14B-Instruct-AWQ (model) | huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct-AWQ | Apache-2.0 | HF API `cardData.license` | 2026-08-24 | §2 big-arm fallback proposer, ACTIVE (9B §4.7 probe failed both ways; findings S2.5 §6-7) |
| trl | github.com/huggingface/trl | Apache-2.0 | `gh api repos/huggingface/trl --jq .license.spdx_id` | 2026-08-24 | `serve` extra: `SFTTrainer`/`SFTConfig` in `crucible/sleep/train.py` (v1.10.0 at verify) |
| datasets | github.com/huggingface/datasets | Apache-2.0 | `gh api repos/huggingface/datasets --jq .license.spdx_id` | 2026-08-24 | `serve` extra: in-memory `Dataset` for sleep-cycle SFT pairs (v5.0.1 at verify) |
| crepes (design reference only) | pypi.org/project/crepes | BSD (PyPI classifier; `gh api` repo 404) | PyPI `License :: OSI Approved :: BSD License` classifier | 2026-08-24 | NOT a dependency: Mondrian/provenance-class conformal *design* informed `crucible/uncertainty/conformal.py`; implementation is stdlib PAVA, no code copied |
| MemOS (schema fields, idea) | github.com/MemTensor/MemOS | Apache-2.0 | `gh api repos/MemTensor/MemOS --jq .license.spdx_id` | 2026-08-24 | field *design* in `crucible/memory/schema.py`: confidence, source locators, status, version, history on memory items; no code copied |
| Graphiti (schema fields, idea) | github.com/getzep/graphiti | Apache-2.0 | `gh api repos/getzep/graphiti --jq .license.spdx_id` | 2026-08-24 | temporal field *design* in `crucible/memory/schema.py`: valid_at/invalid_at/expired_at + episodes lineage; no code copied |
| MIRIX (schema fields, idea) | github.com/Mirix-AI/MIRIX | Apache-2.0 | `gh api repos/Mirix-AI/MIRIX --jq .license.spdx_id` | 2026-08-24 | credibility/evidence/lineage field *design* in `crucible/memory/schema.py`; no code copied |
