# Pillar 1 — Latent Predictors / World Models for Code

**Survey date:** 2026-08-23 · **Scope:** informs crucible Phase B (non-token "intuition" proposer), not the Phase A build.
**Licensing rule applied:** Apache-2.0 / MIT / BSD = ADOPT-eligible · GPL/AGPL/LGPL = copyleft flag · CC-BY-NC / research-only / none = REFERENCE-ONLY.
**Every license below was read from a command run today** (`gh api repos/…/license` → base64-decoded LICENSE text, or the HuggingFace model API). SPDX `NOASSERTION` was always resolved by fetching the actual file. See the Evidence log.

**Headline:** the single most important licensing fact for this pillar is that **Meta's JEPA line is split**. I-JEPA, V-JEPA v1, LeJEPA and TD-JEPA are all **CC-BY-NC-4.0** (verified, four separate LICENSE fetches). **V-JEPA 2 (MIT), EB-JEPA (Apache-2.0), LeWorldModel (MIT), LLM-JEPA (Apache-2.0), stable-worldmodel / stable-pretraining (MIT)** are not. The adoptable path exists and is 2026-current; the famous names are the ones you cannot use.

---

## 1. JEPA family

| Name | URL | License (verified how) | Stars | Last push | Released artifacts | Size | What it gives B-lite | Verdict |
|---|---|---|---|---|---|---|---|---|
| **LeWorldModel (LeWM)** | github.com/lucas-maes/le-wm | **MIT** — `gh api repos/lucas-maes/le-wm/license` → "MIT License / Copyright (c) 2026 Lucas Maes" | 4330 | 2026-05-26 | code + HF checkpoints + datasets (`quentinll/lewm` collection) | **~15M params**, single GPU, few hours | The recipe: end-to-end JEPA world model with **exactly two loss terms** (next-embedding prediction + Gaussian-isotropy regularizer), **no EMA, no stop-grad, no pretrained encoder, no auxiliary supervision**. One tunable loss hyperparameter instead of six. This is the closest published thing to "train a latent predictor from scratch on a laptop-class GPU." | **ADOPT** (deep-read #1) |
| **EB-JEPA** | github.com/facebookresearch/eb_jepa | **Apache-2.0** — `gh api repos/facebookresearch/eb_jepa/license` → "Apache License Version 2.0" | 759 | 2026-07-17 | library + 3 worked examples (image JEPA, video JEPA, action-conditioned video JEPA + planning) | each example "trainable on a single GPU within a few hours" | FAIR's own permissively-licensed JEPA reference implementation, incl. the **action-conditioned** variant — the exact shape crucible needs (`state × action → next state`, where action = candidate edit). Energy-based framing gives a natural verify/score head. Apache-2.0 means you can lift code, not just ideas. | **ADOPT** (deep-read #2) |
| **V-JEPA 2** | github.com/facebookresearch/vjepa2 · huggingface.co/facebook/vjepa2-* | **MIT** (repo, `gh api …vjepa2`) — HF cards mixed: `facebook/vjepa2-vitl-fpc64-256` = `mit`, `facebook/vjepa2-vitg-fpc64-384` = `apache-2.0`, `facebook/vjepa2-vitl-fpc16-256-ssv2` = `mit` (all via HF API) | 4500 | 2026-03-23 | code + weights (ViT-L/H/g) + action-conditioned variant | ViT-L ~300M · ViT-H ~600M · ViT-g ~1B | The **only large Meta JEPA you can actually use commercially.** Useful as: (a) proof the EMA-target + multi-block-mask recipe scales, (b) the V-JEPA2-AC action-conditioning pattern. Vision-domain, so no direct weight reuse for code. | **PORT-PIECES** |
| **LLM-JEPA** | github.com/galilai-group/llm-jepa | **Apache-2.0** — `gh api repos/galilai-group/llm-jepa/license` → "Apache License Version 2.0" (SPDX also Apache-2.0) | 327 | 2026-04-15 | training/fine-tune code (`finetune.py`, `stp.py`), no weights | works on 0.5B–8B backbones (LoRA path provided) | **The only JEPA-for-language codebase with a permissive license.** Critically, its canonical two-view pair is literally **(Text, Code)** — the JEPA loss makes the embedding of the description predict the embedding of the code. Swapping the second view to *execution outcome* is a small, honest change. Also ships "Semantic Tube Prediction" (random-span two-view) which removes the need for paired data, plus JEPA-loss ablations (L2 / MSE / `[PRED]` token / reverse-prediction). | **ADOPT** (deep-read #3) |
| **stable-worldmodel** | github.com/galilai-group/stable-worldmodel | **MIT** — `gh api …/license` → "MIT License / Copyright (c) 2026 GalilAI-group" | 2142 | 2026-08-18 (5 days ago) | platform: env management, planning, evaluation | n/a (harness) | LeWM's substrate. Gives the planning + eval loop around a latent predictor, so you only write the model. Actively maintained. | **PORT-PIECES** |
| **stable-pretraining** | github.com/galilai-group/stable-pretraining | **MIT** — `gh api …/license` → "MIT License / Copyright (c) 2024 rbalestr-lab" | 303 | 2026-07-16 | training library | n/a (harness) | Training-loop half of the same stack. | **PORT-PIECES** |
| **LeJEPA** | github.com/rbalestr-lab/lejepa → redirects to galilai-group/lejepa | **CC-BY-NC-4.0** — `gh api repos/galilai-group/lejepa/license` → "Attribution-NonCommercial 4.0 International". Both the old and new org path return the same NC text. | 1318 | 2026-01-25 | code + benchmark tables, no weights on HF | ViT-L 304M / ConvNeXtV2-H 660M pretrained ~100 ep IN-1K | The **SIGReg** idea (sketched isotropic Gaussian regularization: random 1-D projections + characteristic-function matching, linear time/memory) is the theory behind LeWM's regularizer. ~50 lines of core code — but it is NC, so **read the paper (arXiv:2511.08544) and reimplement; do not copy the file.** | **REFERENCE-ONLY** |
| **lejepa-identifiability** | github.com/klindtlab/lejepa-identifiability | **MIT** — `gh api …/license` → "MIT License" | 133 | 2026-05-27 | simulations + identifiability proof | tiny | An MIT-licensed, independent implementation of the SIGReg maths — a clean-room route to the LeJEPA objective without touching NC code. | **PORT-PIECES** |
| **I-JEPA** | github.com/facebookresearch/ijepa · huggingface.co/facebook/ijepa_vith14_1k | **CC-BY-NC-4.0** — repo `gh api …/license` → "Attribution-NonCommercial 4.0 International"; HF card `license: cc-by-nc-4.0` | 3489 | 2024-05-08, **archived** | code + ViT-H/14 weights | ViT-H ~632M | Origin of the masked-latent-prediction + EMA-target recipe. Archived, NC, vision-only. Historical reference. | **REFERENCE-ONLY** |
| **V-JEPA (v1)** | github.com/facebookresearch/jepa | **CC-BY-NC-4.0** — `gh api repos/facebookresearch/jepa/license` → "Attribution-NonCommercial 4.0 International" | 4097 | 2025-02-27 | code + video weights | ViT-L/H | Superseded by V-JEPA 2 (MIT). No reason to touch the NC version. | **SKIP** |

**Text/code JEPA landscape check.** I searched GitHub (`gh search repos "LLM-JEPA"`, `"LeJEPA"`, `"JEPA text embedding predictive"`) and HF (`api/models?search=jepa&sort=downloads`). Beyond `galilai-group/llm-jepa` (327★) the LLM-JEPA space is a long tail of ≤24★ hobby forks. The HF JEPA weight ecosystem is dominated by **domain ports, not code**: `braindecode/signal-jepa` (EEG, 3.8k dl), `lerobot/VLA-JEPA-Pretrain` (robotics, **apache-2.0**, 1.4k dl, pushed 2026-07-06), `NYUMedML/Neuro-JEPA`, `Flogrammer/Mol-JEPA`, `ProteinJEPA` (arXiv 2605.07554). **There is no published code-domain or execution-domain JEPA with released weights as of 2026-08-23.** That is the gap crucible Phase B would be entering — genuinely open, and genuinely unvalidated.

---

## 2. Latent world models (RL)

| Name | URL | License (verified how) | Stars | Last push | Released artifacts | Size | What it gives B-lite | Verdict |
|---|---|---|---|---|---|---|---|---|
| **DreamerV3** | github.com/danijar/dreamerv3 | **MIT** — `gh api repos/danijar/dreamerv3` → `spdx_id: MIT` | 3690 | 2026-05-25 | JAX code, no weights | XS–XL configs (~8M–200M) | The canonical **RSSM**: discrete latent state + recurrent dynamics + reward head + value head, with symlog/twohot normalization that makes one hyperparameter set work across wildly different domains. Crucible's value function and its latent dynamics head could be near-copies of Dreamer's. Its **discrete (categorical) latents are itself a collapse-prevention mechanism** — a different answer than SIGReg, and arguably a better fit for discrete program state. | **PORT-PIECES** |
| **dreamerv3-torch** | github.com/NM512/dreamerv3-torch | **MIT** — `gh api` → `spdx_id: MIT`, `archived: true` | 887 | 2026-03-08 (**archived**) | PyTorch port | same | PyTorch reading copy of the RSSM if you don't want JAX. Archived — treat as frozen reference, not a dependency. | **REFERENCE-ONLY** |
| **TD-MPC2** | github.com/nicklashansen/tdmpc2 | **MIT** — `gh api` → `spdx_id: MIT` | 930 | 2026-07-13 | code + **300+ checkpoints** + 30/80-task datasets | 1M–**317M** params; 12 GB RAM single-task, 8 GB GPU for eval | Structurally **the closest RL analogue to crucible Phase A**: a *decoder-free* latent dynamics model (no pixel/token reconstruction at all) + TD-learned value + planning by trajectory optimization in latent space. That is exactly "latent predictor + value function + tree search." Single hyperparameter set across 104 tasks. Its multitask 317M model shows the shape scales. | **PORT-PIECES** (structure), high value |
| **TD-JEPA** | github.com/facebookresearch/td_jepa | **CC-BY-NC-4.0** — `gh api repos/facebookresearch/td_jepa/license` → "Attribution-NonCommercial 4.0 International" (SPDX said NOASSERTION) | 56 | 2025-12-22 | PyTorch code | small | Conceptually the best-matched paper in this section (arXiv:2510.00739): explicit state encoder + task encoder + **policy-conditioned multi-step latent predictor** + latent-space policies, learned by TD from offline reward-free transitions. Multi-step latent TD is precisely what "predict the outcome N edits ahead" needs. But NC — read the paper, write your own. | **REFERENCE-ONLY** |
| **LightZero** | github.com/opendilab/LightZero | **Apache-2.0** — `gh api` → `spdx_id: Apache-2.0` | 1636 | **2026-08-21 (2 days ago)** | unified MCTS/MuZero benchmark: MuZero, EfficientZero, Sampled/Stochastic MuZero, Gumbel MuZero | configurable, small | **The only permissively-licensed MuZero-family codebase that is actively maintained.** MuZero's learned latent dynamics + policy/value heads driving MCTS is the exact template for crucible's tree search over edits. Apache-2.0 makes this the default choice over EfficientZero. | **ADOPT** (for the search+value half) |
| **DIAMOND** | github.com/eloialonso/diamond | **MIT** — `gh api` → `spdx_id: MIT` | 2095 | 2024-12-06 | code + weights | small | Diffusion world model. Reconstruction-in-pixel-space — the *opposite* of the JEPA thesis. Included as the counterfactual: it's the evidence that generative world models still win on fidelity when the observation space is renderable. Program state is not renderable, so this is a caution, not a path. | **REFERENCE-ONLY** |
| **IRIS** | github.com/eloialonso/iris | **GPL-3.0** ⚠️ — `gh api` → `spdx_id: GPL-3.0` | 899 | 2024-10-14 | code + weights | ~small | Transformer world model over a discrete autoencoder. **Copyleft — flagged.** Do not vendor. Ideas only. | **REFERENCE-ONLY (copyleft)** |
| **EfficientZero V2** | github.com/Shengjiewang-Jason/EfficientZeroV2 | **GPL-3.0** ⚠️ — `gh api …/license` → "GNU GENERAL PUBLIC LICENSE Version 3" | 123 | 2024-08-09 | code | small | Sample-efficient MuZero. **Copyleft — flagged**, and stale (2024). LightZero (Apache-2.0) covers the same ground. | **SKIP** |
| **EfficientZero (v1)** | github.com/YeWR/EfficientZero | **GPL-3.0** ⚠️ — `gh api` → `spdx_id: GPL-3.0` | 944 | 2023-12-20 | code | small | Same as above, older. | **SKIP** |
| **tinyworlds** (Genie) | github.com/AlmondGod/tinyworlds | **MIT** — `gh api` → `spdx_id: MIT` | 1377 | 2026-04-15 | minimal Genie reimplementation | tiny | Readable minimal implementation of Genie-style latent-action world models. Genie's **latent action model** — inferring an unlabelled action code from consecutive states — is a genuinely interesting trick if you ever want to learn an edit representation rather than hand-specify one. DeepMind released no Genie code; this is the usable substitute. | **REFERENCE-ONLY** |

---

## 3. Code-execution prediction models

| Name | URL | License (verified how) | Stars | Last push | Released artifacts | Size | Token or latent? | Verdict |
|---|---|---|---|---|---|---|---|---|
| **CWM (Meta Code World Model)** | github.com/facebookresearch/cwm · huggingface.co/facebook/cwm | **Split.** Code = **BSD-3** (`gh api repos/facebookresearch/cwm/license` → "BSD License / For cwm software / Copyright (c) Meta Platforms"). Weights = **`fair-noncommercial-research-license`, `gated: manual`** (HF API `cardData.license`). README confirms: "code … BSD-3 … model weights … released under a custom license." | 890 | 2026-07-17 | inference code, evals, **neural-debugger demos**, tech report; weights in 3 variants (pretrain/SFT/instruct) behind manual approval | **32B dense**, 64 blocks, GQA, 8k/131k local/global 3:1, 128k vocab. **Requires ~160 GB VRAM** (2×H100) to run the repo defaults. | **Token space.** Emits execution traces as text. | **REFERENCE-ONLY** — but the single most important reference in this survey |
| **CodeExecutor** | github.com/microsoft/CodeBERT (`/CodeExecutor`) · huggingface.co/microsoft/codeexecutor | Repo **MIT** (`gh api repos/microsoft/CodeBERT` → `spdx_id: MIT`); weights **MIT** (HF API → `license: mit`) | 2786 (parent repo) | 2023-07-09 | code, weights, datasets on Zenodo (SingleLine / Tutorial / CodeNetMut + code-to-code search) | RoBERTa-shaped: **12 layers, hidden 768, 12 heads, vocab 51627 ≈ 125M params** (from `config.json`) | **Token space.** Predicts `<line> <0> <state> s : [ x , y , z ] </state>` sequences. | **PORT-PIECES** — best-licensed prior art at the right *size* |
| **SemCoder** | github.com/ARiSE-Lab/SemCoder · huggingface.co/semcoder/semcoder_s_1030 | Repo **MIT** (`gh api` → `spdx_id: MIT`); weights **NOT MIT** — HF `license: other`, `license_name: "deepseek"`, `license_link` → DeepSeek-Coder LICENSE-MODEL (use-restriction clauses) | 30 | 2024-11-19 | code + 2 checkpoints + PyX corpus | **6.7B** (DeepSeek-Coder base) | **Token space.** "Monologue reasoning" — natural-language narration of execution (forward monologue = state evolution, backward = input inference). | **PORT-PIECES** (data recipe only; weights carry DeepSeek use restrictions — flag) |
| **CRUXEval** | github.com/facebookresearch/cruxeval · huggingface.co/datasets/cruxeval-org/cruxeval | **MIT** — `gh api` → `spdx_id: MIT`, `archived: true` | 173 | 2024-10-11 (**archived**) | 800 functions + I/O pairs, eval harness, **generation pipeline** | benchmark | Benchmark (CRUXEval-I input pred / CRUXEval-O output pred) | **ADOPT** as the eval, see §5 for its generator |
| **TRACED** | github.com/ARiSE-Lab/TRACED_ICSE_24 | **NO LICENSE FILE** — `gh api repos/ARiSE-Lab/TRACED_ICSE_24` → `license: null`; `gh api …/license` → 404. No license = all rights reserved. | 22 | 2024-03-21 | code + data (Figshare mirror), models claimed | encoder-scale | Execution-aware *pre-training*: joint static+dynamic objective (branch coverage + runtime value prediction as auxiliary heads on a code encoder). Closest existing thing to "add execution supervision to a frozen code encoder." | **REFERENCE-ONLY** (unlicensed) |
| **NExT** (Google DeepMind) | arXiv:2404.14662 | n/a — **no code repository found** (web search returned papers/coverage only, no GitHub) | — | — | paper only | PaLM 2-scale | **Token space.** Self-training bootstrap of execution-aware rationales; +26.1% / +14.3% fix rate on MBPP/HumanEval. | **REFERENCE-ONLY** |

**2025–2026 activity note.** This subfield is hot but almost entirely token-space: *"What I cannot execute, I do not understand"* (arXiv:2503.05703), REval / *"Reasoning Runtime Behavior of a Program with LLM"* (2403.16437), *Self-Execution Simulation* (2604.03253), *StepCodeReasoner* (2605.11922, RL against stepwise traces), *Can LLMs Reason About Complex Execution Paths?* (2511.18288), *ReMind* (2511.00488), semantic-equivalence self-play with formal verification (2604.17010). **I found zero papers predicting execution state in a learned latent space for code.** Everyone narrates traces as tokens. That is simultaneously the opportunity and the warning.

---

## 4. Latent reasoning in LMs

| Name | URL | License (verified how) | Stars | Last push | Released artifacts | Size | What it gives B-lite | Verdict |
|---|---|---|---|---|---|---|---|---|
| **Coconut** | github.com/facebookresearch/coconut | **MIT** — `gh api repos/facebookresearch/coconut` → `spdx_id: MIT` | 1687 | **2026-07-02** | training code, no weights | GPT-2 / small-LM scale experiments | The reference implementation of a **non-token reasoning channel**: feed the last hidden state back as the next input embedding, skipping the vocabulary entirely. Multi-stage curriculum that progressively replaces CoT text with continuous thoughts. MIT, from FAIR, still maintained in 2026. If crucible wants the proposer to think in latents *within* a token model rather than beside one, this is the code. | **ADOPT** |
| **Huginn / recurrent-depth** | github.com/seal-rg/recurrent-pretraining · huggingface.co/tomg-group-umd/huginn-0125 | Repo **Apache-2.0** (`gh api` → `spdx_id: Apache-2.0`); weights **apache-2.0** (HF API) | 911 | 2025-12-29 | pretraining + inference code, **weights released** | **3.5B**, trained 800B tokens, depth-recurrent | Test-time compute scaling by *iterating a recurrent block in latent space* rather than emitting more tokens. Fully permissive **including weights** — rare. A 3.5B bf16 model is ~7 GB, i.e. B-mid territory on the 5080. | **PORT-PIECES** |
| **CoLaR** | github.com/xiaomi-research/colar | **Apache-2.0** — `gh api repos/xiaomi-research/colar/license` → "Apache License Version 2.0" | 99 | 2026-06-29 | code (NeurIPS 2025) | fine-tune scale | Dynamic latent compression of reasoning chains with a single-pass compression factor + RL on the latent policy. Permissive; small but real. | **PORT-PIECES** |
| **Awesome-Latent-CoT** | github.com/EIT-NLP/Awesome-Latent-CoT | **Apache-2.0** — `gh api` → `spdx_id: Apache-2.0` | 376 | 2026-06-20 | curated paper list | n/a | Live index of latent-space reasoning work; pairs with the *Survey on Latent Reasoning* (arXiv:2507.06203). Cheapest way to stay current on this pillar. | **ADOPT** (as a reading source) |
| **LatentCoT-Horizon** | github.com/multimodal-art-projection/LatentCoT-Horizon | **NO LICENSE** — `gh api` → `license: null` | 412 | 2025-11-05 | paper list | n/a | Second index, broader but staler and unlicensed (fine to read, don't vendor). | **REFERENCE-ONLY** |
| **SoftCoT / SoftCoT++** | github.com/xuyige/SoftCoT | **NTUITIVE non-commercial** ⚠️ — SPDX said NOASSERTION; `gh api …/license` → "NANYANG TECHNOLOGICAL UNIVERSITY - NTUITIVE PTE LTD Dual License Agreement / **Non-Commercial Use Only**" | 95 | 2025-05-30 | code | small assistant-model + frozen LLM | Soft thought tokens produced by a small assistant model and projected into a frozen LLM's embedding space — architecturally very close to "small trained predictor beside a frozen big model," which is crucible's exact shape. **But the license forbids it.** Read the ACL 2025 paper, reimplement. | **REFERENCE-ONLY** |

---

## 5. Execution-trace data generation

| Name | URL | License (verified how) | Stars | Last push | Released artifacts | Size | What it gives B-lite | Verdict |
|---|---|---|---|---|---|---|---|---|
| **sensorium** (local) | /home/brice/workspace/sensorium | Local project (no LICENSE checked — user's own) | — | active (v0.4.0 line) | PEP 669 `sys.monitoring` recorder → **SQLite** trace per run; CLI: `run/tree/frame/grep/flow/diff/watch/refocus`; 27 source files | zero runtime deps, Py3.12+ | **This is the data-generation organ, already built.** See fit paragraph below. | **ADOPT** |
| **PySnooper** | github.com/cool-RR/PySnooper | **MIT** — `gh api` → `spdx_id: MIT` | 16581 | 2026-06-08 | decorator-based line tracer with local-variable deltas | tiny | The canonical "print every line + changed locals" format. Its *output format* is a good spec for what a state snapshot should contain; its `sys.settrace` mechanism is slower than PEP 669. | **REFERENCE-ONLY** (format inspiration) |
| **VizTracer** | github.com/gaogaotiantian/viztracer | **Apache-2.0** — `gh api` → `spdx_id: Apache-2.0` | 7722 | 2026-08-18 | C-extension tracer, JSON/perfetto output | low overhead | Fastest general Python tracer, Apache-2.0, actively maintained. The fallback if you need throughput sensorium doesn't give — but it records *timing/calls*, not local-variable state, so it's a poorer fit for state prediction. | **PORT-PIECES** |
| **snoop** | github.com/alexmojaki/snoop | **MIT** — `gh api` → `spdx_id: MIT` | 1461 | 2026-07-18 | PySnooper successor; pairs with `executing` for exact AST-node attribution | tiny | `executing`'s **AST-node-level attribution** is the piece worth stealing: it maps a runtime event to the exact expression node, which is how you align a *latent code embedding* (AST-indexed) with a *latent state embedding* (event-indexed). | **PORT-PIECES** |
| **CRUXEval generator** | github.com/facebookresearch/cruxeval | **MIT** — `gh api` → `spdx_id: MIT` (archived) | 173 | 2024-10-11 | full generation pipeline: LLM generates functions+inputs → **execute to get outputs** → filter for short/low-memory/human-doable → sample 800 | 800 samples released; pipeline unbounded | The **exact blueprint** for synthesizing (code, input) → outcome pairs at scale, including the filtering discipline that keeps traces short enough to embed. Generate with your Phase-A frozen proposer instead of CodeLlama-34B. | **ADOPT** |
| **CWM trace pipeline** | github.com/facebookresearch/cwm (tech report) | Code **BSD-3** (verified above); the *trace corpus itself is not released* | 890 | 2026-07-17 | inference/eval code + neural-debugger demos; mid-training data not published | — | The tech report describes mid-training on two trajectory families: **(1) Python interpreter traces recording local variable state after each executed line, (2) agentic interactions in Dockerized repos capturing edits, shell commands, and test feedback.** That is a validated recipe at 32B scale for exactly the data crucible would generate. Read the report for the trace schema and the observation/action framing. | **REFERENCE-ONLY** (recipe), high value |
| **OnlinePythonTutor** | github.com/hcientist/OnlinePythonTutor (fork; `pgbovine/OnlinePythonTutor` now **404**) | **NO LICENSE** — `gh api repos/hcientist/OnlinePythonTutor` → `license: null` | 197 | 2021-05-04 | step-by-step heap/stack state serializer | tiny | Its JSON state model (stack frames + heap objects + aliasing) is the best-thought-out *human-legible* program-state serialization. Stale (2021), unlicensed, upstream gone. Ideas only. | **REFERENCE-ONLY** |

### sensorium fit (one paragraph)

sensorium is close to a drop-in Phase-B data organ and it is better suited than anything public. It is built on **PEP 669 `sys.monitoring`** (not `sys.settrace`), so per-event overhead is low enough to trace a real test suite; it writes **one SQLite file per run** with calls, returns, raises and handled-exceptions plus captured argument/return values, and `--focus module:qualname` adds **line-level capture with local-variable deltas** for named code — which is exactly the (code, input) → state-sequence supervision a latent predictor needs, at exactly the granularity CWM used at 32B scale. Three properties matter disproportionately for training data: (1) **truncated captures are marked and counted rather than silently dropped**, so you can filter or mask lossy samples instead of training on quiet corruption; (2) **`diff RUN_A RUN_B` already computes causal-stream divergence** between two runs, which is a ready-made label for "did this edit change behavior, and where" — the single most valuable training signal for an edit-conditioned predictor, and normally the hardest part to build; (3) **`refocus` issues a MATCH/DIVERGED/REFUSED verdict on whether a re-run was the same execution**, giving you determinism screening for free, so nondeterministic samples get excluded rather than poisoning the target. The gaps are real but bounded: it is a *query* tool, not a *corpus* tool — there is no batch/parallel harvest mode, no export to a tensor-friendly format, and no state serializer of the OnlinePythonTutor kind (heap graph + aliasing) beyond captured locals; and per-run SQLite files means a 100k-program corpus is 100k files needing a manifest layer. Estimated work to turn it into a corpus generator: a harvest driver + a `traces → (code_span, state_snapshot, outcome)` exporter, not a rewrite.

---

## B-lite sketch

1. **Encoder (frozen).** `jinaai/jina-embeddings-v2-base-code` (**apache-2.0**, HF-verified, 137M, 8k ctx) for code spans; `Qwen/Qwen3-Embedding-0.6B` (**apache-2.0**, verified) if you want 0.6B. bf16, ~0.3–1.2 GB. Do **not** train it — that is what makes the budget work.
2. **State encoder (trained, small).** ~20M transformer over serialized sensorium state snapshots (locals as typed key/value tokens + line id). Trained jointly; this is the half with no pretrained option.
3. **Predictor.** ~12-layer, d=768, ~100M transformer: `(z_code, z_edit, z_input) → ẑ_state_next` and `→ ẑ_outcome`. Action-conditioned exactly as EB-JEPA's `ac_video_jepa` example (Apache-2.0, liftable code).
4. **Collapse prevention — this is the whole game.** Take **LeWM's two-term objective** (MIT, copyable): prediction loss + isotropic-Gaussian regularizer. **No EMA target, no stop-gradient.** LeWM shows this trains stably end-to-end at 15M params. Reimplement SIGReg from arXiv:2511.08544 or lift from `klindtlab/lejepa-identifiability` (MIT) — **never from `lejepa` itself (CC-BY-NC)**.
5. **Grounded head (the anti-collapse insurance).** A small supervised head predicting the *observable* verify-by-execution outcome (test pass/fail, exception type, return-value hash). This is not optional: it is what stops the latent space from becoming an elegant, useless constant, and it is the only output Phase A's search actually consumes.
6. **Training signal.** sensorium traces over a CRUXEval-style generated corpus (MIT pipeline): Phase-A proposer emits functions + inputs → execute → filter short/deterministic/low-memory → record. `sensorium diff` supplies edit-effect labels for free.
7. **Budget.** Frozen encoder ~1.2 GB + predictor 150M×16B (bf16 params + fp32 Adam m/v/master) ≈ 2.4 GB + activations at batch 64, short sequences ≈ 2 GB → **~6 GB peak, comfortably inside 16 GB.** Room to grow the predictor to ~400M before needing B-mid.
8. **Search + value.** Reuse **LightZero** (Apache-2.0, pushed 2 days ago) for MCTS-over-edits and the value head; borrow **TD-MPC2**'s decoder-free latent-dynamics + TD-value structure (MIT) rather than DreamerV3's reconstruction path.

## Risks — honest reasons B loses to a token model at this scale

- **No prior exists.** Zero published code/execution JEPAs with weights (verified across GitHub + HF searches). Every 2025–2026 execution-reasoning result — CWM, SemCoder, NExT, StepCodeReasoner — is token-space. You would be first, on a research spike, with no baseline to inherit.
- **Program state is discrete and adversarial to smooth latents.** An off-by-one changes the outcome categorically. JEPA's inductive bias — "throw away unpredictable detail" — is *correct* for video pixels and *catastrophic* for `i` vs `i+1`. DreamerV3's discrete categorical latents exist precisely because of this; a continuous predictor may learn to discard exactly the bits that decide pass/fail.
- **The scale mismatch is 200×.** CWM needed 32B params and ~160 GB VRAM to be good at trace prediction. A 150M predictor is not a small CWM; it is a different animal, and the honest prior is that it learns a coarse "will this plausibly crash" signal, not execution semantics.
- **Collapse is the default outcome, and it is quiet.** Without EMA/stop-grad you are trusting one regularizer, validated on pixels, to hold on a domain with different statistics. Collapse presents as smoothly decreasing loss and a useless model. Budget for the grounded head and for probe metrics from day one, and pre-register a kill criterion.
- **A frozen code encoder was never trained to represent state.** `jina-embeddings-v2-base-code` embeds code *for retrieval*. Nothing guarantees its space is linearly related to execution behavior. TRACED exists because static code encoders demonstrably lack this — and TRACED is unlicensed, so you cannot even start from its weights.
- **The cheap baseline is very strong.** Fine-tuning a 0.5B code LM to emit `<state>` tokens (the CodeExecutor recipe — MIT weights, 125M params, already trained) costs a fraction of the effort and gives an interpretable, debuggable output. **Run it as the control.** If B cannot beat a 125M token-space model, B has not earned its complexity.
- **License traps are dense here.** The four most-cited artifacts in this pillar (I-JEPA, V-JEPA v1, LeJEPA, TD-JEPA) are all CC-BY-NC, plus SoftCoT (NTUITIVE-NC), CWM weights (FAIR-NC, gated), SemCoder weights (DeepSeek use-restrictions), IRIS/EfficientZero (GPL-3.0), TRACED and OnlinePythonTutor (no license at all). Any casual `git clone` of "the obvious repo" contaminates the project.

---

## Evidence log

All commands run 2026-08-23 on this machine. `gh auth status` → logged in as `bricelancasterwcp-sudo`.

**Repo metadata** — `gh api repos/OWNER/REPO --jq '{license:.license.spdx_id,stars:.stargazers_count,pushed:.pushed_at,desc:.description,archived:.archived}'`:

```
facebookresearch/ijepa      NOASSERTION 3489 2024-05-08 archived:true
facebookresearch/jepa       NOASSERTION 4097 2025-02-27
facebookresearch/vjepa2     MIT         4500 2026-03-23
facebookresearch/coconut    MIT         1687 2026-07-02
facebookresearch/cruxeval   MIT          173 2024-10-11 archived:true
facebookresearch/cwm        NOASSERTION  890 2026-07-17
facebookresearch/eb_jepa    Apache-2.0   759 2026-07-17
facebookresearch/td_jepa    NOASSERTION   56 2025-12-22
galilai-group/lejepa        NOASSERTION 1318 2026-01-25   (rbalestr-lab/lejepa redirects here, same output)
galilai-group/llm-jepa      Apache-2.0   327 2026-04-15
galilai-group/stable-worldmodel  MIT    2142 2026-08-18
galilai-group/stable-pretraining MIT     303 2026-07-16
lucas-maes/le-wm            MIT         4330 2026-05-26
klindtlab/lejepa-identifiability MIT     133 2026-05-27
danijar/dreamerv3           MIT         3690 2026-05-25
NM512/dreamerv3-torch       MIT          887 2026-03-08 archived:true
nicklashansen/tdmpc2        MIT          930 2026-07-13
eloialonso/iris             GPL-3.0      899 2024-10-14
eloialonso/diamond          MIT         2095 2024-12-06
opendilab/LightZero         Apache-2.0  1636 2026-08-21
YeWR/EfficientZero          GPL-3.0      944 2023-12-20
Shengjiewang-Jason/EfficientZeroV2 GPL-3.0 123 2024-08-09
AlmondGod/tinyworlds        MIT         1377 2026-04-15
microsoft/CodeBERT          MIT         2786 2023-07-09
ARiSE-Lab/SemCoder          MIT           30 2024-11-19
ARiSE-Lab/TRACED_ICSE_24    null          22 2024-03-21
seal-rg/recurrent-pretraining Apache-2.0 911 2025-12-29
xiaomi-research/colar       Apache-2.0    99 2026-06-29
xuyige/SoftCoT              NOASSERTION   95 2025-05-30
EIT-NLP/Awesome-Latent-CoT  Apache-2.0   376 2026-06-20
multimodal-art-projection/LatentCoT-Horizon null 412 2025-11-05
cool-RR/PySnooper           MIT        16581 2026-06-08
gaogaotiantian/viztracer    Apache-2.0  7722 2026-08-18
alexmojaki/snoop            MIT         1461 2026-07-18
hcientist/OnlinePythonTutor null         197 2021-05-04
pgbovine/OnlinePythonTutor  → HTTP 404 (upstream gone)
```

**NOASSERTION resolved** — `gh api repos/OWNER/REPO/license --jq .content | base64 -d | head`:

```
facebookresearch/ijepa   → "Attribution-NonCommercial 4.0 International"           = CC-BY-NC-4.0
facebookresearch/jepa    → "Attribution-NonCommercial 4.0 International"           = CC-BY-NC-4.0
galilai-group/lejepa     → "Attribution-NonCommercial 4.0 International"           = CC-BY-NC-4.0
rbalestr-lab/lejepa      → "Attribution-NonCommercial 4.0 International"           = CC-BY-NC-4.0
facebookresearch/td_jepa → "Attribution-NonCommercial 4.0 International"           = CC-BY-NC-4.0
facebookresearch/cwm     → "BSD License / For cwm software / Copyright (c) Meta Platforms, Inc." = BSD-3
xuyige/SoftCoT           → "NANYANG TECHNOLOGICAL UNIVERSITY - NTUITIVE PTE LTD Dual License
                            Agreement / Non-Commercial Use Only"                   = NC, not adoptable
lucas-maes/le-wm         → "MIT License / Copyright (c) 2026 Lucas Maes"           = MIT
galilai-group/stable-worldmodel  → "MIT License / Copyright (c) 2026 GalilAI-group"= MIT
galilai-group/stable-pretraining → "MIT License / Copyright (c) 2024 rbalestr-lab" = MIT
Shengjiewang-Jason/EfficientZeroV2 → "GNU GENERAL PUBLIC LICENSE Version 3"        = GPL-3.0
ARiSE-Lab/TRACED_ICSE_24 → HTTP 404 (no LICENSE file, license:null) = all rights reserved
```

**HuggingFace weights** — `curl -s https://huggingface.co/api/models/ORG/NAME | jq '{license:.cardData.license,...}'`:

```
facebook/cwm                        license=fair-noncommercial-research-license  gated=manual  dl=9609   2025-10-15
facebook/ijepa_vith14_1k            license=cc-by-nc-4.0                         gated=false   dl=36546  2025-08-11
facebook/vjepa2-vitl-fpc64-256      license=mit                                  gated=false   dl=321394 2025-08-11
facebook/vjepa2-vitg-fpc64-384      license=apache-2.0                           gated=false   dl=4656   2025-08-11
facebook/vjepa2-vitl-fpc16-256-ssv2 license=mit                                  gated=false   dl=10657  2025-08-11
microsoft/codeexecutor              license=mit                                  gated=false   dl=125    2023-06-25
semcoder/semcoder_s_1030            license=other  license_name=deepseek
                                    license_link=github.com/deepseek-ai/DeepSeek-Coder/blob/main/LICENSE-MODEL
semcoder/semcoder_1030              license=other  (same)
tomg-group-umd/huginn-0125          license=apache-2.0                           gated=false   dl=13404  2025-07-29
lerobot/VLA-JEPA-Pretrain           license=apache-2.0                           gated=false   dl=1412   2026-07-06
jinaai/jina-embeddings-v2-base-code license=apache-2.0                           dl=406694
Qwen/Qwen3-Embedding-0.6B           license=apache-2.0                           dl=7206735
Qwen/Qwen2.5-Coder-0.5B             license=apache-2.0                           dl=27085
Salesforce/codet5p-110m-embedding   license=bsd-3-clause                         dl=30709
nomic-ai/CodeRankEmbed              license=mit                                  dl=416885
```

`curl .../microsoft/codeexecutor/raw/main/config.json` → `{"model_type":"roberta","num_hidden_layers":12,"hidden_size":768,"num_attention_heads":12,"vocab_size":51627}` ≈ 125M params.

**Searches** — `gh search repos <q> --sort stars --limit N` (search API is 30/min and was rate-limited twice; queries with 5+ words return empty because gh treats them as exact phrases):
`"LeJEPA"` → galilai-group/lejepa 1318★ top, rest ≤133★ · `"LLM-JEPA"` → galilai-group/llm-jepa 327★ top, rest ≤24★ · `"code world model"` → facebookresearch/cwm 890★ top · `"genie world model"` → AlmondGod/tinyworlds 1377★ top · `"EfficientZero V2"` → Shengjiewang-Jason 123★ · `chain of latent reasoning compression CoLaR` → xiaomi-research/colar 99★ · `online python tutor visualizer` → hcientist fork 197★.
`curl "https://huggingface.co/api/models?search=jepa&limit=25&sort=downloads"` → top code/text-relevant hits are all domain ports (signal-jepa 3856, VLA-JEPA-LIBERO 2339, VLA-JEPA-Pretrain 1412, Neuro-JEPA 738); **no code/execution JEPA weights exist.**

**READMEs read in full** (`gh api repos/OWNER/REPO/readme --jq .content | base64 -d`): `facebookresearch/cwm` (license split + 160 GB VRAM + mid-training on Python interpreter traces & Dockerized agentic interactions), `lucas-maes/le-wm` (15M params, 2 loss terms, 1 hyperparameter, builds on stable-worldmodel/stable-pretraining), `facebookresearch/eb_jepa` (3 examples incl. action-conditioned video JEPA + planning, single GPU), `galilai-group/llm-jepa` (Text↔Code two-view JEPA loss, Semantic Tube Prediction, LoRA, JEPA-loss ablations), `galilai-group/lejepa` (SIGReg, ~50 lines core, ViT-L 304M / ConvNeXtV2-H 660M IN-1K tables), `facebookresearch/cruxeval` (800 samples; generation = CodeLlama-34B generates → execute → filter), `microsoft/CodeBERT/CodeExecutor` (trace token format `<line> <0> <state> s : [ x , y , z ] </state>`, Zenodo datasets), `nicklashansen/tdmpc2` (317M multitask, 300+ checkpoints, 8–12 GB).

**Web searches** (no repo / 2026 discovery): LeJEPA → arXiv:2511.08544 · LeWorldModel → arXiv:2603.19312, code at lucas-maes/le-wm · EB-JEPA → arXiv:2602.03604 · TD-JEPA → arXiv:2510.00739 · CWM license → FAIR Non-Commercial Research License, 32B, mid-trained on Python interpreter traces recording locals after each line · NExT → arXiv:2404.14662, **no GitHub repo found** · TRACED → arXiv:2306.07487, repo ARiSE-Lab/TRACED_ICSE_24 + Figshare mirror · 2026 execution-reasoning papers: 2503.05703, 2403.16437 (REval), 2604.03253, 2605.11922, 2511.18288, 2511.00488, 2604.17010 — all token-space.

**Local:** `cat /home/brice/workspace/sensorium/README.md` (PEP 669 recorder, SQLite per run, `--focus` line-level locals, `diff` causal divergence, `refocus` MATCH/DIVERGED/REFUSED); `find src -name '*.py' | wc -l` → 27; `src/` = `cli.py paths.py query/ record/ store/`.
