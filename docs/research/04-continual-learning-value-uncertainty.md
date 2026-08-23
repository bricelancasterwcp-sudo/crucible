# 04 — Continual Learning, Value Functions, Uncertainty

**Project:** crucible (research spike — frozen 1.5–3B proposer + memory organ + execution-scored tree search + structural uncertainty)
**Survey date:** 2026-08-23
**Hardware target:** 1× RTX 5080 16 GB VRAM, 29 GB system RAM
**Licensing rule applied:** Apache-2.0 / MIT / BSD = adoptable · GPL/AGPL/LGPL = copyleft flag · CC-BY-NC / research-only / custom / none = reference-only

> Every license in this document was read from the GitHub API (`.license.spdx_id`) and, for all top picks, re-verified against the repo's actual `LICENSE` file via `gh api repos/OWNER/REPO/license`. Nothing here is from memory. Commands and outputs are in the [Evidence log](#evidence-log).

---

## (a) Continual / online learning from agent experience

The "sleep" step: periodically fold **verified-only** episodes back into weights via LoRA, without catastrophic forgetting.

| Name | URL | License (verified how) | Stars | Last push | What it gives us | Verdict |
|---|---|---|---|---|---|---|
| **TRL** | https://github.com/huggingface/trl | Apache-2.0 — `gh api repos/huggingface/trl/license` → LICENSE, "Apache License 2.0" | 19133 | 2026-08-23 | `SFTTrainer` + `GRPOTrainer` accepting arbitrary Python reward callables; PEFT/LoRA, gradient checkpointing, activation offloading, chunked CE, vLLM colocate + sleep mode. The consolidation engine. | **ADOPT** |
| **SEAL (Self-Adapting LMs)** | https://github.com/Continual-Intelligence/SEAL | MIT — `gh api .../license` → LICENSE, "MIT License" | 1849 | 2025-08-01 | Full RL loop where the model emits "self-edits", they are trained into LoRAs, LoRAs are **evaluated**, and only winners are behavior-cloned back (ReST-EM). Runs on Llama-3.2-**1B**, LoRA r=16. Closest published analogue to our sleep step. | **PORT-PIECES** |
| **PEFT** | https://github.com/huggingface/peft | Apache-2.0 — `gh api repos/huggingface/peft --jq .license.spdx_id` → `Apache-2.0` | 21580 | 2026-08-22 | `LoraModel.add_weighted_adapter(combination_type=…)` — ties / dare / svd / cat LoRA merging under a permissive license. Our "LoRA soup" without the mergekit copyleft. | **ADOPT** |
| **marc (test-time training)** | https://github.com/ekinakyurek/marc | MIT — `gh api repos/ekinakyurek/marc --jq .license.spdx_id` → `MIT` | 354 | 2025-11-10 | The TTT-LoRA harness SEAL's few-shot code was adopted from; per-task LoRA fit + vLLM `LoRARequest` serving of many adapters. | **PORT-PIECES** |
| **verl** | https://github.com/volcengine/verl | Apache-2.0 — `gh api .../license` → LICENSE, "Apache License 2.0" | 23082 | 2026-08-22 | Production RLVR (PPO/GRPO) with Ray. Correct at scale, heavyweight for one GPU. | REFERENCE-ONLY |
| **OpenRLHF** | https://github.com/OpenRLHF/OpenRLHF | Apache-2.0 — `gh api repos/OpenRLHF/OpenRLHF --jq .license.spdx_id` → `Apache-2.0` | 9945 | 2026-08-13 | Ray-based agentic RL (PPO/DAPO/REINFORCE++). Same multi-GPU assumption as verl. | REFERENCE-ONLY |
| **open-r1** | https://github.com/huggingface/open-r1 | Apache-2.0 — `gh api repos/huggingface/open-r1 --jq .license.spdx_id` → `Apache-2.0` | 26441 | 2026-04-02 | Recipe-level reference for GRPO-on-verifiable-rewards config; thin wrapper over TRL. Stale ~5 months. | REFERENCE-ONLY |
| **agent-workflow-memory** | https://github.com/zorazrw/agent-workflow-memory | Apache-2.0 — `gh api repos/zorazrw/agent-workflow-memory --jq .license.spdx_id` → `Apache-2.0` | 460 | 2025-12-22 | Induce reusable *workflows* from past trajectories, abstract out instance context, inject into prompt. Procedural-memory design, offline **and** online variants. | **PORT-PIECES** |
| **ExpeL** | https://github.com/LeapLabTHU/ExpeL | Apache-2.0 — `gh api repos/LeapLabTHU/ExpeL --jq .license.spdx_id` → `Apache-2.0` | 236 | 2024-12-20 | Cross-episode insight extraction + retrieval from a success/failure pool. In-context, no weight updates. Design reference for the semantic-memory writer. | REFERENCE-ONLY |
| **Reflexion** | https://github.com/noahshinn/reflexion | MIT — `gh api repos/noahshinn/reflexion --jq .license.spdx_id` → `MIT` | 3239 | 2025-01-14 | Verbal self-critique loop; the "learning part" is an episodic buffer replayed in-context. Superseded by AWM/ExpeL for our purpose. | REFERENCE-ONLY |
| **O-LoRA** | https://github.com/cmnfriend/O-LoRA | MIT — `gh api .../license` → LICENSE, "MIT License" | 212 | 2024-07-13 | Orthogonal-subspace constraint across sequential LoRAs — the cheapest anti-forgetting regularizer to port (a loss term, ~30 lines). Old but the method is small. | **PORT-PIECES** |
| **InfLoRA** | https://github.com/liangyanshuo/InfLoRA | MIT — `gh api repos/liangyanshuo/InfLoRA --jq .license.spdx_id` → `MIT` | 114 | 2025-03-13 | Interference-free LoRA (CVPR'24). Vision-CL codebase; method transfers, code does not. | REFERENCE-ONLY |
| **mergekit** | https://github.com/arcee-ai/mergekit | **LGPL-3.0** — `gh api repos/arcee-ai/mergekit --jq .license.spdx_id` → `LGPL-3.0` | 7302 | 2026-06-17 | TIES/DARE/task-arithmetic merging. **Copyleft flag** — PEFT covers our merge needs under Apache-2.0. | SKIP (copyleft) |
| **SPIN** | https://github.com/uclaml/SPIN | Apache-2.0 — `gh api repos/uclaml/SPIN --jq .license.spdx_id` → `Apache-2.0` | 1254 | 2024-05-08 | Self-play fine-tuning against your own prior generations. No verifier in the loop — wrong shape for us (we *have* ground truth from tests). | SKIP |
| **STaR / Quiet-STaR** | https://github.com/ezelikman/STaR · https://github.com/ezelikman/quiet-star | Apache-2.0 both — `gh api repos/ezelikman/STaR --jq .license.spdx_id` → `Apache-2.0`; same for `quiet-star` | 230 / 739 | 2023-02-21 / 2024-08-21 | STaR = the original filter-then-SFT loop (our sleep step, minus infrastructure). Quiet-STaR is token-level latent thoughts — orthogonal. | REFERENCE-ONLY |

### Top pick 1 — SEAL (MIT)

**Why it is the closest thing that exists to our sleep step.** SEAL's few-shot pipeline is literally: generate candidate self-edits → train a LoRA per edit → **evaluate each LoRA** → keep only the edits whose LoRAs improved measured performance → behavior-clone on the survivors (`BC-self-edit.py`, ReST-EM) → repeat. Swap "ARC task solved" for "pytest suite green" and that is crucible's consolidation loop.

**Reuse:** the four-stage script skeleton (`self-edit.py` → `eval-self-edits.py` → `BC-self-edit.py` → next iteration), `few-shot/ttt.py`'s LoRA config plumbing, and the vLLM multi-adapter serving in `few-shot/inference/` (`LoRARequest`, `max_lora_rank`).

**Change:** everything domain-specific. `arclib/` is ARC grid handling — delete. The self-edit *content* becomes our verified episode records (bug, repair diff, test outcome, provenance) rather than free-form "finetuning directives". SEAL's reward is task accuracy; ours is the test-suite verdict, which is cheaper and binary, so the filtering step gets simpler and more trustworthy.

**VRAM/data:** the README's headline claim is "2 A100/H100 GPUs" — **but that is for the full RL sweep**. The actual few-shot commands run `meta-llama/Llama-3.2-1B-Instruct` with `--lora_rank=16 --lora_alpha=16 --per_device_train_batch_size=5`. At our 1.5–3B target with LoRA r=16–32 in bf16: base weights 3–6 GB, LoRA + optimizer state well under 1 GB, activations 2–4 GB with gradient checkpointing at 2–4k context. **The SFT half fits 16 GB with room to spare.** Data: SEAL runs its first iteration on **12 problems × 15 self-edits** — encouragingly small, which matters because verified episodes will be scarce early.

**Gotchas:** (1) `requirements.txt` pins `torch==2.7.0`, `vllm==0.9.1`, `trl==0.18.1`, `transformers==4.52.4` — these predate mature Blackwell (`sm_120`) support; install current cu128 wheels and expect the pins to fight you. (2) Requires `OPENAI_API_KEY` in `.env` and imports `openai`/`litellm` — a hosted-model dependency that must be cut for a self-contained local spike. (3) All shell scripts carry SLURM directives. (4) Last push 2025-08-01 — a year stale; treat as a paper artifact to port, not a dependency to pin.

### Top pick 2 — TRL (Apache-2.0)

**Reuse:** `SFTTrainer` for the LoRA consolidation pass, and `GRPOTrainer` (`trl/trainer/grpo_trainer.py`, confirmed by code search) if we later want the proposer to *learn to propose* rather than only be distilled into. `GRPOTrainer` takes arbitrary reward functions, so our sandboxed-pytest verdict drops straight in.

**Change:** TRL ships **no execution-based reward**. `docs/source/rewards.md` lists exactly `accuracy_reward`, `reasoning_accuracy_reward`, `get_cosine_scaled_reward`, `think_format_reward`, `get_repetition_penalty_reward`, `get_soft_overlong_punishment` — all math/format. We write the pytest-in-sandbox reward ourselves. Also no replay/anti-forgetting: mixing a rehearsal buffer into the SFT dataset is our code (trivial), and an O-LoRA orthogonality penalty is a custom loss if we need it.

**VRAM:** the memory levers are all documented and real (`docs/source/reducing_memory_usage.md`: PEFT, packing, Liger, chunked cross-entropy, padding-free, activation offloading, gradient checkpointing, **vLLM sleep mode**). LoRA-SFT of a 3B model at 16 GB is comfortable. **GRPO is the tight one**: policy + reference + vLLM KV cache co-resident. Use colocate mode *plus* sleep mode, or drop to 1.5B, or run generation as a separate pass. Our **29 GB system RAM** is arguably the binding constraint once vLLM, the trainer, and the episode store coexist — budget for it.

**Verdict framing:** ADOPT as the training substrate; the interesting research content is the episode-selection policy we wrap around it, not the optimizer.

---

## (b) Process reward models / value functions for code from outcome labels

| Name | URL | License (verified how) | Stars | Last push | What it gives us | Verdict |
|---|---|---|---|---|---|---|
| **ThinkPRM** | https://github.com/mukhal/thinkprm | MIT — `gh api .../license` → LICENSE, "MIT License" | 91 | 2026-07-30 | Generative long-CoT PRM trained on **1K** synthetic verification CoTs filtered against 8K PRM800K labels. Ships **ThinkPRM-1.5B** (HF card: `apache-2.0`, base R1-Distill-Qwen-1.5B). Proof a ≤1.5B verifier is viable. | **ADOPT (model) + PORT-PIECES (recipe)** |
| **OpenR** | https://github.com/openreasoner/openr | MIT — `gh api .../license` → LICENSE, "MIT License" | 1853 | 2025-01-17 | **OmegaPRM** MCTS-based automatic step labeling, discriminative + generative PRM training (`prm/code`, `gen_rm/`), and Best-of-N / beam / MCTS search that consumes the PRM. The most complete open PRM stack. | **PORT-PIECES** |
| **RLHF-Reward-Modeling** | https://github.com/RLHFlow/RLHF-Reward-Modeling | Apache-2.0 — `gh api repos/RLHFlow/RLHF-Reward-Modeling --jq .license.spdx_id` → `Apache-2.0` | 1541 | 2025-04-24 | Clean, permissive reward/PRM training recipes (incl. Math-Shepherd-style automatic labeling). Good scaffolding for a discriminative head. | **PORT-PIECES** |
| **PRM800K** | https://github.com/openai/prm800k | MIT — `gh api repos/openai/prm800k --jq .license.spdx_id` → `MIT` | 2150 | 2023-06-01 (**archived: true**) | 800K human step-labels on MATH. Data + label schema only; wrong domain, archived. | REFERENCE-ONLY |
| **rStar** | https://github.com/microsoft/rStar | MIT — `gh api repos/microsoft/rStar --jq .license.spdx_id` → `MIT` | 1425 | 2025-09-12 | MCTS + mutual-consistency discriminator; the rStar-Math PPM pairwise-preference idea (rank steps rather than regress scores) is the transferable bit given noisy labels. | **PORT-PIECES** |
| **PURE** | https://github.com/CJReinforce/PURE | **NONE** — `gh api repos/CJReinforce/PURE --jq .license.spdx_id` → `null`; root listing has no LICENSE/COPYING file | 172 | 2025-10-23 | Min-form credit assignment for PRMs (use `min` over step rewards, not sum) — a one-line idea worth stealing conceptually. | REFERENCE-ONLY (no license) |
| **PRMBench** | https://github.com/ssmisya/PRMBench | Apache-2.0 — `gh api repos/ssmisya/PRMBench --jq .license.spdx_id` → `Apache-2.0` | 94 | 2025-02-15 | Fine-grained PRM evaluation harness — how to *measure* whether our value function is any good. | **PORT-PIECES** |
| **CodePRM** | https://github.com/SIMONLQY/CodePRM | **NONE** — `gh api repos/SIMONLQY/CodePRM --jq .license.spdx_id` → `null`; root listing has no LICENSE/COPYING file | 10 | 2025-05-26 | Nominally the closest name-match (PRM for code). Unlicensed, 10 stars. | REFERENCE-ONLY (no license) |
| **CodePRM-DataKit** | https://github.com/Drnaive/CodePRM-DataKit | MIT — `gh api repos/Drnaive/CodePRM-DataKit --jq .license.spdx_id` → `MIT` | 3 | 2026-05-18 | Step-level preference data construction for code-agent PRMs. Permissive but 3 stars — inspect before trusting. | REFERENCE-ONLY |
| **SWE-Gym / R2E-Gym / SWE-smith** | https://github.com/SWE-Gym/SWE-Gym · https://github.com/R2E-Gym/R2E-Gym · https://github.com/SWE-bench/SWE-smith | Apache-2.0 / Apache-2.0 / MIT — `gh api repos/OWNER/REPO --jq .license.spdx_id` → `Apache-2.0`, `Apache-2.0`, `MIT` | 723 / 323 / 748 | 2025-07-29 / 2025-07-13 / 2026-08-17 | Executable Python-repo environments with test verdicts — the **episode source** that feeds both the value function and the sleep step. SWE-smith is the actively maintained one and does bug *synthesis*. | **ADOPT (episode source)** |

### Top pick 1 — ThinkPRM (MIT code, Apache-2.0 weights)

**The finding that de-risks (b):** ThinkPRM trained a usable process verifier from **1,000** synthetic verification CoTs, filtered by requiring every step-judgment to match gold PRM800K labels. Released at 1.5B / 7B / 14B; **ThinkPRM-1.5B is Apache-2.0**, finetuned from DeepSeek-R1-Distill-Qwen-1.5B. So: **yes, a ≤1.5B value model is realistic**, and the data requirement is ~10³ examples, not ~10⁵.

**Reuse:** the data-construction recipe — sample a verification CoT over a candidate's steps, emit `\boxed{correct}` / `\boxed{incorrect}` per step, and **keep only chains whose step judgments match ground truth**. In crucible we have something better than PRM800K's human labels: test execution. Per-step gold is still not free (tests give an *outcome*, not per-step credit), so pair this with OmegaPRM-style MCTS attribution or rStar-style pairwise preference to get step-level signal.

**Change:** it is math-domain and it is *generative* (a verifier that thinks costs a full decode per scoring call). For tree search over repair candidates we need many cheap scores per node, so use the generative PRM as a **label generator / occasional oracle** and distill into a small discriminative head for the inner loop.

**VRAM:** 1.5B verifier in bf16 ≈ 3 GB, or ~1 GB at 4-bit — co-resident with a 3B proposer inside 16 GB. Serving is via sglang in the repo (they use two venvs to dodge dependency conflicts — expect the same friction).

**Gotchas:** the repo's own two-virtualenv install instruction is a smell for dependency fragility; the long-CoT verifier is slow (that is the paper's whole point — scaling verifier compute); and its filtering assumes gold step labels exist, which is precisely what we do not have for code.

### Top pick 2 — OpenR / OmegaPRM (MIT)

**Reuse:** `prm/code` (discriminative PRM training), `gen_rm/` (generative RM), the OmegaPRM automatic-labeling implementation, and the search side (Best-of-N, beam, MCTS) that already consumes a PRM — which overlaps our reasoning-as-tree-search component, so we get labeler and consumer from one MIT codebase.

**Change:** math-only I/O throughout; the "step" abstraction must become our repair-step abstraction (edit hunk / hypothesis / test-run). OmegaPRM's labeling assumes a cheap correctness oracle at rollout leaves — for us that is running the test suite, which is *far* more expensive than checking a math answer. Budget the MCTS accordingly, and cache aggressively per (repo-state, patch) hash.

**Data note:** OmegaPRM's value comes from generating step labels from *outcome* labels via rollout statistics — exactly the "value function trained on real outcomes" the project calls for. This is the mechanism to port.

**Gotchas:** last push 2025-01-17 (19 months stale) — a reference implementation to port, not a dependency. It also benchmarks against `peiyi9979/math-shepherd-mistral-7b-prm`; note that **Math-Shepherd has no maintained first-party training repo** — searches surfaced only unrelated or toy forks, so treat Math-Shepherd as a *method* (automatic step labels from completion-rate rollouts) reimplemented via OpenR/RLHFlow, not as adoptable code.

---

## (c) Calibration / uncertainty for LLM-produced claims

| Name | URL | License (verified how) | Stars | Last push | What it gives us | Verdict |
|---|---|---|---|---|---|---|
| **LM-Polygraph** | https://github.com/IINemo/lm-polygraph | MIT — `gh api .../license` → LICENSE.md, "MIT License" | 500 | 2026-08-19 | ~45 UE estimators incl. `semantic_entropy.py`, `p_true.py`, `verbalized_1s/2s.py`, Mahalanobis; a **claim-level** estimator package (`estimators/claim/`); and **calibration normalizers** (`isotonic_pcc.py`, `binned_pcc.py`, `quantile.py`) that map raw UE scores → interpretable confidence. TACL 2025 benchmark. | **ADOPT** |
| **crepes** | https://github.com/henrikbostrom/crepes | BSD-3-Clause — `gh api .../license` → LICENSE, "BSD 3-Clause…" | 579 | 2026-07-08 | Conformal classifiers/regressors/**predictive systems** on top of any model, with **Mondrian (category-conditional)** variants — coverage guarantees *per provenance class*, which is exactly our structural-uncertainty shape. Small, pure, no sklearn-estimator ceremony. | **ADOPT** |
| **MAPIE** | https://github.com/scikit-learn-contrib/MAPIE | BSD-3-Clause — `gh api .../license` → LICENSE, "BSD 3-Clause 'New' or 'Revised' License" | 1583 | 2026-08-14 | Mature scikit-learn-contrib conformal library: prediction sets, **risk control** (LTT / CRC), well-documented. Best-supported option if we want a maintained dependency. | **ADOPT** |
| **TorchCP** | https://github.com/ml-stat-Sustech/TorchCP | **LGPL-3.0** — `gh api repos/ml-stat-Sustech/TorchCP/license` → LICENSE, "GNU Lesser General Public License v3.0" | 477 | 2026-08-05 | PyTorch-native conformal prediction, incl. LLM-oriented scores. **Copyleft flag** — crepes/MAPIE cover the same ground under BSD. | SKIP (copyleft) |
| **semantic_uncertainty** (Farquhar et al.) | https://github.com/jlko/semantic_uncertainty | **BSD-3-Clause-Clear** — `gh api repos/jlko/semantic_uncertainty/license` → LICENSE, "BSD 3-Clause Clear License" | 422 | 2024-04-12 | Reference implementation of semantic entropy (Nature 2024). Clear variant = BSD + explicit *no patent grant*; still permissive, adoptable. Stale. | REFERENCE-ONLY (LM-Polygraph reimplements it under MIT) |
| **long_hallucinations** | https://github.com/jlko/long_hallucinations | **NONE** — `gh api repos/jlko/long_hallucinations --jq .license.spdx_id` → `null`; root contents listing shows **no LICENSE file** | 83 | 2024-04-12 | Paragraph-length semantic-entropy experiments (claim decomposition + per-claim entropy). | REFERENCE-ONLY (unlicensed — do not vendor) |
| **LLM-Uncertainty-Bench** | https://github.com/smartyfh/LLM-Uncertainty-Bench | (surfaced via search; not adopted — see note) | 263 | 2026-08-04 | Conformal-prediction-based UQ benchmark for LLMs. Benchmark, not a library we need. | SKIP |
| **ConformalLLM** | https://github.com/bhaweshiitk/ConformalLLM | (surfaced via search; not adopted — see note) | 69 | 2026-08-20 | Conformal prediction over LLM outputs; small, research-grade. | SKIP |

> Note: the two SKIP rows above were identified by `gh search repos` but not license-verified, because they are skipped on capability grounds regardless of license. **Do not treat their license as known.** Verify before any use.

### Top pick 1 — LM-Polygraph (MIT)

**Why it wins (c).** It is the only actively maintained (pushed 2026-08-19), permissively licensed library that covers all three things we need at once:

1. **Pre-execution confidence on a repair candidate** — white-box sequence-level estimators computed from the proposer's own logits during generation: `semantic_entropy.py`, `token_entropy.py`, `perplexity.py`, `p_true.py`, `mahalanobis_distance.py`, `self_certainty.py`. Cheap ones (max-prob, perplexity, self-certainty) are essentially free at generation time — usable as a **tree-search prior** to order candidates *before* paying for a test run. That is a direct compute saving in the search loop, not just a reporting nicety.
2. **Confidence on a stored semantic claim** — the `estimators/claim/` package scores individual claims rather than whole generations, which maps onto our semantic-memory records.
3. **Actual calibration** — `normalizers/isotonic_pcc.py` and `binned_pcc.py` fit a monotone map from raw UE score → calibrated confidence on held-out data. This is the piece most UQ repos omit and the piece we specifically need, because "0.83" must mean something.

**Reuse:** import as a library; take the estimator + normalizer abstractions and the `stat_calculators` design (compute logits/embeddings once, feed many estimators). **Change:** it is built around NLG benchmarks (QA/summarization/translation) with `generation_metrics` to match; our "correctness" signal is a test verdict, so we supply a custom metric and recalibrate on our own episodes. Sampling-based estimators (semantic entropy, MC entropy) need *k* generations per candidate — with a 1.5–3B proposer and vLLM that is affordable, but it is the dominant cost; the deterministic estimators are not.

**VRAM/gotchas:** white-box estimators need the *same* model that generated the text (hidden states/logits), so the proposer must be served in-process or via an adapter exposing logprobs — plan the serving layer accordingly. Python 3.12 per the badge. `unbabel-comet` extra pins `numpy<2.0` and conflicts with vLLM — do not install the `[comet]` extra.

### Top pick 2 — crepes (BSD-3-Clause), with MAPIE as the maintained fallback

**Why crepes specifically.** Its **Mondrian** conformal predictors give coverage guarantees *conditional on a category you choose*. Our category is the provenance/verification status of the claim — `verified-by-test`, `believed-unverified`, `unknown`. That means we can honestly say "within the *unverified* bucket, this confidence has 90% coverage", instead of a single global guarantee that averages the buckets together and hides exactly the distinction the project is built to expose. It also implements conformal **predictive systems** (full distributions, not just sets), which suits a value function's scalar output.

**Reuse:** wrap the value head's score as the non-conformity measure; calibrate on a held-out slice of verified episodes; emit prediction sets. **Abstention falls out for free**: an empty prediction set at the chosen error rate *is* "I don't know", and it is a principled abstention rather than a threshold we invented. **Change:** conformal validity assumes exchangeability between calibration and test data — an assumption our setup **violates by design**, since the memory organ and the sleep step deliberately shift the distribution over time. Mitigate by recalibrating after every sleep cycle, and by holding out a rolling recent window rather than a fixed early split. Treat this as a first-class experimental risk, not a footnote.

**MAPIE fallback:** if we want a dependency with more maintenance surface (scikit-learn-contrib, pushed 2026-08-14), MAPIE covers the same conformal ground and adds **risk control** (bounding a chosen loss, not just miscoverage) — useful if we later want to bound "fraction of confidently-wrong repairs" directly. Same BSD-3-Clause. Both are small; adopting one and keeping the other in reserve costs nothing.

---

## Recommended path

1. **Sleep step:** TRL `SFTTrainer` + PEFT LoRA (r=16–32, bf16) over verified-only episodes; port SEAL's four-stage ReST-EM skeleton (propose → train LoRA → *evaluate* → behavior-clone survivors) as the control flow.
2. Guard forgetting cheaply first: a rehearsal buffer mixed into each SFT pass, plus PEFT `add_weighted_adapter` to merge/average successive LoRAs. Add O-LoRA's orthogonality penalty only if measured forgetting demands it.
3. Defer GRPO. Filtered-SFT (ReST-EM) gets most of the gain at a fraction of the 16 GB pressure; revisit `GRPOTrainer` once the episode store is non-trivial.
4. **Value function:** two tiers — a small discriminative head over proposer hidden states for the inner search loop, labeled by OmegaPRM-style MCTS rollout statistics ported from OpenR; ThinkPRM-1.5B (Apache-2.0) as an occasional oracle and label generator. Evaluate with PRMBench's methodology.
5. Use rStar-style **pairwise preference** over candidate repairs rather than absolute score regression — more robust to the noisy step-credit we get from binary test outcomes.
6. Episodes come from SWE-smith (MIT, actively maintained) for mutation-style bug synthesis in real Python repos.
7. **Uncertainty:** LM-Polygraph estimators for pre-execution candidate confidence (cheap deterministic ones inside the search loop, sampling-based ones for stored claims) + its isotonic normalizer for calibration.
8. Wrap the calibrated score in crepes Mondrian conformal prediction, with provenance/verification status as the Mondrian category — abstention = empty prediction set.
9. Recalibrate after every sleep cycle; exchangeability is violated by design and must be monitored as a first-class risk.
10. **Nothing here is a framework commitment:** TRL/PEFT/LM-Polygraph/crepes are libraries; SEAL/OpenR are code we port. No copyleft enters the tree (mergekit and TorchCP both excluded, both replaceable).

## Gaps — what we build ourselves

- **A PRM/value function for program repair does not exist in open source.** `gh search repos "process reward model program repair"`, `"value network program repair"`, and `"learned verifier code repair"` all returned **zero results**. Every open PRM is math-domain. Our step-credit-assignment-from-test-outcomes design is net-new.
- **Execution-based reward function.** TRL ships six reward functions, all math/format (`docs/source/rewards.md`) — none execute code. The sandboxed-pytest reward, its timeout/resource limits, and its result caching are ours.
- **Conformal prediction over code generation is unaddressed.** `gh search repos "conformal prediction code generation"` returned nothing relevant. Choosing the non-conformity score for a repair candidate is an open design question.
- **The structural-uncertainty layer is entirely ours.** No surveyed repo derives confidence from *provenance + verification status*. LM-Polygraph gives model-internal uncertainty; conformal gives coverage; the mapping from "where did this belief come from and was it checked" → a three-way `verified` / `believed-unchecked` / `unknown` output has no prior art here.
- **Verified-episode selection policy.** SEAL filters on task accuracy; we must define what makes an episode worth consolidating (passing tests is necessary, not sufficient — overfitted or trivially-passing repairs would poison the weights).
- **Anti-forgetting evaluation.** No surveyed repo measures forgetting for an agent learning from its own experience over long horizons. We need a held-out regression suite run every sleep cycle.
- **Blackwell (sm_120) toolchain.** Every candidate's pins predate the RTX 5080; expect to rebuild the torch/vLLM/flash-attn stack and to treat all pinned requirements as advisory.

---

## Evidence log

All commands run 2026-08-23 from a shell with `gh` authenticated as `bricelancasterwcp-sudo` (verified via `gh auth status`).

**Environment**
- `gh auth status` → `✓ Logged in to github.com account bricelancasterwcp-sudo`, scopes `gist, read:org, repo, workflow`.
- `gh api rate_limit --jq '{core:.resources.core, search:.resources.search}'` → `core limit 5000`, `search limit 30`. *Search bucket (30/min) was exhausted twice during this survey; core bucket carried the license verification.*

**Method gotcha worth recording:** `gh search repos "quoted multi word phrase"` returns **zero results**; unquoted terms work (they are ANDed). Long descriptions blow past output limits — use `--json fullName,stargazersCount,updatedAt --jq`.

**Repo searches run** (`gh search repos <terms> --sort stars --limit N`)
`STaR self-taught reasoner` · `quiet-star` · `open-r1` · `V-STaR verifier self-taught` · `SPIN self-play finetuning LLM` · `ReST expert iteration language model` · `SEAL self-adapting language models` · `ExpeL experiential learning agent` · `agent workflow memory` · `continual learning LoRA O-LoRA orthogonal` · `InfLoRA continual learning` · `LoRA merging model soup mergekit` · `elastic weight consolidation LLM continual` · `experience replay rehearsal LLM finetuning` · `CodePRM` · `process reward model code` · `Math-Shepherd` · `value network program repair` · `learned verifier code repair` · `conformal prediction language model` · `selective prediction abstention LLM` · `semantic entropy hallucination` · `LLM uncertainty quantification benchmark` · `conformal prediction LLM` · `GRPO unit test reward code` · `RLVR code execution reward` · `process reward model program repair` · `test time training LoRA` · `conformal prediction code generation`

**Zero-result searches (load-bearing negative evidence):**
`value network program repair` → *(empty)* · `learned verifier code repair` → *(empty)* · `process reward model program repair` → *(empty)* · `RLVR code execution reward` → *(empty)* · `conformal prediction code generation` → only `VAIXLNS/VAIXLNS` (2 stars, unrelated) and `MayaVB/conformal-doa-tracking` (0 stars, DOA tracking). `Math-Shepherd` → top hit `aaronlyt/math-prm-openai-2023` (9 stars); no first-party repo.

**Code searches run** (`gh search code`)
- `GRPOTrainer repo:huggingface/trl` → `trl/trainer/grpo_trainer.py: class GRPOTrainer(_BaseTrainer)`, `trl/experimental/gspo_token/grpo_trainer.py`, `trl/__init__.py: "GRPOTrainer"`, docs references.
- `add_weighted_adapter repo:huggingface/peft` → `src/peft/tuners/lora/model.py: def add_weighted_adapter(`, `docs/source/developer_guides/model_merging.md` ("the specific model merging method is specified in the `combination_type` parameter").
- `LoRA repo:Continual-Intelligence/SEAL` → `few-shot/ttt.py: # lora config`, `few-shot/inference/engine_no_lora.py: from vllm.lora.request import LoRARequest`, `max_lora_rank: int = 64`, README `--lora_rank=16 --lora_alpha=16`.
- `"def compute_reward" pytest` → only third-party framework docs (`math-inc/OpenGauss`, `redai-infra/Relax`); **no reusable pytest-execution reward for TRL**.
- `conformal abstention language:python` → scattered research scripts (`Imbernoulli/MLS-Bench`, `debajyotidasgupta/eu-halt`, `AidanSYu/fMRI-Abstention`); no library.

**License / metadata verification** (`gh api repos/OWNER/REPO --jq '{license:.license.spdx_id,stars:…,pushed:…,archived:…}'`)

| Repo | spdx_id | stars | pushed | archived |
|---|---|---|---|---|
| ezelikman/STaR | Apache-2.0 | 230 | 2023-02-21 | false |
| ezelikman/quiet-star | Apache-2.0 | 739 | 2024-08-21 | false |
| uclaml/SPIN | Apache-2.0 | 1254 | 2024-05-08 | false |
| LeapLabTHU/ExpeL | Apache-2.0 | 236 | 2024-12-20 | false |
| noahshinn/reflexion | MIT | 3239 | 2025-01-14 | false |
| zorazrw/agent-workflow-memory | Apache-2.0 | 460 | 2025-12-22 | false |
| Continual-Intelligence/SEAL | MIT | 1849 | 2025-08-01 | false |
| volcengine/verl | Apache-2.0 | 23082 | 2026-08-22 | false |
| huggingface/trl | Apache-2.0 | 19133 | 2026-08-23 | false |
| OpenRLHF/OpenRLHF | Apache-2.0 | 9945 | 2026-08-13 | false |
| huggingface/open-r1 | Apache-2.0 | 26441 | 2026-04-02 | false |
| huggingface/peft | Apache-2.0 | 21580 | 2026-08-22 | false |
| openai/prm800k | MIT | 2150 | 2023-06-01 | **true** |
| microsoft/rStar | MIT | 1425 | 2025-09-12 | false |
| openreasoner/openr | MIT | 1853 | 2025-01-17 | false |
| mukhal/thinkprm | MIT | 91 | 2026-07-30 | false |
| RLHFlow/RLHF-Reward-Modeling | Apache-2.0 | 1541 | 2025-04-24 | false |
| arcee-ai/mergekit | **LGPL-3.0** | 7302 | 2026-06-17 | false |
| cmnfriend/O-LoRA | MIT | 212 | 2024-07-13 | false |
| liangyanshuo/InfLoRA | MIT | 114 | 2025-03-13 | false |
| SIMONLQY/CodePRM | **null** | 10 | 2025-05-26 | false |
| Drnaive/CodePRM-DataKit | MIT | 3 | 2026-05-18 | false |
| CJReinforce/PURE | **null** | 172 | 2025-10-23 | false |
| ssmisya/PRMBench | Apache-2.0 | 94 | 2025-02-15 | false |
| ekinakyurek/marc | MIT | 354 | 2025-11-10 | false |
| SWE-Gym/SWE-Gym | Apache-2.0 | 723 | 2025-07-29 | false |
| R2E-Gym/R2E-Gym | Apache-2.0 | 323 | 2025-07-13 | false |
| SWE-bench/SWE-smith | MIT | 748 | 2026-08-17 | false |
| scikit-learn-contrib/MAPIE | BSD-3-Clause | 1583 | 2026-08-14 | false |
| ml-stat-Sustech/TorchCP | **LGPL-3.0** | 477 | 2026-08-05 | false |
| henrikbostrom/crepes | BSD-3-Clause | 579 | 2026-07-08 | false |
| IINemo/lm-polygraph | MIT | 500 | 2026-08-19 | false |
| jlko/semantic_uncertainty | BSD-3-Clause-Clear | 422 | 2024-04-12 | false |
| jlko/long_hallucinations | **null** | 83 | 2024-04-12 | false |
| valeman/awesome-conformal-prediction | NOASSERTION | 1290 | 2026-07-09 | false |

**Null-license resolution** (rule: null → fetch LICENSE file and state what it is)
- `gh api repos/jlko/long_hallucinations/contents/LICENSE` → empty; `gh api repos/jlko/long_hallucinations/contents --jq '.[].name'` → `.gitignore, README.md, data.py, environment.yaml, environment_export.yaml, eval_utils.py, hallucination.py, models.py, notebooks, utils.py` — **no LICENSE file. Unlicensed → reference-only, do not vendor.**
- `gh api repos/SIMONLQY/CodePRM/contents --jq '.[].name' | grep -iE 'licen|copying'` → **no match. Unlicensed → reference-only.**
- `gh api repos/CJReinforce/PURE/contents --jq '.[].name' | grep -iE 'licen|copying'` → **no match. Unlicensed → reference-only.**

**LICENSE-file re-verification of top picks** (`gh api repos/OWNER/REPO/license --jq '"\(.license.spdx_id)\t\(.license.name)\t\(.path)"`)
```
Continual-Intelligence/SEAL     MIT            MIT License                                      LICENSE
IINemo/lm-polygraph             MIT            MIT License                                      LICENSE.md
mukhal/thinkprm                 MIT            MIT License                                      LICENSE
huggingface/trl                 Apache-2.0     Apache License 2.0                               LICENSE
volcengine/verl                 Apache-2.0     Apache License 2.0                               LICENSE
openreasoner/openr              MIT            MIT License                                      LICENSE
cmnfriend/O-LoRA                MIT            MIT License                                      LICENSE
henrikbostrom/crepes            BSD-3-Clause   BSD 3-Clause "New" or "Revised" License          LICENSE
scikit-learn-contrib/MAPIE      BSD-3-Clause   BSD 3-Clause "New" or "Revised" License          LICENSE
ml-stat-Sustech/TorchCP         LGPL-3.0       GNU Lesser General Public License v3.0           LICENSE
jlko/semantic_uncertainty       BSD-3-Clause-Clear  BSD 3-Clause Clear License                  LICENSE
```

**Model-weight license** (WebFetch, since HF is not on GitHub)
- `https://huggingface.co/launch/ThinkPRM-1.5B` → license tag **`apache-2.0`**; base model **DeepSeek-R1-Distill-Qwen-1.5B**.

**READMEs / file listings read** (`gh api repos/OWNER/REPO/readme --jq .content | base64 -d`, and `…/contents/PATH`)
- **SEAL** README → "framework for training language models via RL to generate self-edits"; "All experiments can be run with 2 A100/H100 GPUs"; requires `OPENAI_API_KEY`; SLURM directives in every `.sh`.
- **SEAL** `requirements.txt` → `torch==2.7.0, transformers==4.52.4, vllm==0.9.1, trl==0.18.1, peft==0.15.2, openai==1.86.0, litellm==1.72.4, flashinfer-python==0.2.2`.
- **SEAL** `few-shot/` listing → `self-edit.py, eval-self-edits.py, BC-self-edit.py, ttt.py, arclib/, inference/, launch.sh`. `few-shot/README.md` → `--model_name=meta-llama/Llama-3.2-1B-Instruct --n_tasks=12 --n_self_edits_per_task=15`; "RestEM on Iteration 1"; `--lora_rank=16 --lora_alpha=16 --per_device_train_batch_size=5`; "Code is adopted from Ekin's Repo" (ekinakyurek/marc).
- **ThinkPRM** README → trained on 1K synthetic verification CoTs "filtered based on only on 8K process labels from PRM800K"; releases at 1.5B/7B/14B from R1-Distill-Qwen; install requires two uv venvs (sglang conflicts).
- **OpenR** README → OmegaPRM row ("Automated Process Supervision"), "Generative and Discriminative PRM Training" (`prm/code`, `gen_rm/`), search strategies "Greedy / Best-of-N / Beam / MCTS / rStar"; compares against `peiyi9979/math-shepherd-mistral-7b-prm`.
- **TRL** `docs/source/rewards.md` headings → `accuracy_reward, reasoning_accuracy_reward, get_cosine_scaled_reward, think_format_reward, get_repetition_penalty_reward, get_soft_overlong_punishment` — **no execution-based reward**.
- **TRL** `docs/source/reducing_memory_usage.md` headings → `Truncation, Packing, PEFT, Liger, Chunked cross-entropy, Padding-free, Activation offloading, Disabling model gathering for generation in online methods, vLLM sleep mode, Gradient checkpointing`.
- **TRL** `docs/source/grpo_trainer.md` → vLLM server vs **colocate** mode ("vLLM runs inside the trainer process and shares GPU memory… may lead to memory contention"; colocate is default).
- **LM-Polygraph** README → MIT badge, Python 3.12, TACL 2025 benchmark; `[comet]` extra "pins `numpy<2.0` which may conflict with packages like vLLM".
- **LM-Polygraph** `src/lm_polygraph/estimators/` → 45 files incl. `semantic_entropy.py, p_true.py, verbalized_1s.py, verbalized_2s.py, mahalanobis_distance.py, self_certainty.py, token_entropy.py, perplexity.py, claim/`. `estimators/claim/` → 13 claim-level estimators. `normalizers/` → `base.py, binned_pcc.py, isotonic_pcc.py, minmax.py, quantile.py`. Package root → `estimators, generation_metrics, model_adapters, normalizers, stat_calculators, ue_metrics`.
- **crepes** README → "conformal classifiers, regressors, and predictive systems on top of any standard classifier and regressor"; "standard and Mondrian conformal classifiers as well as standard, normalized and Mondrian conformal regressors"; PyPI 0.9.1, BSD-3-clause badge.
- **agent-workflow-memory** README → "induce, integrate, and utilize workflows via an agent memory"; "operate in both offline and online settings"; WebArena + Mind2Web pipelines.
- **ExpeL** README → "LLM Agents are Experiential Learners", AAAI 2024 Oral, Apache-2.0 badge.
