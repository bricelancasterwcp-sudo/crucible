# Pillar 3 — Search + Verify-by-Execution: open-source component survey

Survey date: **2026-08-23**. All license/star/push facts below were read from the GitHub API
during this survey (see **Evidence log**); none are from memory.

---

## Summary

- Execution-scored search over code repair is a solved-in-parts problem: the **search policies**
  (MCTS/UCT, Thompson-sampling bandit, best-first) and the **test-execution harnesses** both exist
  under permissive licenses, but almost nobody ships them wired together with a *learned* value
  function trained on real execution outcomes.
- Adopt outright: `REx` (45-line Thompson-sampling refinement scheduler, MIT),
  `mini-swe-agent` (~200-line agent loop + local/Docker exec env, MIT),
  `SWE-smith` (procedural mutation bug-injection + validation harness, MIT) — SWE-smith's
  `bug_gen/procedural` **is** our task generator for mutation-injected Python bugs.
- Port pieces: `moatless-tools`/`moatless-tree-search` (UCT selector, node/tree, pytest log
  parsers, local swebench runtime — Apache-2.0/MIT) and `Code-AI-Tree-Search` (P-UCT with an
  actual value-model head and pass-rate reward, MIT).
- Reference-only on license: **AlphaCodium (AGPL-3.0)**, **SWE-RL (CC-BY-NC-4.0)**,
  **auto-code-rover (Sonar Source-Available, non-compete)**, **ReST-MCTS\* (no LICENSE file)**,
  **PSearch (no LICENSE file)**.
- We build from scratch: the execution-outcome value function for *repair* (not competition
  codegen), structural-uncertainty-aware node scoring, and the frozen-1.5–3B proposer interface
  (every repo here is hard-wired to a hosted-API LLM).

---

## Candidate table

| Name | Repo URL | License (verified) | Stars | Last push | What it gives us | Verdict |
|---|---|---|---|---|---|---|
| REx | https://github.com/haotang1995/REx | MIT (`.license.spdx_id`) | 6 | 2024-12-07 | 45-line arm-acquiring bandit / Thompson-sampling refinement scheduler + greedy/BFS/fixed-width baselines behind one 24-line `_Domain` interface | **ADOPT** |
| mini-swe-agent | https://github.com/SWE-agent/mini-swe-agent | MIT | 6,695 | 2026-08-17 | Minimal agent loop (`agents/default.py`, 8 KB), `environments/{local,docker,singularity}.py`, swebench batch runners | **ADOPT** |
| SWE-smith | https://github.com/SWE-bench/SWE-smith | MIT | 748 | 2026-08-17 | `bug_gen/procedural/python/*` mutation operators + `harness/{valid,eval,grading,repair}.py`: inject bugs into any repo, keep only those that break ≥1 test | **ADOPT** |
| moatless-tools | https://github.com/aorwall/moatless-tools | MIT | 642 | 2025-09-01 | Maintained successor to moatless-tree-search: `flow/search_tree.py`, `selector/`, `node.py`, `runtime/local.py` (no k8s), `testing/python/*_parser.py` | **PORT-PIECES** |
| moatless-tree-search (SWE-Search) | https://github.com/aorwall/moatless-tree-search | Apache-2.0 | 142 | 2025-06-06 | Reference MCTS impl for SWE-bench: `search_tree.py` (853 L), `selector/selector.py` UCT + 12 bonus/penalty terms, `value_function/coding.py` | **PORT-PIECES** |
| Code-AI-Tree-Search (PG-TD) | https://github.com/shunzh/Code-AI-Tree-Search | MIT | 118 | 2024-07-17 | Token-level P-UCT MCTS where reward = **actual test pass-rate** (`eval/compute_reward.py`), with an optional learned `value_model` head | **PORT-PIECES** |
| rStar-Math (branch `rStar-math`) | https://github.com/microsoft/rStar/tree/rStar-math | MIT (LICENSE on that branch) | 1,425 (repo) | 2025-09-12 (main) | `rstar_deepthink/agents/{mcts,beam_search,tree}.py`, `nodes/mcts_node.py`, `tools/python_tool.py`; PPM trained from MCTS Q-values | **PORT-PIECES** |
| SWE-bench (harness) | https://github.com/SWE-bench/SWE-bench | MIT | 5,689 | 2026-08-18 | `harness/{grading.py,log_parsers/,run_evaluation.py}` — the canonical F2P/P2P resolution logic and per-framework test-log parsers | **PORT-PIECES** |
| CodeTree | https://github.com/SalesforceAIResearch/CodeTree | Apache-2.0 | 38 | 2026-06-02 | Agent-guided tree search (`strategy.py`, `bfs.py`, `dfs_real.py`) with a real `executors/py_executor.py` (subprocess + timeout) | **PORT-PIECES** |
| SWE-Gym | https://github.com/SWE-Gym/SWE-Gym | Apache-2.0 | 723 | 2025-07-29 | 2.4K executable Python repo tasks + the *verifier* recipe (train a reranker on rollout outcomes) | **REFERENCE-ONLY** (data/recipe) |
| LATS | https://github.com/lapisrocks/LanguageAgentTreeSearch | MIT | 852 | 2024-07-30 | `programming/mcts.py` + `executors/` — MCTS over agent trajectories; reward blends self-reflection with generated tests | **REFERENCE-ONLY** (stale, LLM-judge reward) |
| ReST-MCTS\* | https://github.com/THUDM/ReST-MCTS | **No LICENSE file** (`.license` null; root listing has none) | 711 | 2025-01-20 | Process-reward-guided tree search + self-training loop | **REFERENCE-ONLY** (unlicensed) |
| AlphaCodium | https://github.com/Codium-ai/AlphaCodium | **AGPL-3.0** | 3,965 | 2024-11-25 | Test-anchored iterative flow engineering (generate tests → iterate on public/AI tests) | **REFERENCE-ONLY** (copyleft) |
| SWE-RL | https://github.com/facebookresearch/swe-rl | **NOASSERTION → CC-BY-NC-4.0** (LICENSE text reads "Attribution-NonCommercial 4.0 International") | 719 | 2025-03-16 | RL on software evolution with a similarity reward | **REFERENCE-ONLY** (non-commercial) |

### Also verified, rejected quickly

| Name | Repo | License (verified) | Stars / push | Why not |
|---|---|---|---|---|
| auto-code-rover | AutoCodeRoverSG/auto-code-rover | **NOASSERTION → Sonar Source-Available License v1.0** (LICENSE text) | 3,099 / 2025-04-24 | Not open source; explicit non-compete clause. Reference only. |
| PSearch (ASE 2026) | iSEngLab/Psearch | **No LICENSE file** | 2 / 2026-07-09 | Most on-topic 2026 work (`swe_mcts.py`, `condefects_mcts.py`, `vul4j_mcts.py`) but unlicensed → reference only. |
| SWE-agent (full) | SWE-agent/SWE-agent | MIT | 20,109 / 2026-08-17 | Superset of mini-swe-agent; too much surface for a harness. Use mini. |
| OpenHands | OpenHands/OpenHands | MIT | 84,822 / 2026-08-22 | Full IDE-grade agent platform; wrong altitude for a research spike. |
| Agentless | OpenAutoCoder/Agentless | MIT | 2,101 / 2024-12-22 | No tree search; localize→repair→rerank pipeline. Its reproduction-test reranking is a useful idea only. |
| Reflexion | noahshinn/reflexion | MIT | 3,239 / 2025-01-14 | Linear self-refine, no tree, no learned value. Baseline only. |
| Tree of Thoughts | princeton-nlp/tree-of-thought-llm | MIT | 6,052 / 2025-01-16 | Toy-task BFS/DFS with LLM self-evaluation — no execution. |
| RepairAgent | sola-st/RepairAgent | NOASSERTION → **MIT** (LICENSE text) | 106 / 2026-05-31 | Java/Defects4J, single-trajectory FSM agent, no search. |
| verl / rllm / SkyRL | verl-project/verl, rllm-org/rllm, NovaSky-AI/SkyRL | Apache-2.0 / Apache-2.0 / Apache-2.0 | 23,082 / 5,796 / 2,187, all pushed 2026-08-2x | Correct home for unit-test-reward GRPO later; far too heavy for one 16 GB 5080. Cloud-rental phase only. |
| augment-swebench-agent | augmentcode/augment-swebench-agent | NOASSERTION → **MIT** (LICENSE text) | 882 / 2026-08-21 | Ensembling + majority vote, no tree search or value model. |

---

## Top picks — detail

### 1. REx — `haotang1995/REx` (MIT) — the search policy we should start with

**Architecture.** `acr/run.py` picks a domain and a scheduler. A scheduler is a bare function over a
`_Domain` (`acr/domains/base.py`, 24 lines) exposing `reset(problem_id) -> [(idx, name, heuristic)]`,
`step(idx) -> (reward, done, new_actions)`, `get_metrics()`. `acr/scheduler/rex.py` (45 lines) keeps a
flat list of "arms" — every program ever produced is an arm whose refinement can be sampled — each
with Beta(α, β) parameters seeded from the program's heuristic reward (test pass rate:
`alpha = smoothing + C*h`, `beta = smoothing + C*(1-h)`). Each step: `argmax` of a Beta draw, run the
refine action, execute tests, update α/β with the observed reward, append the children as new arms.
Baselines `greedy.py`, `bfs.py`, `fw.py` share the same interface — free ablations.

**Reuse.** `acr/scheduler/*.py` (all four, ~170 lines total) and `acr/domains/base.py` verbatim.
`acr/domains/apps/main.py` as the template for a `MutationRepairDomain`.

**Change.** Write our own domain: `reset` = load the mutated repo + failing test set; `step` = ask the
frozen proposer for a patch, apply, run pytest, return `reward = fraction of target tests passing`.
Replace `acr/utils/llm` (OpenAI/tiktoken) with a local vLLM/llama.cpp client.

**Gotchas.** conda `environment.yml` only, no pyproject. Caching via `dill` pickles keyed on prompt
strings — fine locally, brittle across model swaps. Reward must be in [0,1] or the Beta update breaks.
No tree structure is materialized; if we want provenance edges (pillar 1) we must add the parent
pointer ourselves — it's 3 lines.

### 2. mini-swe-agent — `SWE-agent/mini-swe-agent` (MIT) — the execution harness

**Architecture.** Three orthogonal protocols: `Model`, `Environment`, `Agent`. `agents/default.py`
(8,043 bytes) is a `while True: step()` loop with a Jinja2 `StrictUndefined` template for the system and
instance messages, hard limits on steps/cost/wall-time/consecutive format errors, and a `Submitted`
exception raised by the *environment* when the model emits a sentinel line. `environments/local.py`
runs bash via `subprocess` with timeout and returns `{output, returncode, exception_info}`;
`docker.py` and `singularity.py` are drop-in replacements. `run/benchmarks/swebench.py` is the batch
runner.

**Reuse.** `environments/local.py` + `environments/docker.py` (our sandbox), `agents/default.py` as the
per-node rollout body, `exceptions.py`, `utils/serialize.py`.

**Change.** The agent currently *is* the whole trajectory; for crucible each search node needs one
bounded rollout, so we call `step()` from our own tree loop rather than `run()`. Swap the `Model`
implementation for a local frozen 1.5–3B (the protocol is tiny — one `query()` + `format_message()`).

**Gotchas.** Hard dependency on `litellm >= 1.75.5` even for local models (they do route local via
litellm's `openai/` provider, so a vLLM OpenAI-compatible server works). Python ≥3.10. `LocalEnvironment`
runs commands **on the host with `os.environ`** — for mutation-repair on real repos we must use the
Docker env or run inside our own container. Default timeout is 30 s.

### 3. SWE-smith — `SWE-bench/SWE-smith` (MIT) — the task generator

**Architecture.** `swesmith/bug_gen/` has three bug sources: `procedural/` (AST-level modifiers),
`llm/` (model-written bugs), `combine/` (multi-hunk). `procedural/base.py` defines
`ProceduralModifier` with `can_change(code_entity)` gated on tags + a cyclomatic-complexity window and
`modify(code_entity) -> BugRewrite`; `procedural/python/` implements `classes.py`, `control_flow.py`,
`operations.py`, `remove.py`. `swesmith/harness/valid.py` then runs the repo's suite and **keeps only
mutants that break ≥1 test**; `eval.py` + `grading.py` score candidate patches; `profiles/` maps a repo
to its Docker image and test command.

**Reuse.** `bug_gen/procedural/*` (our mutation operators, seeded/deterministic via
`random.Random(seed)`), `harness/valid.py` (the "is this a real bug" filter), `harness/grading.py`,
`profiles/` as the shape of our repo registry.

**Change.** We only need the Python modifiers; drop the 8 other language dirs. We want the mutant's
*ground-truth* fix (the inverse patch), which SWE-smith records as `BugRewrite` — good enough.

**Gotchas.** Docker is mandatory and the authors state Linux-only (Ubuntu 22.04 tested) — fine on this
box. Python ≥3.10. The published 52k-instance dataset pulls per-repo Docker images from
`SWE-smith-envs` (250+ images) — do **not** pull all of them; build 3–5 repos ourselves.

### 4. moatless-tools / moatless-tree-search — `aorwall/*` (MIT / Apache-2.0) — the tree machinery

**Architecture.** `moatless-tree-search` is the SWE-Search paper code: `search_tree.py` implements the
literal MCTS loop (`_select` → `_expand` → `_simulate` → `_backpropagate`, plus
`get_best_trajectory`/`is_finished`); `node.py` (881 lines) is the node with `visits`, `reward`,
`file_context`, `observation`; `selector/selector.py` has `uct_score()` decomposed into
`calculate_exploitation/exploration/depth_bonus/diversity_bonus/duplicate_child_penalty/…` plus
`BestFirstSelector`, `SoftmaxSelector`, `LLMSelector`; `runtime/runtime.py` defines the 42-line
`RuntimeEnvironment` ABC (`run_tests(patch, test_files) -> [TestResult]`).
`moatless-tools` is the maintained MIT successor: same ideas under `flow/search_tree.py` +
`flow/trajectory_tree.py`, a slimmer `selector/{base,simple}.py`, a **`runtime/local.py`** that runs
swebench tests without Kubernetes, and standalone `testing/python/{pytest,django,sympy,seaborn}_parser.py`.

**Reuse.** `runtime/runtime.py` (`TestResult`/`TestStatus`/`RuntimeEnvironment` — steal verbatim, it is
the right seam), the UCT term decomposition in `selector/selector.py`, the select/expand/backprop
skeleton of `search_tree.py`, and **`moatless/testing/python/*_parser.py`** from moatless-tools (a
genuinely reusable pytest-output → structured-result parser).

**Change.** The value function is the problem. `value_function/base.py` requires a `completion_model`
and prompts an LLM for the reward — i.e. an **LLM judge**, exactly what we're rejecting.
`value_function/coding.py` only contributes deterministic bonuses/penalties for action-level failures
(`FAILURE_VALUES = {"MAJOR": -50, "MINOR": -25}`) plus test outcomes. And moatless-tools'
`value_function/swebench.py` is oracle-contaminated by construction — it reads `expected_spans` from the
benchmark instance and its own docstring says *"intended for evaluation purposes only. Using it outside
of an evaluation scenario may contaminate the results."* We must replace the whole `ValueFunction` with
an execution-outcome model.

**Gotchas.** moatless-tree-search's `runtime/testbed.py` imports `testbeds.sdk` → the separate
`aorwall/moatless-testbeds` (MIT, 14★) which wants a **Kubernetes cluster**; prefer moatless-tools'
`runtime/local.py`. Dependency weight is severe: moatless-tools pins `swebench==3.0.17`,
`llama-index`, `faiss-cpu`, `voyageai`, `redis`, `psycopg2-binary`, `gunicorn`, `sqlalchemy`,
`opentelemetry-*`. moatless-tree-search additionally needs a **Voyage AI key** for its pre-embedded
vector stores. Take files, not the package. Python <3.14, ≥3.10. Pydantic v2 throughout.

### 5. Code-AI-Tree-Search (PG-TD) — `shunzh/Code-AI-Tree-Search` (MIT) — the only real execution-trained value function

**Architecture.** `generate/program_env.py` is a Gym-style env whose state is a token list, action is a
token, and *reward is the test pass rate of the completed program* — `get_reward` → `eval/compute_reward.py`
→ `check_correctness()` which forks a `multiprocessing.Process` with a 10 s global timeout around
`eval/testing_util.py:run_test`. Returns `pass_rate` plus `{compile_error, runtime_error}` fractions.
`dyna_gym/agents/{uct,mcts}.py` (from SuReLI/dyna-gym) provide `uct` / `p_uct` / `var_p_uct` tree
policies. `generate/default_pi.py:APPSHeuristic` is the LLM prior: `get_top_k_predict` supplies the
policy prior, and `get_value(state)` runs an optional **separate `value_model` transformer** that scores
a partial program — trained on real APPS pass rates, which is precisely pillar-3 item 2.

**Reuse.** `eval/compute_reward.py` + the `ProgramEnv` reward/transition contract; `dyna_gym/agents/uct.py`
tree-policy variants; the `use_value` + `new_token_num` shortened-horizon pattern (rollout `k` tokens,
then bootstrap from the value head) — this is the correct shape for our budgeted value function.

**Change.** Everything token-level must become patch-level: our action is a candidate patch, not a token,
so `transition` and `get_top_k_predict` get replaced by "sample N patches from the frozen proposer".
`testing_util.run_test` is APPS-specific (stdin/stdout + `assert` call modes); swap for pytest.

**Gotchas.** Pinned `torch==1.12.1+cu116`, `transformers` unpinned, `gym` (not gymnasium), and `pyext`
— this will **not** install as-is on a 5080 (sm_120); treat as a source of algorithms, not a runnable
package. Fine-tuned GPT-2 1.5B / GPT-Neo 2.7B weights come from a Google Drive link. Last push
2024-07-17. Superseded ergonomics live in the same author's `shunzh/mcts-for-llm`.

---

## Gaps — what we must build

1. **An execution-outcome value function for repair.** Nothing here trains a value model on
   *repair* rollouts. PG-TD's value head is competition-codegen and token-level; ReST-MCTS\*'s PRM is
   unlicensed; SWE-Gym publishes a verifier *recipe* but for full agent trajectories. We build:
   features = (tests passing before/after, F2P delta, traceback class, diff size, files touched,
   provenance/confidence of the retrieved memory that motivated the patch) → predicted probability the
   subtree yields a full fix; trained on our own logged rollouts. Small MLP or LoRA head, not a 7B judge.
2. **Structural-uncertainty-aware node scoring.** No repo scores a node by *how well-verified its
   supporting evidence is*. The UCT term decomposition in moatless' `selector.py` is the right place to
   graft a `calculate_provenance_confidence()` term, but the term itself is ours.
3. **A patch-level search node with provenance edges.** REx has arms but no tree; moatless has a tree but
   nodes carry file-context, not memory citations. We need `Node = (patch, parent, test_report,
   memory_citations, verification_status)`.
4. **A frozen-small-proposer interface.** Every candidate assumes a hosted API (litellm/OpenAI/Anthropic)
   or a `transformers` model loaded per-process. We need one adapter serving a 1.5–3B model to N
   concurrent tree rollouts on one 16 GB card (vLLM OpenAI-compatible server + prefix caching), plus a
   deterministic seed policy so search is reproducible.
5. **A cheap local test runner with a per-node budget.** SWE-bench's harness is one Docker build per
   instance; that is far too slow for hundreds of tree nodes. We need warm containers with the repo
   pre-installed, `pytest --last-failed`-style target selection, and a hard per-node wall-clock budget —
   assembled from mini-swe-agent's `DockerEnvironment` + moatless' pytest parser + SWE-bench's
   grading semantics.
6. **Budget accounting for the baseline comparison.** The thesis needs "same verification budget"
   enforced identically for the small+memory system and the bigger frozen baseline. No repo exposes a
   verification-budget meter; REx's `max_steps` is the closest and it counts LLM calls, not test runs.

---

## Evidence log

Environment: `gh auth status` → logged in as `bricelancasterwcp-sudo`, token active.
Note: `gh search` is capped at 30 req/min (hit HTTP 403 twice, backed off); `gh api repos/...`
uses the 5,000/hr core limit (`gh api rate_limit` → core 5000).
Note on `gh search repos`: a single quoted argument is sent as an exact phrase, so 4+ word queries
returned zero rows; effective form was `gh search repos <terms> --match name,description`.

**Searches run**
- `gh search repos "language agent tree search" --sort stars --limit 15` → lapisrocks/LanguageAgentTreeSearch (top), weill-labs/lats, 12 forks.
- `gh search repos "rStar-Math"` → ai-in-pm/rStar-Math, hpitta26/LLM-Inference-Techniques, pierre-roth/rstar-arc (no official Microsoft hit under that name).
- `gh search repos "AlphaCodium"` → Codium-ai/AlphaCodium + 10 forks.
- `gh search repos "tree of thoughts"` → princeton-nlp/tree-of-thought-llm 6,052★, kyegomez/tree-of-thoughts.
- `gh search repos "ReST-MCTS"` → THUDM/ReST-MCTS 711★.
- `gh search repos "CodeTree"` → SalesforceAIResearch/CodeTree.
- `gh search repos "SWE-agent" / "mini-swe-agent" / "OpenHands" / "Agentless" / "SWE-smith" / "RepairAgent"` → the canonical repos plus SWE-Gym/SWE-Gym, augmentcode/augment-swebench-agent, china-qijizhifeng/agentic-harness-engineering.
- `gh search repos "verl"` → verl-project/verl 23,082★, rllm-org/rllm, TIGER-AI-Lab/verl-tool, langfengQ/verl-agent.
- `gh search repos "SWE-bench"` → SWE-bench/SWE-bench, AutoCodeRoverSG/auto-code-rover, openai/SWELancer-Benchmark.
- `gh search repos "reflexion language agents"` → noahshinn/reflexion 3,239★.
- `gh search repos mcts code --match name,description --sort stars --limit 15` → facebookresearch/LaMCTS, YuxiXie/MCTS-DPO, cavaunpeu/mcts-llm-codegen, jokieleung/I-MCTS.
- `gh search repos "tree search" code --match name,description --sort stars --limit 15` → **shunzh/Code-AI-Tree-Search**, kohjingyu/search-agents, OSU-NLP-Group/llm-planning-eval, nicoladainese96/code-world-models.
- `gh search repos "process reward model" --match name,description` → RyanLiu112/Awesome-Process-Reward-Models, mukhal/ThinkPRM, CJReinforce/PURE, WindyLee0822/Process_Q_Model (all math/VLM, none code-execution).
- `gh search repos SWE-RL --match name,description` → facebookresearch/swe-rl, zhenyuhe00/SWE-Swiss.
- `gh search repos "unit test" reward RL code --match description` → THUDM/CodeRM-NT ("Reward Model for Code RL **without** Unit Tests" — opposite of what we want).
- `gh search repos "moatless tree search"` → aorwall/moatless-tree-search.
- `gh search code "class MCTS code repair language:python" --limit 15` → huyuelin/agentless_MCTS, **iSEngLab/Psearch** (`swe_mcts.py`, `vul4j_mcts.py`), StevenZHB/DVO, DhanHaidar/auto_code_repair.
- `gh search code "uct_score pytest reward language:python" --limit 15` → **aorwall/moatless-tree-search:tests/test_selector.py `test_uct_score`**, AutoGPT `prompt_strategies/lats.py`, NumberChiffre/mcts-llm.

**License / metadata verification** (`gh api repos/OWNER/REPO --jq '{license:.license.spdx_id,stars:.stargazers_count,pushed:.pushed_at,desc:.description}'`)
```
lapisrocks/LanguageAgentTreeSearch   MIT          852     2024-07-30T20:29:27Z
SalesforceAIResearch/CodeTree        Apache-2.0    38     2026-06-02T18:55:14Z
THUDM/ReST-MCTS                      null         711     2025-01-20T06:01:03Z
Codium-ai/AlphaCodium                AGPL-3.0    3965     2024-11-25T13:09:34Z
princeton-nlp/tree-of-thought-llm    MIT         6052     2025-01-16T20:02:00Z
SWE-agent/SWE-agent                  MIT        20109     2026-08-17T22:33:19Z
SWE-agent/mini-swe-agent             MIT         6695     2026-08-17T22:33:23Z
OpenHands/OpenHands                  MIT        84822     2026-08-22T17:02:25Z
OpenHands/software-agent-sdk         MIT         1018     2026-08-23T01:53:19Z
OpenAutoCoder/Agentless              MIT         2101     2024-12-22T19:29:31Z
SWE-bench/SWE-smith                  MIT          748     2026-08-17T20:11:51Z
SWE-bench/SWE-bench                  MIT         5689     2026-08-18T23:53:40Z
sola-st/RepairAgent                  NOASSERTION  106     2026-05-31T19:27:01Z
verl-project/verl                    Apache-2.0 23082     2026-08-22T09:24:25Z
rllm-org/rllm                        Apache-2.0  5796     2026-08-23T02:36:13Z
NovaSky-AI/SkyRL                     Apache-2.0  2187     2026-08-22T01:02:39Z
SWE-Gym/SWE-Gym                      Apache-2.0   723     2025-07-29T17:38:25Z
aorwall/moatless-tree-search         Apache-2.0   142     2025-06-06T13:16:21Z
aorwall/moatless-tools               MIT          642     2025-09-01T04:30:25Z
aorwall/moatless-testbeds            MIT           14     2025-04-09T03:52:02Z
a-antoniades/swe-search              MIT           14     2024-11-05T06:38:11Z
noahshinn/reflexion                  MIT         3239     2025-01-14T07:54:02Z
AutoCodeRoverSG/auto-code-rover      NOASSERTION 3099     2025-04-24T07:58:24Z
augmentcode/augment-swebench-agent   NOASSERTION  882     2026-08-21T06:53:01Z
shunzh/Code-AI-Tree-Search           MIT          118     2024-07-17T05:50:41Z
cavaunpeu/mcts-llm-codegen           null          17     2023-12-01T14:08:25Z
nicoladainese96/code-world-models    Apache-2.0    20     2025-02-21T14:07:51Z
kohjingyu/search-agents              MIT          223     2024-07-25T01:54:49Z
OSU-NLP-Group/llm-planning-eval      null          54     2024-02-23T14:12:15Z
haotang1995/REx                      MIT            6     2024-12-07T05:30:30Z
microsoft/rStar                      MIT         1425     2025-09-12T08:26:02Z
zhentingqi/rStar                     MIT          972     2025-01-23T22:02:48Z
facebookresearch/swe-rl              NOASSERTION  719     2025-03-16T21:31:36Z
iSEngLab/Psearch                     null           2     2026-07-09T06:32:55Z
0xWJ/code-judge                      null          24     2025-10-10T06:38:15Z
```

**null / NOASSERTION resolution** (fetched the actual LICENSE)
- `gh api repos/THUDM/ReST-MCTS/contents --jq '.[].name'` → `.DS_Store CoT MCTS PRM README.md ToT assets data eval_vm.py evaluate.py figures models requirements_*.txt self_train tasks utils` — **no LICENSE file**. `gh api repos/THUDM/ReST-MCTS/license` → HTTP 404. ⇒ unlicensed, NOT adoptable.
- `gh api repos/sola-st/RepairAgent/contents/LICENSE --jq .content | base64 -d | head -20` → `MIT License / Copyright (c) 2024 ISLEM BOUZENIA @ SOLA-ST`. ⇒ **MIT**.
- `gh api repos/AutoCodeRoverSG/auto-code-rover/license ... | base64 -d | head -12` → `SONAR Source-Available License v1.0 ... "Competing" means marketing a product or service as a substitute for the functionality or value of SonarQube`. ⇒ **not open source**, reference-only.
- `gh api repos/augmentcode/augment-swebench-agent/license ... | base64 -d | head -12` → `MIT License / Copyright (c) 2025 Augment Code`. ⇒ **MIT**.
- `gh api repos/facebookresearch/swe-rl/license ... | base64 -d | head -5` → `Attribution-NonCommercial 4.0 International`. ⇒ **CC-BY-NC-4.0**, NOT adoptable.
- `gh api repos/iSEngLab/Psearch/contents --jq '.[].name'` → no LICENSE among 30 entries. ⇒ unlicensed.
- `gh api "repos/microsoft/rStar/contents/LICENSE?ref=rStar-math" | base64 -d | head -5` → `MIT License / Copyright (c) Microsoft Corporation.` ⇒ **MIT on the rStar-math branch**.
- `head -3 /…/scratchpad/mts/LICENSE` (cloned moatless-tree-search) → `Apache License Version 2.0`.

**Structure / code reads**
- Cloned to scratchpad: `moatless-tree-search` (22 MB, 162 .py), `Code-AI-Tree-Search` (4.1 MB), `REx` (20 MB), `CodeTree` (27 MB).
- `wc -l mts/moatless/search_tree.py mts/moatless/node.py` → 853 / 881.
- `grep -n "def " mts/moatless/search_tree.py` → `run_search, _select, _expand, _simulate, _backpropagate, get_best_trajectory, is_finished`.
- `grep -n "def " mts/moatless/selector/selector.py` → `uct_score, calculate_exploitation, calculate_exploration, calculate_depth_bonus, calculate_diversity_bonus, calculate_duplicate_child_penalty, …`; classes `BestFirstSelector, SoftmaxSelector, LLMSelector`.
- `head -50 mts/moatless/value_function/base.py` → `class ValueFunction(BaseModel)` with required `completion_model: CompletionModel` ⇒ **LLM-judge reward**.
- `head -70 mts/moatless/value_function/coding.py` → `FAILURE_REWARDS`, `FAILURE_VALUES = {"MAJOR": -50, "MINOR": -25}`, `is_test()`.
- `head -60 mts/moatless/runtime/runtime.py` → `TestStatus`, `TestResult`, `RuntimeEnvironment.run_tests(patch, test_files)`, `NoEnvironment`.
- `sed -n '1,60p' mts/moatless/runtime/testbed.py` → `from testbeds.sdk import TestbedSDK` ⇒ external k8s testbed service.
- `sed -n '1,60p' mts/pyproject.toml` → py3.10–3.13, litellm, instructor, llama-index, faiss-cpu, voyageai embeddings, `moatless-testbeds`.
- `gh api repos/aorwall/moatless-tools/contents/moatless --jq '.[].name'` → `flow/ selector/ runtime/ environment/ node.py expander/ discriminator/ …`; `flow/` contains `search_tree.py`, `trajectory_tree.py`; `runtime/` contains **`local.py`**; `testing/python/` contains `pytest_parser.py, django_parser.py, sympy_parser.py, seaborn_parser.py, parser_registry.py`.
- `gh api …/moatless-tools/contents/moatless/value_function/swebench.py | base64 -d | head -70` → `SwebenchValueFunction` weights `{identify:0.4, patch:0.3, test:0.3}` and docstring: *"intended for evaluation purposes only… may contaminate the results."*
- `gh api …/moatless-tools/contents/moatless/runtime/local.py | base64 -d | head -50` → `SweBenchLocalEnvironment`, imports `swebench.harness.grading.get_eval_report` and `make_test_spec`.
- `gh api …/moatless-tools/contents/pyproject.toml` → pins `swebench==3.0.17`, faiss-cpu, voyageai, redis, psycopg2-binary, gunicorn, sqlalchemy, opentelemetry.
- `head -60 cats/eval/compute_reward.py` → `check_correctness()` forks a `multiprocessing.Process` with `p.join(timeout=10)`; `compute_reward` returns `pass_rate` + `{compile_error, runtime_error}`.
- `head -70 cats/generate/program_env.py` → `ProgramEnv` docstring: *"Reward: pass rate of the program (on the training set in training, and on the test set in testing)"*.
- `head -45 cats/dyna_gym/agents/uct.py` → `uct_tree_policy`, `p_uct_tree_policy`, `var_p_uct_tree_policy`, `class UCT`.
- `sed -n '177,195p' cats/generate/default_pi.py` → `get_value(state)` → `self.value_model(input_ids).logits.item()`; `use_value = (self.value_model is not None)`.
- `cat cats/requirements.txt` → `torch==1.12.1+cu116`, `gym`, `pyext`, `transformers`.
- `cat rex/acr/scheduler/rex.py` (45 lines) → Beta(α,β) with `alpha = smoothing + constant*heuristic_reward`, `action = max(actions, key=lambda a: rng.beta(a.alpha, a.beta))`, `alpha += reward; beta += 1-reward`.
- `cat rex/acr/domains/base.py` (24 lines) → `_Domain.reset/step/summarize_results/set_seed`.
- `sed -n '1,90p' rex/acr/domains/apps/main.py` → `InitAPPS` / `RefineAPPS` actions, `compute_reward`, `compute_heuristic`, `get_new_actions`.
- `gh api repos/SWE-bench/SWE-smith/contents/swesmith --jq '.[].name'` → `bug_gen build_repo constants.py harness issue_gen profiles train`; `harness/` → `eval.py gather.py grading.py repair.py utils.py valid.py`; `bug_gen/procedural/python/` → `classes.py control_flow.py operations.py remove.py`.
- `gh api …/SWE-smith/contents/swesmith/bug_gen/procedural/base.py | base64 -d | head -60` → `class ProceduralModifier(ABC)` with `can_change()`/`modify() -> BugRewrite`, `random.Random(seed)`.
- SWE-smith README: *"Keep tasks that break 1+ unit tests"*, "requires Docker… developed and tested on Ubuntu 22.04.4 LTS… do *not* plan on supporting Windows or MacOS", license MIT.
- `gh api repos/SWE-agent/mini-swe-agent/contents/src/minisweagent/... ` → `agents/{default,interactive}.py`, `environments/{local,docker,singularity}.py`, `models/litellm_model.py`, `run/benchmarks/{swebench,swebench_single,programbench}.py`; `agents/default.py` size 8,043 bytes; pyproject deps `pyyaml requests jinja2 pydantic>=2 litellm>=1.75.5 tenacity rich typer textual`, `requires-python = ">=3.10"`.
- `gh api repos/SWE-bench/SWE-bench/contents/swebench/harness --jq '.[].name'` → `constants/ docker_utils.py grading.py infra_failure.py log_parsers/ modal_eval/ run_evaluation.py reporting.py`.
- `gh api repos/lapisrocks/LanguageAgentTreeSearch/contents/programming --jq '.[].name'` → `mcts.py dfs.py reflexion.py executors/ generators/ human-eval/ benchmarks/`.
- `gh api repos/SalesforceAIResearch/CodeTree/contents --jq '.[].name'` → `strategy.py bfs.py dfs_real.py reflexion.py executors/ generators/ llm_agent_guide.py`; `executors/py_executor.py` (296 lines) writes `temp_files/temp_test{pid}.py` and runs with a timeout; per-file SPDX header `Apache-2`.
- `gh api "repos/microsoft/rStar/branches" --jq '.[].name'` → includes **`rStar-math`**, `rStar-mutualreasoning`; `contents?ref=rStar-math` → `rstar_deepthink/` with `agents/{mcts.py,beam_search.py,tree.py}`, `nodes/mcts_node.py`, `tools/python_tool.py`. `main` is now rStar2-Agent (`code-judge` + `verl` submodules → `0xWJ/code-judge` unlicensed, `J-shang/verl`).
- `gh api repos/SWE-Gym/SWE-Gym/readme` → "2.4K real tasks from 11 Python repos", "Training Software Engineering Agents **and Verifiers**".

**Web**
- WebSearch "REx refine explore exploit LLM code repair Tang bandit Thompson sampling github" → NeurIPS 2024 paper `arXiv:2405.17503`, official repo **github.com/haotang1995/REx** (confirmed via API above).
- WebSearch "2026 open source MCTS program repair execution reward github" → **CodePilot**, `arXiv:2602.00129` (submitted 2026-01-28), MCTS + execution-feedback reward, 24.67% on SWE-bench Lite with open-weight models — **no public repo found**; tracked as a paper-only reference.
