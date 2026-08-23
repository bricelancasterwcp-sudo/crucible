# Pillar 2 — The Memory Organ: open-source component survey

Survey date: 2026-08-23. All license/star/push facts below were read from the GitHub API
or from a fetched `LICENSE` file during this survey; see the Evidence log. Nothing here is
quoted from memory.

## Summary

Typed episodic/semantic/procedural separation is a solved, adoptable problem — MIRIX and MemOS
both ship it, and MemOS's record metadata already carries `confidence`, `source` locators,
`status`, and a versioned `history`. Bi-temporal fact validity with provenance back to raw
episodes is solved by Graphiti (`valid_at`/`invalid_at`/`expired_at`/`episodes`). Consolidation
episodic→weights is **not** solved by anything mature: MemOS's `parametric/lora.py` is an explicit
placeholder that writes the literal bytes `b"Placeholder"`, and its `dream/` consolidation module
is beta and disabled by default. The single closest thing to the crucible thesis is `cls-ledger`
(MIT, 5 stars, 1.5k LOC) — fact cards with supersedence distilled into a LoRA with
provenance-based unlearning — but it is MLX/Apple-only and unreviewed. We adopt schemas, port
the consolidation blueprint, and build the verification coupling ourselves.

## Candidate table

| Name | Repo | License (verified) | Stars | Last push | What it gives us | Verdict |
|---|---|---|---|---|---|---|
| MemOS | github.com/MemTensor/MemOS | Apache-2.0 (api) | 10927 | 2026-08-21 | Richest record metadata found: `confidence`, `source`, `sources[]` locators, `status`, `version`, `history[]`; `dream/` consolidation pipeline | PORT-PIECES |
| Graphiti | github.com/getzep/graphiti | Apache-2.0 (api) | 30204 | 2026-08-21 | Bi-temporal fact edges with validity windows + provenance to episodes | PORT-PIECES |
| MIRIX | github.com/Mirix-AI/MIRIX | Apache-2.0 (api) | 3434 | 2026-08-20 | 6 separate typed memory tables incl. procedural; `skill_experience` w/ credibility + lineage | PORT-PIECES |
| cls-ledger | github.com/caiovicentino/cls-ledger | MIT (api) | 5 | 2026-07-16 | Fact cards → selective LoRA distillation; provenance-based unlearning; LoRA slot fusion | PORT-PIECES |
| cognee | github.com/topoteretes/cognee | Apache-2.0 (api) | 30189 | 2026-08-22 | `DataPoint` with deterministic `identity_fields` ids → idempotent merge across runs | PORT-PIECES |
| A-MEM | github.com/agiresearch/A-mem | MIT (api) | 1152 | 2025-12-12 | Zettelkasten note w/ `evolution_history`, `retrieval_count`, link generation | REFERENCE-ONLY |
| mem0 | github.com/mem0ai/mem0 | Apache-2.0 (api) | 63847 | 2026-08-23 | Extract → compare → ADD/UPDATE/DELETE conflict-resolution loop | REFERENCE-ONLY |
| Letta (letta-code) | github.com/letta-ai/letta-code | Apache-2.0 (api + LICENSE) | 3089 | 2026-08-23 | Memory blocks, sleep-time agents. **TypeScript**; Python V1 is archived | REFERENCE-ONLY |
| HippoRAG 2 | github.com/OSU-NLP-Group/HippoRAG | MIT (api) | 3957 | 2026-07-29 | Personalized PageRank over a KG for associative recall | REFERENCE-ONLY |
| Voyager | github.com/MineDojo/Voyager | MIT (api) | 7154 | 2024-04-03 | Skill library: add skill only after critic verifies success | REFERENCE-ONLY |
| Agent Workflow Memory | github.com/zorazrw/agent-workflow-memory | Apache-2.0 (api) | 460 | 2025-12-22 | Induce reusable workflows from successful trajectories only | REFERENCE-ONLY |
| Stitch | github.com/mlb2251/stitch | MIT (api) | 99 | 2025-09-10 | Fast compression-based abstraction learning (Rust + py bindings) | PORT-PIECES |
| DreamCoder | github.com/ellisk42/ec | MIT (api) | 555 | 2023-07-01 | Wake/sleep library learning — the original consolidation-of-programs loop | REFERENCE-ONLY |
| prov | github.com/trungdong/prov | MIT (api) | 138 | 2026-08-21 | W3C PROV-DM in Python; PROV-O/JSON/XML export | ADOPT (optional) |
| pyactr | github.com/jakdot/pyactr | **GPL-3.0** (api) | 183 | 2026-08-19 | ACT-R base-level activation/decay equations | REFERENCE-ONLY (copyleft) |

Also verified but not shortlisted: `anthropics/skills` (171063 stars, **no LICENSE file in repo
root** → reference-only by our rule; the `spec/` SKILL.md frontmatter format is the reusable part);
`gabegrand/lilo` (MIT stated in README only, API reports `null`); `SoarGroup/Soar` (BSD, verified
from `LICENSE.md`); `cmekik/pyClarion` (MIT, 68 stars); `langchain-ai/langmem` (MIT, 1621);
`joonspk-research/generative_agents` (Apache-2.0, 21970, recency×importance×relevance scoring);
`zhongwanjun/MemoryBank-SiliconFriend` (MIT, but last push 2023-05-24 — stale);
`agiresearch/AIOS` (**LICENSE file is 1 byte / empty** → treat as unlicensed).
Searching "evidence ledger" / "falsification" surfaced almost entirely 0-star vibe repos
(`crodorg/falsify` MIT 0 stars, `Treibs/kp-build` MIT 0 stars) — nothing adoptable.

## Top picks — detail

### 1. MemOS (`MemTensor/MemOS`) — port the record schema

Data model (`src/memos/memories/textual/item.py`):
- `SourceMessage`: `type` (chat/doc/web/file/system), `role`, `content` (minimal reproducible
  snippet), `chat_time`, `message_id`, `doc_path`, `file_info` — i.e. a locator that points
  back at the origin, not just a string.
- `TextualMemoryMetadata`: `confidence: float`, `source: Literal[...]`, `status: activated |
  resolving | archived | deleted`, `version: int`, `history: list[ArchivedTextualMemory]`,
  `evolve_to: list[str]`, `tags`, `updated_at`, `visibility`.
- `ArchivedTextualMemory`: `update_type: conflict | duplicate | extract | unrelated | feedback`,
  `memory_form: state | event`, `archived_memory_id`, `timespec`.
- `TreeNodeTextualMemoryMetadata` adds `sources: list[SourceMessage]`, `memory_type` lifecycle,
  `usage: list[str]`, `background`.

Reuse: the metadata Pydantic models, near-verbatim. `update_type` and `memory_form` are exactly
the ledger-vs-lesson and event-vs-state distinctions we want, and `history` gives supersedence
for free. Change: `confidence` must become a function of verification (tests passed) rather than
an LLM's self-report, and we add `last_verified_at` + `falsified_by`, which do not exist.

Gotchas — important: **`memories/parametric/lora.py` is a stub.** Its header says "This file
currently serves as a placeholder… Please do not use this as a functional module yet", and
`dump()` writes `b"Placeholder"`. `ParametricMemoryItem` is `{id, memory: Any, metadata: dict}`.
So MemOS's headline "parametric memory" gives us nothing executable. The `dream/` module is
real code but is beta, disabled by default (`MEMOS_ENABLED_PLUGINS=dream`), LLM-driven, and
produces "insights" and a "diary" — it consolidates text into more text, not into weights.
Also a large repo with graph DBs, vector DBs, schedulers, and an API server; we want the
schema files, not the platform.

### 2. Graphiti (`getzep/graphiti`) — port the temporal fact model

Data model (`graphiti_core/edges.py`):
- `Edge`: `uuid`, `group_id` (partition), `source_node_uuid`, `target_node_uuid`, `created_at`.
- `EntityEdge(Edge)`: `name`, `fact: str`, `fact_embedding`, `episodes: list[str]` (**provenance
  — the raw episodes that produced this fact**), `expired_at` (when the system learned it was
  no longer true), `valid_at` / `invalid_at` (when it was true in the world), `reference_time`,
  `attributes: dict`.
- Separate `EpisodicEdge`, `CommunityEdge`, `HasEpisodeEdge`, `NextEpisodeEdge` types.

This is a genuine bi-temporal model with an explicit ledger (episodes) / lesson (facts) split,
and it is the best answer in the survey to "when did we learn this stopped being true".

Reuse: the four-timestamp pattern and `episodes: list[str]` backlink. Change: there is **no
confidence field** anywhere on the edge, and invalidation is decided by an LLM judging
contradiction between facts — for crucible, falsification must be a re-run of the test, so we
replace the invalidation trigger entirely while keeping the fields.

Gotchas: `neo4j>=5.26.0` and `openai>=1.91.0` are **unconditional** dependencies in
`pyproject.toml`, plus `posthog` telemetry. Embedded options exist but are awkward: the Kuzu
extra is marked "the upstream Kuzu project is unmaintained; this extra will be removed", leaving
`falkordblite` (py≥3.12) as the only no-server path. Ingestion is LLM-call-heavy per episode.

### 3. MIRIX (`Mirix-AI/MIRIX`) — port the typed table split

Data model (`mirix/orm/`), one SQLAlchemy table per memory type — the cleanest separation found:
- `episodic_memory`: `occurred_at`, `actor`, `event_type` (user_message/inference/…), `summary`,
  `details`, `last_modify {timestamp, operation}`, `filter_tags`.
- `semantic_memory`: `name`, `summary`, `details`, **`source`** (origin reference), `created_at`.
- `procedural_memory`: `name`, `entry_type` (workflow/guide/script), `description`,
  `instructions`, `triggers: list`, `examples: list`, `version` (semver).
- `skill_experience`: `session_id` ("Provenance: the session this experience was distilled
  from"), `experience_type: worth_learning | worth_avoiding`, `importance: float[0,1]`,
  **`credibility: float[0,1]`** ("how well-grounded in a direct signal"), `evidence` (JSON
  `{quote, signal_type}`), `status: pending | consumed | superseded`, `consumed_by`
  (the evolution-run id that consumed it), `influenced_skill_ids` (lineage).

`skill_experience` is the single best schema match in this survey: it is literally a lesson
record with provenance, two separate confidence axes, an evidence pointer, a consumption
lifecycle, and forward lineage into the skills it changed.

Reuse: the table split and `skill_experience` verbatim. Change: `procedural_memory` stores
`instructions` as a **string of prose**, not an executable program, and carries no pass-rate —
for crucible a skill must be a runnable artifact with a scored execution history, so we replace
that column set. Gotchas: Postgres-native (BM25 full-text, `init.sql`), Docker-compose deployment
with a dashboard, and the quick-start is wired to Gemini for both LLM and embeddings.

### 4. cls-ledger (`caiovicentino/cls-ledger`) — port the consolidation blueprint

The only repo found that implements verified-selective episodic→parametric consolidation.
1466 LOC total, readable end to end.

Data model (`clsledger/ledger.py`): `Card{entity, attribute, value, day, episode_id, usage,
superseded_by}`, with `key = norm(entity).norm(attribute)` and
`card_id = key@day.episode_id`. `Ledger.add()` supersedes the previous card for the same key,
carries `usage` forward ("usage follows the fact, not the copy"), and handles out-of-order
older statements by superseding the *new* card instead. `current_cards()` = non-superseded only.

`system.py` states the design directly: hippocampus = episodic store, cortex = LoRA written by
distillation of *selected* cards, ledger = maps every training example back to cards and
episodes. Three stated properties: (1) state-level not episode-level — only current values are
consolidated; (2) churn-gated selection — facts observed to change often stay episodic
("freshness in weights is a losing game"); (3) reversibility — drop an entity's cards and
re-distill. `slots.py` implements one LoRA per entity fused by rank concatenation, so deleting a
slot is a re-fusion rather than a retrain (O(fusion), not O(training)).

Reuse: the selection policy, the card→training-example→ledger mapping, and the slot-fusion idea.
Change: `slots.py` is built on **MLX** (`import mlx.core as mx`) — Apple Silicon only — so on the
RTX 5080 it must be reimplemented against PEFT/torch. Its notion of "verified" is supersedence
and churn, not test execution; we substitute test outcomes. Gotchas: 5 stars, no reviews, no
CI visible, research artifact tied to its own `agentlife` benchmark harness; treat as a design
document with working reference code, not a dependency.

### 5. cognee (`topoteretes/cognee`) — one idea worth taking

`DataPoint` (`cognee/infrastructure/engine/models/DataPoint.py`) documents a trap we would
otherwise hit: a random default UUID means a node "has NO stable identity, so such a node never
deduplicates/merges across runs or mentions and cannot be looked up by recomputing its id".
Declaring `identity_fields` in metadata derives the id deterministically, namespaced by class
name. For a memory that is written after every episode and must merge repeated observations of
the same claim, content-addressed record ids are the right default. Take the idea; the rest of
cognee is a large ECL platform with its own DB abstractions.

## Relation to witness / sensorium

Read read-only: `/home/brice/workspace/witness/ORIGIN.md` and
`/home/brice/workspace/sensorium/README.md`. These two local projects already occupy the two
halves of pillar 2 that the open-source field does worst, and they were designed on exactly the
stance this survey was asked to enforce. Witness is the licensing discipline: an append-only
ledger of what was actually retrieved (files opened, commands run, timestamps, hashes) against
which every sentence must be licensed, with the model allowed to *request* a read while the
witness records whether the read happened — and it explicitly rejects "cloud 'memory layer' that
embeds old chats and retrieves a vibe", which is the same rejection driving crucible's design
stance. Sensorium is the episodic recorder for precisely crucible's domain: it wraps one Python
run, streams every call/return/exception with captured values into a SQLite trace, and answers
only as a deterministic function of that trace, marking truncations and labelling a rerun that
turned out to be a different execution. Since a crucible episode *is* a Python test run against a
mutation-injected bug, sensorium is the natural episodic-ledger instrument — it already produces
the raw trace that a lesson would be derived from, with the honesty properties (never answer from
data you do not have) that the semantic store's `confidence` field needs in order to mean
anything. The gap between them and pillar 2 is the layer above: neither derives typed semantic
claims, neither maintains a procedural skill library, and neither consolidates anything into
weights. Practically: sensorium supplies the ledger, witness supplies the claim→evidence
licensing rule, and crucible builds lesson-derivation, skill scoring, and consolidation on top.
Neither was modified.

## Gaps — what we build from scratch

The prior was: provenance-aware semantic store, falsification checks, verified-only consolidation
trigger. Partly refuted, partly confirmed.

- **Provenance-aware semantic store — REFUTED (port, don't build).** MemOS's
  `TextualMemoryMetadata` (confidence, source, sources[] locators, status, version, history[])
  plus Graphiti's four-timestamp `EntityEdge` with `episodes: list[str]` together cover this.
  Missing fields we add: `last_verified_at`, `falsified_by`, and a `verification_method`.
- **Falsification checks — CONFIRMED, build.** Every system found decides invalidation by asking
  an LLM whether two text facts contradict (Graphiti, mem0, MemOS `update_type=conflict`). None
  re-executes anything to test a claim. Crucible's falsification is a test run, and the scheduler
  that decides *when* a claim is stale enough to re-verify does not exist anywhere in this survey.
- **Verified-only consolidation trigger — CONFIRMED, build, with a blueprint.** cls-ledger is the
  only implementation and it is MLX-only, 5 stars, and gates on supersedence/churn rather than
  verification. MemOS's LoRA path is a placeholder. We port cls-ledger's selection policy and
  ledger→training-example mapping onto PEFT/torch and swap the gate to "episode's tests passed".
- **Procedural memory as scored executable skills — CONFIRMED, build.** MIRIX stores skills as
  prose `instructions` with no success rate; Voyager stores executable code but retrieves it by
  OpenAI embedding similarity and keeps no per-skill score history; AWM induces workflows from
  successful trajectories but as text. A skill library where each entry is a runnable program with
  an execution/pass-rate record does not exist off the shelf.
- **Local-model-only operation — build/adapt.** Graphiti hard-depends on `openai`, MIRIX's
  quick-start is Gemini-wired, Voyager hardcodes `ChatOpenAI`/`OpenAIEmbeddings`, A-MEM defaults
  to `gpt-4o-mini`. Every adopted component needs its LLM/embedding calls rerouted to the frozen
  local proposer or removed.
- **Decay/activation math — reference, then implement.** pyactr has the ACT-R base-level equations
  but is **GPL-3.0**, so we read it and implement independently; Soar (BSD) and pyClarion (MIT)
  are safe to read for the declarative/procedural split.

## Evidence log

Commands run (terse; outputs are the recorded facts above).

```
ls /home/brice/workspace/crucible/docs/research/          -> empty dir exists
head /home/brice/workspace/witness/ORIGIN.md              -> read (ledger/licensing design)
head /home/brice/workspace/sensorium/README.md            -> read (PEP 669 SQLite trace recorder)
gh api rate_limit --jq .resources                         -> search 30/min shared; core 5000/hr
```

`gh api repos/OWNER/REPO --jq '{license:.license.spdx_id,stars,pushed,desc}'` results:

| repo | license (spdx) | stars | pushed |
|---|---|---|---|
| letta-ai/letta | Apache-2.0 | 24356 | 2026-08-16 |
| letta-ai/letta-code | Apache-2.0 | 3089 | 2026-08-23 |
| mem0ai/mem0 | Apache-2.0 | 63847 | 2026-08-23 |
| getzep/graphiti | Apache-2.0 | 30204 | 2026-08-21 |
| getzep/zep | Apache-2.0 | 4858 | 2026-08-19 |
| topoteretes/cognee | Apache-2.0 | 30189 | 2026-08-22 |
| OSU-NLP-Group/HippoRAG | MIT | 3957 | 2026-07-29 |
| agiresearch/A-mem | MIT | 1152 | 2025-12-12 |
| WujiangXu/A-mem | MIT | 945 | 2026-03-05 |
| zhongwanjun/MemoryBank-SiliconFriend | MIT | 445 | 2023-05-24 |
| joonspk-research/generative_agents | Apache-2.0 | 21970 | 2024-08-05 |
| langchain-ai/langmem | MIT | 1621 | 2026-08-11 |
| BAI-LAB/MemoryOS | Apache-2.0 | 1558 | 2026-07-07 |
| MineDojo/Voyager | MIT | 7154 | 2024-04-03 |
| ellisk42/ec | MIT | 555 | 2023-07-01 |
| mlb2251/stitch | MIT | 99 | 2025-09-10 |
| gabegrand/lilo | **null** | 100 | 2024-08-29 |
| zorazrw/agent-workflow-memory | Apache-2.0 | 460 | 2025-12-22 |
| LeapLabTHU/ExpeL | Apache-2.0 | 236 | 2024-12-20 |
| jakdot/pyactr | **GPL-3.0** | 183 | 2026-08-19 |
| trungdong/prov | MIT | 138 | 2026-08-21 |
| MemTensor/MemOS | Apache-2.0 | 10927 | 2026-08-21 |
| Mirix-AI/MIRIX | Apache-2.0 | 3434 | 2026-08-20 |
| anthropics/skills | **null** | 171063 | 2026-08-21 |
| cmekik/pyClarion | MIT | 68 | 2026-08-06 |
| SoarGroup/Soar | **NOASSERTION** | 431 | 2026-07-20 |
| memodb-io/memobase | Apache-2.0 | 2853 | 2026-01-11 |
| agiresearch/AIOS | **NOASSERTION** | 6272 | 2026-07-20 |
| caiovicentino/cls-ledger | MIT | 5 | 2026-07-16 |
| qpiai/Proced_mem_bench | Apache-2.0 | 6 | 2026-03-18 |
| dog-last/E-mem | Apache-2.0 | 28 | 2026-05-03 |
| TiMEM-AI/TiMEM | NOASSERTION | 174 | 2026-05-26 |
| Kyros-494/kyros-ai | Apache-2.0 | 96 | 2026-08-11 |
| NirDiamant/Agent_Memory_Techniques | Apache-2.0 | 926 | 2026-08-15 |
| crodorg/falsify | MIT | 0 | 2026-07-10 |
| Treibs/kp-build | MIT | 0 | 2026-07-11 |

License resolution for null / NOASSERTION (per the rule, fetched the actual file):

```
gh api repos/SoarGroup/Soar/contents/LICENSE.md | base64 -d
  -> "Soar is distributed under the BSD License ... Copyright 2000-2012 Regents of
     University of Michigan"                                    => BSD, adoptable
gh api repos/agiresearch/AIOS/contents/LICENSE --jq '{size,encoding}'
  -> {"enc":"base64","size":1}; curl raw -> empty              => empty file, NOT adoptable
gh api repos/anthropics/skills/contents --jq '.[].name'
  -> .claude-plugin, .gitignore, README.md, THIRD_PARTY_NOTICES.md, skills, spec, template
     (no LICENSE file; README has no license section)          => unlicensed, reference-only
gh api repos/gabegrand/lilo/readme | base64 -d | grep -i licen
  -> "# License / MIT License Copyright (c) 2023 Gabriel Grand" => MIT (README-only assertion)
gh api repos/letta-ai/letta/contents/LICENSE | base64 -d       => Apache License 2.0
curl raw letta-ai/letta-code/main/LICENSE                      => Apache License 2.0
```

Searches (`gh search repos <terms> --sort stars --limit N`) — note: quoting multi-word queries
makes gh send them as an exact phrase and return nothing; unquoted short terms work:

```
"agent memory episodic semantic"          -> 2 hits, both minor
memory consolidation agent                -> surfaced MIRIX, cls-ledger's neighbours, ~12 vibe repos
procedural memory LLM                     -> surfaced Proced_mem_bench, awesome-llm-agent-skills-papers
episodic memory LLM agent                 -> surfaced cls-ledger, E-mem, Agent_Memory_Techniques
falsification claim verification agent    -> crodorg/falsify (0*), Xconmax245/Popper, Treibs/kp-build (0*)
evidence ledger provenance AI             -> all 0-2 star; nothing adoptable
```

Source files read (via `gh api repos/.../contents/PATH | base64 -d`):

```
getzep/graphiti      graphiti_core/edges.py        -> EntityEdge{name,fact,fact_embedding,
                                                      episodes[],expired_at,valid_at,invalid_at,
                                                      reference_time,attributes}; no confidence field
getzep/graphiti      pyproject.toml                -> unconditional deps: neo4j>=5.26, openai>=1.91,
                                                      posthog; kuzu extra marked unmaintained
getzep/graphiti      graphiti_core/driver/         -> neo4j, falkordb, kuzu, neptune drivers
MemTensor/MemOS      memories/textual/item.py      -> SourceMessage; TextualMemoryMetadata w/
                                                      confidence, source, status, version, history[];
                                                      ArchivedTextualMemory.update_type
MemTensor/MemOS      memories/parametric/item.py   -> ParametricMemoryItem{id, memory:Any, metadata:dict}
MemTensor/MemOS      memories/parametric/lora.py   -> "TODO ... placeholder ... do not use";
                                                      dump() writes b"Placeholder" (file size 1397)
MemTensor/MemOS      dream/README.md               -> beta, disabled by default, LLM insights+diary
Mirix-AI/MIRIX       orm/{episodic,semantic,procedural}_memory.py, orm/skill_experience.py
                                                   -> typed tables; skill_experience has session_id,
                                                      importance, credibility, evidence, status,
                                                      consumed_by, influenced_skill_ids
agiresearch/A-mem    agentic_memory/memory_system.py -> MemoryNote{content,keywords,links,context,
                                                      category,tags,timestamp,last_accessed,
                                                      retrieval_count,evolution_history}; ChromaDB+BM25
MineDojo/Voyager     voyager/agents/skill.py       -> SkillManager: skills.json + Chroma w/
                                                      OpenAIEmbeddings; hardcoded ChatOpenAI
topoteretes/cognee   .../models/DataPoint.py       -> identity_fields => deterministic node ids
letta-ai/letta       README.md                     -> "This repository now serves as a landing page";
                                                      code moved to letta-ai/letta-code (TypeScript);
                                                      Python V1 preserved on `archive` branch, unsupported
trungdong/prov       README.md                     -> W3C PROV-DM, PROV-O/XML/JSON/JSONLD, MIT, py3
```

Cloned (small, MIT) for full read:

```
git clone --depth 1 https://github.com/caiovicentino/cls-ledger.git
  clsledger/ledger.py  (135 LOC) -> Card{entity,attribute,value,day,episode_id,usage,superseded_by}
  clsledger/system.py  (632 LOC) -> hippocampus/cortex/ledger; state-level + churn-gated selection
  clsledger/slots.py   (265 LOC) -> one LoRA per entity, fused by rank concatenation; MLX-only
  clsledger/{replay,unlearn_eval,unlearn_slots,analyze_h1}.py -> 434 LOC; total 1466 LOC
```
