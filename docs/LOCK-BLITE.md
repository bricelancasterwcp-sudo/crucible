# Pillar-1 B-lite lock record — `prereg-lock-blite`

**Locked 2026-08-27.** Spec: `docs/superpowers/specs/2026-08-25-crucible-pillar1-blite-prereg.md`
(final pre-lock text incl. the five §12 amendments: determinism denominator, battery
enrichment, replay-fix provenance, remote-training environment, and the LLM-pass
abandonment recorded inside the battery amendment). After this tag, spec §6–§8 are
immutable. Instrument: 24 commits, 255 tests, three whole-branch/fable reviews, every
live-fire defect (dedup, INPUTS parsing, relative-path store nesting, harvest replay)
fixed and mutation-pinned pre-lock; no gating number existed before this tag.

| Item | Value |
|---|---|
| **Corpus** | `runs/blite-corpus` (git-ignored); samples.jsonl sha256 `289e56ebc53bd8a864f3d7ed7a7f6722bf8b1178f4bc4a667fc18455b9768df7`; **5,000 functions / 61,299 samples** (train 49,236 · val 6,000 · test 6,063 by function-level split 4017/488/495) |
| Class balance | binary 0:42,187 / 1:19,112 (max-share 68.8% ≤ 80% ✓); multiclass **TypeError 42,186 · return 19,112 · ZeroDivisionError 1** — disclosed: the battery's type-probes make the exception mass essentially one class |
| Floors (§4/§8) | floor_functions **PASS** (5,000 ≥ 3,000); nondet_rate **0.0** (**PASS**, denominator per §12 over reharvest+battery buckets); stats_sources = [reharvest_stats.json, battery_stats.json] |
| Sample provenance | every sample from the post-replay-fix reharvest (1,301) + deterministic battery (59,998); corrupt first pass archived `samples.jsonl.replay-corrupt`; LLM minority attempt archived `minority_stats.llm-attempt.json` |
| Split | seed 0, sha256(f"{seed}:{fn_id}") big-endian recipe (golden-pinned); assignment hash `59a2372e4a647c9b879c17e0bf60501c` |
| Encoder | `jinaai/jina-embeddings-v2-base-code` revision `516f4baf13dec4ddddda8631e019b5737c8bc250` (frozen; mask-weighted mean-pool, value-pinned) |
| Control | `microsoft/codeexecutor` revision `fcaa2615bd918a68e8c0a478934cfacfe423028e` + fresh binary head |
| Configs | `crucible/latent/config.py` @ this commit is the exhaustive chosen-numbers record (training: LR 3e-4, BATCH 64, MAX_STEPS 20k, EVAL_EVERY 500, PATIENCE 5, SEED 0; control: 2e-5, ≤5 epochs, MAXLEN 512; N_UNROLL_STEPS 8; battery values/cap; caps 24/96/128) |
| Gate method | paired DeLong (structural components, midranks; asymmetric-case and paired-SE pinned); P1 = diff ≥ 2·SE_diff; floors per §6; one-read enforced (out_path + .lock sentinel) |
| **Environment** | training + scoring BOTH arms on RunPod pod `egz3gwbzkhvygd` (RTX 3090 24 GB, SECURE, image `runpod/pytorch:1.1.0-cu1290-torch291-ubuntu2404`), per the §12 amendment; exact library versions echoed into GATE-P1; corpus/harvest/eval local |

**Standing disclosures:** identical-arms degenerate case (diff=0 ∧ SE=0 → 0≥0) passes P1
by the literal rule — practically impossible with distinct trained arms; sensorium-internal
exit-1 crashes count as truncated_rejected (conservative direction); samples.jsonl row
order canonicalized by sorted atomic rewrite.
