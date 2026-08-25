# S3 — A_full ops smoke: full organ loop runs end-to-end on the box

**Date:** 2026-08-24
**Slice:** S3 (branch `s3-memory`), ops record for spec
`docs/superpowers/specs/2026-08-24-crucible-s3-memory-design.md` and plan Task 12.
**Verdict:** **PASS — 30/30 phase-1 tasks, 4 sleep cycles (train → holdout gate →
accept → runtime hot-load) completed in one arm-run process beside the live server;
every exit criterion below verified against the records.** This is an OPS smoke of the
instrument, not a capability measurement: sleep-threshold 4 and n=30 are smoke settings,
NOT the pre-registered N=16/full-stream protocol, and no capability claim is made from
its success rate.

## 1. Setup

- Server: `Qwen/Qwen2.5-Coder-1.5B-Instruct` chat-served via vLLM on `:8010`,
  `gpu-memory-utilization 0.45` (the S3 co-residency entry in `crucible/run/serving.py`),
  `VLLM_ALLOW_RUNTIME_LORA_UPDATING=true`.
- Arm: `crucible arm run streams/1158e92f40ad --arm A_full --tasks runs/s3-smoke-keys.txt
  --out runs/s3-smoke4 --sleep-threshold 4`, K=8, seed 0, under the 12G scope cap with
  disk `TMPDIR` (R-T2-6 discipline).
- Tasks: the same 30 phase-1 keys the S2.5 ceiling pilot used (`runs/s3-smoke-keys.txt`).

## 2. Getting it to fit: three OOMs, three distinct causes

A 16 GiB card hosting BOTH the serving engine and the sleep trainer is the whole ops
question of S3, and it failed three different ways before it held. Each fix is committed
on the branch; the sequence is the finding.

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | First sleep dies allocating during training; server held 9.75 GiB | Serve util 0.6 leaves < 5 GiB for training | SERVE entry 0.6 → **0.45** (`4207bd6`) |
| 2 | First sleep dies in the training step at 6.27 GiB; ~0.7 GiB reserved-but-unallocated | TRL default `per_device_train_batch_size=8` puts eight 2048-token activation sets on the card at once (the synthetic trainer smoke's tiny pairs hid this; real episode prompts are 2–6k chars) | **batch 1 + `gradient_accumulation_steps=8`** (same effective batch) + `gradient_checkpointing=True` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (`3e6bb09`) |
| 3 | Sleep 0 trains, accepts, hot-loads — then sleep 1 OOMs at LoRA injection with 5.92 GiB already PyTorch-allocated (clean-room load of the same path measures 2.88 GiB) | `LoraTrainer.train` never freed the base model, so sleep N+1 loaded a SECOND 2.9 GiB copy into the same process | **try/finally** releasing model/peft/trainer refs + `gc` + `empty_cache`; ast test pins the finally, mutation-tested (`686c541`) |

Sizing rule this bought: on this card the resident split that holds is server ≈ 8.1 GiB
(util 0.45; vLLM's actual hold, not the util fraction) + trainer peak ≈ 5.2 GiB
(batch 1 + checkpointing) + desktop ≈ 1 GiB ≈ 14.5/15.47 GiB. util 0.6 or TRL's default
batch each break it on their own.

## 3. The passing run

`EXIT=0`, wall 519 s. Records under `runs/s3-smoke4/A_full/`:

| Criterion | Observed |
|---|---|
| ≥1 honest sleep accept/reject | **4 sleeps, all accepted**; holdout slice measured before AND after each (3→3, 7→7, 10→10, 9→9 under `ACCEPT_MAX_DROP=1`); cumulative episode selection 4→8→12→16 |
| Refalsify tally ≥1 | Ran every sleep: checked 4/8/12/16 (40 total), all passed, 0 falsified, 0 infra, 0 broken-citation |
| Records parse | task (30), sleep (4), exec (30), registry (4) all parse; lens round-trips exactly |
| Lens `adapter_ids` | Full lineage in order: `ad-a11b1589b5930498, ad-363ab9fc0f76d6df, ad-797dc366c1ec6f01, ad-102ec8b9041fe7a7` |
| `value_update_misses == 0` | 0 miss lines in the run log |
| No task charged > K=8 | max `executions_charged` = 8 |
| Abstain sanity | `abstain_rate 0.0` — plausible under the calibrated 0.2 gate with a small early window |
| Memory store | 30 episodic (one per task), **17 semantic = exactly the hidden-pass count** (verified-only distillation held), 0 procedural (by design, unpopulated) |
| Task-record honesty | `infra_error: None ×30`, `tampered: False ×30`, confidence populated ×30; statuses split `believed 12 / verified_visible 18` |

`gpu_s` on sleep records is `None` **by design** (plan Task 10: training happens behind
the `Trainer` seam; unmeasured-is-None, never fake-zero).

Uncharged numbers, for orientation only (smoke ≠ measurement): `succ_overall 0.567`
(17/30 hidden), `landing_rate 1.0`, `infra_rate 0.0`.

## 4. Paths exercised live (not just in tests)

- **Runtime LoRA hot-swap** under `VLLM_ALLOW_RUNTIME_LORA_UPDATING`: 4 adapters loaded
  into the running server; requests routed by `model == adapter_id`.
- **Idempotent adapter load:** adapter ids are content-addressed, so smoke4's sleep 0/1
  reproduced ids already loaded by earlier attempts byte-for-byte; the server's HTTP 400
  on re-POST was taken through the verify-idempotent path (`d847491`) twice.
- **Adapter lineage stamping:** 11 pre-adapter tasks stamped `adapter_id: None`, then
  6/5/6/2 tasks under each successive adapter.
- **Retrieval:** 17/30 tasks carried non-empty `retrieved_ids`.

## 5. Teardown

EngineCore killed by pid (nvidia-smi compute-apps listing, not pkill), VRAM verified back
to ~1 GiB desktop baseline, ollama restarted and answering on :11434.
