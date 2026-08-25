# S4 — pre-gate ops hardening: probe, gpu_s, no-resume ruling, rehearsal

**Date:** 2026-08-24
**Slice:** S4 (branch `s4-ops`) — the minimal set a multi-hour gating run needs, per the
post-S3 recommendation Brice approved verbatim ("do exactly as you recommended").

## 1. Adapter-accumulation probe: risk REFUTED, nothing built

The S3 CARRIED-DEBT feared ~15–25 accepted adapters accumulating in the vLLM server over
a full gating run (the smoke only ever proved 4). Measured live (1.5B entry, util 0.45,
`VLLM_ALLOW_RUNTIME_LORA_UPDATING=true`):

| Step | Observation |
|---|---|
| Baseline after ready | 9360 MiB |
| 30 × `POST /v1/load_lora_adapter` (distinct names, same rank-16 adapter) | **9360 MiB after every load — zero growth**, zero non-200s |
| Completion routed via adapter #30 | HTTP 200 |
| `/v1/models` | 31 ids advertised (base + 30) |
| `POST /v1/unload_lora_adapter` | HTTP 200, adapter removed from the advertised list, VRAM unchanged |

vLLM registers runtime adapters and pages them through pre-allocated LoRA slots on
demand; the util-0.45 budget already contains the slot memory. **Unload-on-supersede was
therefore NOT built** — building against a refuted risk is how instruments grow untested
code. The unload endpoint is verified to exist should a future slice need it.

Lens note: VRAM was sampled with adapters idle plus one activation of the newest; the
S3 smoke separately proved serial activation across 4 successive adapters. Nothing in the
gating protocol activates two adapters concurrently.

## 2. `gpu_s` measured (pre-reg §3 reporting commitment)

`SleepRecord.gpu_s` was `None`-always; pre-reg §3 declares A_full's sleep GPU time an
asymmetry that "is **reported** as GPU-minutes per arm." Now: wall-clock seconds of the
`Trainer.train` call via `time.monotonic` (named lens: co-resident wall time, not
exclusive GPU occupancy; `None` only in pre-S4 records). The module's no-clock pin was
tightened, not dropped: `datetime` banned, `time` admissible for `monotonic` only.
Mutation-tested both ways (commit c0f7ae0).

## 3. Ruling R-S4-1: gating runs are never resumed

An infrastructure kill of a gating run = clean rerun from zero (already sanctioned by the
pre-reg's infrastructure-kill rule; a completed measurement is still never re-rolled).
Rationale: a resumed run would re-warm the value model and calibrator from scratch against
a warm memory/adapter state — a mid-run instrument change nobody pre-registered. This
ruling also makes the parked crash-window self-exemplar leak (S3 final review M-1)
unreachable in any gating run, and removes value/calibrator resume persistence from the
gate's critical path. Mechanical guard: `scripts/run_arm_detached.sh` — the ONLY sanctioned way to launch a
rehearsal or gating run — refuses a non-empty out dir (exit 3, tested live). Boundary
named honestly: `crucible arm run` invoked directly still permits a coherent resume BY
DESIGN (smokes and debugging need it); the guard is mechanical at the launcher layer and
procedural below it, and the gating protocol is "launch through the launcher, nothing
else".

## 4. Dress rehearsal (100 tasks, threshold 16)

(appended after the run)
