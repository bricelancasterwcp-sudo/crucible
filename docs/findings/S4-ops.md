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

## 4. Dress rehearsal (100 tasks, threshold 16): PASS

First 100 phase-1 keys in stream order (derived from `streams/1158e92f40ad/manifest.json`,
phase==1), A_full, `--sleep-threshold 16` (the REAL pre-registered cadence — the S3 smoke
used 4), launched through `scripts/run_arm_detached.sh`. `EXIT=0`, wall 1045 s.

| Check | Observed |
|---|---|
| Tasks | 100/100, landing 1.0, infra 0, max charged K=8 |
| Sleeps at threshold 16 | 2, both accepted (slice 8→8, 7→7); episodes 16 then 32 (cumulative) |
| `gpu_s` (S4 fix, live) | **16.218 s and 24.625 s measured** — the field the gate reports GPU-minutes from |
| Refalsify | 48/48 passed, 0 falsified, 0 broken-citation |
| Adapter lineage | 29 tasks pre-adapter → 40 under `ad-8779b09…` → 31 under `ad-f643fdc…`; lens carries both |
| Abstention | rate 0.04 — the calibrated 0.2 gate visibly active (smoke's tiny windows never tripped it) |
| Memory | 100 episodic / 45 semantic (= hidden-pass count) / 0 procedural |

Uncharged orientation numbers (rehearsal ≠ measurement): `succ_phase1 0.45`.

Cost extrapolation for the gate (§8 preflight arithmetic): phase-1 tasks averaged
~10.5 s at K=8 including sleeps. Full protocol = 450 tasks/arm; sleep training grows
cumulatively (~8 s per additional sleep at this LoRA size). Estimates: **A_full ≈ 80–95
min; A_noMem ≈ 70 min** on the 1.5B server. **B (14B-AWQ) is extrapolated from its own
small-N preflight before launch** (per §8) — chat serving on this card is materially
slower per token and the landing probe's 30 tasks are the anchor.

