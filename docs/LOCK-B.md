# Phase-B lock record — `prereg-lock-b`

**Locked 2026-08-25.** Spec: `docs/superpowers/specs/2026-08-25-crucible-phase-b-prereg.md`
(final pre-lock text at this commit). Instrument branch d6a9482..681f05d (8 commits, final
whole-branch review CLEAN, 5/5 mutation pins killed, full suite green under the 4G scope).
After this tag, spec §5–§7 are immutable.

| Item | Value |
|---|---|
| **Δ_min (derived)** | **0.0807** = 2·√(2·p̄(1−p̄)/C), p̄ = 0.795 (frozen control succ_phase1), C = 200 |
| Frozen control | GATE-A B_search lens, sha256 `bd1147ab17f41e5702905acd6b99a518d25cf1f1bd5dbaed6a72d0ace163c4c4` (n=450, phase-1 0.7950, second 0.7850, novel 0.6800; within-arm Δ −0.0100) — **never re-run** |
| Frozen exploratory comparator | A_mem_nosleep lens, sha256 `e76e7ab19407fa221e92cb341d4a6fd8352c0d1182fff7fe708456606dfc536e` (phase-1 0.4250, second 0.5600, novel 0.4000, abstain 0.087) |
| Stream | `1158e92f40ad7ebb184b3a79a1472d2660bb087c498e461dda0370aa30ec7cf8` (dir `streams/1158e92f40ad`), task set `--tasks all` (450) |
| Arms | `B_mem` (gating), `A_mem_exactonly` (exploratory) |
| B-arm model | `Qwen/Qwen2.5-Coder-14B-Instruct-AWQ`, HF revision `eb3172f06a6d6b3a15f08947b0668d782e4d2d2c`, weights sha256/16 `aacd6553a9ca787e` / `735a941a5b54c0ea` / `1c0174225b114921` (shards 1–3), served under `Qwen/Qwen2.5-Coder-14B-Instruct` |
| Small-arm model | `Qwen/Qwen2.5-Coder-1.5B-Instruct`, HF revision `2e1fd397ee46e1388853d2af2c993145b0f1098a`, weights sha256/16 `c1b9b30e90795051` |
| Serving | vLLM 0.27.1; 14B: `--max-model-len 16384 --gpu-memory-utilization 0.90 --enforce-eager`; 1.5B: `--max-model-len 16384 --gpu-memory-utilization 0.45 --enable-lora --max-lora-rank 32`; port 8010, VLLM_USE_FLASHINFER_SAMPLER=0 |
| Seed / budget | seed 0, k=8, width=4 (both arms — B_search's identity, pinned by test) |

**Serve-window disclosure (stated plainly per the final review):** the frozen control ran
at `--max-model-len 8192`; B_mem runs at 16384. The change is pre-registered (spec §4.4)
as pure KV capacity — memory-augmented refinement prompts exceed 8192−2048, the same
overflow that infra-killed A_full's first gate attempt. It is sampling-neutral: no request
the control made was affected by the window (it completed 450/450 with zero infra at
8192), and the §5 gating endpoint is WITHIN-arm (B_mem second vs B_mem phase-1), so the
window is common to both sides of the gating comparison.

**Pre-lock smoke (2026-08-25):** 16 tasks (8 firsts + their 8 second exposures), EXIT=0,
16/16 measured, 0 infra, 0 context overflows, all 8 seconds carried non-empty
`retrieved_ids`, `adapter_id` None on every record, 0 KV preemptions, 6.1 s/task.
Window verdict: **16384 stands** (no 12288 fallback needed).
