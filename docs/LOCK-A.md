# Phase-A lock record (`prereg-lock-a`)

Everything pre-reg §12 requires the lock tag to sit on, in one place. The tag
`prereg-lock-a` points at the commit introducing this file; nothing in pre-reg §§2–7
changes after it (amendments A1–A5 predate the lock and are footnoted in place).

| §12 item | Value |
|---|---|
| Stream hash | `1158e92f40ad7ebb184b3a79a1472d2660bb087c498e461dda0370aa30ec7cf8` (dir `streams/1158e92f40ad`) |
| C (eligible classes) | 200 |
| N_nov | 50 |
| Rung | `stack2` (hardening ladder rung (i); fixed by re-pilot, findings `S2.5-stack2.md`) |
| p0 (ceiling pilot) | **0.267** (8/30, A_noMem, K=8, seed 0 — `runs/pilot-stack2-15bc/A_noMem/`) |
| Accept gate (was Δ_min) | `ACCEPT_MAX_DROP = 1` on a `min(12, |solved|)` holdout slice [amendment A4] |
| Sleep trigger | verified-episode threshold **N = 16** [amendment A2] |
| Proposer (small arms) | `Qwen/Qwen2.5-Coder-1.5B-Instruct`, HF revision `2e1fd397ee46e1388853d2af2c993145b0f1098a`, weights sha256/16 `c1b9b30e90795051` (model.safetensors) |
| Baseline (B arm) | `Qwen/Qwen2.5-Coder-14B-Instruct-AWQ`, HF revision `eb3172f06a6d6b3a15f08947b0668d782e4d2d2c`, weights sha256/16 `aacd6553a9ca787e` / `735a941a5b54c0ea` / `1c0174225b114921` (shards 1–3) |
| Server kind | vLLM **0.27.1** (OpenAI-compatible, runtime LoRA hot-swap; SERVE table `crucible/run/serving.py` is the flag authority) |
| Families map | ARITH 120 · BOOL 65 · CMP 41 · FLOW 2 · SDL 217 · UNARY 5 (450 tasks: 200 phase-1, 250 phase-2) |
| Sampler | temp 0.7, top_p 0.95, max_new_tokens 2048 [A1], thinking OFF, seed = hash(run_id, task_key, node_id, k) |
| K / abstention enrichment | K = 8 executions/task; enrichment ≥ 2× (pre-reg §1, unchanged) |
| Run protocol | gating runs launch ONLY via `scripts/run_arm_detached.sh` (fresh-dir guard, pid + `.DONE`); never resumed (R-S4-1); infra kill = clean rerun from zero |

Instrument state at lock: 575 tests green under the R-T2-6 cap; A_full ops smoke
(`docs/findings/S3-smoke.md`) and the 100-task threshold-16 dress rehearsal
(`docs/findings/S4-ops.md` §4) both passed on this code.
