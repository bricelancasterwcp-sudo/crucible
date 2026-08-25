# Phase-C lock record — `prereg-lock-c`

**Locked 2026-08-25.** Spec: `docs/superpowers/specs/2026-08-25-crucible-phase-c-prereg.md`
(final pre-lock text at this commit, incl. the §11 self-citation amendment). Instrument
branch ecd49ea..343cf5f (11 commits; per-task reviews with 4 fix rounds, final
whole-branch review CLEAN, 6/6 mutation sweep dead, suite 623+/green under the 4G scope).
After this tag, spec §5–§7 are immutable — τ included.

| Item | Value |
|---|---|
| **Δ_min(pool), derived** | **0.0764** = 2·√(2·p̄(1−p̄)/250), p̄ = 0.7600 (frozen B_mem pool) → C1 bar **0.8364** |
| **Δ_min(repeat guard), derived** | **0.0586** = 2·√(2·0.905·0.095/200) |
| **τ (locked)** | **0.8051** = P95 of the unrelated-pair score distribution (spec §4.4 rule) |
| τ calibration data | the four Phase-A/B memory DBs (gate-a-full, abl-mem-nosleep, abl-mem-exactonly, gate-b2-mem), locked stream unit sources; **n_unrelated = 292,660, n_related = 921** (self-citation pairs excluded per §11 amendment) |
| τ distributions | unrelated p50/p90/p99 = 0.5517 / 0.7432 / 0.8540; related p50/p90/p99 = 0.8750 / 0.9428 / 0.9692; percentiles via `statistics.quantiles(n=100)` (p95 = index 94) |
| **Ranker-sanity gate (§7)** | **PASS** — median related 0.8750 > τ 0.8051. Disclosed: unrelated p99 (0.8540) > τ, so ≈1% of cross-family noise pairs clear the gate by design of the P95 rule |
| Frozen gating control | B_mem (GATE-B), lens sha256 `bc8c9ba41b4f1ea5bdb1714906e8ff4b05bb3eb4e9ab0c616fa23b92d46bde5d`; **pool recomputed from records: 190/250 = 0.7600**; second 0.9050; novel 0.6800; task wall 2459 s — **never re-run** |
| Frozen exploratory comparator | A_mem_nosleep (ABLATIONS-A), lens sha256 `e76e7ab19407fa221e92cb341d4a6fd8352c0d1182fff7fe708456606dfc536e`; pool recomputed 105/250 = 0.4200 |
| Stream | `1158e92f40ad7ebb184b3a79a1472d2660bb087c498e461dda0370aa30ec7cf8`, `--tasks all` (450) |
| Arms | `B_symmem` (gating), `A_symmem` (exploratory) |
| Models / serving | unchanged from LOCK-B verbatim: 14B AWQ rev `eb3172f06a6d…` shards `aacd6553a9ca787e`/`735a941a5b54c0ea`/`1c0174225b114921` at 16384/0.90/eager; 1.5B rev `2e1fd397…` at 16384/0.45/lora; vLLM 0.27.1, port 8010, seed 0, k=8, w=4 |

**Pre-lock smoke (2026-08-25):** 20 tasks (8 first + 8 second + 4 novel), EXIT=0, 20/20
measured, 0 infra, 0 overflows, 0 KV preemptions, `symptom_probes.txt` = 20 (== tasks),
6/8 seconds exact-path blocks (consistent with the control's 78% phase-1 solve rate),
firsts/novels silent under τ with the smoke's near-empty store. 6.3 s/task.
