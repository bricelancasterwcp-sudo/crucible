# crucible

**A completed, pre-registered research program (2026-08) on non-LLM-centric machinery for
code repair, run end-to-end on consumer hardware.** The founding thesis, written before any
instrument existed: *stop putting knowledge in weights, put learning in the loop.* Five
pre-registered gates later, that thesis has measured answers — some of them yes, the
sharpest ones no, and every number below was locked before it was read.

The architecture under test: a small **frozen** proposer wrapped by structured memory
written continuously, reasoning as tree search scored by *executing* tests, a value
function trained on real outcomes, uncertainty derived from provenance — and, as the
second pillar, a non-token **latent execution predictor** meant to eventually replace the
token proposer entirely. Hardware: one RTX 5080 (16 GB) + 29 GB RAM, plus a single
$0.50/h rented RTX 3090 for the final pillar ($2.97 total).

## The five gates

| Gate | Question | Verdict | Headline number |
|---|---|---|---|
| **A** (`prereg-lock-a`) | Does the memory loop work at all? | **PARTIAL** | second-exposure repair +14.0 pp over control (bar 8.8); but the 14B baseline wins overall |
| **Ablations** (pre-declared) | *Which* mechanism carries it? | — | retrieval alone: +13.5 of the +14.0 pp; sleep-LoRA alone: +1.0 pp (noise) — **the store carries it; consolidation bought speed, not accuracy** |
| **B** (`prereg-lock-b`) | Is the store model-agnostic? | **GO** | store-only retrieval lifts the 14B's repeats **0.780 → 0.905** (+12.5 pp, bar 8.1), zero training |
| **C** (`prereg-lock-c`) | Can retrieval *transfer* across units (symptom matching + learned silence, τ = P95 of 292,660 noise pairs)? | **NO-GO** | non-repeat pool +1.2 pp vs a 7.6 pp bar. Matcher precision **99.4%**, reach 35% — transfer fails for want of *material*, not matching |
| **P1** (`prereg-lock-blite`) | Can a 120M **non-token** latent predictor beat a same-size token model at predicting execution outcomes? | **NO-GO** | latent unroll from (code, input) alone: AUROC **0.99981** — real, collapse-free execution intuition. Token control: **0.99998** (paired DeLong z = −4.07). Reading the text beat simulating the program |

Full records: [`docs/findings/GATE-A.md`](docs/findings/GATE-A.md) ·
[`ABLATIONS-A.md`](docs/findings/ABLATIONS-A.md) ·
[`GATE-B.md`](docs/findings/GATE-B.md) · [`GATE-C.md`](docs/findings/GATE-C.md) ·
[`GATE-P1.md`](docs/findings/GATE-P1.md). Lock records: [`docs/LOCK-A.md`](docs/LOCK-A.md)
· [`LOCK-B.md`](docs/LOCK-B.md) · [`LOCK-C.md`](docs/LOCK-C.md) ·
[`LOCK-BLITE.md`](docs/LOCK-BLITE.md).

## What the program established

**1. An explicit episodic store is a model-agnostic, zero-training upgrade on
re-exposure.** Mechanically distilled lessons (verified fix + failing tests, no LLM in
the distillation), retrieved deterministically into the prompt, converted at +13.5 pp on
a 1.5B and +12.5 pp on a 14B ten times its size. Weight consolidation (LoRA "sleep")
added ~1 pp of accuracy but cut wall-clock ~40% — memory belongs in the loop; the weights
mostly bought speed. *The thesis's "learning in the loop" half: measured yes, for
experience you have already had.*

**2. Nothing generalized to strangers, and the failure is precisely localized.** Novel
units never moved — not under family-wide lessons, not under silence, not under
symptom-conditioned cross-unit retrieval, not under consolidation, on either model
(the 14B's novel rate was 0.68 in three separate arms, exactly). Phase-C's diagnostic
pair is the program's crispest negative: when the lexical matcher spoke it was right
99.4% of the time, but it could speak for only 35% of tasks — in a 179-unit corpus,
transferable repair material barely exists. *Generalization kept collapsing back into
the network; the mechanical organs could bookkeep experience but not manufacture
transfer.*

**3. Latent execution intuition works — and lost anyway.** The B-lite predictor (frozen
137M code encoder + 120M trained, LeWM two-term objective, no EMA/stop-grad, autoregressive
latent unroll with **no trace access at inference**) reached test AUROC 0.9998 predicting
run outcomes of unseen functions. No published latent-space execution predictor existed to
inherit (survey: `docs/research/05`). It still lost, decisively, to a same-size token
encoder — because this corpus's outcomes (type-crash dominated, by construction of its
adversarial-input battery) are largely readable off the surface text, which is the token
model's home turf. The revival experiment, if ever run, needs semantically hard outcomes
that must be *executed* to be known.

## The other contribution: the measurement discipline, and what it caught

Every gate was pre-registered with derived (not chosen) thresholds, locked by tag before
any number was read, evaluated once, and never re-run. Chosen constants are marked as
chosen; unmeasured is `None`, never zero; every lens is named. Tests are mutation-tested
(a test that survives its own mutant is a defect). The paper trail includes a corrupt
data file and an abandoned approach, both archived rather than deleted.

Defects this discipline caught **before** any locked number existed — each one would have
silently fabricated a result:

- **Label leakage by input shape** (final review): the latent model originally scored from
  recorded traces, whose emptiness encoded the label. The gate would have been a readout.
  Fixed with the unroll contract: the model predicts; it never reads a trace.
- **Silent replay corruption** (implementer smoke test): a shared scratch dir plus an
  unchecked exit code made every harvest after the first return the first one's result —
  all 1,000 initial samples were byte-identical. Archived as
  `samples.jsonl.replay-corrupt`; corpus rebuilt from fixed code.
- **A saturated, unsatisfiable endpoint bar** (Gate A, E3b: bar computed to 100.4%) —
  reported FAIL-as-written with the unthresholded observation alongside, then amended in
  the *next* pre-registration only.
- A dozen surviving test mutants (paired-SE vs unpaired DeLong, endianness of split
  hashing, cosine denominators, causal-mask alignment, truncation byte semantics, …)
  found by adversarial reviewers and killed before the code measured anything.

## Repo map

- `crucible/` — the instrument: stream building & sandboxed execution, search, memory
  organ (store/distill/retrieve/symmatch), value + conformal uncertainty, sleep/LoRA,
  arms & driver, `latent/` (B-lite: harvest→corpus→model→train→eval).
- `docs/superpowers/specs/` — the four pre-registrations, amendments inline, never
  silently edited. `docs/findings/` — gate records + ops findings. `docs/LOCK-*.md` —
  what each tag froze. `docs/research/01..05` — component surveys with every license
  verified by command. `docs/CARRIED-DEBT.md`, `docs/WITHDRAWN-CLAIMS.md`.
- Reproducibility pins: stream hash, HF model revisions + shard digests, τ = 0.8051 with
  its derivation population, split-hash golden values, seeds, serve flags — in the lock
  records. Run data (`runs/`, `streams/`) is git-ignored by design; the locks carry the
  hashes.

## Status

The program is complete: both founding pillars measured, five verdicts on the board.
There is no active phase. Any revival — corpus expansion for transfer, semantically hard
outcomes for the latent pillar — starts with a new pre-registration, per the standing
rule that made every number here mean something: **the point estimate decides.**
