# crucible (working name)

A research spike on a non-LLM-centric AI architecture: a small **frozen** proposer wrapped by
structured memory written continuously, reasoning as tree search scored by *executing* tests,
a value function trained on real outcomes, and uncertainty derived from provenance +
verification status. The thesis in one line: **stop putting knowledge in weights, put learning
in the loop.**

Phase A tests the loop (memory + verify-by-execution search) with a borrowed small LLM as the
proposer, against a bigger frozen model at equal verification budget. Phase B (only on GO)
replaces the token proposer with a latent "intuition" predictor.

- Pre-registration + design of record: `docs/superpowers/specs/2026-08-23-crucible-phase-a-prereg.md`
- Open-component surveys (licenses verified by command): `docs/research/01..05-*.md`
- Third-party artifacts and their licenses: `THIRD_PARTY.md` (maintained at merge)
- Debt and withdrawn claims: `docs/CARRIED-DEBT.md`, `docs/WITHDRAWN-CLAIMS.md`

Target hardware: one RTX 5080 (16 GB) + 29 GB RAM; consumer-class by design.
