# Third-party artifacts in crucible

Every adopted, vendored, or ported artifact. License is what the command returned on the date shown, not what anyone remembered.

| Artifact | Source | License | Verified by | Date | What we take |
|---|---|---|---|---|---|
| cosmic-ray | github.com/sixty-north/cosmic-ray | MIT | `gh api repos/sixty-north/cosmic-ray --jq .license.spdx_id` | 2026-08-23 | library dependency: `mutate_code`, operators, parso AST helpers |
| parso | pypi.org/project/parso | MIT | PyPI `.info.license` | 2026-08-23 | library dependency (cosmic-ray's parser) |
| pytest-timeout | pypi.org/project/pytest-timeout | MIT | PyPI classifier | 2026-08-23 | per-test wall-clock kill in the sandbox |
| EvalPlus HumanEval+ v0.1.10 | github.com/evalplus/humanevalplus_release | Apache-2.0 | HF API `cardData.license` on `evalplus/humanevalplus` | 2026-08-23 | seed units (data) |
| EvalPlus MBPP+ v0.2.0 | github.com/evalplus/mbppplus_release | Apache-2.0 | HF API `cardData.license` on `evalplus/mbppplus` | 2026-08-23 | seed units (data) |
| MutPy `StatementDeletion` (idea) | github.com/mutpy/mutpy | Apache-2.0 | LICENSE file fetched (API said NOASSERTION) | 2026-08-23 | operator *design* reimplemented in `crucible/stream/sdl.py` on cosmic-ray's ABC; no code copied |
| mini-swe-agent `LocalEnvironment` (pattern) | github.com/SWE-agent/mini-swe-agent | MIT | `gh api` | 2026-08-23 | subprocess-isolation pattern in `crucible/sandbox/exec.py`; no code copied |
