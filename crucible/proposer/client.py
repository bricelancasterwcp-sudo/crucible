"""``Proposer`` protocol + vLLM adapter: turn model completions into ranked ``Candidate``s.

The search loop draws repair attempts from *a* proposer without caring how they are served;
:class:`Proposer` is the structural contract it depends on (a ``model`` name plus a
``generate``). :class:`VLLMProposer` is the one implementation the spike ships: it talks to a
vLLM OpenAI-compatible ``/v1/completions`` endpoint over stdlib ``urllib`` (no ``requests``
dependency, matching :mod:`crucible.proposer.identity`).

Construction asserts *served identity* (``assert_identity``) so a run can never silently draw
candidates from the wrong checkpoint -- the whole spike hinges on measuring one frozen model,
so a mismatched server must fail loud and early, not at analysis time.

Each choice becomes a :class:`Candidate` carrying two pre-execution confidence signals lifted
from ``logprobs=1``:

- ``mean_logprob`` -- length-normalised: the mean of the choice's ``token_logprobs``.
- ``self_certainty`` -- a cheap prior in ``[0, 1]``: the mean of ``exp(token_logprob)`` over
  tokens, i.e. the average probability the model assigned to the tokens it actually emitted.

Both are ``None`` when the serving path returns no logprobs, so search treats an absent score
as *unknown* rather than as zero (see :class:`crucible.run.types.Candidate`). The completion
text is run through ``extract_module`` first, so ``Candidate.text`` is decoded module source
(the last fenced block, or the raw text on fallback) -- never the `````python`` wrapper.
"""
from __future__ import annotations

import json
import math
import urllib.request
from typing import Protocol, runtime_checkable

from crucible.proposer.codec import extract_module
from crucible.proposer.identity import assert_identity
from crucible.run.types import Candidate

# Generation on a real server is slow; the fake-server tests answer instantly. A generous cap
# keeps a wedged server from hanging a run forever without tripping on legitimate long batches.
_REQUEST_TIMEOUT_S = 600.0


@runtime_checkable
class Proposer(Protocol):
    """What the search loop needs from any candidate source: a name and ``generate``."""

    model: str

    def generate(
        self, prompt: str, *, n: int, seed: int, max_tokens: int = 1024, temperature: float = 0.7
    ) -> list[Candidate]: ...


def _mean(xs: list[float]) -> float | None:
    """Arithmetic mean, or ``None`` for an empty sequence (an absent, not zero, signal)."""
    return sum(xs) / len(xs) if xs else None


def _scores(logprobs: dict | None) -> tuple[float | None, float | None]:
    """``(mean_logprob, self_certainty)`` for one choice's ``logprobs`` block.

    ``token_logprobs`` is the per-token logprob of the *chosen* token. The mean is the
    length-normalised sequence logprob; ``mean(exp(lp))`` is the average chosen-token
    probability -- a certainty proxy that lands in ``[0, 1]`` because each ``exp(lp) <= 1``.
    ``None`` entries (e.g. a leading token with no predecessor) are dropped; an absent or
    empty block yields ``(None, None)`` so the score reads as unknown downstream.
    """
    if not logprobs:
        return None, None
    toks = [lp for lp in (logprobs.get("token_logprobs") or []) if lp is not None]
    if not toks:
        return None, None
    return _mean(toks), _mean([math.exp(lp) for lp in toks])


class VLLMProposer:
    """Draws ``Candidate``s from a vLLM OpenAI-compatible ``/v1/completions`` endpoint."""

    def __init__(self, base_url: str, model: str) -> None:
        # Fail loud and early if the server serves a different checkpoint than we expect.
        assert_identity(base_url, model)
        self.base_url = base_url.rstrip("/")
        self.model = model

    def generate(
        self, prompt: str, *, n: int, seed: int, max_tokens: int = 1024, temperature: float = 0.7
    ) -> list[Candidate]:
        """POST one completion request for ``n`` samples; decode each choice to a ``Candidate``."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "n": n,
            "logprobs": 1,
            "seed": seed,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        body = self._post("/v1/completions", payload)
        return [self._candidate(choice) for choice in body["choices"]]

    def _post(self, path: str, payload: dict) -> dict:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT_S) as r:
            return json.loads(r.read().decode("utf-8"))

    @staticmethod
    def _candidate(choice: dict) -> Candidate:
        """One choice -> ``Candidate``: codec-decoded source + the two logprob scores."""
        landed = extract_module(choice.get("text", ""))
        mean_logprob, self_certainty = _scores(choice.get("logprobs"))
        return Candidate(landed.module_src, mean_logprob, self_certainty)
