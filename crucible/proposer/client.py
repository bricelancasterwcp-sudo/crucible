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

# The pinned per-completion token budget (spec S3). Amendment A1 (2026-08-23, pre-lock): raised
# 1024 -> 2048 after the S4.7 landing pre-check found the 1024 cap was the dominant landing
# failure -- the full-module-rewrite codec re-emits the module AND the whole visible test
# harness in one block, which routinely overflows 1024 tokens and truncates the completion with
# its fence still open (recorded in docs/findings/S2-ceiling-pilot.md). Named, not inlined, so
# the pin lives in one place and a test can guard it against silent drift.
MAX_NEW_TOKENS = 2048


@runtime_checkable
class Proposer(Protocol):
    """What the search loop needs from any candidate source: a name and ``generate``."""

    model: str

    def generate(
        self, prompt: str, *, n: int, seed: int, max_tokens: int = MAX_NEW_TOKENS, temperature: float = 0.7
    ) -> list[Candidate]: ...


def _mean(xs: list[float]) -> float | None:
    """Arithmetic mean, or ``None`` for an empty sequence (an absent, not zero, signal)."""
    return sum(xs) / len(xs) if xs else None


def _scores_from_toks(toks: list) -> tuple[float | None, float | None]:
    """``(mean_logprob, self_certainty)`` from a list of chosen-token logprobs.

    The per-token value is the logprob of the *chosen* token. ``mean(logprob)`` is the
    length-normalised sequence logprob; ``mean(exp(lp))`` is the average chosen-token
    probability -- a certainty proxy in ``[0, 1]`` because each ``exp(lp) <= 1``. ``None``
    entries (e.g. a leading token with no predecessor) are dropped; an empty list yields
    ``(None, None)`` so the score reads as *unknown* downstream. Shared by both serving
    shapes (raw completions and chat) so the two paths score identically.
    """
    toks = [lp for lp in toks if lp is not None]
    if not toks:
        return None, None
    return _mean(toks), _mean([math.exp(lp) for lp in toks])


def _scores(logprobs: dict | None) -> tuple[float | None, float | None]:
    """Raw ``/v1/completions`` logprobs shape: ``{"token_logprobs": [lp, ...]}``."""
    if not logprobs:
        return None, None
    return _scores_from_toks(logprobs.get("token_logprobs") or [])


def _chat_scores(logprobs: dict | None) -> tuple[float | None, float | None]:
    """Chat ``/v1/chat/completions`` logprobs shape: ``{"content": [{"logprob": lp}, ...]}``.

    Chat completions carry per-token logprobs under ``content`` rather than a flat
    ``token_logprobs`` list. Absent (some servers omit them in chat mode) yields
    ``(None, None)`` -- a Candidate then reads as unknown, which the search already tolerates
    (the scores are not load-bearing until S3's value/uncertainty organs).
    """
    if not logprobs:
        return None, None
    return _scores_from_toks([c.get("logprob") for c in (logprobs.get("content") or [])])


class VLLMProposer:
    """Draws ``Candidate``s from a vLLM OpenAI-compatible endpoint.

    ``chat`` selects the serving surface. A *base* model (e.g. Qwen3.5-2B) is served in raw
    ``/v1/completions`` mode: the prompt is sent verbatim. An *instruct* model
    (Qwen2.5-Coder-1.5B-Instruct, the amended small-arm proposer -- spec §2 amendment A2) must
    be served in ``/v1/chat/completions`` mode so vLLM applies its chat template; serving an
    instruct model raw makes it emit empty completions (~6% here), which the §4.7 landing
    pre-check caught (see docs/findings/S2-ceiling-pilot.md §7). Either surface returns
    codec-decoded ``Candidate``s with the same two logprob scores.
    """

    def __init__(self, base_url: str, model: str, *, chat: bool = False) -> None:
        # Fail loud and early if the server serves a different checkpoint than we expect.
        assert_identity(base_url, model)
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.chat = chat

    def generate(
        self, prompt: str, *, n: int, seed: int, max_tokens: int = MAX_NEW_TOKENS, temperature: float = 0.7
    ) -> list[Candidate]:
        """POST one request for ``n`` samples; decode each choice to a ``Candidate``.

        Routes to chat or raw completions per ``self.chat``; both honour the same pinned
        ``temperature``/``max_tokens`` and request per-token logprobs for the Candidate scores.
        """
        if self.chat:
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "n": n,
                "logprobs": True,
                "top_logprobs": 1,
                "seed": seed,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            body = self._post("/v1/chat/completions", payload)
            return [self._candidate_chat(choice) for choice in body["choices"]]
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
        """One raw choice -> ``Candidate``: codec-decoded ``text`` + the two logprob scores."""
        landed = extract_module(choice.get("text", ""))
        mean_logprob, self_certainty = _scores(choice.get("logprobs"))
        return Candidate(landed.module_src, mean_logprob, self_certainty)

    @staticmethod
    def _candidate_chat(choice: dict) -> Candidate:
        """One chat choice -> ``Candidate``: codec-decoded ``message.content`` + logprob scores.

        A missing/None ``content`` (an instruct model can still return an empty message) decodes
        to a non-landing Candidate, exactly as an empty raw completion does.
        """
        landed = extract_module((choice.get("message") or {}).get("content") or "")
        mean_logprob, self_certainty = _chat_scores(choice.get("logprobs"))
        return Candidate(landed.module_src, mean_logprob, self_certainty)
