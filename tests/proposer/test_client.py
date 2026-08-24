import json, math, threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from crucible.proposer.client import Proposer, VLLMProposer
from crucible.run.types import Candidate

MODEL = "Qwen/Qwen3.5-2B"

# Canned completion: prose then a fenced module. The codec must strip the fence and keep the
# body, so `Candidate.text` is runnable module source -- never the raw ```python wrapper.
_COMPLETION = "let me think...\n```python\ndef add(a, b):\n    return a + b\n```\n"
# The chosen-token logprobs vLLM returns under `logprobs=1`. Both scores derive from these.
_TOKEN_LOGPROBS = [-0.10, -0.20, -0.05, -0.30]
_NO_LOGPROBS_PROMPT = "NOLOGPROBS"  # sentinel: server omits the logprobs block for this prompt


def _serve(served_model=MODEL):
    """A fake vLLM: /v1/models for the identity assert, /v1/completions echoing `n` choices.

    The completions handler reads `n` from the POST body and returns that many choices, so a
    proposer that ignores `n` is observable (the 2-candidate test would see 1). /props 404s so
    the identity probe classifies this as vLLM (it checks llama.cpp's /props first)."""

    class H(BaseHTTPRequestHandler):
        def _json(self, code, body):
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

        def do_GET(self):
            if self.path == "/v1/models":
                self._json(200, {"data": [{"id": served_model}]})
            else:
                self._json(404, {})  # /props -> not llama.cpp

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            self.captured.clear()
            self.captured.update(req)
            self.captured["__path__"] = self.path
            if self.path == "/v1/chat/completions":
                # Chat shape: text under message.content, logprobs under a `content` list.
                choice = {"message": {"role": "assistant", "content": _COMPLETION},
                          "finish_reason": "stop"}
                choice["logprobs"] = {
                    "content": [{"token": "t", "logprob": lp} for lp in _TOKEN_LOGPROBS]
                }
                self._json(200, {"choices": [dict(choice) for _ in range(req.get("n", 1))]})
                return
            choice = {"text": _COMPLETION, "finish_reason": "stop"}
            if req.get("prompt") != _NO_LOGPROBS_PROMPT:
                choice["logprobs"] = {
                    "token_logprobs": list(_TOKEN_LOGPROBS),
                    "top_logprobs": [{"t": lp} for lp in _TOKEN_LOGPROBS],
                }
            self._json(200, {"choices": [dict(choice) for _ in range(req.get("n", 1))]})

        def log_message(self, *a):
            pass

    H.captured = {}
    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}", H


def test_generate_returns_n_codec_extracted_candidates():
    srv, url, _ = _serve()
    try:
        p = VLLMProposer(url, MODEL)
        cands = p.generate("fix the bug", n=2, seed=7)
        assert len(cands) == 2
        for c in cands:
            assert isinstance(c, Candidate)
            assert "```" not in c.text  # fence stripped by the codec
            assert c.text.strip() == "def add(a, b):\n    return a + b"
    finally:
        srv.shutdown()


def test_mean_logprob_is_mean_of_token_logprobs():
    srv, url, _ = _serve()
    try:
        c = VLLMProposer(url, MODEL).generate("x", n=1, seed=1)[0]
        assert c.mean_logprob == pytest.approx(sum(_TOKEN_LOGPROBS) / len(_TOKEN_LOGPROBS))
    finally:
        srv.shutdown()


def test_self_certainty_is_mean_exp_of_token_logprobs_in_unit_range():
    srv, url, _ = _serve()
    try:
        c = VLLMProposer(url, MODEL).generate("x", n=1, seed=1)[0]
        expected = sum(math.exp(lp) for lp in _TOKEN_LOGPROBS) / len(_TOKEN_LOGPROBS)
        assert c.self_certainty == pytest.approx(expected)
        assert 0.0 <= c.self_certainty <= 1.0
    finally:
        srv.shutdown()


def test_scores_are_none_when_logprobs_absent():
    srv, url, _ = _serve()
    try:
        c = VLLMProposer(url, MODEL).generate(_NO_LOGPROBS_PROMPT, n=1, seed=1)[0]
        assert c.mean_logprob is None and c.self_certainty is None
    finally:
        srv.shutdown()


def test_generate_passes_seed_temperature_and_logprobs_to_server():
    srv, url, H = _serve()
    try:
        VLLMProposer(url, MODEL).generate("p", n=3, seed=42, max_tokens=256, temperature=0.4)
        assert H.captured["n"] == 3
        assert H.captured["seed"] == 42
        assert H.captured["temperature"] == 0.4
        assert H.captured["max_tokens"] == 256
        assert H.captured["logprobs"] == 1
    finally:
        srv.shutdown()


def test_model_attribute_and_protocol_conformance():
    srv, url, _ = _serve()
    try:
        p = VLLMProposer(url, MODEL)
        assert p.model == MODEL
        assert isinstance(p, Proposer)  # structural: has generate + model
    finally:
        srv.shutdown()


def test_construction_asserts_served_identity():
    from crucible.proposer.identity import IdentityMismatch

    srv, url, _ = _serve(served_model="Qwen/Qwen3.5-9B")
    try:
        with pytest.raises(IdentityMismatch):
            VLLMProposer(url, MODEL)
    finally:
        srv.shutdown()


def test_default_max_tokens_is_the_pinned_budget():
    """The pinned S3 token budget (amendment A1) reaches the server by default and is guarded
    against silent drift: a bare generate() sends exactly MAX_NEW_TOKENS, whose value is 2048."""
    from crucible.proposer.client import MAX_NEW_TOKENS

    assert MAX_NEW_TOKENS == 2048  # the amended pin; changing it is a pre-registration decision
    srv, url, H = _serve()
    try:
        VLLMProposer(url, MODEL).generate("p", n=1, seed=1)
        assert H.captured["max_tokens"] == MAX_NEW_TOKENS
    finally:
        srv.shutdown()


def test_chat_mode_posts_to_chat_endpoint_with_messages_and_decodes_content():
    """chat=True routes to /v1/chat/completions, sends the prompt as a user message, and decodes
    message.content through the codec (the fix for instruct models emitting empty raw completions)."""
    srv, url, H = _serve()
    try:
        cands = VLLMProposer(url, MODEL, chat=True).generate("fix the bug", n=2, seed=7)
        assert H.captured["__path__"] == "/v1/chat/completions"
        assert H.captured["messages"] == [{"role": "user", "content": "fix the bug"}]
        assert "prompt" not in H.captured  # chat mode must not send a raw prompt
        assert len(cands) == 2
        for c in cands:
            assert isinstance(c, Candidate)
            assert c.text.strip() == "def add(a, b):\n    return a + b"  # fence stripped
    finally:
        srv.shutdown()


def test_chat_mode_scores_come_from_content_logprobs():
    """Chat logprobs live under logprobs.content[].logprob, not a flat token_logprobs list;
    the two Candidate scores must still be computed from them identically to the raw path."""
    srv, url, _ = _serve()
    try:
        c = VLLMProposer(url, MODEL, chat=True).generate("x", n=1, seed=1)[0]
        assert c.mean_logprob == pytest.approx(sum(_TOKEN_LOGPROBS) / len(_TOKEN_LOGPROBS))
        expected = sum(math.exp(lp) for lp in _TOKEN_LOGPROBS) / len(_TOKEN_LOGPROBS)
        assert c.self_certainty == pytest.approx(expected)
    finally:
        srv.shutdown()


def test_chat_mode_empty_content_is_a_nonlanding_candidate():
    """An instruct model can still return an empty message; that must decode to a non-landing
    Candidate (source that does not parse), exactly like an empty raw completion."""
    from crucible.proposer.codec import extract_module

    srv, url, H = _serve()
    try:
        # Point the fake at an empty-content response by monkeypatching the module constant.
        import tests.proposer.test_client as T
        saved = T._COMPLETION
        T._COMPLETION = ""
        try:
            c = VLLMProposer(url, MODEL, chat=True).generate("x", n=1, seed=1)[0]
            assert not extract_module(c.text or "").ok
        finally:
            T._COMPLETION = saved
    finally:
        srv.shutdown()
