import json, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import pytest
from crucible.proposer.identity import probe, assert_identity, IdentityMismatch

def _serve(routes):
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            body = routes.get(self.path)
            self.send_response(200 if body is not None else 404); self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(json.dumps(body or {}).encode())
        def log_message(self, *a): pass
    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}"

def test_probe_vllm():
    srv, url = _serve({"/v1/models": {"data": [{"id": "Qwen/Qwen3.5-2B"}]}})
    ident = probe(url)
    assert ident.kind == "vllm" and ident.model == "Qwen/Qwen3.5-2B"
    assert assert_identity(url, "Qwen/Qwen3.5-2B").model == "Qwen/Qwen3.5-2B"
    with pytest.raises(IdentityMismatch):
        assert_identity(url, "Qwen/Qwen3.5-9B")
    srv.shutdown()

def test_probe_llamacpp():
    # A real llama.cpp server ALSO serves an OpenAI-compatible /v1/models (ruling R6),
    # so the fake server exposes both: probe must trust /props first, else it misreads
    # llama.cpp as vLLM. Without the /v1/models route here the ordering is untested.
    srv, url = _serve({
        "/props": {"model_path": "/models/Qwen3.5-2B-Q6_K.gguf", "default_generation_settings": {"model": "/models/Qwen3.5-2B-Q6_K.gguf"}},
        "/v1/models": {"object": "list", "data": [{"id": "/models/Qwen3.5-2B-Q6_K.gguf", "object": "model"}]},
    })
    ident = probe(url)
    assert ident.kind == "llamacpp" and ident.model.endswith("Qwen3.5-2B-Q6_K.gguf")
    assert assert_identity(url, "Qwen3.5-2B-Q6_K.gguf").kind == "llamacpp"
    srv.shutdown()

def test_probe_unknown_server_raises():
    srv, url = _serve({})
    with pytest.raises(IdentityMismatch):
        probe(url)
    srv.shutdown()

def test_probe_lists_every_advertised_model_id():
    # vLLM with a runtime LoRA loaded: the BASE is first, the adapter after.
    srv, url = _serve({"/v1/models": {"data": [{"id": "Qwen/Qwen2.5-Coder-1.5B-Instruct"},
                                               {"id": "ad-0123456789abcdef"}]}})
    ident = probe(url)
    assert ident.model == "Qwen/Qwen2.5-Coder-1.5B-Instruct"        # primary = the base
    assert ident.models == ("Qwen/Qwen2.5-Coder-1.5B-Instruct", "ad-0123456789abcdef")
    srv.shutdown()

def test_assert_identity_matches_a_loaded_adapter_not_just_the_first_id():
    # THE S3 fix: an arm (or the sleep slice) serving an accepted adapter asks for it BY NAME.
    # Matching only data[0].id raised IdentityMismatch on the first adapter ever trained --
    # inside maybe_sleep, after the training cost had already been paid.
    srv, url = _serve({"/v1/models": {"data": [{"id": "Qwen/Qwen2.5-Coder-1.5B-Instruct"},
                                               {"id": "ad-0123456789abcdef"}]}})
    assert assert_identity(url, "ad-0123456789abcdef").kind == "vllm"
    assert assert_identity(url, "Qwen/Qwen2.5-Coder-1.5B-Instruct").kind == "vllm"
    with pytest.raises(IdentityMismatch):                            # still strict
        assert_identity(url, "ad-notloadedatall")
    srv.shutdown()
