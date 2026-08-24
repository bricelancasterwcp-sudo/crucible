"""Task 11 -- the model-serving harness: S1 launch flags + the ``wait_ready`` poller.

Two load-bearing properties, both traceable to S1 (docs/findings/S1-serving.md):

* ``SERVE`` carries the S1-measured launch flags verbatim -- the 9B's FP8 repo, its
  ``--enforce-eager`` (CUDA-graph capture OOMs beside 12 GiB of FP8 weights on the 16 GiB
  card), and every gpu-memory-utilization / max-model-len value. Drift here silently
  changes what the arms actually serve.
* ``scripts/serve_model.sh`` disables the FlashInfer sampler JIT
  (``VLLM_USE_FLASHINFER_SAMPLER=0``) -- the S1 gotcha: that JIT needs ninja+nvcc, absent on
  this box, so engine init crashes without it. Step 5's mutation removes this and this test
  must go red.

``wait_ready`` is polling logic only -- exercised against a FAKE in-process ``http.server``
(no GPU, no vLLM), so this whole module runs UNWRAPPED. The pid-death check must DOMINATE
server readiness: a dead launcher returns False even while the fake server answers 200 (that
is the property the pid-death mutation breaks).
"""
import os
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from crucible.run.serving import SERVE, ServeSpec, serve_command, wait_ready


class _ModelsHandler(BaseHTTPRequestHandler):
    """Fake vLLM: answers ``GET /v1/models`` with 200 + a minimal model list, else 404."""

    def do_GET(self):  # noqa: N802 (http.server API)
        if self.path == "/v1/models":
            body = b'{"data": [{"id": "Qwen/Qwen3.5-2B"}]}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *_a):  # silence stderr noise during the test
        pass


@pytest.fixture
def fake_server():
    srv = HTTPServer(("127.0.0.1", 0), _ModelsHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    host, port = srv.server_address
    try:
        yield f"http://{host}:{port}"
    finally:
        srv.shutdown()
        srv.server_close()
        t.join(timeout=2)


def _closed_port_url() -> str:
    """A loopback URL that refuses connections (bound then released -> nothing listening)."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return f"http://127.0.0.1:{port}"


def test_wait_ready_true_when_server_up(fake_server):
    # pid = this process (guaranteed alive); the fake server answers /v1/models 200.
    assert wait_ready(fake_server, os.getpid(), timeout_s=5.0, interval_s=0.05) is True


def test_wait_ready_false_when_pid_dead(fake_server, monkeypatch):
    # Server is UP, but the launcher pid is dead -> pid-death must DOMINATE readiness.
    # (Mutation: drop the pid check -> the live server makes this return True -> test fails.)
    def _dead(_pid, _sig):
        raise ProcessLookupError

    monkeypatch.setattr(os, "kill", _dead)
    assert wait_ready(fake_server, 999_999, timeout_s=5.0, interval_s=0.05) is False


def test_wait_ready_false_on_timeout_no_server():
    # Nothing listening; pid alive -> must time out to False (and not hang).
    assert wait_ready(_closed_port_url(), os.getpid(), timeout_s=0.3, interval_s=0.05) is False


def test_serve_2b_carries_s1_flags():
    s = SERVE["Qwen/Qwen3.5-2B"]
    assert isinstance(s, ServeSpec)
    assert s.served_name == "Qwen/Qwen3.5-2B"
    assert s.hf_id == "Qwen/Qwen3.5-2B"
    assert s.extra_args == [
        "--max-model-len", "8192",
        "--gpu-memory-utilization", "0.6",
        "--enable-lora", "--max-lora-rank", "32",
    ]
    assert s.port == 8010


def test_serve_9b_carries_s1_fp8_flags():
    s = SERVE["Qwen/Qwen3.5-9B"]
    assert s.served_name == "Qwen/Qwen3.5-9B"
    assert s.hf_id == "lovedheart/Qwen3.5-9B-FP8"  # FP8, Blackwell-native (S1 §7)
    assert s.extra_args == [
        "--max-model-len", "4096",
        "--gpu-memory-utilization", "0.90",
        "--enforce-eager",
    ]
    assert s.port == 8010


def test_serve_1_5b_proposer_carries_a2_flags():
    # A2 (docs/findings/S2-ceiling-pilot.md §7-§8): Qwen2.5-Coder-1.5B-Instruct, chat-served
    # by the client -- but the vllm-serve surface is the same OpenAI server as the 2B, so it
    # gets the 2B's proven flag shape (2B-parity defaults, to be confirmed live at first serve).
    s = SERVE["Qwen/Qwen2.5-Coder-1.5B-Instruct"]
    assert isinstance(s, ServeSpec)
    assert s.served_name == "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    assert s.hf_id == "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    assert s.extra_args == [
        "--max-model-len", "8192",
        "--gpu-memory-utilization", "0.6",
        "--enable-lora", "--max-lora-rank", "32",
    ]
    assert s.port == 8010


def test_serve_command_prefixes_vllm_and_names_the_model():
    argv = serve_command("Qwen/Qwen3.5-9B")
    assert argv[:3] == ["vllm", "serve", "lovedheart/Qwen3.5-9B-FP8"]
    assert "--served-model-name" in argv
    assert argv[argv.index("--served-model-name") + 1] == "Qwen/Qwen3.5-9B"
    assert argv[-2:] == ["--port", "8010"]
    # every S1 flag rides through, in order
    assert "--enforce-eager" in argv


def test_serve_command_for_a2_proposer():
    argv = serve_command("Qwen/Qwen2.5-Coder-1.5B-Instruct")
    assert argv[:3] == ["vllm", "serve", "Qwen/Qwen2.5-Coder-1.5B-Instruct"]
    assert "--served-model-name" in argv
    name_idx = argv.index("--served-model-name") + 1
    assert argv[name_idx] == "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    assert argv[-2:] == ["--port", "8010"]
    assert "--enable-lora" in argv


def test_serve_script_disables_flashinfer_sampler():
    # The S1 gotcha (Step 5 mutation check): the JIT sampler needs ninja+nvcc, absent here.
    # Key on the FUNCTIONAL launch (the exec line applies the env var to the served process),
    # not merely a mention -- a comment/echo copy must not mask its removal from the launch.
    script = Path(__file__).resolve().parents[2] / "scripts" / "serve_model.sh"
    lines = script.read_text(encoding="utf-8").splitlines()
    launch = [ln for ln in lines if ln.strip().startswith("exec ")]
    assert launch, "serve_model.sh must exec the server"
    assert all("VLLM_USE_FLASHINFER_SAMPLER=0" in ln for ln in launch)


def test_serve_script_is_executable():
    script = Path(__file__).resolve().parents[2] / "scripts" / "serve_model.sh"
    assert os.access(script, os.X_OK)
