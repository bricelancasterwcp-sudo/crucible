"""The model-serving harness: the S1-measured launch flags per served model, plus the
readiness poller the driver waits on after it (re)starts a server between arms.

S2's arms run one model at a time -- the 2B proposer and the 9B baseline do not co-reside in
16 GiB (S1 §7: 2B bf16 ~9.6 GiB at util 0.6; 9B FP8 ~14.2 GiB at util 0.90) -- so the driver
SWAPS models between arms: stop the server, launch the next one, wait for it to come up. This
module owns the two halves the unit tests can pin without a GPU:

* ``SERVE`` -- the exact ``vllm serve`` launch flags for each served model, measured live in
  S1 (docs/findings/S1-serving.md §7). Load-bearing: the 9B is served from the
  ``lovedheart/Qwen3.5-9B-FP8`` repo (bf16 9B ~18 GiB does not fit) with ``--enforce-eager``
  (CUDA-graph capture OOMs beside 12 GiB of FP8 weights on the 16 GiB card) at util 0.90; the
  2B runs bf16 at util 0.6 with runtime-LoRA enabled. The served *name* is stamped as the
  arm's model id and later identity-asserted (:mod:`crucible.proposer.identity`).

* ``wait_ready`` -- poll ``GET {base}/v1/models`` until the server answers 200 (ready), the
  launcher pid dies (crashed -> give up), or the timebox elapses. The pid-death check
  DOMINATES: a dead launcher returns False even if a stale server on the port still answers,
  so the driver never proceeds against the wrong (or a half-dead) server.

The ACTUAL start/stop of a live server is operational (Task 14/16), not here: the tests drive
``wait_ready`` against a fake in-process http.server, and ``scripts/serve_model.sh`` is the
runnable launcher (it disables the FlashInfer sampler JIT -- the S1 ninja+nvcc gotcha -- and
reads its argv from :func:`serve_command` so the flags never drift from this table).
"""
from __future__ import annotations

import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

DEFAULT_PORT = 8010  # 8001 was already bound on this box (S1 §7)


@dataclass(frozen=True)
class ServeSpec:
    """One served model's launch identity: the name the arm asserts, the HF repo vLLM loads,
    the extra ``vllm serve`` flags, and the port. Frozen so a spec cannot drift mid-run."""

    served_name: str
    hf_id: str
    extra_args: list[str]
    port: int = DEFAULT_PORT


SERVE: dict[str, ServeSpec] = {
    # 2B proposer: bf16, runtime-LoRA on (rank 32), 8k context, util 0.6 -> ~9.6 GiB (S1 §7).
    "Qwen/Qwen3.5-2B": ServeSpec(
        "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-2B",
        ["--max-model-len", "8192", "--gpu-memory-utilization", "0.6",
         "--enable-lora", "--max-lora-rank", "32"],
    ),
    # 9B baseline: FP8 (bf16 ~18 GiB does not fit); eager (graph capture OOMs beside the
    # weights); util 0.90, 4k context -> ~14.2 GiB. Served-name asserted as the plain id (S1 §7).
    "Qwen/Qwen3.5-9B": ServeSpec(
        "Qwen/Qwen3.5-9B", "lovedheart/Qwen3.5-9B-FP8",
        ["--max-model-len", "4096", "--gpu-memory-utilization", "0.90", "--enforce-eager"],
    ),
}


def serve_command(served_name: str) -> list[str]:
    """The ``vllm serve`` argv for ``SERVE[served_name]`` (no env prefix, no shell quoting).

    Single source of truth for the launch flags: ``scripts/serve_model.sh`` reads this argv
    (one token per line) and prepends ``VLLM_USE_FLASHINFER_SAMPLER=0``, so the S1 flags never
    drift between this table and the launcher.
    """
    spec = SERVE[served_name]
    return ["vllm", "serve", spec.hf_id,
            "--served-model-name", spec.served_name,
            *spec.extra_args, "--port", str(spec.port)]


def _pid_alive(pid: int) -> bool:
    """True if ``pid`` is a live process. ``os.kill(pid, 0)`` sends no signal -- it only
    probes existence: ``ProcessLookupError`` -> dead; ``PermissionError`` -> alive (exists,
    not ours). This is the launcher-crashed check that lets ``wait_ready`` give up early."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _server_ok(url: str, timeout_s: float = 2.0) -> bool:
    """True iff a GET of ``url`` returns HTTP 200. Any connection/HTTP error -> not up yet.
    ``urllib.error.HTTPError`` subclasses ``URLError``, so both readiness-miss cases fall here."""
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError):
        return False


def wait_ready(base_url: str, pid: int, *, timeout_s: float, interval_s: float = 2.0) -> bool:
    """Poll until the server is ready, its launcher dies, or the timebox elapses.

    Returns True as soon as ``GET {base_url}/v1/models`` answers 200. Returns False if the
    launcher ``pid`` is dead (checked FIRST each pass, so a crashed launcher loses even to a
    stale server still answering on the port) or if ``timeout_s`` elapses first. Polls every
    ``interval_s`` seconds, never sleeping past the deadline.
    """
    deadline = time.monotonic() + timeout_s
    url = base_url.rstrip("/") + "/v1/models"
    while True:
        if not _pid_alive(pid):
            return False
        if _server_ok(url):
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(interval_s, remaining))


def _emit_command(argv: list[str]) -> int:
    """Print ``serve_command`` one token per line for ``scripts/serve_model.sh`` (``mapfile``)."""
    if len(argv) != 2:
        print("usage: python -m crucible.run.serving <served-model-name>", file=sys.stderr)
        return 2
    name = argv[1]
    if name not in SERVE:
        print(f"unknown served model {name!r}; known: {sorted(SERVE)}", file=sys.stderr)
        return 2
    print("\n".join(serve_command(name)))
    return 0


if __name__ == "__main__":
    raise SystemExit(_emit_command(sys.argv))
