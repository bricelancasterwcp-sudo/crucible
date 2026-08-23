"""Served-model identity assertion: verify *which* model a server actually serves.

Before any run reads a completion from an OpenAI-compatible endpoint we assert the
server's advertised model matches what the run expects. Two server kinds are supported:

- **vLLM** advertises via ``GET /v1/models`` → ``data[0].id`` (the ``--served-model-name``).
- **llama.cpp** advertises via ``GET /props`` → ``model_path`` / ``default_generation_settings.model``.

llama.cpp *also* serves an OpenAI-compatible ``/v1/models`` (ledger ruling R6), so a plain
``/v1/models`` probe would misclassify it as vLLM. ``probe`` therefore checks llama.cpp's
own ``/props`` first and only falls back to ``/v1/models`` for the vLLM case.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass


class IdentityMismatch(RuntimeError):
    """Raised when no server is recognised, or the served model is not the expected one."""


@dataclass(frozen=True)
class ServedIdentity:
    kind: str  # {"vllm", "llamacpp"}
    model: str
    extra: dict


def _get(url: str, timeout_s: float):
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as r:
            return json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, OSError):
        return None


def probe(base_url: str, timeout_s: float = 5.0) -> ServedIdentity:
    base = base_url.rstrip("/")
    # llama.cpp ALSO serves /v1/models (OpenAI-compatible), so its own /props must be checked first.
    p = _get(f"{base}/props", timeout_s)
    if p and (p.get("model_path") or p.get("default_generation_settings", {}).get("model")):
        model = p.get("model_path") or p["default_generation_settings"]["model"]
        return ServedIdentity("llamacpp", model, {k: p[k] for k in ("total_slots",) if k in p})
    v = _get(f"{base}/v1/models", timeout_s)
    if v and v.get("data"):
        return ServedIdentity("vllm", v["data"][0]["id"], {"n_models": len(v["data"])})
    raise IdentityMismatch(f"no recognisable server at {base_url}")


def assert_identity(base_url: str, expected_model: str) -> ServedIdentity:
    ident = probe(base_url)
    if not (ident.model == expected_model or ident.model.endswith(expected_model)):
        raise IdentityMismatch(f"served {ident.model!r} at {base_url}, expected {expected_model!r}")
    return ident
