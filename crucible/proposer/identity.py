"""Served-model identity assertion: verify *which* model a server actually serves.

Before any run reads a completion from an OpenAI-compatible endpoint we assert the
server's advertised model matches what the run expects. Two server kinds are supported:

- **vLLM** advertises via ``GET /v1/models`` → ``data[0].id`` (the ``--served-model-name``).
- **llama.cpp** advertises via ``GET /props`` → ``model_path`` / ``default_generation_settings.model``.

llama.cpp *also* serves an OpenAI-compatible ``/v1/models`` (ledger ruling R6), so a plain
``/v1/models`` probe would misclassify it as vLLM. ``probe`` therefore checks llama.cpp's
own ``/props`` first and only falls back to ``/v1/models`` for the vLLM case.

*A server can advertise MORE THAN ONE id, and the expected one is not always first (S3).*
Once a LoRA adapter is loaded at runtime, vLLM lists the base model AND every loaded adapter
under its own name -- base first, adapters after -- and the arm legitimately asks for the
adapter by name (that is how vLLM routes a request to it). Matching only ``data[0].id``
therefore rejected exactly the case S3's sleep loop creates: the first accepted adapter would
raise :class:`IdentityMismatch` the moment anything tried to serve it, AFTER the training
cost had already been paid. :func:`assert_identity` matches the expected model against ANY
advertised id, and is otherwise unchanged -- an id the server does not advertise at all still
raises. ``ServedIdentity.model`` remains the PRIMARY (first) id, which is the base checkpoint;
``ServedIdentity.models`` carries every advertised id, so a caller can see the whole list.
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
    """What a server says it is serving.

    ``model`` is the PRIMARY advertised id (vLLM lists the base checkpoint first);
    ``models`` is every id it advertises -- the base plus any runtime-loaded LoRA adapters.
    Trailing and defaulted so a pre-S3 construction still means what it meant; empty is read
    as "just the primary" by :func:`assert_identity`.
    """

    kind: str  # {"vllm", "llamacpp"}
    model: str
    extra: dict
    models: tuple[str, ...] = ()


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
        return ServedIdentity("llamacpp", model,
                              {k: p[k] for k in ("total_slots",) if k in p}, (model,))
    v = _get(f"{base}/v1/models", timeout_s)
    if v and v.get("data"):
        ids = tuple(str(m["id"]) for m in v["data"] if "id" in m)
        return ServedIdentity("vllm", ids[0], {"n_models": len(v["data"])}, ids)
    raise IdentityMismatch(f"no recognisable server at {base_url}")


def assert_identity(base_url: str, expected_model: str) -> ServedIdentity:
    """Assert ``base_url`` advertises ``expected_model``; return what it is serving.

    The match is against EVERY advertised id, not just the first: a vLLM with a runtime LoRA
    loaded lists the base first and the adapter after, and an arm serving that adapter asks
    for it by name (see the module docstring). Still strict -- a model the server does not
    advertise at all raises, and the error names every id it does.
    """
    ident = probe(base_url)
    advertised = ident.models or (ident.model,)
    if not any(m == expected_model or m.endswith(expected_model) for m in advertised):
        raise IdentityMismatch(f"served {list(advertised)!r} at {base_url}, "
                               f"expected {expected_model!r}")
    return ident
