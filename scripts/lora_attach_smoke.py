#!/usr/bin/env python3
"""LoRA-attach smoke test for the crucible proposer (Qwen3.5-2B).

Question this answers: can PEFT attach a LoRA to Qwen3.5-2B's language model, run a
forward+backward, generate once with the adapter active, and save it -- and (optionally)
does a running server report the identity we expect before we trust its completions?

Decision rule (spec §2 fallback): if attach + fwd/bwd + generate + save all succeed the
proposer is ``Qwen/Qwen3.5-2B``; if any step fails and cannot be fixed inside the timebox,
re-run with ``Qwen/Qwen2.5-Coder-1.5B-Instruct`` and record that as the proposer.

RUNTIME: this is the *live* half of the S1 serving spike. It needs the ``serve`` extra
(``torch``, ``transformers``, ``peft``, ``accelerate``) and a CUDA GPU, so it is run by the
operator on the GPU box -- NOT in CI and NOT headless. The heavy imports live inside the
functions so this file imports (and ``--help`` works) with none of them installed; only
``crucible.proposer.identity`` (stdlib-only) is imported at module load.

Examples
--------
    # local attach smoke only (no server):
    python scripts/lora_attach_smoke.py --model Qwen/Qwen3.5-2B

    # also assert a running server's identity before trusting it, then POST the adapter:
    python scripts/lora_attach_smoke.py --model Qwen/Qwen3.5-2B \
        --base-url http://127.0.0.1:8001 --served-model Qwen/Qwen3.5-2B --load-into-server
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `crucible` importable when invoked as `python scripts/lora_attach_smoke.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crucible.proposer.identity import ServedIdentity, assert_identity  # stdlib-only

# Attention + MLP projection leaves LoRA should target; anything else (e.g. a vision
# tower's linears) is deliberately excluded via the "vis" filter in _lora_targets.
_PROJ_KEYS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "in_proj", "out_proj")
_LORA_R = 16
_LORA_ALPHA = 16
_PROMPT = "def add(a, b):\n    return a + b\n"
_DEFAULT_OUT = "runs/lora-smoke"


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="LoRA-attach smoke test for the crucible proposer.")
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B", help="HF id or local path of the base model.")
    ap.add_argument("--out", default=_DEFAULT_OUT, help="Directory to save the trained adapter.")
    ap.add_argument("--base-url", default=None, help="If set, assert this server's served identity first.")
    ap.add_argument("--served-model", default=None, help="Expected served model name (defaults to --model).")
    ap.add_argument("--load-into-server", action="store_true",
                    help="After saving, POST the adapter to a vLLM server (/v1/load_lora_adapter).")
    return ap.parse_args(argv)


def _lora_targets(model, torch) -> list[str]:
    """Distinct linear-leaf names to adapt, excluding any vision-tower linears."""
    leaves = sorted({
        name.split(".")[-1]
        for name, mod in model.named_modules()
        if isinstance(mod, torch.nn.Linear) and "vis" not in name.lower()
    })
    targets = [t for t in leaves if any(k in t for k in _PROJ_KEYS)]
    return targets or leaves[:4]


def _load_base(model_name: str):
    """Load the base causal-LM on CUDA; fall back to the image-text-to-text head if needed."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(model_name)
    print("config class:", type(cfg).__name__, "| architectures:", getattr(cfg, "architectures", None))
    try:
        return AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda")
    except Exception as exc:  # noqa: BLE001 -- record and try the multimodal head
        print("AutoModelForCausalLM failed:", exc)
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda")


def attach_and_step(model_name: str, out_dir: str) -> dict:
    """Attach a LoRA, run one fwd/bwd, generate once with it active, and save it."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer

    model = _load_base(model_name)
    targets = _lora_targets(model, torch)
    print("LoRA targets:", targets)
    peft_model = get_peft_model(model, LoraConfig(r=_LORA_R, lora_alpha=_LORA_ALPHA,
                                                  target_modules=targets, task_type="CAUSAL_LM"))
    peft_model.print_trainable_parameters()

    tok = AutoTokenizer.from_pretrained(model_name)
    batch = tok([_PROMPT] * 2, return_tensors="pt").to("cuda")
    out = peft_model(**batch, labels=batch["input_ids"])
    out.loss.backward()  # proves the adapter participates in autograd

    peft_model.eval()
    with torch.no_grad():  # one generation with the adapter active proves it loads + runs
        gen = peft_model.generate(**tok([_PROMPT], return_tensors="pt").to("cuda"), max_new_tokens=8)
    sample = tok.decode(gen[0], skip_special_tokens=True)

    Path(out_dir).parent.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(out_dir)
    result = {
        "loss": float(out.loss),
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
        "sample": sample,
        "adapter_dir": out_dir,
        "targets": targets,
    }
    print("loss:", result["loss"], "| peak VRAM GB:", result["peak_vram_gb"])
    print("adapter-active sample:", repr(sample))
    print("saved:", out_dir)
    return result


def check_served_identity(base_url: str, expected_model: str) -> ServedIdentity:
    """Assert the server serves the expected model before we trust it. Raises on mismatch."""
    ident = assert_identity(base_url, expected_model)
    print(f"served identity OK: kind={ident.kind} model={ident.model!r}")
    return ident


def load_into_server(base_url: str, adapter_dir: str, ident: ServedIdentity, name: str = "smoke") -> None:
    """POST the saved adapter to the server. vLLM only; llama.cpp needs a GGUF-converted adapter."""
    import json
    import urllib.request

    if ident.kind != "vllm":
        print(f"skip server load: {ident.kind} needs a GGUF-converted adapter "
              "(convert_lora_to_gguf.py) + --lora-adapters at launch; record that path in findings.")
        return
    body = json.dumps({"lora_name": name, "lora_path": str(Path(adapter_dir).resolve())}).encode()
    req = urllib.request.Request(f"{base_url.rstrip('/')}/v1/load_lora_adapter", data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as resp:
        print("load_lora_adapter:", resp.status, resp.read().decode("utf-8", "replace")[:200])


def main(argv=None) -> int:
    args = parse_args(argv)
    ident = None
    if args.base_url:
        ident = check_served_identity(args.base_url, args.served_model or args.model)
    result = attach_and_step(args.model, args.out)
    if args.load_into_server:
        if ident is None:
            print("--load-into-server requires --base-url; skipping server load.")
        else:
            load_into_server(args.base_url, result["adapter_dir"], ident)
    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
