#!/usr/bin/env bash
# Launch vLLM for one S2 served model with the S1-measured flags (docs/findings/S1-serving.md).
#
# The launch argv (HF repo, --served-model-name, per-model flags, --port) comes straight from
# crucible/run/serving.py's SERVE table -- the single source of truth -- so the flags can never
# drift between the Python code and this script.
#
# The FlashInfer sampler JIT compiles a top-k/top-p CUDA kernel at engine init, which needs
# ninja + nvcc. This box has neither, so engine-core init crashes without the workaround: the
# launch below disables it (native torch sampling; n-best and logprobs are unaffected). Do not
# remove that env var -- it is the S1 gotcha this harness exists to encode.
#
# Usage: scripts/serve_model.sh <served-model-name>      # e.g. Qwen/Qwen3.5-2B or Qwen/Qwen3.5-9B
# Operational (Task 14/16), GPU-committing -- NOT run by the unit tests.
set -euo pipefail

MODEL="${1:?usage: serve_model.sh <served-model-name>   (e.g. Qwen/Qwen3.5-2B)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"
PY="${PYTHON:-$REPO_ROOT/.venv/bin/python}"

# Stop Ollama first so its VRAM is freed (S1 §5); then confirm the GPU is near-idle.
systemctl --user stop ollama 2>/dev/null || pkill -f 'ollama serve' 2>/dev/null || true

# Pull the exact vllm-serve argv (one token per line) from the SERVE table.
mapfile -t ARGV < <("$PY" -m crucible.run.serving "$MODEL")

echo "[serve_model] launching: VLLM_USE_FLASHINFER_SAMPLER=0 ${ARGV[*]}" >&2
exec env VLLM_USE_FLASHINFER_SAMPLER=0 "${ARGV[@]}"
