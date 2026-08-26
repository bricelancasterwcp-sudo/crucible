#!/usr/bin/env bash
# Detached B-lite corpus generation (Pillar-1 prereg §4). Mirrors run_arm_detached.sh:
# refuses a non-empty out dir (one corpus, never resumed/spliced), OS-detaches, writes
# <out>.pid / <out>.log / <out>.DONE. The `timeout 21600` is the budgeted generation
# window the prereg's floors are judged against (6 h wall; T2 ledger: generate_corpus
# trusts the caller's bound). Usage: gen_corpus_detached.sh <out-dir> [target-functions]
set -u
OUT="$(realpath -m "$1")"; TARGET="${2:-}"   # ABSOLUTE: harvest+sensorium compose paths off it
cd /home/brice/workspace/crucible
if [ -e "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then
  echo "REFUSED: out dir $OUT exists and is non-empty (one corpus; pick a fresh dir)" >&2
  exit 3
fi
mkdir -p "$OUT" runs/tmp
setsid nohup bash -c '
  start=$(date +%s)
  PYTHONDONTWRITEBYTECODE=1 TMPDIR=/home/brice/workspace/crucible/runs/tmp \
    timeout 21600 systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 --quiet \
    .venv/bin/python -c "
from pathlib import Path
from crucible.proposer.client import VLLMProposer
from crucible.latent import config
from crucible.latent.gen import generate_corpus
target = int(\"'"${TARGET:-0}"'\") or config.TARGET_FUNCTIONS
proposer = VLLMProposer(\"http://127.0.0.1:8010\", \"Qwen/Qwen2.5-Coder-1.5B-Instruct\", chat=True)
stats = generate_corpus(proposer, target, Path(\"'"$OUT"'\"), seed=0)
print(\"GEN_STATS:\", stats)
"
  rc=$?
  end=$(date +%s)
  echo "EXIT=$rc WALL_S=$((end-start))" >> '"$OUT"'.log
  echo "$rc" > '"$OUT"'.DONE
' >> "$OUT".log 2>&1 &
echo $! > "$OUT".pid
echo "launched corpus-gen out=$OUT pid=$(cat "$OUT".pid)"
