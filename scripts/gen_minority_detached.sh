#!/usr/bin/env bash
# Detached B-lite MINORITY-INPUT second pass (Pillar-1 prereg S4, spec S12
# pre-lock amendment). Mirrors gen_corpus_detached.sh's detach/pid/log/DONE
# mechanics, but the refusal direction is FLIPPED: OUT must ALREADY exist and
# already contain functions.jsonl (this pass enriches an existing corpus, it
# never starts one from scratch), and it refuses outright if
# minority_stats.json already exists in OUT -- this is a ONE-PASS amendment,
# never resumed/re-run over the same corpus (generate_minority_inputs itself
# also dedups against samples.jsonl, not just functions.jsonl's original
# args_literals, as a second line of defense against a crash-then-retry, but
# the launcher's own refusal is the first one). The `timeout 7200` is this
# pass's own budgeted window (2h; distinct from the first pass's 6h
# generate_corpus window) -- generate_minority_inputs trusts the caller's
# bound, same as generate_corpus does. Usage: gen_minority_detached.sh <out-dir>
set -u
OUT="$(realpath -m "$1")"   # ABSOLUTE: harvest+sensorium compose paths off it
cd /home/brice/workspace/crucible
if [ ! -d "$OUT" ] || [ ! -f "$OUT/functions.jsonl" ]; then
  echo "REFUSED: $OUT does not exist or has no functions.jsonl (this pass enriches an EXISTING corpus -- run gen_corpus_detached.sh first)" >&2
  exit 3
fi
if [ -e "$OUT/minority_stats.json" ]; then
  echo "REFUSED: $OUT/minority_stats.json already exists (one minority pass only; pick a fresh corpus or move the old stats file aside)" >&2
  exit 3
fi
mkdir -p runs/tmp
setsid nohup bash -c '
  start=$(date +%s)
  PYTHONDONTWRITEBYTECODE=1 TMPDIR=/home/brice/workspace/crucible/runs/tmp \
    timeout 7200 systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 --quiet \
    .venv/bin/python -c "
from pathlib import Path
from crucible.proposer.client import VLLMProposer
from crucible.latent.gen import generate_minority_inputs
proposer = VLLMProposer(\"http://127.0.0.1:8010\", \"Qwen/Qwen2.5-Coder-1.5B-Instruct\", chat=True)
stats = generate_minority_inputs(proposer, Path(\"'"$OUT"'\"), seed=0)
print(\"MINORITY_STATS:\", stats)
"
  rc=$?
  end=$(date +%s)
  echo "EXIT=$rc WALL_S=$((end-start))" >> '"$OUT"'.minority.log
  echo "$rc" > '"$OUT"'.minority.DONE
' >> "$OUT".minority.log 2>&1 &
echo $! > "$OUT".minority.pid
echo "launched minority-pass out=$OUT pid=$(cat "$OUT".minority.pid)"
