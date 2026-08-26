#!/usr/bin/env bash
# Detached B-lite DETERMINISTIC BATTERY pass (Pillar-1 prereg S4, spec S12
# amendment, controller ruling round 2). Mirrors gen_minority_detached.sh's
# detach/pid/log/DONE mechanics and refusal shape (OUT must ALREADY exist and
# already contain functions.jsonl; refuses if OUT/battery_stats.json already
# exists -- one pass only), but launches NO proposer/server at all: this pass
# is a fixed, no-model enumeration over each function's own arity
# (crucible.latent.gen_battery.generate_battery_inputs), replacing the
# LLM-proposed minority pass after its live-fire failure (93% parse_fail, 0
# accepted_minority -- see runs/blite-corpus/minority_stats.llm-attempt.json
# and crucible/latent/gen_battery.py's module docstring). The `timeout 7200`
# is this pass's own budgeted window (2h, same as the LLM minority pass's
# window -- generate_battery_inputs trusts the caller's bound, same as its
# siblings). Usage: gen_battery_detached.sh <out-dir>
set -u
OUT="$(realpath -m "$1")"   # ABSOLUTE: harvest+sensorium compose paths off it
cd /home/brice/workspace/crucible
if [ ! -d "$OUT" ] || [ ! -f "$OUT/functions.jsonl" ]; then
  echo "REFUSED: $OUT does not exist or has no functions.jsonl (this pass enriches an EXISTING corpus -- run gen_corpus_detached.sh first)" >&2
  exit 3
fi
if [ -e "$OUT/battery_stats.json" ]; then
  echo "REFUSED: $OUT/battery_stats.json already exists (one battery pass only; pick a fresh corpus or move the old stats file aside)" >&2
  exit 3
fi
mkdir -p runs/tmp
setsid nohup bash -c '
  start=$(date +%s)
  PYTHONDONTWRITEBYTECODE=1 TMPDIR=/home/brice/workspace/crucible/runs/tmp \
    timeout 7200 systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 --quiet \
    .venv/bin/python -c "
from pathlib import Path
from crucible.latent.gen_battery import generate_battery_inputs
stats = generate_battery_inputs(Path(\"'"$OUT"'\"), seed=0)
print(\"BATTERY_STATS:\", stats)
"
  rc=$?
  end=$(date +%s)
  echo "EXIT=$rc WALL_S=$((end-start))" >> '"$OUT"'.battery.log
  echo "$rc" > '"$OUT"'.battery.DONE
' >> "$OUT".battery.log 2>&1 &
echo $! > "$OUT".battery.pid
echo "launched battery-pass out=$OUT pid=$(cat "$OUT".battery.pid)"
