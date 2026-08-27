#!/usr/bin/env bash
# Detached, PARALLEL re-harvest of an already-generated B-lite corpus
# (Pillar-1 prereg S4, round-3 CRITICAL fix -- see crucible/latent/harvest.py's
# module docstring and .superpowers/sdd/2026-08-26-crucible-pillar1-blite/
# minority-pass-report.md). harvest()'s pre-fix scratch-dir-reuse bug meant
# every sample after the first successful harvest() call in a shared scratch
# dir silently replayed that first call's result -- so an EXISTING corpus's
# samples.jsonl (from generate_corpus and/or the minority/battery enrichment
# passes) may be entirely fabricated. This script re-runs every (function,
# input) pair in functions.jsonl from scratch against the FIXED harvest(),
# archiving the old samples.jsonl (crucible.latent.reharvest.reharvest_samples
# never deletes it) and replacing it with a freshly-measured one.
#
# Mirrors gen_battery_detached.sh's detach/pid/log/DONE mechanics and refusal
# shape (OUT must ALREADY exist and already contain functions.jsonl; refuses
# if OUT/reharvest_stats.json already exists -- one pass only, same as every
# other pass in this package), but launches NO proposer/server at all: this
# calls crucible.latent.reharvest.reharvest_samples directly, which runs its
# own ThreadPoolExecutor internally (JOBS below, default 8). `timeout 14400`
# (4h) is this pass's own budgeted window -- reharvest_samples trusts the
# caller's bound, same as every sibling launcher.
#
# Usage: reharvest_detached.sh <out-dir> [jobs]
set -u
OUT="$(realpath -m "$1")"   # ABSOLUTE: harvest+sensorium compose paths off it
JOBS="${2:-8}"
cd /home/brice/workspace/crucible
if [ ! -d "$OUT" ] || [ ! -f "$OUT/functions.jsonl" ]; then
  echo "REFUSED: $OUT does not exist or has no functions.jsonl (reharvest re-measures an EXISTING corpus -- run gen_corpus_detached.sh first)" >&2
  exit 3
fi
if [ -e "$OUT/reharvest_stats.json" ]; then
  echo "REFUSED: $OUT/reharvest_stats.json already exists (one reharvest pass only; pick a fresh corpus or move the old stats file aside)" >&2
  exit 3
fi
mkdir -p runs/tmp
setsid nohup bash -c '
  start=$(date +%s)
  PYTHONDONTWRITEBYTECODE=1 TMPDIR=/home/brice/workspace/crucible/runs/tmp \
    timeout 14400 systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 --quiet \
    .venv/bin/python -c "
from pathlib import Path
from crucible.latent.reharvest import reharvest_samples
stats = reharvest_samples(Path(\"'"$OUT"'\"), jobs='"$JOBS"')
print(\"REHARVEST_STATS:\", stats)
"
  rc=$?
  end=$(date +%s)
  echo "EXIT=$rc WALL_S=$((end-start))" >> '"$OUT"'.reharvest.log
  echo "$rc" > '"$OUT"'.reharvest.DONE
' >> "$OUT".reharvest.log 2>&1 &
echo $! > "$OUT".reharvest.pid
echo "launched reharvest out=$OUT jobs=$JOBS pid=$(cat "$OUT".reharvest.pid)"
