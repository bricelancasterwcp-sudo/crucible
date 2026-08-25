#!/usr/bin/env bash
# Detached arm run with the R-S4-1 guard: gating/rehearsal runs are NEVER resumed --
# a non-empty out dir is refused (infra kill = clean rerun from zero, new out dir).
# Usage: run_arm_detached.sh <arm> <out-dir> <tasks-file> [extra crucible-arm-run args...]
# Writes <out-dir>.pid and <out-dir>.DONE (marker carries the exit code); OS-detached
# per pre-reg §8 (runs > 2 h outlive the session harness).
set -u
ARM="$1"; OUT="$2"; TASKS="$3"; shift 3
cd /home/brice/workspace/crucible
if [ -e "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then
  echo "REFUSED: out dir $OUT exists and is non-empty (R-S4-1: never resume; pick a fresh dir)" >&2
  exit 3
fi
mkdir -p "$OUT" runs/tmp
setsid nohup bash -c '
  start=$(date +%s)
  PYTHONDONTWRITEBYTECODE=1 TMPDIR=/home/brice/workspace/crucible/runs/tmp \
    systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 --quiet \
    .venv/bin/crucible arm run streams/1158e92f40ad --arm '"$ARM"' \
    --base-url http://127.0.0.1:8010 --tasks '"$TASKS"' --out '"$OUT"' '"$*"'
  rc=$?
  end=$(date +%s)
  echo "EXIT=$rc WALL_S=$((end-start))" >> '"$OUT"'.log
  echo "$rc" > '"$OUT"'.DONE
' >> "$OUT".log 2>&1 &
echo $! > "$OUT".pid
echo "launched arm=$ARM out=$OUT pid=$(cat "$OUT".pid)"
