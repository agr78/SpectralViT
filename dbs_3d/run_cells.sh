#!/usr/bin/env bash
# Launch cell workers concurrently on the GPU, JOBS at a time.
# usage: run_cells.sh [jobs]
set -u
JOBS=${1:-6}
PY=${PY:-/opt/dpeekenv/bin/python}
HERE="$(cd "$(dirname "$0")" && pwd)"
LOG="$HERE/../docs/dbs_3d/cells.log"
mkdir -p "$(dirname "$LOG")"
: > "$LOG"

for ordering in rowmajor l1 lap; do
  for K in 8 16; do
    for weight in none inv lap learned; do
      while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 2; done
      "$PY" -u "$HERE/cell_worker.py" "$ordering" "$K" "$weight" 20 >> "$LOG" 2>&1 &
    done
  done
done
wait
echo "ALL CELLS DONE" >> "$LOG"
