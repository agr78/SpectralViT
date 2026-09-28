#!/usr/bin/env bash
# Label permutations for one model, in chunks of ten, many at once.
# Chunk c uses RandomState(2000 + c), so a longer run extends a shorter one rather
# than replacing it: the first chunks reproduce exactly.
# usage: run_perm.sh [model] [K] [jobs] [chunks]
set -u
PY=/opt/dpeekenv/bin/python; H="$(cd "$(dirname "$0")" && pwd)"
LOG="$H/../docs/dbs_3d/perm.log"; : > "$LOG"
M=${1:-SpectralViT}; K=${2:-16}; JOBS=${3:-10}; CHUNKS=${4:-100}
for ((c=0; c<CHUNKS; c++)); do
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 2; done
  "$PY" -u "$H/sc_worker.py" perm "$M" "$K" "$c" 10 >> "$LOG" 2>&1 &
done
wait
echo "PERM DONE" >> "$LOG"
