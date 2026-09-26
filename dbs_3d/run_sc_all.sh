#!/usr/bin/env bash
# multi-model scorecard, cells in parallel then permutations in parallel
set -u
PY=/opt/dpeekenv/bin/python; H="$(cd "$(dirname "$0")" && pwd)"
LOG="$H/../docs/dbs_3d/sc.log"; mkdir -p "$(dirname "$LOG")"; : > "$LOG"
JOBS=${1:-12}; NS=${2:-10}; NPERM_CHUNK=${3:-10}
MODELS=("Clinical" "Spectral LR" "Spectral MLP" "Spatial ViT" "SpectralViT")
# phase 1: every (model, K) cell
for m in "${MODELS[@]}"; do
  for K in 8 16 32; do
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 2; done
    "$PY" -u "$H/sc_worker.py" cell "$m" "$K" "$NS" >> "$LOG" 2>&1 &
  done
done
wait
echo "CELLS DONE" >> "$LOG"
# phase 2: permutations for each model at its best-internal K
for m in "${MODELS[@]}"; do
  BEST=$("$PY" - "$m" <<'PYEOF'
import sys, glob, pickle, os, numpy as np
m = sys.argv[1].replace(' ', '_')
best, bk = -1, 8
for p in glob.glob(os.path.join('/workspace/docs/dbs_3d/sc', f'cell_{m}_*.pkl')):
    r = pickle.load(open(p, 'rb'))
    v = float(np.mean(r['internal']))
    if v > best: best, bk = v, r['K']
print(bk)
PYEOF
)
  for c in 0 1 2 3 4 5; do
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 2; done
    "$PY" -u "$H/sc_worker.py" perm "$m" "$BEST" "$c" "$NPERM_CHUNK" >> "$LOG" 2>&1 &
  done
done
wait
echo "PERMS DONE" >> "$LOG"
