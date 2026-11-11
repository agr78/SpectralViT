#!/bin/bash
# Reporting runs for the ladder sweep: 100 seeds at n=25 and n=50, 20 seeds for n>=100.
# Fewer seeds at large n because the per-draw variance falls with n and the cost rises with
# it; 20 seeds resolves ~0.025, which is adequate where the crossover sits. Waits on the
# CV log MARKER, never on a pgrep of a script name (that self-matches and deadlocks).
set -u
PY=/opt/dpeekenv/bin/python
PAR=${PAR:-4}
for N in 25 50 100 200 400 800; do
  while ! grep -q -- "--- n=$N done" /workspace/results/ladder_sweep.log 2>/dev/null; do sleep 60; done
  SEEDS=20; [ "$N" -le 50 ] && SEEDS=100
  echo "=== n=$N selection ready, $SEEDS seeds $(date +%H:%M:%S)"
  $PY /workspace/select_ladder.py "$N" > /workspace/results/ladder_report_n$N.tsv || continue
  cat /workspace/results/ladder_report_n$N.tsv
  while IFS=$'\t' read -r ds p tag cfg e npz; do
    o=/workspace/results/ladder_100/$tag/$ds
    [ -f "$o/medmnist_perseed.csv" ] && { echo "skip $tag"; continue; }
    while [ "$(jobs -rp|wc -l)" -ge "$PAR" ]; do wait -n; done
    mkdir -p "$o"
    ( env $e MEDMNIST_PATCH=$p MEDMNIST_ARM=$tag MEDMNIST_NPZ=$npz MEDMNIST_OUT=$o \
          MEDMNIST_N=$N MEDMNIST_SEED0=5 MEDMNIST_SEEDS=$SEEDS \
          $PY /workspace/medmnist_sweep.py > "$o/log.txt" 2>&1 \
      && echo "ok n=$N $tag $ds" || echo "FAIL n=$N $tag $ds" ) &
  done < /workspace/results/ladder_report_n$N.tsv
  wait
  echo "=== n=$N reporting done $(date +%H:%M:%S)"
done
echo "=== all reporting done $(date +%H:%M:%S)"
