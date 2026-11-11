#!/bin/bash
# 8 independent inner-fold shuffles of the magnitude-family CV on fracture.
# MEDMNIST_CVSEED varies the StratifiedKFold assignment only: the draw, the basis, the
# encoder and the optimiser are identical to the reporting run, and the test split is
# never read. Averaging over shuffles tests whether fracture's within-family misordering
# is fold-assignment variance (removable) or bias (not).
set -u
PY=/opt/dpeekenv/bin/python
OUT=/workspace/results/medmnist_cvshuf
TSV=/tmp/claude-6269505/-workspace/cb72acaf-48d3-4307-9277-3cbe75d63ca5/scratchpad/magfam.tsv
PAR=${PAR:-6}
export MEDMNIST_N=25 MEDMNIST_SEEDS=5 MEDMNIST_CVK=5
run(){ local cfg=$1 e=$2 cs=$3
  local o=$OUT/$cfg/cv$cs
  [ -f "$o/medmnist_cv.csv" ] && { echo "skip $cfg cv$cs"; return 0; }
  mkdir -p "$o"
  env $e MEDMNIST_CVSEED=$cs MEDMNIST_PATCH=7 MEDMNIST_ARM=$cfg \
      MEDMNIST_NPZ=/workspace/data/medmnist/fracturemnist3d.npz MEDMNIST_OUT=$o \
      $PY medmnist_cv.py > "$o/log.txt" 2>&1 && echo "ok $cfg cv$cs" || echo "FAIL $cfg cv$cs"; }
echo "=== start $(date +%H:%M:%S)"
for cs in 0 1 2 3 4 5 6 7; do
  while IFS=$'\t' read -r cfg e; do
    while [ "$(jobs -rp|wc -l)" -ge "$PAR" ]; do wait -n; done
    run "$cfg" "$e" "$cs" &
  done < $TSV
done
wait
echo "=== done $(date +%H:%M:%S)"
