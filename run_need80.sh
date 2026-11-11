#!/bin/bash
# 100-seed reporting runs for the configs the 10-DRAW CV selects. PAR kept low so the
# 100-draw CV job (blocks 10-19) keeps most of the GPU.
set -u
PY=/opt/dpeekenv/bin/python
TSV=/tmp/claude-6269505/-workspace/cb72acaf-48d3-4307-9277-3cbe75d63ca5/scratchpad/need80.tsv
PAR=${PAR:-3}
echo "=== start $(date +%H:%M:%S)"
while IFS=$'\t' read -r ds p tag cfg e npz; do
  o=/workspace/results/medmnist_100/$tag/$ds
  [ -f "$o/medmnist_perseed.csv" ] && { echo "skip $tag $ds"; continue; }
  while [ "$(jobs -rp|wc -l)" -ge "$PAR" ]; do wait -n; done
  mkdir -p "$o"
  ( env $e MEDMNIST_PATCH=$p MEDMNIST_ARM=$tag MEDMNIST_NPZ=$npz MEDMNIST_OUT=$o \
        MEDMNIST_N=25 MEDMNIST_SEED0=5 MEDMNIST_SEEDS=100 \
        $PY medmnist_sweep.py > "$o/log.txt" 2>&1 && echo "ok $tag $ds" || echo "FAIL $tag $ds" ) &
done < $TSV
wait
echo "=== done $(date +%H:%M:%S)"
