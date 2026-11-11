#!/bin/bash
# 100-seed reporting runs for the (dataset, config, PATCH) triples that draw-averaged CV
# selects. The patch matters: 40-draw CV moves the spatial baseline to patch 4 on organ,
# adrenal and synapse, where 5-draw CV had chosen 14, 7 and 14. Patch 4 is the stronger
# baseline (pooled mean 0.648 vs 0.621 at patch 14), so these runs can only shrink our
# margins relative to the current tables -- which is the correct direction for a selector
# chosen on a mechanism rather than on an outcome.
# Output key carries the patch (cfg_P<patch>) so the old runs stay valid and findable.
set -u
PY=/opt/dpeekenv/bin/python
TSV=/tmp/claude-6269505/-workspace/cb72acaf-48d3-4307-9277-3cbe75d63ca5/scratchpad/need100.tsv
PAR=${PAR:-5}
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
