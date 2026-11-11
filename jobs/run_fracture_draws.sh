#!/bin/bash
# Does the fracture misordering come from the DRAW, not the folds?
# Selection has only ever averaged over 5 draws of 25 volumes (seeds 0-4). MEDMNIST_CVSEED
# shuffles inner folds WITHIN those same five draws, so 32 shuffles cannot touch draw noise.
# Here each config is scored over 40 fresh draws (seeds 105-144), disjoint from the
# reporting draws 5-104, single fold shuffle each. If the CV preference flips to
# magblockpca_c16_s7, the misordering was draw variance and selection does work.
set -u
PY=/opt/dpeekenv/bin/python
OUT=/workspace/results/medmnist_cvdraws
TSV=/tmp/claude-6269505/-workspace/cb72acaf-48d3-4307-9277-3cbe75d63ca5/scratchpad/magfam.tsv
PAR=${PAR:-6}
# 8 blocks of 5 draws so partial progress is usable and each block is an independent estimate
for blk in 0 1 2 3 4 5 6 7; do
  s0=$((105 + blk*5))
  while IFS=$'\t' read -r cfg e; do
    o=$OUT/$cfg/d$s0
    [ -f "$o/medmnist_cv.csv" ] && continue
    while [ "$(jobs -rp|wc -l)" -ge "$PAR" ]; do wait -n; done
    mkdir -p "$o"
    env $e MEDMNIST_CV_SEED0=$s0 MEDMNIST_SEEDS=5 MEDMNIST_CVK=5 MEDMNIST_N=25 \
        MEDMNIST_CVSEED=0 MEDMNIST_PATCH=7 MEDMNIST_ARM=$cfg \
        MEDMNIST_NPZ=/workspace/data/medmnist/fracturemnist3d.npz MEDMNIST_OUT=$o \
        $PY medmnist_cv.py > "$o/log.txt" 2>&1 &
  done < $TSV
done
wait
echo "=== done $(date +%H:%M:%S)"
