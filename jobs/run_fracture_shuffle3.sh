#!/bin/bash
# Bring the other 12 magnitude-family configs from 8 to 32 shuffles, so the argmax at any
# k<=32 is taken over the WHOLE family rather than the two leaders. Without this, a k=10
# answer would be conditioned on only the top two having been extended.
set -u
PY=/opt/dpeekenv/bin/python
OUT=/workspace/results/medmnist_cvshuf
TSV=/tmp/claude-6269505/-workspace/cb72acaf-48d3-4307-9277-3cbe75d63ca5/scratchpad/magfam.tsv
PAR=${PAR:-6}
export MEDMNIST_N=25 MEDMNIST_SEEDS=5 MEDMNIST_CVK=5
for cs in $(seq 8 31); do
  while IFS=$'\t' read -r cfg e; do
    case "$cfg" in magblockpca_c16_s7|magblockpca_c8_s14) continue;; esac
    o=$OUT/$cfg/cv$cs
    [ -f "$o/medmnist_cv.csv" ] && continue
    while [ "$(jobs -rp|wc -l)" -ge "$PAR" ]; do wait -n; done
    mkdir -p "$o"
    env $e MEDMNIST_CVSEED=$cs MEDMNIST_PATCH=7 MEDMNIST_ARM=$cfg \
        MEDMNIST_NPZ=/workspace/data/medmnist/fracturemnist3d.npz MEDMNIST_OUT=$o \
        $PY medmnist_cv.py > "$o/log.txt" 2>&1 &
  done < $TSV
done
wait
echo "=== done $(date +%H:%M:%S)"
