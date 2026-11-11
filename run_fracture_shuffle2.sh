#!/bin/bash
# Shuffles 8-31 for the two configs the 8-shuffle CV cannot separate (0.5904 vs 0.5892).
# If their inner-CV means stay equal as the SE shrinks, the inner score carries no
# information about a 0.013 test difference -- bias, not variance, and no number of
# shuffles recovers the fracture win.
set -u
PY=/opt/dpeekenv/bin/python
OUT=/workspace/results/medmnist_cvshuf
PAR=${PAR:-6}
export MEDMNIST_N=25 MEDMNIST_SEEDS=5 MEDMNIST_CVK=5
declare -A E=(
 [magblockpca_c16_s7]="MEDMNIST_BASIS=magblockpca MEDMNIST_COMP=16 MEDMNIST_BLOCK=14 MEDMNIST_STRIDE=7 MEDMNIST_FREQ=3"
 [magblockpca_c8_s14]="MEDMNIST_BASIS=magblockpca MEDMNIST_COMP=8 MEDMNIST_BLOCK=14 MEDMNIST_STRIDE=14 MEDMNIST_FREQ=3"
)
for cs in $(seq 8 31); do
  for cfg in magblockpca_c16_s7 magblockpca_c8_s14; do
    while [ "$(jobs -rp|wc -l)" -ge "$PAR" ]; do wait -n; done
    o=$OUT/$cfg/cv$cs
    [ -f "$o/medmnist_cv.csv" ] && continue
    mkdir -p "$o"
    env ${E[$cfg]} MEDMNIST_CVSEED=$cs MEDMNIST_PATCH=7 MEDMNIST_ARM=$cfg \
        MEDMNIST_NPZ=/workspace/data/medmnist/fracturemnist3d.npz MEDMNIST_OUT=$o \
        $PY medmnist_cv.py > "$o/log.txt" 2>&1 &
  done
done
wait
echo "=== done $(date +%H:%M:%S)"
