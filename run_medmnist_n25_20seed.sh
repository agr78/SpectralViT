#!/bin/bash
# n=25 at 20 seeds: the regime the paper's claim lives in, powered enough to separate
# close arms. Five seeds left every effect but one under-significance; n=25 cells are
# ~2 s so the extra seeds are nearly free.
#
#   pca16   16 global PCA components, the published setting
#   dct     8 blocks of 14^3, 3^3 lowest cosine frequencies, 216 coefficients
#   concat  both, standardised separately and concatenated, 232 features
set -u
PY=/opt/dpeekenv/bin/python
D=/workspace/data/medmnist
OUT=/workspace/results/medmnist_n25_20s

run () {  # tag  env...
  local tag=$1; shift
  for ds in organmnist3d nodulemnist3d adrenalmnist3d fracturemnist3d vesselmnist3d synapsemnist3d; do
    npz=$D/$ds.npz
    [ "$ds" = organmnist3d ] && npz=/agent-home/medmnist/organmnist3d.npz
    echo "--- $tag $ds $(date -u +%H:%M:%S)"
    env "$@" MEDMNIST_NPZ="$npz" MEDMNIST_N=25 MEDMNIST_SEEDS=20 \
        MEDMNIST_OUT="$OUT/$tag/$ds" $PY -u medmnist_sweep.py 2>&1 \
      | grep -vE "UserWarning|self.transformer"
  done
}

run pca16  MEDMNIST_BASIS=pca    MEDMNIST_COMP=16
run dct    MEDMNIST_BASIS=dct    MEDMNIST_BLOCK=14 MEDMNIST_STRIDE=14 MEDMNIST_FREQ=3
run concat MEDMNIST_BASIS=concat MEDMNIST_BLOCK=14 MEDMNIST_STRIDE=14 MEDMNIST_FREQ=3 MEDMNIST_COMP=16
echo "=== all done $(date -u +%H:%M:%S)"
