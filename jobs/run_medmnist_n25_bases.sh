#!/bin/bash
# Every basis at n=25, the size where the data-efficiency claim lives.
#
# Arms are matched at ~216 features except the two global PCA arms, which cannot reach
# that count: whole-volume PCA on 25 samples has rank 25, so 16 and 24 are the only
# meaningful settings. Block PCA is the point of the comparison -- it is the data-driven
# twin of the block cosine transform at identical locality and count, and pooling blocks
# across volumes lifts the rank cap that binds the global arms.
#
#   pca16        global, 16 comps          16 features
#   pca24        global, at the rank cap   24 features
#   blockpca     8 x 14^3 blocks, 28 comps 224 features
#   blockpca_ov  27 x 14^3 @7, 8 comps     216 features
#   dct          8 x 14^3 blocks, 3^3 freq 216 features
#   dct_ov       27 x 14^3 @7, 2^3 freq    216 features
set -u
PY=/opt/dpeekenv/bin/python
D=/workspace/data/medmnist
OUT=/workspace/results/medmnist_n25

run() {  # tag  env-assignments...
  local tag=$1; shift
  for ds in organmnist3d nodulemnist3d adrenalmnist3d fracturemnist3d vesselmnist3d synapsemnist3d; do
    npz=$D/$ds.npz
    [ "$ds" = organmnist3d ] && npz=/agent-home/medmnist/organmnist3d.npz
    echo "--- $tag $ds $(date -u +%H:%M:%S)"
    env "$@" MEDMNIST_NPZ="$npz" MEDMNIST_N=25 MEDMNIST_SEEDS=5 \
        MEDMNIST_OUT="$OUT/$tag/$ds" $PY -u medmnist_sweep.py 2>&1 \
      | grep -vE "UserWarning|self.transformer"
  done
}

run pca16       MEDMNIST_BASIS=pca      MEDMNIST_COMP=16
run pca24       MEDMNIST_BASIS=pca      MEDMNIST_COMP=24
run blockpca    MEDMNIST_BASIS=blockpca MEDMNIST_BLOCK=14 MEDMNIST_STRIDE=14 MEDMNIST_COMP=28
run blockpca_ov MEDMNIST_BASIS=blockpca MEDMNIST_BLOCK=14 MEDMNIST_STRIDE=7  MEDMNIST_COMP=8
run dct         MEDMNIST_BASIS=dct      MEDMNIST_BLOCK=14 MEDMNIST_STRIDE=14 MEDMNIST_FREQ=3
run dct_ov      MEDMNIST_BASIS=dct      MEDMNIST_BLOCK=14 MEDMNIST_STRIDE=7  MEDMNIST_FREQ=2
echo "=== all done $(date -u +%H:%M:%S)"
