#!/bin/bash
# Rerun the two block-PCA arms at the full 5 seeds. Their earlier numbers were single-seed:
# a local named 'pool' inside the blockpca branch shadowed the class-to-indices dict the
# subsample draw uses, so every seed after the first died with IndexError.
set -u
PY=/opt/dpeekenv/bin/python
D=/workspace/data/medmnist
OUT=/workspace/results/medmnist_n25
run () {
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
run blockpca    MEDMNIST_BASIS=blockpca MEDMNIST_BLOCK=14 MEDMNIST_STRIDE=14 MEDMNIST_COMP=28
run blockpca_ov MEDMNIST_BASIS=blockpca MEDMNIST_BLOCK=14 MEDMNIST_STRIDE=7  MEDMNIST_COMP=8
echo "=== all done $(date -u +%H:%M:%S)"
