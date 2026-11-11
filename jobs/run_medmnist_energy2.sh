#!/bin/bash
set -u
PY=/opt/dpeekenv/bin/python
D=/workspace/data/medmnist
OUT=/workspace/results/medmnist_n25_paired
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
B="MEDMNIST_BLOCK=14 MEDMNIST_STRIDE=14 MEDMNIST_FREQ=3 MEDMNIST_COMP=16"
#run energy     MEDMNIST_BASIS=energy $B
#run energy_cat MEDMNIST_BASIS=energy_cat $B
echo "=== all done $(date -u +%H:%M:%S)"
run dct_energy MEDMNIST_BASIS=dct_energy $B
run pca_energy MEDMNIST_BASIS=pca_energy $B
echo "=== all done $(date -u +%H:%M:%S)"
