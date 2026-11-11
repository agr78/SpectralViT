#!/bin/bash
# The arms that matter, at 10 seeds. n=25 cells are seconds and the spatial arm is cached
# per (dataset, n, seed), so the first arm pays for seeds 5-9 and the rest reuse them --
# which also keeps the comparison paired across every arm.
set -u
PY=/opt/dpeekenv/bin/python
D=/workspace/data/medmnist
OUT=/workspace/results/medmnist_n25_10s
B="MEDMNIST_BLOCK=14 MEDMNIST_STRIDE=14 MEDMNIST_FREQ=3 MEDMNIST_COMP=16"
run () {
  local tag=$1; shift
  for ds in organmnist3d nodulemnist3d adrenalmnist3d fracturemnist3d vesselmnist3d synapsemnist3d; do
    npz=$D/$ds.npz
    [ "$ds" = organmnist3d ] && npz=/agent-home/medmnist/organmnist3d.npz
    echo "--- $tag $ds $(date -u +%H:%M:%S)"
    env $B "$@" MEDMNIST_NPZ="$npz" MEDMNIST_N=25 MEDMNIST_SEEDS=10 \
        MEDMNIST_OUT="$OUT/$tag/$ds" $PY -u medmnist_sweep.py 2>&1 \
      | grep -vE "UserWarning|self.transformer"
  done
}
run pca16      MEDMNIST_BASIS=pca
run energy     MEDMNIST_BASIS=energy
run pca_energy MEDMNIST_BASIS=pca_energy
run concat     MEDMNIST_BASIS=concat
run energy_cat MEDMNIST_BASIS=energy_cat
run dct_energy MEDMNIST_BASIS=dct_energy
run dct        MEDMNIST_BASIS=dct
echo "=== all done $(date -u +%H:%M:%S)"
