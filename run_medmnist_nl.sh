#!/bin/bash
# Every nonlinear-combination variant at n=25, against the linear baselines already run.
set -u
PY=/opt/dpeekenv/bin/python
D=/workspace/data/medmnist
OUT=/workspace/results/medmnist_n25
COMMON="MEDMNIST_BLOCK=14 MEDMNIST_STRIDE=14 MEDMNIST_FREQ=3 MEDMNIST_COMP=16"

run () {  # tag  env...
  local tag=$1; shift
  for ds in organmnist3d nodulemnist3d adrenalmnist3d fracturemnist3d vesselmnist3d synapsemnist3d; do
    npz=$D/$ds.npz
    [ "$ds" = organmnist3d ] && npz=/agent-home/medmnist/organmnist3d.npz
    echo "--- $tag $ds $(date -u +%H:%M:%S)"
    env $COMMON "$@" MEDMNIST_NPZ="$npz" MEDMNIST_N=25 MEDMNIST_SEEDS=5 \
        MEDMNIST_OUT="$OUT/$tag/$ds" $PY -u medmnist_sweep.py 2>&1 \
      | grep -vE "UserWarning|self.transformer"
  done
}

run cross4  MEDMNIST_BASIS=cross MEDMNIST_CROSS=4
run cross16 MEDMNIST_BASIS=cross MEDMNIST_CROSS=16
run square  MEDMNIST_BASIS=square
run rbf     MEDMNIST_BASIS=rbf
run triple  MEDMNIST_BASIS=triple
echo "=== all done $(date -u +%H:%M:%S)"
