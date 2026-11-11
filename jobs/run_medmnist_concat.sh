#!/bin/bash
# The union arm at n=25: pca16 and the 216-coefficient block DCT concatenated, 232
# features. Tests whether combining removes the need to select, given the oracle ceiling
# over choosing between them is only +0.029.
set -u
PY=/opt/dpeekenv/bin/python
D=/workspace/data/medmnist
for ds in organmnist3d nodulemnist3d adrenalmnist3d fracturemnist3d vesselmnist3d synapsemnist3d; do
  npz=$D/$ds.npz
  [ "$ds" = organmnist3d ] && npz=/agent-home/medmnist/organmnist3d.npz
  echo "--- $ds $(date -u +%H:%M:%S)"
  MEDMNIST_BASIS=concat MEDMNIST_BLOCK=14 MEDMNIST_STRIDE=14 MEDMNIST_FREQ=3 \
    MEDMNIST_COMP=16 MEDMNIST_NPZ="$npz" MEDMNIST_N=25 MEDMNIST_SEEDS=5 \
    MEDMNIST_OUT=/workspace/results/medmnist_n25/concat/$ds $PY -u medmnist_sweep.py 2>&1 \
    | grep -vE "UserWarning|self.transformer"
done
echo "=== all done $(date -u +%H:%M:%S)"
