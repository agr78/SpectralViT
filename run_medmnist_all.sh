#!/bin/bash
# Every MedMNIST3D set at the paper's sizes plus its own full training count, so each
# dataset is anchored at both ends of the data-efficiency curve rather than stopping at
# n=500. The full count differs per set, hence the explicit pairing below.
#
# Runs on whatever GPU is visible and costs minutes: 28^3 volumes, ~1k training samples.
set -u
PY=/opt/dpeekenv/bin/python
D=/workspace/data/medmnist
OUTROOT=/workspace/results/medmnist

run() {  # name npz full_n
  local name=$1 npz=$2 full=$3
  echo "=== $name (full n=$full) $(date -u +%H:%M:%S)"
  MEDMNIST_NPZ="$npz" MEDMNIST_N="25,50,100,200,500,$full" MEDMNIST_SEEDS=5 \
    MEDMNIST_OUT="$OUTROOT/$name" $PY -u medmnist_sweep.py 2>&1 \
    | grep -vE "UserWarning|self.transformer"
}

run organmnist3d   /agent-home/medmnist/organmnist3d.npz 971
run nodulemnist3d  "$D/nodulemnist3d.npz"   1158
run adrenalmnist3d "$D/adrenalmnist3d.npz"  1188
run fracturemnist3d "$D/fracturemnist3d.npz" 1027
run vesselmnist3d  "$D/vesselmnist3d.npz"   1335
run synapsemnist3d "$D/synapsemnist3d.npz"  1230
echo "=== all done $(date -u +%H:%M:%S)"
