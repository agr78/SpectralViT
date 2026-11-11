#!/bin/bash
set -u
OUTROOT=/workspace/results/medmnist_pool
PY=/opt/dpeekenv/bin/python
export MEDMNIST_BASIS=pool MEDMNIST_N=25 MEDMNIST_SEEDS=5 MEDMNIST_SELK=312
for sel in cons cons_lite; do
  for npz in /workspace/data/medmnist/*.npz /agent-home/medmnist/organmnist3d.npz; do
    ds=$(basename "$npz" .npz); out="$OUTROOT/$sel/$ds"
    [ -f "$out/medmnist_perseed.csv" ] && continue
    mkdir -p "$out"; echo "--- $sel $ds $(date +%H:%M:%S)"
    MEDMNIST_SELECT=$sel MEDMNIST_NPZ="$npz" MEDMNIST_OUT="$out" \
      $PY medmnist_sweep.py 2>&1 | grep -vE "UserWarning|self.transformer"
  done
done
echo "=== cons done $(date +%H:%M:%S)"
