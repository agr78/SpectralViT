#!/bin/bash
# Pool the tokens of every basis considered, then select inside the pool.
# SELK is fixed at 312 a priori -- the feature count of dct_energy, the best single arm --
# so a win cannot come from the selected pool simply being larger.
# The spatial arm is read from the shared cache, so every cell is paired with every
# previously measured arm on the same (dataset, n, seed) spatial prediction.
set -u
OUTROOT=/workspace/results/medmnist_pool
PY=/opt/dpeekenv/bin/python
export MEDMNIST_BASIS=pool MEDMNIST_N=25 MEDMNIST_SEEDS=5 MEDMNIST_SELK=312
mkdir -p "$OUTROOT"
for sel in none group mrmr stab fscore mi pcared cons cons_lite; do
  for npz in /workspace/data/medmnist/*.npz /agent-home/medmnist/organmnist3d.npz; do
    ds=$(basename "$npz" .npz)
    out="$OUTROOT/$sel/$ds"
    [ -f "$out/medmnist_perseed.csv" ] && continue
    mkdir -p "$out"
    echo "--- $sel $ds $(date +%H:%M:%S)"
    MEDMNIST_SELECT=$sel MEDMNIST_NPZ="$npz" MEDMNIST_OUT="$out" \
      $PY medmnist_sweep.py 2>&1 | grep -vE "UserWarning|self.transformer"
  done
done
echo "=== all done $(date +%H:%M:%S)"
