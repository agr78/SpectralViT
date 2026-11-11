#!/bin/bash
# Bases that carry more than a marginal band power. Reference is dct_energy (+0.031).
set -u
OUTROOT=/workspace/results/medmnist_bands
PY=/opt/dpeekenv/bin/python
export MEDMNIST_N=25 MEDMNIST_SEEDS=5
for b in energy_gram dct_scatter scatter dct_oriented oriented dct_energy_scatter; do
  for npz in /workspace/data/medmnist/*.npz /agent-home/medmnist/organmnist3d.npz; do
    ds=$(basename "$npz" .npz); out="$OUTROOT/$b/$ds"
    [ -f "$out/medmnist_perseed.csv" ] && continue
    mkdir -p "$out"; echo "--- $b $ds $(date +%H:%M:%S)"
    MEDMNIST_BASIS=$b MEDMNIST_NPZ="$npz" MEDMNIST_OUT="$out" \
      $PY medmnist_sweep.py 2>&1 | grep -vE "Warning|warn|self.transformer"
  done
done
echo "=== bands done $(date +%H:%M:%S)"
