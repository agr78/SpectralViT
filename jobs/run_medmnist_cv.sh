#!/bin/bash
# Cross-validate every basis inside the 25 training volumes, per dataset and seed.
# Selection score is the pooled out-of-fold AUC over all 25; the MedMNIST test split is
# never read. The env per arm reproduces the reporting runs exactly.
set -u
OUTROOT=/workspace/results/medmnist_cv
PY=/opt/dpeekenv/bin/python
export MEDMNIST_N=25 MEDMNIST_SEEDS=5 MEDMNIST_CVK=5
declare -A ARMS=(
  [pca16]="MEDMNIST_BASIS=pca MEDMNIST_COMP=16"
  [pca24]="MEDMNIST_BASIS=pca MEDMNIST_COMP=24"
  [pca_fullrank]="MEDMNIST_BASIS=pca MEDMNIST_COMP=0"
  [dct]="MEDMNIST_BASIS=dct"
  [dct_ov]="MEDMNIST_BASIS=dct MEDMNIST_BLOCK=14 MEDMNIST_STRIDE=7 MEDMNIST_FREQ=2"
  [blockpca]="MEDMNIST_BASIS=blockpca MEDMNIST_BLOCK=14 MEDMNIST_STRIDE=14 MEDMNIST_COMP=28"
  [blockpca_ov]="MEDMNIST_BASIS=blockpca MEDMNIST_BLOCK=14 MEDMNIST_STRIDE=7 MEDMNIST_COMP=8"
  [energy]="MEDMNIST_BASIS=energy"
  [concat]="MEDMNIST_BASIS=concat"
  [concat_rank]="MEDMNIST_BASIS=concat MEDMNIST_RANK=1"
  [triple]="MEDMNIST_BASIS=triple"
  [cross4]="MEDMNIST_BASIS=cross MEDMNIST_CROSS=4"
  [cross8]="MEDMNIST_BASIS=cross MEDMNIST_CROSS=8"
  [cross16]="MEDMNIST_BASIS=cross MEDMNIST_CROSS=16"
  [rbf]="MEDMNIST_BASIS=rbf"
  [square]="MEDMNIST_BASIS=square"
  [pca_energy]="MEDMNIST_BASIS=pca_energy"
  [pca_sq_energy]="MEDMNIST_BASIS=pca_sq_energy"
  [energy_cat]="MEDMNIST_BASIS=energy_cat"
  [dct_energy]="MEDMNIST_BASIS=dct_energy"
  [scatter]="MEDMNIST_BASIS=scatter"
  [dct_scatter]="MEDMNIST_BASIS=dct_scatter"
  [oriented]="MEDMNIST_BASIS=oriented"
  [dct_oriented]="MEDMNIST_BASIS=dct_oriented"
  [energy_gram]="MEDMNIST_BASIS=energy_gram"
  [dct_energy_scatter]="MEDMNIST_BASIS=dct_energy_scatter"
  [pool_none]="MEDMNIST_BASIS=pool MEDMNIST_SELECT=none"
  [pool_pcared]="MEDMNIST_BASIS=pool MEDMNIST_SELECT=pcared MEDMNIST_SELK=312"
)
for arm in "${!ARMS[@]}"; do
  for npz in /workspace/data/medmnist/*.npz /agent-home/medmnist/organmnist3d.npz; do
    ds=$(basename "$npz" .npz); out="$OUTROOT/$arm/$ds"
    [ -f "$out/medmnist_oof.csv" ] && continue
    mkdir -p "$out"
    echo "--- $arm $ds $(date +%H:%M:%S)"
    env ${ARMS[$arm]} MEDMNIST_ARM="$arm" MEDMNIST_NPZ="$npz" MEDMNIST_OUT="$out" \
      $PY medmnist_cv.py 2>&1 | grep -E "pooled OOF|Traceback|Error|no inner split"
  done
done
echo "=== cv done $(date +%H:%M:%S)"
