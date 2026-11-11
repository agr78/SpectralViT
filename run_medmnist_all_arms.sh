#!/bin/bash
# Every arm at n=25, with the spatial arm cached per (dataset, n, seed) so the comparison
# is genuinely paired. Supersedes results/medmnist_n25, where the spatial column carried
# up to +-0.003 of per-run noise against arm differences spanning 0.013.
set -u
PY=/opt/dpeekenv/bin/python
D=/workspace/data/medmnist
OUT=/workspace/results/medmnist_n25_paired
B14="MEDMNIST_BLOCK=14 MEDMNIST_STRIDE=14"
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
run pca16        MEDMNIST_BASIS=pca MEDMNIST_COMP=16
run pca24        MEDMNIST_BASIS=pca MEDMNIST_COMP=24
run pca_fullrank MEDMNIST_BASIS=pca MEDMNIST_COMP=0
run dct          MEDMNIST_BASIS=dct $B14 MEDMNIST_FREQ=3
run dct_ov       MEDMNIST_BASIS=dct MEDMNIST_BLOCK=14 MEDMNIST_STRIDE=7 MEDMNIST_FREQ=2
run blockpca     MEDMNIST_BASIS=blockpca $B14 MEDMNIST_COMP=28
run blockpca_ov  MEDMNIST_BASIS=blockpca MEDMNIST_BLOCK=14 MEDMNIST_STRIDE=7 MEDMNIST_COMP=8
run concat       MEDMNIST_BASIS=concat $B14 MEDMNIST_FREQ=3 MEDMNIST_COMP=16
run concat_rank  MEDMNIST_BASIS=concat $B14 MEDMNIST_FREQ=3 MEDMNIST_COMP=16 MEDMNIST_RANK=1
run cross4       MEDMNIST_BASIS=cross $B14 MEDMNIST_FREQ=3 MEDMNIST_COMP=16 MEDMNIST_CROSS=4
run cross8       MEDMNIST_BASIS=cross $B14 MEDMNIST_FREQ=3 MEDMNIST_COMP=16 MEDMNIST_CROSS=8
run cross16      MEDMNIST_BASIS=cross $B14 MEDMNIST_FREQ=3 MEDMNIST_COMP=16 MEDMNIST_CROSS=16
run square       MEDMNIST_BASIS=square $B14 MEDMNIST_FREQ=3 MEDMNIST_COMP=16
run rbf          MEDMNIST_BASIS=rbf $B14 MEDMNIST_FREQ=3 MEDMNIST_COMP=16
run triple       MEDMNIST_BASIS=triple $B14 MEDMNIST_FREQ=3 MEDMNIST_COMP=16
echo "=== all done $(date -u +%H:%M:%S)"
