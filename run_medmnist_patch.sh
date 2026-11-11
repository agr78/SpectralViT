#!/bin/bash
# Give the spatial baseline the same per-dataset capacity choice the spectral arm gets.
# Patch side must divide 28. The spectral arm is pinned to a cheap basis here because only
# the spat_* columns are read; the spatial predictions are cached per (dataset,n,seed,patch).
set -u
MODE=${1:-test}
PY=/opt/dpeekenv/bin/python
PAR=${PAR:-4}
if [ "$MODE" = cv ]; then
  OUTROOT=/workspace/results/medmnist_patchcv; SCRIPT=medmnist_cv.py; MARK=medmnist_oof.csv
  EXTRA="MEDMNIST_CV_ARM=spatial"
else
  OUTROOT=/workspace/results/medmnist_patch;   SCRIPT=medmnist_sweep.py; MARK=medmnist_perseed.csv
  EXTRA="MEDMNIST_CV_ARM=spectral"
fi
NPZS="/workspace/data/medmnist/adrenalmnist3d.npz /workspace/data/medmnist/fracturemnist3d.npz
/workspace/data/medmnist/nodulemnist3d.npz /workspace/data/medmnist/synapsemnist3d.npz
/workspace/data/medmnist/vesselmnist3d.npz /agent-home/medmnist/organmnist3d.npz"
echo "=== $MODE start $(date +%H:%M:%S)"
cell() {
  local p=$1 npz=$2 ds=$3
  local out="$OUTROOT/patch$p/$ds"
  mkdir -p "$out"
  if env $EXTRA MEDMNIST_PATCH=$p MEDMNIST_BASIS=pca MEDMNIST_COMP=4 \
         MEDMNIST_ARM="patch$p" MEDMNIST_NPZ="$npz" MEDMNIST_OUT="$out" \
         MEDMNIST_N=25 MEDMNIST_SEEDS=5 MEDMNIST_CVK=5 \
         "$PY" "$SCRIPT" > "$out/log.txt" 2>&1; then
    echo "ok   patch$p $ds"
  else
    echo "FAIL patch$p $ds"
  fi
}
for p in 2 4 7 14 28; do
  for npz in $NPZS; do
    ds=$(basename "$npz" .npz)
    [ -f "$OUTROOT/patch$p/$ds/$MARK" ] && continue
    while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do wait -n; done
    cell "$p" "$npz" "$ds" &
  done
done
wait
echo "=== $MODE done $(date +%H:%M:%S)"
