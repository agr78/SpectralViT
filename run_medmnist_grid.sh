#!/bin/bash
# Run the basis x capacity grid, either on the MedMNIST test split (ground truth) or as
# inner cross-validation inside the 25 training volumes (the selection score).
#   ./run_medmnist_grid.sh test     -> results/medmnist_grid/<config>/<dataset>/
#   ./run_medmnist_grid.sh cv       -> results/medmnist_gridcv/<config>/<dataset>/
# Six cells at a time on the one Blackwell: the models are ~1.2M parameters at batch 128,
# so a single process leaves the GPU mostly idle, and the spatial predictions for every
# (dataset, n=25, seed) are already cached, so concurrent cells only ever read that cache.
set -u
MODE=${1:-test}
PY=/opt/dpeekenv/bin/python
PAR=${PAR:-6}
if [ "$MODE" = cv ]; then
  OUTROOT=/workspace/results/medmnist_gridcv; SCRIPT=medmnist_cv.py; MARK=medmnist_oof.csv
else
  OUTROOT=/workspace/results/medmnist_grid;   SCRIPT=medmnist_sweep.py; MARK=medmnist_perseed.csv
fi
NPZS="/workspace/data/medmnist/adrenalmnist3d.npz /workspace/data/medmnist/fracturemnist3d.npz
/workspace/data/medmnist/nodulemnist3d.npz /workspace/data/medmnist/synapsemnist3d.npz
/workspace/data/medmnist/vesselmnist3d.npz /agent-home/medmnist/organmnist3d.npz"

JOBS=$(mktemp)
while IFS=$'\t' read -r name envs; do
  for npz in $NPZS; do
    ds=$(basename "$npz" .npz)
    [ -f "$OUTROOT/$name/$ds/$MARK" ] && continue
    printf '%s\t%s\t%s\t%s\n' "$name" "$envs" "$npz" "$ds" >> "$JOBS"
  done
done < results/medmnist_grid_configs.tsv
TOT=$(wc -l < "$JOBS" 2>/dev/null || echo 0)
echo "=== $MODE: $TOT cells, $PAR at a time, start $(date +%H:%M:%S)"

cell() {
  local name=$1 envs=$2 npz=$3 ds=$4
  local out="$OUTROOT/$name/$ds"
  mkdir -p "$out"
  # envs is deliberately word-split: it is a list of KEY=VALUE assignments.
  if env $envs MEDMNIST_ARM="$name" MEDMNIST_NPZ="$npz" MEDMNIST_OUT="$out" \
         MEDMNIST_N=25 MEDMNIST_SEEDS=5 MEDMNIST_CVK=5 \
         "$PY" "$SCRIPT" > "$out/log.txt" 2>&1; then
    echo "ok   $name $ds"
  else
    echo "FAIL $name $ds  (see $out/log.txt)"
  fi
}

while IFS=$'\t' read -r name envs npz ds; do
  while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do wait -n; done
  cell "$name" "$envs" "$npz" "$ds" &
done < "$JOBS"
wait
rm -f "$JOBS"
echo "=== $MODE done $(date +%H:%M:%S)"
