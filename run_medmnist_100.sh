#!/bin/bash
# Two-stage evaluation. Stage 1 (done): the basis grid and the patch grid were selected on
# seeds 0-4. Stage 2 (here): the best config of each FAMILY, plus the selected patch, are
# evaluated on seeds 5-104 -- draws disjoint from the ones selection saw. The per-seed
# family probe is applied afterwards, so the reported number is the whole procedure's
# performance on fresh data, not one frozen basis.
set -u
PY=/opt/dpeekenv/bin/python
OUT=/workspace/results/medmnist_100
PAR=${PAR:-5}
run(){ local ds=$1 p=$2 fam=$3 cfg=$4 e=$5
  local npz=/workspace/data/medmnist/$ds.npz
  [ "$ds" = organmnist3d ] && npz=/agent-home/medmnist/organmnist3d.npz
  local o=$OUT/$cfg/$ds; [ -f "$o/medmnist_perseed.csv" ] && { echo "skip $cfg $ds"; return 0; }
  mkdir -p "$o"
  env $e MEDMNIST_PATCH=$p MEDMNIST_ARM=$cfg MEDMNIST_NPZ=$npz MEDMNIST_OUT=$o \
      MEDMNIST_N=25 MEDMNIST_SEED0=5 MEDMNIST_SEEDS=100 \
      $PY medmnist_sweep.py > "$o/log.txt" 2>&1 && echo "ok $cfg $ds" || echo "FAIL $cfg $ds"; }
echo "=== start $(date +%H:%M:%S)"
while IFS=$'\t' read -r ds p fam cfg e; do
  while [ "$(jobs -rp|wc -l)" -ge "$PAR" ]; do wait -n; done
  run "$ds" "$p" "$fam" "$cfg" "$e" &
done < /tmp/claude-6269505/-workspace/cb72acaf-48d3-4307-9277-3cbe75d63ca5/scratchpad/jobs100.tsv
wait
echo "=== done $(date +%H:%M:%S)"
