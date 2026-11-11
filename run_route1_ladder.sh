#!/bin/bash
# Route 1 regenerated under the curve's protocol: 40-draw CV at n=25 over all FOUR family
# ladders, so the per-seed family probe can be applied post hoc exactly as before while the
# config and patch come from the same selector the curve uses. 22 configs x 6 datasets.
set -u
PY=/opt/dpeekenv/bin/python
OUT=/workspace/results/route1_cv
TSV=/workspace/results/ladder4_jobs.tsv
PAR=${PAR:-6}
sem(){ while [ "$(jobs -rp|wc -l)" -ge "$PAR" ]; do wait -n; done; }
echo "=== start $(date +%H:%M:%S)"
for blk in $(seq 0 7); do
  s0=$((105 + blk*5))
  while IFS=$'\t' read -r ds p fam cfg e npz; do
    o=$OUT/$ds/$cfg/d$s0
    [ -f "$o/medmnist_cv.csv" ] && continue
    sem; mkdir -p "$o"
    env $e MEDMNIST_CV_SEED0=$s0 MEDMNIST_SEEDS=5 MEDMNIST_CVK=5 MEDMNIST_N=25 \
        MEDMNIST_CVSEED=0 MEDMNIST_PATCH=$p MEDMNIST_ARM=$cfg \
        MEDMNIST_NPZ=$npz MEDMNIST_OUT=$o $PY medmnist_cv.py > "$o/log.txt" 2>&1 &
  done < $TSV
  echo "block $blk (draws $s0-$((s0+4))) dispatched $(date +%H:%M:%S)"
done
wait
echo "=== done $(date +%H:%M:%S)"
