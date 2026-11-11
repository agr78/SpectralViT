#!/bin/bash
# Fill in the SPECTRAL blocks of the 100-draw CV. Separate from run_draws_all.sh because
# that script is mid-execution and bash re-reads a running script by byte offset -- editing
# it corrupted a run once already. The job list lives in results/ now, not the session
# scratchpad, which is wiped when the session restarts (that is what emptied it today).
# Blocks 0-16 (seeds 105-185) are already complete for all 181 pairs.
set -u
PY=/opt/dpeekenv/bin/python
OUT=/workspace/results/medmnist_cvdraws
TSV=/workspace/results/draws_all_jobs.tsv
PAR=${PAR:-5}
echo "=== start $(date +%H:%M:%S)"
for s0 in 190 195 200; do
  while IFS=$'\t' read -r ds p cfg e npz; do
    o=$OUT/$ds/$cfg/d$s0
    [ -f "$o/medmnist_cv.csv" ] && continue
    while [ "$(jobs -rp|wc -l)" -ge "$PAR" ]; do wait -n; done
    mkdir -p "$o"
    env $e MEDMNIST_CV_SEED0=$s0 MEDMNIST_SEEDS=5 MEDMNIST_CVK=5 MEDMNIST_N=25 \
        MEDMNIST_CVSEED=0 MEDMNIST_PATCH=$p MEDMNIST_ARM=$cfg \
        MEDMNIST_NPZ=$npz MEDMNIST_OUT=$o \
        $PY medmnist_cv.py > "$o/log.txt" 2>&1 &
  done < $TSV
  echo "block seeds $s0-$((s0+4)) dispatched $(date +%H:%M:%S)"
done
wait
echo "=== done $(date +%H:%M:%S)"
