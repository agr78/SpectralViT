#!/bin/bash
# MedMNIST3D sample-size sweep with re-selection at every n on a fixed capacity ladder.
#
# Design: the basis FAMILY comes from the training-split property rule and is fixed; within
# it the config is re-selected at each n from a 5-6 rung capacity ladder, and the spatial
# patch is re-selected over {4,7,14} by the same CV. The number of CV draws shrinks as n
# grows so that draws x n ~ 1000 at every rung -- the selector's total sample budget is held
# constant, which is the right scaling because inner-CV noise falls as n rises (at n=25 each
# of 5 folds validates on 5 volumes; at n=800 on 160).
#
# n:      25  50  100  200  400  800
# draws:  40  20   10    5    2    1
#
# Draws are seeds 105+, always disjoint from the reporting draws 5-104. Top rung is 800, not
# 1000, because organ has only 971 training volumes.
# Job list is in results/, not the session scratchpad (that is wiped on restart).
# Do not edit while running: bash re-reads a running script by byte offset.
set -u
PY=/opt/dpeekenv/bin/python
TSV=/workspace/results/ladder_jobs.tsv
ROOT=${LADDER_ROOT:-/workspace/results/ladder_cv}
PAR=${PAR:-6}
sem(){ while [ "$(jobs -rp|wc -l)" -ge "$PAR" ]; do wait -n; done; }
run_n(){
  local N=$1 DRAWS=$2 OUT=$ROOT/n$1
  # 5 draws per block; a final short block carries the remainder (DRAWS<5 -> one short block)
  local full=$((DRAWS/5)) rem=$((DRAWS%5)) i s0
  echo "--- n=$N, $DRAWS draws ($full full blocks + remainder $rem) $(date +%H:%M:%S)"
  for ((i=0;i<full;i++)); do
    s0=$((105+i*5))
    while IFS=$'\t' read -r ds p cfg e npz; do
      local o=$OUT/$ds/$cfg/d$s0
      [ -f "$o/medmnist_cv.csv" ] && continue
      sem; mkdir -p "$o"
      env $e MEDMNIST_CV_SEED0=$s0 MEDMNIST_SEEDS=5 MEDMNIST_CVK=5 MEDMNIST_N=$N \
          MEDMNIST_CVSEED=0 MEDMNIST_PATCH=$p MEDMNIST_ARM=$cfg \
          MEDMNIST_NPZ=$npz MEDMNIST_OUT=$o $PY medmnist_cv.py > "$o/log.txt" 2>&1 &
    done < $TSV
    for ds in organmnist3d nodulemnist3d adrenalmnist3d fracturemnist3d vesselmnist3d synapsemnist3d; do
      local npz=/workspace/data/medmnist/$ds.npz
      [ "$ds" = organmnist3d ] && npz=/agent-home/medmnist/organmnist3d.npz
      for p in 4 7 14; do
        local o=$OUT/$ds/spatial_p$p/d$s0
        [ -f "$o/medmnist_cv.csv" ] && continue
        sem; mkdir -p "$o"
        env MEDMNIST_CV_ARM=spatial MEDMNIST_CV_SEED0=$s0 MEDMNIST_SEEDS=5 MEDMNIST_CVK=5 \
            MEDMNIST_N=$N MEDMNIST_CVSEED=0 MEDMNIST_PATCH=$p MEDMNIST_ARM=spatial_p$p \
            MEDMNIST_NPZ=$npz MEDMNIST_OUT=$o $PY medmnist_cv.py > "$o/log.txt" 2>&1 &
      done
    done
  done
  if [ $rem -gt 0 ]; then
    s0=$((105+full*5))
    while IFS=$'\t' read -r ds p cfg e npz; do
      local o=$OUT/$ds/$cfg/r$s0
      [ -f "$o/medmnist_cv.csv" ] && continue
      sem; mkdir -p "$o"
      env $e MEDMNIST_CV_SEED0=$s0 MEDMNIST_SEEDS=$rem MEDMNIST_CVK=5 MEDMNIST_N=$N \
          MEDMNIST_CVSEED=0 MEDMNIST_PATCH=$p MEDMNIST_ARM=$cfg \
          MEDMNIST_NPZ=$npz MEDMNIST_OUT=$o $PY medmnist_cv.py > "$o/log.txt" 2>&1 &
    done < $TSV
    for ds in organmnist3d nodulemnist3d adrenalmnist3d fracturemnist3d vesselmnist3d synapsemnist3d; do
      local npz=/workspace/data/medmnist/$ds.npz
      [ "$ds" = organmnist3d ] && npz=/agent-home/medmnist/organmnist3d.npz
      for p in 4 7 14; do
        local o=$OUT/$ds/spatial_p$p/r$s0
        [ -f "$o/medmnist_cv.csv" ] && continue
        sem; mkdir -p "$o"
        env MEDMNIST_CV_ARM=spatial MEDMNIST_CV_SEED0=$s0 MEDMNIST_SEEDS=$rem MEDMNIST_CVK=5 \
            MEDMNIST_N=$N MEDMNIST_CVSEED=0 MEDMNIST_PATCH=$p MEDMNIST_ARM=spatial_p$p \
            MEDMNIST_NPZ=$npz MEDMNIST_OUT=$o $PY medmnist_cv.py > "$o/log.txt" 2>&1 &
      done
    done
  fi
  wait
  echo "--- n=$N done $(date +%H:%M:%S)"
}
echo "=== ladder sweep start $(date +%H:%M:%S)"
run_n 25 40
run_n 50 20
run_n 100 10
run_n 200 5
run_n 400 2
run_n 800 1
echo "=== ladder sweep done $(date +%H:%M:%S)"
