#!/usr/bin/env bash
# Re-run the IXI full-cohort cells under one known partition.
#
# Why: ixi_nested.py's folds() stratifies whenever the class counts allow, which they do
# at n=566, but the saved n=566 cells straddle the script's last edit (spec 09:51 and spat
# 14:23 against a 15:28 mtime), so which partition each arm used cannot be established.
# This re-runs all six arms x five seeds so the whole curve is one documented protocol.
#
# Output goes to *_strat directories, NOT over results/ixi_curve: the current cells are
# what the paper's table reports, and they stay untouched until these are complete and
# checked. Promote them deliberately afterwards, never from this script.
#
# usage: bash run_ixi_566.sh [concurrency]      # default 1
#   IXI_CACHE_PRE  volume cache prefix (default picks the local one if present)
#   IXI_ARMS_LIST  override the arm list
set -u

# Paths come from where this script lives, not a fixed prefix: the repo is /workspace in
# the container and ~/SpectralViT on cadiz and durian, and hardcoding the former made every
# redirect fail instantly on the host.
ROOT=${IXI_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
if [ -x /opt/dpeekenv/bin/python ]; then
  PY=${PY:-/opt/dpeekenv/bin/python}
else
  PY=${PY:-$HOME/miniforge3/envs/dpeekenv/bin/python}
fi
# The conda env's libstdc++ has to lead, or torch fails to import on durian (GLIBCXX_3.4.29).
export LD_LIBRARY_PATH="$(dirname "$PY")/../lib:${LD_LIBRARY_PATH:-}"
JOBS=${1:-1}
CACHE_PRE=${IXI_CACHE_PRE:-$HOME/spacelink/ixi_vols_96}
ARMS=${IXI_ARMS_LIST:-"spec spat swin unet logreg mlp"}
SEEDS=${IXI_SEEDS_LIST:-"0 1 2 3 4"}
OUTD=$ROOT/results/ixi_nested_strat
MIRD=$ROOT/results/ixi_curve_strat
LOGD=$ROOT/results/ixi566_logs

if [ ! -f "$CACHE_PRE.npy" ]; then
  echo "no volume cache at $CACHE_PRE.npy"
  echo "build it where /checkpoints is readable:  $PY ixi_cache.py [prefix]"
  exit 1
fi
if ! mkdir -p "$OUTD" "$MIRD" "$LOGD"; then
  echo "cannot create output dirs under $ROOT/results -- wrong ROOT?"; exit 1
fi

echo "root    $ROOT"
echo "python  $PY"
echo "cache   $CACHE_PRE.npy"
echo "arms    $ARMS"
echo "seeds   $SEEDS"
echo "out     $MIRD  (results/ixi_curve is left alone)"
echo "jobs    $JOBS"
"$PY" -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),
    torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

run_cell() {
  local arm=$1 seed=$2
  local log="$LOGD/${arm}_n566_s${seed}.log"
  if [ -f "$MIRD/oof_${arm}_n566_s${seed}.npz" ]; then
    echo "  skip $arm s$seed (already written)"
    return 0
  fi
  local t0=$SECONDS
  IXI_N=566 IXI_SEED="$seed" IXI_ARMS="$arm" IXI_SUFFIX="_n566_s${seed}" \
  IXI_CACHE="$CACHE_PRE" IXI_OUT="$OUTD" IXI_MIRROR="$MIRD" \
    "$PY" ixi_nested.py > "$log" 2>&1
  local rc=$? el=$((SECONDS - t0))
  if [ $rc -eq 0 ] && [ -f "$MIRD/oof_${arm}_n566_s${seed}.npz" ]; then
    echo "  done $arm s$seed  ${el}s"
  else
    echo "  FAIL $arm s$seed  rc=$rc ${el}s -- see $log"
  fi
}

n=0
for seed in $SEEDS; do
  for arm in $ARMS; do
    run_cell "$arm" "$seed" &
    n=$((n + 1))
    if [ $((n % JOBS)) -eq 0 ]; then wait; fi
  done
done
wait

echo
echo "landed $(ls "$MIRD"/oof_*_n566_s*.npz 2>/dev/null | wc -l) of 30 cells in $MIRD"
