#!/bin/bash
# Chain the 40-draw CV across sample sizes. Waits on the LOG MARKER rather than on a
# pgrep of a script name -- a pgrep pattern matches this script's own command line and
# deadlocks (that happened twice on 2026-10-22/23).
set -u
for N in 100 200; do
  prev=$1; shift || true
  while ! grep -q "=== n=$prev done" /workspace/results/draws_n$prev.log 2>/dev/null; do sleep 60; done
  echo "n=$prev finished; starting n=$N at $(date +%H:%M:%S)"
  MEDMNIST_SWEEP_N=$N bash /workspace/run_draws_n.sh > /workspace/results/draws_n$N.log 2>&1
  set -- $N
done
echo "=== queue done $(date +%H:%M:%S)"
