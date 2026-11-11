#!/bin/bash
# Wait on the running job by PID (NOT by pgrep of a script name -- that matches this very
# command line and deadlocks, which it already did once today), then resume to add blocks 8-9.
set -u
while kill -0 "$1" 2>/dev/null; do sleep 30; done
echo "primary job $1 finished $(date +%H:%M:%S); resuming for blocks 8-9"
exec bash /workspace/run_draws_all.sh
