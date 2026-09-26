#!/usr/bin/env bash
# Full metric table, every model at the capacity its own internal cross-validation
# selects. Capacities are derived, never hardcoded: hardcoding them is how the table and
# the scorecard drifted apart. Reports both operating points, raw and quantile.
set -u
PY=/opt/dpeekenv/bin/python; H="$(cd "$(dirname "$0")" && pwd)"
LOG="$H/../docs/dbs_3d/metrics2.log"; : > "$LOG"
NS=${1:-10}
while pgrep -f sc_worker.py >/dev/null; do sleep 15; done
"$PY" "$H/select_capacity.py" | while IFS=: read -r m k; do
  "$PY" -u "$H/metrics2.py" "$m" "$k" "$NS" >> "$LOG" 2>&1
done
echo "METRICS2 DONE" >> "$LOG"
