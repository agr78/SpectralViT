#!/usr/bin/env bash
# Live GPU usage by user, cluster-wide. squeue sees everyone's jobs even though sacct -a
# is silently filtered to your own, so this is the only route to real contention figures.
# Pass a partition to narrow: whoisusing.sh rtx8000
P=${1:-}
F=(-a -h -t RUNNING -O "username:20,partition:16,tres-alloc:100")
[ -n "$P" ] && F+=(-p "$P")

echo "== GPUs in use by user${P:+ on $P} =="
squeue "${F[@]}" | gawk '{ n=0; if (match($0, /gres\/gpu=[0-9]+/)) n = substr($0, RSTART+9, RLENGTH-9)
    g[$1] += n; if (n > 0) j[$1]++ }   # count GPU jobs only; CPU jobs would inflate it
  END { for (u in g) if (g[u] > 0) printf "%-16s %4d GPUs  %4d jobs\n", u, g[u], j[u] }' \
 | sort -k2 -rn
echo
echo "== pending for Resources or Priority, by user (genuine queueing) =="
squeue -a -h -t PENDING -O "username:20,partition:16,reason:24" \
 | gawk '$3 ~ /Resources|Priority/ { c[$1]++ } END { for (u in c) printf "%-16s %4d\n", u, c[u] }' \
 | sort -k2 -rn
echo "  (empty means nobody is actually blocked)"
