#!/usr/bin/env bash
# When is the cluster actually busy? Mean queue wait by hour and weekday over the GPU
# partitions. Reported twice: everyone, and everyone EXCEPT you.
#
# The split matters. Bulk array submissions sit on JobArrayTaskLimit, which sacct counts
# as queue wait, so your own throttled arrays can make a quiet hour look congested -- a
# 36 h mean on rtx8000 came out that way while a job submitted there started in 2 minutes.
# Only the "others" column says whether anyone else was blocked.
DAYS=${1:-14}
ME=$(whoami)

report () {   # $1 = label, $2 = awk guard on user field
  echo "== $1 =="
  sacct -a -X -S "now-${DAYS}days" -P -n -o Submit,Start,Partition,User \
   | gawk -F'|' -v me="$ME" -v mode="$2" '
       $2 ~ /^2/ && $3 ~ /rtx|dgx|lcna|pubgpu/ {
         if (mode == "others" && $4 == me) next
         h = substr($1,12,2); a=$1; b=$2; gsub(/[-T:]/," ",a); gsub(/[-T:]/," ",b)
         w = mktime(b) - mktime(a); if (w >= 0) { s[h]+=w; c[h]++ } }
       END { for (h in c) printf "%s:00  %5d jobs  %7.2f h\n", h, c[h], s[h]/c[h]/3600 }' \
   | sort
  echo
}

report "all users, by hour (last $DAYS days)" all
report "EXCLUDING $ME, by hour" others

echo "== excluding $ME, by weekday =="
sacct -a -X -S "now-${DAYS}days" -P -n -o Submit,Start,Partition,User \
 | gawk -F'|' -v me="$ME" '
     $2 ~ /^2/ && $3 ~ /rtx|dgx|lcna|pubgpu/ && $4 != me {
       a=$1; b=$2; gsub(/[-T:]/," ",a); gsub(/[-T:]/," ",b)
       d = strftime("%a", mktime(a)); w = mktime(b) - mktime(a)
       if (w >= 0) { s[d]+=w; c[d]++ } }
     END { for (d in c) printf "%-4s %5d jobs  %7.2f h\n", d, c[d], s[d]/c[d]/3600 }' \
 | sort -k3 -rn
