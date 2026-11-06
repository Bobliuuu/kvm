#!/usr/bin/env bash
# One-window config (live_cap >= tasks): trace + hop/timeline analysis on the cached baseline binary.
cd /src; D=results/2026-11-06-admission; . $D/jobs/common.sh; mkdir -p /tmp/w
genbundles; build_cached base; wait
for c in mlp xfmr; do
  if [ $c = mlp ]; then A="--live-cap 32768 --ring-words 2097152"; else A="--live-cap 65536 --ring-words 4194304"; fi
  /tmp/ke_base $A --trace-csv /tmp/w/$c.csv /tmp/w/t_$c/bundle 2>&1 | grep "kernel done"
  python3 -I $D/hopan.py /tmp/w/t_$c/bundle /tmp/w/$c.csv | tee $OUT/hopan_$c.txt
  python3 -I $D/timeline.py /tmp/w/$c.csv | tee $OUT/timeline_$c.txt
  gzip -c /tmp/w/$c.csv > $OUT/${c}_trace.csv.gz
done
