#!/bin/bash
r() { local n=$1; shift; timeout 600 "$@" > "/root/reads/vs1/run-20261111T124620Z/$n.log" 2>&1; echo "rc $?" >> "/root/reads/vs1/run-20261111T124620Z/$n.log"; }
r memcheck  /usr/local/cuda/bin/compute-sanitizer --tool memcheck  --error-exitcode 9 /root/reads/vs1/run-20261111T124620Z/split_bed --hidden 96 --depth 2 --virtual 6 --shares 2 --steps 2 --check 2 --warm-ms 0
r racecheck /usr/local/cuda/bin/compute-sanitizer --tool racecheck --error-exitcode 9 /root/reads/vs1/run-20261111T124620Z/split_bed --hidden 96 --depth 2 --virtual 6 --shares 2 --steps 2 --check 2 --warm-ms 0
for s in "200 1" "224 2" "256 4"; do set -- $s
  for S in 18 9 6 4 3 2; do r eq-h$1d$2-s$S /root/reads/vs1/run-20261111T124620Z/split_bed --hidden $1 --depth $2 --virtual 18 --shares $S --steps 2 --check 3 --warm-ms 0 --dump /root/reads/vs1/run-20261111T124620Z/h$1d$2-s$S.bin; done
done
for s in "200 1 2" "200 1 18" "224 2 4" "224 2 18" "256 4 4" "256 4 18"; do set -- $s
  r t-new-h$1d$2-s$3 /root/reads/vs1/run-20261111T124620Z/split_bed --hidden $1 --depth $2 --virtual 18 --shares $3 --steps 200 --check 1
  r t-old-h$1d$2-s$3 /root/reads/vs1/run-20261111T124620Z/split_bed_old --hidden $1 --depth $2 --shares $3 --steps 200 --check 1
done
date -u +%FT%TZ > "/root/reads/vs1/run-20261111T124620Z/gpu.end"
