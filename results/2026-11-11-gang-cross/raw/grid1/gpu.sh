#!/bin/bash
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,power.draw --format=csv,noheader > "/root/work/grid1/run-20261111T064816Z/gpu.txt" 2>&1 || echo "nvidia-smi not found" > "/root/work/grid1/run-20261111T064816Z/gpu.txt"
run() { local log=$1 rc; shift; "$@" > "/root/work/grid1/run-20261111T064816Z/$log.log" 2>&1; rc=$?
  echo "$log rc=$rc: $(grep -E 'check ok|CHECK FAILED|does not fit|cannot be co-resident|does not take' "/root/work/grid1/run-20261111T064816Z/$log.log" | head -1 | cut -c1-60); $(grep -oE '[0-9.]+ us a step in the kernel' "/root/work/grid1/run-20261111T064816Z/$log.log")"; }
round() { local k=$1 only order name bed args arms arm B; shift; only="$*"; order="new"; [ -n "/root/work/pod-ctrl" ] && { [ $((k % 2)) = 1 ] && order="new ctrl" || order="ctrl new"; }
  while read -r name bed args; do
    [ -n "$only" ] && ! echo " $only " | grep -q " $name " && continue
    case "$args" in *candidate-only) args=${args% candidate-only}; arms=new;; *) arms=$order;; esac
    for arm in $arms; do B="/root/work/pod-next"; [ $arm = ctrl ] && B="/root/work/pod-ctrl"; [ $arm = ctrl ] && [ -z "/root/work/pod-ctrl" ] && continue
      run ${name}_${arm}_r$k "$B/$bed" $args --steps 50 --check 3; done
  done < "/root/work/grid1/run-20261111T064816Z/rows.txt"; }
for k in 1 2 3 4 5; do round $k; done
ext=$(/root/py/bin/python3 "/root/work/grid1/run-20261111T064816Z/slow.py" "/root/work/grid1/run-20261111T064816Z" --extend)
if [ -n "$ext" ]; then echo "a slow run in 5 rounds, so 4 more for: $ext"; for k in 6 7 8 9; do round $k $ext; done; fi
date -u +%FT%TZ > "/root/work/grid1/run-20261111T064816Z/gpu.end"
echo BEDS_NEXT_GPU_END
