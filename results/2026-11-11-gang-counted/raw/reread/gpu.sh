#!/bin/bash
cd /root/work/tree || exit 2
for rb in "mlp256-r1 /root/work/gangcnt/pr/bundles-6528ca77af9f9191/panel/h256/bundle" "ho-mlp-h576-d1-r1 /root/work/gangcnt/pr/bundles-6528ca77af9f9191/ho576/bundle"; do
  set -- $rb
  echo "== $1 $(date -u +%T) $(cat /proc/loadavg | cut -d' ' -f1-3) load; throttled periods so far $(awk '/nr_throttled/{print $2}' /sys/fs/cgroup/cpu/cpu.stat 2>/dev/null)"
  /root/py/bin/python3 -I /root/work/ab_bench.py --exe /root/builds/e5b215c6ad3c05c5612594e8/kvm_exec --cand-exe /root/builds/10a883c7bede10e9df070e9b/kvm_exec "--flags=--warm-ms 500" --reps 9 --gate-prefix w. --gate-tol 0 --pool mlp --out /root/work/gangcnt/rr/run-20261111T151036Z/ab_$1.json --row $1 $2 $2 2>&1 | grep -vE '^rep [0-9]'
  echo "ROW $1 rc ${PIPESTATUS[0]}"
done
echo "throttled periods at the end $(awk '/nr_throttled/{print $2}' /sys/fs/cgroup/cpu/cpu.stat 2>/dev/null) of $(awk '/nr_periods/{print $2}' /sys/fs/cgroup/cpu/cpu.stat 2>/dev/null)"
echo GANG_CNT_R_END
