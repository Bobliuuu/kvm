#!/bin/bash
/root/py/bin/python3 -I /root/reads/ab_bench.py --exe /root/builds/ee481fa1490c83f35629dcda/kvm_exec --cand-exe /root/builds/f590861a18c7331e8a759a42/kvm_exec "--flags=--warm-ms 500" --reps 5 --gate-prefix loss. --gate-tol 2e-3 --pool mlp   --out /root/reads/form1m2/run-20261111T134055Z/vsread_ab_mlp72-r32.json --row mlp72-r32 /root/reads/panel-count1/h72_r32 /root/reads/panel-count1/h72_r32 > /root/reads/form1m2/run-20261111T134055Z/vsread_ab_mlp72-r32.log 2>&1; echo "rc $?" >> /root/reads/form1m2/run-20261111T134055Z/vsread_ab_mlp72-r32.log
for n in 8 32; do /root/py/bin/python3 -I /root/reads/rep_check.py /root/trees/form1-633499a /root/reads/form1m2/run-20261111T134055Z/rep-h48-d2 /root/builds/5d5938db2e59a4aad12e3ac6/kvm_exec $n --hidden 48 --depth 2 > /root/reads/form1m2/run-20261111T134055Z/fix-rep-h48-d2-x$n.log 2>&1; echo "rc $?" >> /root/reads/form1m2/run-20261111T134055Z/fix-rep-h48-d2-x$n.log; done
# The virtual shares' own effect, against the head they are stacked on (the granule fix, 5d5938db), in one hold: four single runs (S = Sv,
# whose cost the bed put at 0-3%) and the two multi-run draws (the gangs' turns against the shared waves). Every row bitwise: virtual
# shares keep each run's arithmetic.
O=/root/reads/form1m2/run-20261111T134055Z; V=/root/reads/vsread/bundles-4cec0fbd9b692803; P=/root/reads/panel-count1
for r in "mlp256-r1 $P/h256/bundle" "mlp1024-r1 $P/h1024/bundle" "ho-mlp-h576-d1-r1 $V/ho576/bundle" "ho-mlp-h128-d4-r1 $V/ho128/bundle" "ho-mlp-h200-d1-r16 $V/ho200_r16" "ho-mlp-h224-d2-r8 $V/ho224_r8"; do set -- $r
  /root/py/bin/python3 -I /root/reads/ab_bench.py --exe /root/builds/5d5938db2e59a4aad12e3ac6/kvm_exec --cand-exe /root/builds/f590861a18c7331e8a759a42/kvm_exec "--flags=--warm-ms 500" --reps 5 \
    --gate-prefix w. --gate-tol 0 --pool mlp --out "$O/vsbase_ab_$1.json" --row "$1" "$2" "$2" > "$O/vsbase_ab_$1.log" 2>&1; echo "rc $?" >> "$O/vsbase_ab_$1.log"
done
date -u +%FT%TZ > /root/reads/form1m2/run-20261111T134055Z/gpu.end
