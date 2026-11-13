#!/bin/bash
ab() { /root/py/bin/python3 -I /root/reads/ab_bench.py --exe /root/builds/052204dffcda37f8bbd99109/kvm_exec --cand-exe /root/builds/5d5938db2e59a4aad12e3ac6/kvm_exec "--flags=--warm-ms 500" --reps 5 "$@"; }
ab --pool mlp --gate-prefix w. --gate-tol 0 --row mlp72-r1 /root/reads/panel-count1/h72/bundle /root/reads/panel-count1/h72/bundle --row mlp72-r8 /root/reads/panel-count1/h72_r8 /root/reads/panel-count1/h72_r8    --row mlp-hetero-r8 /root/reads/panel-count1/hetero_r8 /root/reads/panel-count1/hetero_r8 --out /root/reads/form1s/run-20261111T131033Z/ab_bitwise.json > /root/reads/form1s/run-20261111T131033Z/ab_bitwise.log 2>&1
echo "rc $?" >> /root/reads/form1s/run-20261111T131033Z/ab_bitwise.log
ab --pool mlp --gate-prefix loss. --gate-tol 2e-3 --row mlp72-r32 /root/reads/panel-count1/h72_r32 /root/reads/panel-count1/h72_r32 --out /root/reads/form1s/run-20261111T131033Z/ab_r32.json > /root/reads/form1s/run-20261111T131033Z/ab_r32.log 2>&1
echo "rc $?" >> /root/reads/form1s/run-20261111T131033Z/ab_r32.log
for b in cand:/root/builds/5d5938db2e59a4aad12e3ac6/kvm_exec ctrl:/root/builds/052204dffcda37f8bbd99109/kvm_exec; do n=${b%%:*}; x=${b#*:}; $x --static /root/reads/panel-count1/h72_r32 > /dev/null 2> /root/reads/form1s/run-20261111T131033Z/report_r32_$n.log; echo "rc $?" >> /root/reads/form1s/run-20261111T131033Z/report_r32_$n.log; done
for c in "72 2 32" "72 2 8" "48 2 4"; do set -- $c
  /root/py/bin/python3 -I /root/reads/rep_check.py /root/trees/form1stack-5a90557 /root/reads/form1s/run-20261111T131033Z/rep-h$1-d$2 /root/builds/5d5938db2e59a4aad12e3ac6/kvm_exec $3 --hidden $1 --depth $2 > /root/reads/form1s/run-20261111T131033Z/rep-h$1-d$2-x$3.log 2>&1; echo "rc $?" >> /root/reads/form1s/run-20261111T131033Z/rep-h$1-d$2-x$3.log
done
(cd /root/trees/form1stack-5a90557 && KVM_EXE=/root/builds/5d5938db2e59a4aad12e3ac6/kvm_exec KVM_PY=/root/py/bin/python3 bash tests/static/run_check.sh 1 4 16 107 108 > /root/reads/form1s/run-20261111T131033Z/run_check.log 2>&1; echo "rc $?" >> /root/reads/form1s/run-20261111T131033Z/run_check.log)
date -u +%FT%TZ > /root/reads/form1s/run-20261111T131033Z/gpu.end
