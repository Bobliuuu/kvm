#!/bin/bash
cd /root/trees/ceblocks-71af38a && env KVM_EXE=/root/builds/e8b69fa3e9ebd2b927d38f68/kvm_exec KVM_PY=/root/py/bin/python3 bash tests/static/run_check.sh 1 4 16 107 108 > /root/reads/cefix/run-20261111T144557Z/run_check.log 2>&1; echo "rc $?" >> /root/reads/cefix/run-20261111T144557Z/run_check.log
for x in fix:/root/builds/e8b69fa3e9ebd2b927d38f68/kvm_exec main:/root/builds/ee481fa1490c83f35629dcda/kvm_exec; do a=${x%%:*}; e=${x#*:}
  for b in main_h72d2 pr39_h72d2; do $e --static --dump-buffers /root/reads/cefix/bundles/$b/bundle > /root/reads/cefix/run-20261111T144557Z/dump_${b}_$a.out 2> /root/reads/cefix/run-20261111T144557Z/dump_${b}_$a.log; echo "rc $?" >> /root/reads/cefix/run-20261111T144557Z/dump_${b}_$a.log; done
done
one() { /root/py/bin/python3 -I /root/reads/ab_bench.py --exe /root/builds/ee481fa1490c83f35629dcda/kvm_exec --cand-exe /root/builds/e8b69fa3e9ebd2b927d38f68/kvm_exec "--flags=--warm-ms 500" --reps 5 --gate-prefix $3 --gate-tol $4 --pool mlp --out /root/reads/cefix/run-20261111T144557Z/ab_$1.json --row $1 $2 $2 > /root/reads/cefix/run-20261111T144557Z/ab_$1.log 2>&1; echo "rc $?" >> /root/reads/cefix/run-20261111T144557Z/ab_$1.log; }
one pr39-mlp72-r1 /root/reads/m39/pr39/panel/h72/bundle loss. 2e-3
one pr39-mlp72-r8 /root/reads/m39/pr39/panel/h72_r8 loss. 2e-3
one pr39-mlp72-r32 /root/reads/m39/pr39/panel/h72_r32 loss. 2e-3
one pr39-mlp-hetero-r8 /root/reads/m39/pr39/panel/hetero_r8 loss. 2e-3
one main-mlp72-r1 /root/reads/m39/main/panel/h72/bundle w. 0
date -u +%FT%TZ > /root/reads/cefix/run-20261111T144557Z/gpu.end
