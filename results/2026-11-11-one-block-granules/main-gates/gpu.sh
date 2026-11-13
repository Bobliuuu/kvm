#!/bin/bash
cd /root/trees/main-4bbdaab
for s in 72:2 48:2; do IFS=: read -r H D <<< "$s"; n=h$H-d$D
  /root/py/bin/python3 -I bench/exec_vs_torch/run_mlp.py /root/reads/form1m/run-20261111T133103Z/$n --exe /root/builds/ee481fa1490c83f35629dcda/kvm_exec --hidden $H --depth $D --steps 50 --runs 1 --reps 1 --no-golden --skip seq_graph,multi_graph,batched,batched_compile,eager,compile,compile_graph --out /root/reads/form1m/run-20261111T133103Z/$n.json > /root/reads/form1m/run-20261111T133103Z/$n.log 2>&1
  echo "rc $?" >> /root/reads/form1m/run-20261111T133103Z/$n.log
done
for c in "72 2 8" "72 2 32" "48 2 8" "48 2 32"; do set -- $c
  /root/py/bin/python3 -I /root/reads/rep_check.py /root/trees/main-4bbdaab /root/reads/form1m/run-20261111T133103Z/rep-h$1-d$2 /root/builds/ee481fa1490c83f35629dcda/kvm_exec $3 --hidden $1 --depth $2 > /root/reads/form1m/run-20261111T133103Z/rep-h$1-d$2-x$3.log 2>&1; echo "rc $?" >> /root/reads/form1m/run-20261111T133103Z/rep-h$1-d$2-x$3.log
done
# each binary's fusion report on the one-block rows: main, the stack (#33's code) and this PR's head (the fix)
P=/root/reads/panel-count1; O=/root/reads/form1m/run-20261111T133103Z
for r in h72/bundle:r1 h72_r8:r8 hetero_r8:hetero-r8 h72_r32:r32; do b=${r%%:*}; n=${r#*:}
  for a in main:/root/builds/ee481fa1490c83f35629dcda/kvm_exec stack:/root/builds/052204dffcda37f8bbd99109/kvm_exec fix:/root/builds/5d5938db2e59a4aad12e3ac6/kvm_exec; do
    arm=${a%%:*}; x=${a#*:}; "$x" --static "$P/$b" > /dev/null 2> "$O/report_${n}_$arm.log"; echo "rc $?" >> "$O/report_${n}_$arm.log"
  done
done
date -u +%FT%TZ > /root/reads/form1m/run-20261111T133103Z/gpu.end
