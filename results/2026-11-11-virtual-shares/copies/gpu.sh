#!/bin/bash
cd /root/trees/vshare-3836127
for s in 256:4 224:2 200:1 576:1 128:4; do IFS=: read -r H D <<< "$s"; n=h$H-d$D
  /root/py/bin/python3 -I bench/exec_vs_torch/run_mlp.py /root/reads/vsread/form1-20261111T133749Z/$n --exe /root/builds/f590861a18c7331e8a759a42/kvm_exec --hidden $H --depth $D --steps 50 --runs 1 --reps 1 --no-golden --skip seq_graph,multi_graph,batched,batched_compile,eager,compile,compile_graph --out /root/reads/vsread/form1-20261111T133749Z/$n.json > /root/reads/vsread/form1-20261111T133749Z/$n.log 2>&1
  echo "rc $?" >> /root/reads/vsread/form1-20261111T133749Z/$n.log
done
for c in "200 1 16" "224 2 8" "224 2 4" "256 4 32" "72 2 32"; do set -- $c
  /root/py/bin/python3 -I /root/reads/rep_check.py /root/trees/vshare-3836127 /root/reads/vsread/form1-20261111T133749Z/rep-h$1-d$2 /root/builds/f590861a18c7331e8a759a42/kvm_exec $3 --hidden $1 --depth $2 > /root/reads/vsread/form1-20261111T133749Z/rep-h$1-d$2-x$3.log 2>&1; echo "rc $?" >> /root/reads/vsread/form1-20261111T133749Z/rep-h$1-d$2-x$3.log
done
for b in ho576 ho128; do
  /root/builds/f590861a18c7331e8a759a42/kvm_exec --static --dump-buffers /root/reads/gangfix/bundles-ddf9d6b225d9c667/$b/bundle > /root/reads/vsread/form1-20261111T133749Z/$b.new.out 2> /root/reads/vsread/form1-20261111T133749Z/$b.new.log; echo "rc $?" >> /root/reads/vsread/form1-20261111T133749Z/$b.new.log
  /root/builds/052204dffcda37f8bbd99109/kvm_exec --static --dump-buffers /root/reads/gangfix/bundles-ddf9d6b225d9c667/$b/bundle > /root/reads/vsread/form1-20261111T133749Z/$b.old.out 2> /root/reads/vsread/form1-20261111T133749Z/$b.old.log; echo "rc $?" >> /root/reads/vsread/form1-20261111T133749Z/$b.old.log
done
/root/builds/f590861a18c7331e8a759a42/kvm_exec --static --dump-buffers /root/reads/panel-count1/h256/bundle > /root/reads/vsread/form1-20261111T133749Z/mlp256.new.out 2> /root/reads/vsread/form1-20261111T133749Z/mlp256.new.log; echo "rc $?" >> /root/reads/vsread/form1-20261111T133749Z/mlp256.new.log
/root/builds/052204dffcda37f8bbd99109/kvm_exec --static --dump-buffers /root/reads/panel-count1/h256/bundle > /root/reads/vsread/form1-20261111T133749Z/mlp256.old.out 2> /root/reads/vsread/form1-20261111T133749Z/mlp256.old.log; echo "rc $?" >> /root/reads/vsread/form1-20261111T133749Z/mlp256.old.log
date -u +%FT%TZ > /root/reads/vsread/form1-20261111T133749Z/gpu.end
