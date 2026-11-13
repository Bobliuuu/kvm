#!/bin/bash
cd /root/work/tree || exit 2
for rbt in "gpt-tiny /root/work/gangcnt/fgs/run-20261111T142648Z/w_tiny/bundle 2e-3" "gpt-small /root/work/gangcnt/fgs/run-20261111T142648Z/w_small/bundle 2e-3" "ho-gpt-d128 /root/work/gangcnt/fgs/run-20261111T142648Z/w_ho128/bundle 1e-2"; do
  set -- $rbt
  /root/py/bin/python3 -I /root/work/ab_bench.py --exe /root/builds/e5b215c6ad3c05c5612594e8/kvm_exec --cand-exe /root/builds/68a5d88f397bcef25685e10c/kvm_exec "--flags=--warm-ms 500" --reps 9 --gate-prefix loss. --gate-tol $3 --pool exec --out /root/work/w12x/run-20261111T155134Z/ab_$1.json --row $1 $2 $2 2>&1 | grep -vE '^rep [0-9]'
  echo "ROW $1 rc ${PIPESTATUS[0]}"
done
for wt in "w_tiny 2e-3" "w_small 2e-3" "w_ho128 1e-2"; do
  set -- $wt
  /root/py/bin/python3 -I /root/work/tree/bench/exec_vs_torch/run_gpt.py /root/work/gangcnt/fgs/run-20261111T142648Z/$1 --exe /root/builds/68a5d88f397bcef25685e10c/kvm_exec --modes eager --reps 3 --tol $2 --out /root/work/w12x/run-20261111T155134Z/gpt_$1.json > /root/work/w12x/run-20261111T155134Z/gpt_$1.log 2>&1
  echo "RUN_GPT $1 rc $?: $(grep -iE 'max|loss|tol|PASS|FAIL' /root/work/w12x/run-20261111T155134Z/gpt_$1.log | tail -2 | tr '\n' ' ')"
done
/root/py/bin/python3 -I /root/work/tools/num1_exes.py --work /root/work/ho128-golden1 --ctl-exe /root/builds/e5b215c6ad3c05c5612594e8/kvm_exec --cand-exe /root/builds/68a5d88f397bcef25685e10c/kvm_exec   --bench /root/work/referee-tree/bench/vs_pytorch --tools /root/work --rule /root/work/split --out /root/work/w12x/run-20261111T155134Z/num1-ho128.json > /root/work/w12x/run-20261111T155134Z/num1-ho128.log 2>&1
echo W12X_END
