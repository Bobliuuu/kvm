#!/bin/bash
cd /root/work/tree || exit 2
for rb in "gpt-tiny /root/work/gangcnt/fgs/run-20261111T142648Z/w_tiny/bundle" "gpt-small /root/work/gangcnt/fgs/run-20261111T142648Z/w_small/bundle" "ho-gpt-d128 /root/work/gangcnt/fgs/run-20261111T142648Z/w_ho128/bundle"; do
  set -- $rb
  /root/py/bin/python3 -I /root/work/ab_bench.py --exe /root/builds/e5b215c6ad3c05c5612594e8/kvm_exec --cand-exe /root/builds/2f08f189ce49b786adfb83c4/kvm_exec "--flags=--warm-ms 500" --reps 9 --gate-prefix w. --gate-tol 0 --pool exec --out /root/work/gangcnt/entry/run-20261111T160629Z/ab_$1.json --row $1 $2 $2 2>&1 | grep -vE '^rep [0-9]'
  echo "ROW $1 rc ${PIPESTATUS[0]}"
done
/root/py/bin/python3 -I /root/work/tools/bitwise_exes.py --ctl-exe /root/builds/e5b215c6ad3c05c5612594e8/kvm_exec --cand-exe /root/builds/2f08f189ce49b786adfb83c4/kvm_exec /root/work/gangcnt/fgs/run-20261111T142648Z/w_tiny/bundle /root/work/gangcnt/fgs/run-20261111T142648Z/w_small/bundle /root/work/gangcnt/fgs/run-20261111T142648Z/w_ho128/bundle
echo "bitwise rc $?"
echo GANG_CNT_E_END
