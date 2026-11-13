#!/bin/bash
cd "/root/trees/vshare-3836127" || exit 2
"/root/py/bin/python3" -I "/root/reads/vsread/run-20261111T132656Z/report.py" "/root/reads/vsread/run-20261111T132656Z" "/root/builds/f590861a18c7331e8a759a42/kvm_exec" "/root/builds/ee481fa1490c83f35629dcda/kvm_exec" "/root/reads/ab_bench.py" || echo "REPORT FAILED"
rc_check() { name=$1; shift; env KVM_EXE="/root/builds/f590861a18c7331e8a759a42/kvm_exec" KVM_PY="/root/py/bin/python3" "$@" > "/root/reads/vsread/run-20261111T132656Z/rc_$name.log" 2>&1; rc=$?; echo "$name $rc" >> "/root/reads/vsread/run-20261111T132656Z/rc.txt"
  echo "run_check $name rc=$rc: $(grep -c '^ok: ' "/root/reads/vsread/run-20261111T132656Z/rc_$name.log") ok, $(grep -c '^FAIL' "/root/reads/vsread/run-20261111T132656Z/rc_$name.log") FAIL; $(tail -1 "/root/reads/vsread/run-20261111T132656Z/rc_$name.log")"; }
rc_check default tests/static/run_check.sh 1 4 16 107 108
rc_check h256d4 KVM_CHECK_MLP_HIDDEN=256 KVM_CHECK_MLP_DEPTH=4 KVM_CHECK_MLP_DIR=/root/reads/goldens-count1/h256d4 tests/static/run_check.sh 107 108
rc_check h128d4 KVM_CHECK_MLP_HIDDEN=128 KVM_CHECK_MLP_DEPTH=4  tests/static/run_check.sh 48 107 108
rc_check h1024d2 KVM_CHECK_MLP_HIDDEN=1024 KVM_CHECK_MLP_DIR=/root/reads/goldens-count1/h1024d2 tests/static/run_check.sh 107 108
one() { row=$1; pool=$2; b=$3; prefix=$4; tol=$5; reps=$6; out=$("/root/py/bin/python3" -I "/root/reads/ab_bench.py" --exe "/root/builds/ee481fa1490c83f35629dcda/kvm_exec" --cand-exe "/root/builds/f590861a18c7331e8a759a42/kvm_exec" "--flags=--warm-ms 500" --reps $reps         --gate-prefix $prefix --gate-tol $tol --pool $pool --out "$7" --row $row $b $b 2>&1); rc=$?
  echo "$out" | grep -vE '^rep [0-9]'; [ $rc -eq 0 ] || echo "ROW FAILED $row rc=$rc"; return $rc; }
ab() { local row=$1 pool=$2 b=$3 g; g=$(awk -v r="$row" '$1 == r {print $2, $3}' "/root/reads/vsread/run-20261111T132656Z/gates.txt"); [ -n "$g" ] || { echo "ROW FAILED $row: no gate"; return 1; }
  one $row $pool $b $g 5 "/root/reads/vsread/run-20261111T132656Z/ab_$row.json" || return
  if "/root/py/bin/python3" "/root/reads/vsread/run-20261111T132656Z/slow_any.py" "/root/reads/vsread/run-20261111T132656Z/ab_$row.json"; then echo "$row: a slow run in 5 rounds, so 4 more"; one $row $pool $b $g 4 "/root/reads/vsread/run-20261111T132656Z/ab_$row.ext.json"; fi; }
while read -r row pool b; do ab $row $pool $b; done < "/root/reads/vsread/run-20261111T132656Z/rows.txt"
date -u +%FT%TZ > "/root/reads/vsread/run-20261111T132656Z/gpu.end"
echo PR_READ_GPU_END
