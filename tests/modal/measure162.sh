#!/usr/bin/env bash
# gpt_162m 1-step tape on the executor: base (tests/modal/base = origin/perf/combined's kvm_exec.cu + include/) against this tree, same bundle,
# same process-free alternation, kernel_ms of kvm_exec plus the dispatch profile of one traced run each. For tests/modal/run_suite.py --suite custom.
set -u
export KVM_ARCH=sm_80
NV="nvcc -O3 -arch=sm_80 -std=c++17 --fmad=false -Xcompiler -ffp-contract=off -lineinfo -diag-suppress 177"
python3 bench/vs_pytorch/gen.py w162 --model gpt_162m --steps 1 --mode tf32 --tiling a100 2>&1 | tail -n 2 | cut -c1-300
( cd tests/modal/base && time $NV -Iinclude kvm_exec.cu -o ../../../kvm_exec_base 2>&1 | grep -E "error" -A3 ) &
( time $NV -Iinclude kvm_exec.cu -o kvm_exec_new 2>&1 | grep -E "error" -A3 ) &
wait
ls -la kvm_exec_base kvm_exec_new
for rep in 1 2 3 4 5; do
  for v in base new; do
    echo "== $v rep $rep"; ./kvm_exec_$v w162/bundle --max-tasks 524288 --quiet 2>&1 | tail -n 1
  done
done
for v in base new; do
  echo "== traced $v"; ./kvm_exec_$v w162/bundle --max-tasks 524288 --quiet --trace-csv /tmp/t_$v.csv 2>&1 | tail -n 1
  python3 tests/gpu/trace_dispatch.py /tmp/t_$v.csv
done
echo "== defaults (no --max-tasks) base/new"
for v in base new; do ./kvm_exec_$v w162/bundle --quiet 2>&1 | tail -n 1; done
