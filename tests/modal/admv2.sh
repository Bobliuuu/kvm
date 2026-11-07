#!/usr/bin/env bash
# admission-v2 A/B: executor from /src/_main (origin/main) vs this tree, same gpt_162m bundles (f32 and bf16s), interleaved.
# Run: modal run --detach tests/modal/run_suite.py --suite custom --tag NAME --cmd 'bash tests/modal/admv2.sh'
set -u
export PATH=$PATH:/usr/local/cuda/bin
O=/logs/$TAG; mkdir -p $O /w
F="-O3 -arch=sm_80 -std=c++17 -Iinclude --fmad=false -Xcompiler -ffp-contract=off -lineinfo -diag-suppress 177 ${EXTRA:-}"
(cd _main && nvcc $F kvm_exec.cu -o /w/ex_main > $O/build_main.log 2>&1; echo "main $?") &
(nvcc $F kvm_exec.cu -o /w/ex_new > $O/build_new.log 2>&1; echo "new $?") &
M='{"name":"gpt_162m","vocab":50304,"d_model":768,"layers":12,"heads":12,"seq":1024,"batch":4'
g() { python3 bench/vs_pytorch/gen.py /w/$1 --model "$M,\"precision\":\"$2\"}" --steps 1 --mode tf32 --tiling a100 > $O/gen_$1.log 2>&1; echo "gen $1 $?"; }
g f32 f32 & g b16 bf16s & wait
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv,noheader > $O/env.txt
for r in 1 2 3 4 5; do for v in main:ex_main new:ex_new; do for b in b16 f32; do
  n=${v%%:*}; e=${v##*:}
  echo "rep $r $n $b $(timeout 900 /w/$e --quiet /w/$b/bundle 2>&1 | grep -E '^(OK|FAIL):' | tail -1)" | tee -a $O/ab.txt
done; done; done
for b in b16; do /w/ex_new --quiet --trace-csv /w/t.csv /w/$b/bundle 2>&1 | tail -n 1; python3 bench/vs_pytorch/trace_summary.py /w/t.csv 1 2>&1 | head -n 20 | tee $O/trace_$b.txt; done
