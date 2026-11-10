#!/usr/bin/env bash
# Dev loop: sync include/ + tests/ to a private scratch dir on the GPU box, build, run under the GPU lock.
# Usage: KVM_HOST=... [KVM_KEY=...] [NVFLAGS="-DMM_MINB=2"] tests/gpu_ops/dev_bench.sh [mm|bench|test_ops|test_extra]
#   mm = bench_mm.cu (matmul vs cuBLAS, same process, fast build); bench = bench_ops.cu with -DBENCH_CUBLAS (BENCH_ONLY=mm|attn|mem)
set -euo pipefail
DIR="$(cd "$(dirname "$0")/../.." && pwd)"; R="${KVM_SCRATCH:-kvm-perf2-scratch}"; W="${1:-mm}"
SSH="ssh -o BatchMode=yes ${KVM_KEY:+-i $KVM_KEY}"
rsync -az -e "$SSH" --exclude .git --exclude results "$DIR/include" "$DIR/tests" "$KVM_HOST:$R/"
$SSH "$KVM_HOST" "cd $R/tests/gpu_ops && export PATH=\$PATH:/usr/local/cuda/bin && case $W in
 mm) nvcc -O2 -arch=sm_80 -std=c++17 -lineinfo -Xptxas -v ${NVFLAGS:-} -I../../include -I. bench_mm.cu -lcublas -o bench_mm 2>&1 | grep -E 'error|spill|registers' | sort | uniq -c; flock -w 900 ~/.gpu.lock ./bench_mm;;
 bench) nvcc -O2 -arch=sm_80 -std=c++17 -lineinfo -DBENCH_CUBLAS ${NVFLAGS:-} -I../../include -I. bench_ops.cu -lcublas -o bench_ops 2>&1 | grep -E 'error|spill'; flock -w 900 ~/.gpu.lock env BENCH_ONLY=\${BENCH_ONLY:-mm} BENCH_REPS=\${BENCH_REPS:-5} ./bench_ops;;
 test_ops) nvcc -O2 -arch=sm_80 -std=c++17 -lineinfo ${NVFLAGS:-} -I../../include -I. test_ops.cu -o test_ops 2>&1 | grep -E 'error'; flock -w 900 ~/.gpu.lock \${WRAP:-} ./test_ops | tail -\${TAILN:-5};;
 test_extra) nvcc -O2 -arch=sm_80 -std=c++17 -lineinfo --fmad=false -Xcompiler -ffp-contract=off ${NVFLAGS:-} -I../../include -I. test_extra.cu -o test_extra 2>&1 | grep -E 'error'; flock -w 900 ~/.gpu.lock \${WRAP:-} ./test_extra | tail -\${TAILN:-5};;
 esac"
