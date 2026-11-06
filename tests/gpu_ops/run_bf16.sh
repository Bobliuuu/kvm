#!/usr/bin/env bash
# Build and run the bf16 tensor-core GEMM suite on the GPU box: tests/gpu_ops/test_bf16.cu (matmul_mma_bf16 and the bf16-storage paths of the
# GPT-pack ops against float64 references) and tests/gpu_ops/bench_mma_bf16.cu (TFLOP/s against cuBLAS bf16, with a correctness check per row).
# Usage: KVM_HOST=user@host [KVM_ARCH=sm_80] tests/gpu_ops/run_bf16.sh [--sanitize] [--no-bench]
# ONE GPU lock hold covers all runs (test, and with --sanitize memcheck / racecheck / synccheck; then the benchmark); KVM_NO_LOCK=1 skips it.
# The build runs before the lock is taken. Remote scratch: private ~/kvm-scratch-bf16-$USER-$$ (removed on success; KVM_KEEP_SCRATCH=1 keeps it).
set -euo pipefail
HOST="${KVM_HOST:?set KVM_HOST=user@host (an environment variable, never a committed file)}"
ARCH="${KVM_ARCH:-sm_80}"
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=no ${KVM_KEY:+-i $KVM_KEY}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
. "$ROOT/tests/remote_lib.sh"; kvm_scratch bf16; kvm_lock "${KVM_LOCK_WAIT:-7200}"
rsync -az -e "$SSH" --exclude .git --exclude results --exclude logs --exclude docs "$ROOT/" "$HOST:$REMOTE/"
$SSH "$HOST" "cd $REMOTE/tests/gpu_ops && NVCC=\$(command -v nvcc || ls /usr/local/cuda*/bin/nvcc | tail -1) &&
  \$NVCC -O2 -arch=$ARCH -std=c++17 --fmad=false -Xcompiler -ffp-contract=off -lineinfo -I../../include -I. test_bf16.cu -o test_bf16 &&
  \$NVCC -O2 -arch=$ARCH -std=c++17 --fmad=false -Xcompiler -ffp-contract=off -lineinfo -I../../include -I. bench_mma_bf16.cu -lcublas -o bench_mma_bf16"
RUN="./test_bf16"
if [ "${1:-}" = "--sanitize" ]; then
  RUN="./test_bf16; for t in memcheck racecheck synccheck; do echo == \$t; compute-sanitizer --tool \$t --error-exitcode 9 ./test_bf16 | tail -4; echo exit \${PIPESTATUS[0]}; done"
  shift
fi
[ "${1:-}" = "--no-bench" ] || RUN="$RUN; ./bench_mma_bf16"
$SSH "$HOST" "cd $REMOTE/tests/gpu_ops && $FLK timeout 3000 bash -c '$RUN'"
kvm_scratch_done
