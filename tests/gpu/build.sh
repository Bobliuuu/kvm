#!/usr/bin/env bash
# Build tests/gpu/test_exec on the machine with the GPU (run from the repository root). Extra args go to nvcc.
# --fmad=false / -ffp-contract=off: per-op IEEE arithmetic, so fp results are compared bitwise with the CPU reference.
set -euo pipefail
NVCC="$(command -v nvcc || ls /usr/local/cuda*/bin/nvcc | tail -1)"
exec "$NVCC" -O3 -arch="${MVM_ARCH:-sm_80}" -std=c++17 -Iinclude -Itests/gpu --fmad=false -Xcompiler -ffp-contract=off -lineinfo \
  -DMVM_EXEC_NO_MAIN -DMVM_EXTRA_OPS_DEF='"mvm_ops_test.def"' -DMVM_EXTRA_OPS_CUH='"mvm_ops_test.cuh"' "$@" tests/gpu/test_exec.cu -o test_exec
