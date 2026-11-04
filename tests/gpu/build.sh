#!/usr/bin/env bash
# Build tests/gpu/test_exec on the machine with the GPU (run from the repository root). Extra args go to nvcc.
# --fmad=false / -ffp-contract=off: per-op IEEE arithmetic, so fp results are compared bitwise with the CPU reference.
set -euo pipefail
NVCC="$(command -v nvcc || ls /usr/local/cuda*/bin/nvcc | tail -1)"
exec "$NVCC" -O3 -arch="${KVM_ARCH:-sm_80}" -std=c++17 -Iinclude -Itests/gpu --fmad=false -Xcompiler -ffp-contract=off -lineinfo \
  -DKVM_EXEC_NO_MAIN -DKVM_EXTRA_OPS_DEF='"kvm_ops_test.def"' -DKVM_EXTRA_OPS_CUH='"kvm_ops_test.cuh"' "$@" tests/gpu/test_exec.cu -o test_exec
