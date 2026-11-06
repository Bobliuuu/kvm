#!/usr/bin/env bash
# Build tests/gpu/test_admit (run from the repository root, on the GPU box). Same flags as build_sched.sh. Extra args go to nvcc
# (-DKVM_ADMIT_PROF adds the admitter's per-phase timers).
set -euo pipefail
NVCC="$(command -v nvcc || ls /usr/local/cuda*/bin/nvcc | tail -1)"
exec "$NVCC" -O3 -arch="${KVM_ARCH:-sm_80}" -std=c++17 -Iinclude -Itests/gpu --fmad=false -Xcompiler -ffp-contract=off -lineinfo \
  -DKVM_EXEC_NO_MAIN -DKVM_EXTRA_OPS_DEF='"kvm_ops_test.def"' -DKVM_EXTRA_OPS_CUH='"kvm_ops_test.cuh"' "$@" tests/gpu/test_admit.cu -o test_admit
