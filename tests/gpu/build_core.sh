#!/usr/bin/env bash
# Fast build of tests/gpu/test_exec (core operators only: no gpt/extra/comm/game packs), about a minute instead of ~24 min.
# Usage (repository root, on the GPU box): tests/gpu/build_core.sh [out] [nvcc flags...]   KVM_ARCH=sm_86 KVM_OPT=-O1
set -euo pipefail
OUT="${1:-test_exec_core}"; shift || true
NVCC="$(command -v nvcc || ls /usr/local/cuda*/bin/nvcc | tail -1)"
INC="$(mktemp -d)"; trap 'rm -rf "$INC"' EXIT
for f in include/*; do case "$(basename "$f")" in kvm_ops_gpt.*|kvm_ops_extra.*|kvm_ops_comm.*|kvm_ops_game.*) ;; *) cp "$f" "$INC/";; esac; done
"$NVCC" "${KVM_OPT:--O3}" -arch="${KVM_ARCH:-sm_80}" -std=c++17 -I"$INC" -Itests/gpu --fmad=false -Xcompiler -ffp-contract=off -lineinfo \
  -DKVM_EXEC_NO_MAIN -DKVM_EXTRA_OPS_DEF='"kvm_ops_test.def"' -DKVM_EXTRA_OPS_CUH='"kvm_ops_test.cuh"' "$@" tests/gpu/test_exec.cu -o "$OUT"
