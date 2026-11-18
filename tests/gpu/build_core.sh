#!/usr/bin/env bash
# Fast build of tests/gpu/test_exec (core operators only: no gpt/extra/comm/game packs), about a minute instead of ~24 min.
# Usage (repository root, on the GPU box): tests/gpu/build_core.sh [out] [nvcc flags...]   MVM_ARCH=sm_86 MVM_OPT=-O1
set -euo pipefail
OUT="${1:-test_exec_core}"; shift || true
NVCC="$(command -v nvcc || ls /usr/local/cuda*/bin/nvcc | tail -1)"
INC="$(mktemp -d)"; trap 'rm -rf "$INC"' EXIT
for f in include/*; do case "$(basename "$f")" in mvm_ops_gpt.*|mvm_ops_extra.*|mvm_ops_comm.*|mvm_ops_game.*) ;; *) cp -r "$f" "$INC/";; esac; done
"$NVCC" "${MVM_OPT:--O3}" -arch="${MVM_ARCH:-sm_80}" -std=c++17 -I"$INC" -Itests/gpu --fmad=false -Xcompiler -ffp-contract=off -lineinfo \
  -DMVM_EXEC_NO_MAIN -DMVM_EXTRA_OPS_DEF='"mvm_ops_test.def"' -DMVM_EXTRA_OPS_CUH='"mvm_ops_test.cuh"' "$@" tests/gpu/test_exec.cu -o "$OUT"
