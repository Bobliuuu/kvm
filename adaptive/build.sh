#!/usr/bin/env bash
# Build adaptive_run on the GPU box (run from the repository root). Needs doom/build/doom_dev.o (doom/build.sh) for the game pack;
# NODOOM=1 builds without it (--no-doom only). Extra objects (doom_dev_i*.o, multi-instance) are picked up when present. Extra args go to nvcc.
set -euo pipefail
NVCC="$(command -v nvcc || ls /usr/local/cuda*/bin/nvcc | tail -1)"
OBJ=""; GAME=""
if [ -z "${NODOOM:-}" ]; then GAME="-rdc=true -DKVM_WITH_GAME"; OBJ="doom/build/doom_dev.o $(ls doom/build/doom_dev_i*.o 2>/dev/null | tr '\n' ' ' || true)"; fi
exec "$NVCC" -O3 -arch="${KVM_ARCH:-sm_80}" -std=c++17 -I. -Iinclude --fmad=false -Xcompiler -ffp-contract=off -lineinfo $GAME "$@" adaptive/adaptive_run.cu $OBJ -o adaptive_run
