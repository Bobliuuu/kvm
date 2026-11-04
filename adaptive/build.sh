#!/usr/bin/env bash
# Build adaptive_run on the GPU box (run from the repository root). Needs the Doom engine objects (doom/build.sh; DOOM_INSTANCES=N adds
# doom_dev_i1..i<N-1>.o, picked up here from DOOM_BUILD_DIR). NODOOM=1 builds without the game pack (--no-doom only). Extra args go to nvcc.
#   DOOM_BUILD_DIR=doom/build  OUT_BIN=adaptive_run  KVM_ARCH=sm_80
set -euo pipefail
NVCC="$(command -v nvcc || ls /usr/local/cuda*/bin/nvcc | tail -1)"
DB="${DOOM_BUILD_DIR:-doom/build}"; OUT_BIN="${OUT_BIN:-adaptive_run}"
OBJ=""; GAME=""
if [ -z "${NODOOM:-}" ]; then
  INST="$(ls "$DB"/doom_dev_i*.o 2>/dev/null | tr '\n' ' ' || true)"
  GAME="-rdc=true -DKVM_WITH_GAME -DKVM_DOOM_INSTANCES=$(( 1 + $(echo $INST | wc -w) ))"; OBJ="$DB/doom_dev.o $INST"
fi
exec "$NVCC" -O3 -arch="${KVM_ARCH:-sm_80}" -std=c++17 -I. -Iinclude --fmad=false -Xcompiler -ffp-contract=off -lineinfo $GAME "$@" adaptive/adaptive_run.cu $OBJ -o "$OUT_BIN"
