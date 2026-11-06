#!/usr/bin/env bash
# usage: build.sh SRC OUT   (SRC = repo copy on the box, OUT = binary path). Same flags as the audit.
set -e; cd "$1"; NVCC=$(command -v nvcc || ls /usr/local/cuda*/bin/nvcc | tail -1)
nice $NVCC -O3 -arch=sm_80 -std=c++17 -Iinclude --fmad=false -Xcompiler -ffp-contract=off -diag-suppress 177 ${3:-} kvm_exec.cu -o "$2" 2>&1 | grep -i error || true
