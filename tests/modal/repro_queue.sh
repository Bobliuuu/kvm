#!/usr/bin/env bash
# Litmus queue property with line-buffered output (which cap/CTA configuration hangs?). Runs inside tests/modal/run_suite.py --suite custom.
set -u
mkdir -p out
nvcc -O3 -arch=sm_80 -std=c++17 -lineinfo -Wno-deprecated-gpu-targets -Iinclude tests/litmus/litmus.cu -o out/litmus_prod 2>&1 | tail -n 2
cd out
for rk in 1 5; do echo "== queue rounds_k=$rk"; stdbuf -oL timeout 240 ./litmus_prod queue $rk; echo "exit: $?"; done
