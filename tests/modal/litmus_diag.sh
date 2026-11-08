#!/usr/bin/env bash
# Which litmus property hangs on this tree? Line-buffered, hard-killed per test (timeout -k). Runs inside run_suite.py --suite custom.
set -u
mkdir -p out
nvcc -O3 -arch=sm_80 -std=c++17 -lineinfo -Wno-deprecated-gpu-targets -Iinclude tests/litmus/litmus.cu -o out/litmus_prod 2>&1 | tail -n 2
cd out
for c in "sb 100 5" "dekker1 100 5" "dekker3 100 5" "mp 100 5" "queue 1" "queue 5"; do
  echo "== $c"; date -u +%T; stdbuf -oL timeout -k 5 300 ./litmus_prod $c 2>&1 | tail -15; echo "exit: ${PIPESTATUS[0]}"; date -u +%T
done
