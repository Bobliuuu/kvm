#!/usr/bin/env bash
# Interleaved A/B of gen.py bundles through ONE kvm_exec binary: REPS rounds, each round runs every bundle once (one process per launch),
# kernel time from the executor's own cudaEvent line. Run it inside one GPU-lock hold:
#   flock -w 7200 ~/.gpu.lock env EXE=./kvm_exec bash bench/vs_pytorch/ab_bundles.sh REPS OUT_DIR bundle_dir...
set -u
EXE="${EXE:-./kvm_exec}"; REPS="${1:?REPS}"; OUT="${2:?OUT_DIR}"; shift 2; mkdir -p "$OUT"
{ date -u +%FT%TZ; nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv; nvcc --version | tail -n 1; } | tee "$OUT/env.txt"
for r in $(seq 1 "$REPS"); do
  for b in "$@"; do
    o=$("$EXE" --quiet "$b" 2>&1 | grep -E '^(OK|FAIL):' | tail -1)
    echo "rep $r $(basename "$(dirname "$b")") $o" | tee -a "$OUT/runs.txt"
  done
done
