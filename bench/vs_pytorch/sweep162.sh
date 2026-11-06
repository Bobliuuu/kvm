#!/usr/bin/env bash
# Executor admission-window sweep on the 1-step gpt_162m bundle (kvm_exec defaults live_cap 16384, ring 65536 words, 262144 slots are far
# below what demo/kvm_run picks for this size). One lock hold; each configuration = 3 launches, best reported.
#   flock -w 14400 ~/.gpu.lock bash bench/vs_pytorch/sweep162.sh WORKLOAD_DIR OUT_DIR
set -uo pipefail
W="${1:?WORKLOAD_DIR}"; OUT="${2:?OUT_DIR}"; mkdir -p "$OUT"
for cfg in "default:" "lc65536:--live-cap 65536 --ring-words 4194304 --max-tasks 524288" \
           "lc65536_slots2M:--live-cap 65536 --ring-words 4194304 --max-tasks 2097152" \
           "lc262144:--live-cap 262144 --ring-words 4194304 --max-tasks 524288" \
           "lc65536_b64:--live-cap 65536 --ring-words 4194304 --max-tasks 524288 --blocks 64"; do
  name="${cfg%%:*}"; args="${cfg#*:}"
  echo "=== $name [$args] $(date -u +%T)" | tee -a "$OUT/sweep.log"
  # shellcheck disable=SC2086
  python3 bench/vs_pytorch/exec_bench.py "$W" ./kvm_exec --reps 3 --out "$OUT/$name.json" -- $args 2>&1 | tee -a "$OUT/sweep.log"
done
