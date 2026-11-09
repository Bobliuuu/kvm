#!/usr/bin/env bash
# One lock hold: whole-step CUDA graph (torch) vs executor builds on the same workloads. Run on the GPU box from the scratch repo root:
#   flock -w 600 ~/.gpu.lock bash bench/exec_vs_graph/run.sh OUT_DIR
# Env: EXES="name=path ..." executors to time (default ./kvm_exec), CONFIGS="tiny small", REPS=7, TRACE=1 for a traced run per exe.
set -uo pipefail
OUT="${1:?OUT_DIR}"; REPS="${REPS:-7}"; mkdir -p "$OUT"
{ date -u +%FT%TZ; nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv,noheader; } | tee "$OUT/env.txt"
for cfg in ${CONFIGS:-tiny small}; do
  W="w_$cfg"; D="$OUT/$cfg"; mkdir -p "$D"; cp "$W/info.json" "$D/"
  echo "=== $cfg $(date -u +%T)"
  python3 bench/vs_pytorch/torch_bench.py "$W" --reps "$REPS" --precision tf32 --modes ${MODES:-graph,compile_graph} --out "$D/torch.json" 2>&1 | grep -v Warning | tee "$D/torch.log"
  for e in ${EXES:-base=./kvm_exec}; do
    n="${e%%=*}"; x="${e#*=}"
    python3 bench/vs_pytorch/exec_bench.py "$W" "$x" --reps "$REPS" --out "$D/kvm_$n.json" 2>&1 | tee "$D/kvm_$n.log"
    if [ -n "${TRACE:-}" ]; then
      "$x" --quiet --trace-csv "$D/trace_$n.csv" "$W/bundle" > "$D/trace_run_$n.txt" 2>&1
      python3 bench/vs_pytorch/trace_summary.py "$D/trace_$n.csv" "$(python3 -c "import json;print(json.load(open('$W/info.json'))['steps'])")" | tee "$D/trace_summary_$n.txt"
    fi
  done
done
echo "=== done $(date -u +%T)"
