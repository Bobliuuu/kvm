#!/usr/bin/env bash
# Box side of the executor-vs-PyTorch comparison. Run from the repository root on the GPU box, in ONE hold of the GPU lock:
#   flock -w 7200 ~/.gpu.lock bash bench/vs_pytorch/campaign.sh OUT_DIR
# Expects ./kvm_exec built (nvcc -O3 -arch=sm_80 -std=c++17 -Iinclude --fmad=false -Xcompiler -ffp-contract=off -lineinfo kvm_exec.cu)
# and workload dirs w_<name>/ made by gen.py. For each: PyTorch modes (one process), then kvm_exec (separate processes, same hold).
set -uo pipefail
OUT="${1:?OUT_DIR}"; REPS="${REPS:-7}"; mkdir -p "$OUT"
{ date -u +%FT%TZ; nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,clocks.sm --format=csv; nvcc --version | tail -1
  python3 -c "import torch,triton;print('torch',torch.__version__,'cuda',torch.version.cuda,'triton',triton.__version__)"
  git rev-parse --short HEAD 2>/dev/null || echo "commit: ${KVM_COMMIT:-unknown}"; } | tee "$OUT/env.txt"
for cfg in ${CONFIGS:-tiny small mid tiny_p}; do
  W="w_$cfg"; D="$OUT/$cfg"; mkdir -p "$D"; cp "$W/info.json" "$D/"
  prec=tf32; [ "$cfg" = tiny_p ] && prec=precise
  echo "=== $cfg ($prec) $(date -u +%T)"
  python3 bench/vs_pytorch/torch_bench.py "$W" --reps "$REPS" --precision "$prec" --out "$D/torch.json" ${MODES:+--modes $MODES} 2>&1 | grep -v Warning | tee "$D/torch.log"
  [ -n "${SKIP_KVM:-}" ] && continue
  python3 bench/vs_pytorch/kvm_bench.py "$W" ./kvm_exec --reps "$REPS" --out "$D/kvm.json" 2>&1 | tee "$D/kvm.log"
  python3 bench/vs_pytorch/kvm_bench.py "$W" ./kvm_exec --reps "$REPS" --out "$D/kvm_b4.json" -- --blocks 4 2>&1 | tee -a "$D/kvm.log"
  if [ -n "${TRACE:-}" ]; then   # one traced run: where the executor's time goes (the CSV is large and stays on the box)
    ./kvm_exec --quiet --trace-csv "$D/trace.csv" "$W/bundle" > "$D/trace_run.txt" 2>&1
    python3 bench/vs_pytorch/trace_summary.py "$D/trace.csv" "$(python3 -c "import json;print(json.load(open('$W/info.json'))['steps'])")" | tee "$D/trace_summary.txt"
  fi
done
echo "=== done $(date -u +%T)"
