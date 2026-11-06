#!/usr/bin/env bash
# gpt_162m timing on the GPU box: each PyTorch variant in its own process, then the executor (5 launches) and one traced run, in ONE lock hold:
#   flock -w 14400 ~/.gpu.lock bash bench/vs_pytorch/campaign162.sh        (from the repository root; needs w_162m/ from gen.py and ./kvm_exec)
# The workload: python3 bench/vs_pytorch/gen.py w_162m --model gpt_162m --steps 3 --mode tf32 --tiling a100
set -uo pipefail
OUT="${OUT:-logs/$(date -u +%Y%m%dT%H%M%SZ)-162m-timing}"; REPS="${REPS:-5}"; mkdir -p "$OUT"
cp w_162m/info.json "$OUT/"
exec > >(tee "$OUT/campaign.log") 2>&1
echo "lock acquired $(date -u +%T)"
{ date -u +%FT%TZ; nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv; nvcc --version | tail -n 1
  python3 -c "import torch,triton;print('torch',torch.__version__,'cuda',torch.version.cuda,'triton',triton.__version__)"
  echo "commit: ${KVM_COMMIT:-unknown}"; } | tee "$OUT/env.txt"
for m in ${MODES:-eager graph compile_mat compile_ro compile_graph}; do
  echo "=== torch $m $(date -u +%T)"
  python3 bench/vs_pytorch/torch_bench.py w_162m --reps "$REPS" --precision tf32 --modes "$m" --out "$OUT/torch_$m.json" 2>&1 | grep -v Warning
done
echo "=== executor $(date -u +%T)"
python3 bench/vs_pytorch/exec_bench.py w_162m ./kvm_exec --reps "$REPS" --out "$OUT/exec.json"
echo "=== executor trace $(date -u +%T)"
./kvm_exec --quiet --trace-csv "$OUT/trace.csv" w_162m/bundle > "$OUT/trace_run.txt" 2>&1; tail -n 2 "$OUT/trace_run.txt"
echo "=== done $(date -u +%T)"
