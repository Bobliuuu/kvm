#!/usr/bin/env bash
# Run this once on a GPU machine with a checkout of Bobliuuu/kvm.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPSTREAM="${1:?Usage: ./run_gpu_matrix.sh /path/to/kvm [sm_80|sm_90]}"
ARCH="${2:-sm_80}"
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 1; }
command -v timeout >/dev/null || { echo 'timeout required' >&2; exit 1; }
mkdir -p "$ROOT/results"
python3 "$ROOT/integration/apply.py" "$UPSTREAM/kvm.cu"
nvcc -O3 -arch="$ARCH" -std=c++17 -DKTILE=32 -DKKC=64 -DKSTG=4 "$UPSTREAM/kvm_ready.cu" -o "$ROOT/results/kvm_ready"
"$ROOT/results/kvm_ready" --selftest | tee "$ROOT/results/selftest.log"
# Small smoke cases first, before expensive benchmarks. Timeouts catch GPU hangs.
for runs in 1 2 4; do
  for steps in 2 10; do
    echo "=== runs=$runs steps=$steps ==="
    timeout 120s "$ROOT/results/kvm_ready" --runs "$runs" --steps "$steps" --ref-steps "$steps" | tee "$ROOT/results/smoke-r${runs}-s${steps}.log"
  done
done
for runs in 1 2 8 32; do
  echo "=== benchmark runs=$runs ==="
  timeout 900s "$ROOT/results/kvm_ready" --runs "$runs" --steps 300 --ref-steps 30 | tee "$ROOT/results/bench-r${runs}.log"
done
if command -v compute-sanitizer >/dev/null; then
  timeout 300s compute-sanitizer --tool memcheck "$ROOT/results/kvm_ready" --runs 2 --steps 2 --ref-steps 2 | tee "$ROOT/results/memcheck.log"
fi
