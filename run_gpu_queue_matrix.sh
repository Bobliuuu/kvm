#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then echo "Usage: $0 /path/to/upstream/mvm [sm_80]" >&2; exit 2; fi
REPO=$(cd "$1" && pwd)
ARCH=${2:-sm_80}
ROOT=$(cd "$(dirname "$0")" && pwd)
OUT="$ROOT/results-queue"
mkdir -p "$OUT"
command -v nvcc >/dev/null || { echo "nvcc not found" >&2; exit 1; }
python3 "$ROOT/integration/apply.py" "$REPO/mvm.cu"
python3 "$ROOT/integration/apply_queue.py" "$REPO/mvm_ready.cu"
nvcc -O3 -std=c++17 -arch="$ARCH" -DKTILE=32 -DKKC=64 -DKSTG=4 "$REPO/mvm_queued.cu" -o "$OUT/mvm_queued"
# All six benchmark modes run per case; timeout prevents infinite GPU hangs.
for runs in 1 2 4 8; do
  for steps in 2 10 40; do
    echo "=== runs=$runs steps=$steps ==="
    timeout 120s "$OUT/mvm_queued" --runs "$runs" --steps "$steps" --ref-steps "$steps" \
      | tee "$OUT/r${runs}-s${steps}.log"
  done
done
if command -v compute-sanitizer >/dev/null; then
  timeout 180s compute-sanitizer --tool memcheck "$OUT/mvm_queued" --runs 2 --steps 2 --ref-steps 2 \
    | tee "$OUT/sanitizer.log"
fi
