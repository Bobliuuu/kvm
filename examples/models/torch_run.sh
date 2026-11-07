#!/usr/bin/env bash
# Executor vs PyTorch for the whole zoo, run ON the GPU machine from the repository root (see torch_report.py for what is compared):
#   examples/models/torch_run.sh OUT_DIR [KVM_EXEC]
# Builds the one-step bundles (CPU), runs each through kvm_exec under ONE hold of the GPU lock (KVM_NO_LOCK=1 skips it), compares with torch.
# Builds kvm_exec for sm_80 (KVM_ARCH) when no binary is given. Scratch is OUT_DIR only.
set -euo pipefail
OUT="${1:?usage: torch_run.sh OUT_DIR [KVM_EXEC]}"; BIN="${2:-}"; ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
mkdir -p "$OUT"
if [ -z "$BIN" ]; then
  NVCC="$(command -v nvcc || ls /usr/local/cuda*/bin/nvcc | tail -1)"
  "$NVCC" -O3 -arch="${KVM_ARCH:-sm_80}" -std=c++17 -Iinclude --fmad=false -Xcompiler -ffp-contract=off -lineinfo -diag-suppress 177 kvm_exec.cu -o "$OUT/kvm_exec"
  BIN="$OUT/kvm_exec"
fi
python3 examples/models/torch_report.py build "$OUT"
cat > "$OUT/gpu.sh" <<GPU
set -uo pipefail
cd "$ROOT"
for c in \$(python3 -c "import json,sys; sys.path.insert(0,'examples/models'); import torch_report as t; print(' '.join(n for n,*_ in t.CASES))"); do
  "$BIN" --quiet --dump-buffers "$OUT/\$c/bundle" > "$OUT/\$c/dump.txt" 2> "$OUT/\$c/err.txt"; echo "\$c exit \$?"
done
GPU
FLK="flock -w 14400 $HOME/.gpu.lock"; [ -z "${KVM_NO_LOCK:-}" ] || FLK=""
$FLK bash "$OUT/gpu.sh" | tee "$OUT/runs.txt"
python3 examples/models/torch_report.py compare "$OUT" --json "$OUT/torch_compare.json" --md "$OUT/torch_compare.md" | tee "$OUT/torch_compare.txt"
