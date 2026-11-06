#!/usr/bin/env bash
# Same checks as run.sh but executed ON the GPU box from a synced repo copy (bundle generation + golden are CPU work, so the Mac stays
# idle): zoo cases x {cpu, a100} tiling at 3xTF32 vs the CPU golden. Build is outside the lock, ONE short lock hold covers the runs.
#   tests/generic/run_box.sh OUTDIR [case...]        (default: every zoo case)      env: KVM_ARCH, KVM_LOCK_WAIT, KVM_NO_LOCK, KVM_PREBUILT
set -euo pipefail
OUT="$(mkdir -p "${1:?outdir}" && cd "$1" && pwd)"; shift || true
ARCH="${KVM_ARCH:-sm_80}"; ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
CASES="${*:-$(python3 -I tests/generic/gen_bundle.py --list | cut -d: -f1)}"
BUNDLES=""
for c in $CASES; do for t in cpu a100 a100_big; do
  python3 -I tests/generic/gen_bundle.py "$OUT/${c}_$t" "$c" --tiling "$t" --precise | tee -a "$OUT/gen.txt"; BUNDLES="$BUNDLES ${c}_$t"
done; done
NVCC="$(command -v nvcc || ls /usr/local/cuda*/bin/nvcc | tail -1)"
[ -n "${KVM_PREBUILT:-}" ] || "$NVCC" -O3 -arch="$ARCH" -std=c++17 -I. -Iinclude --fmad=false -Xcompiler -ffp-contract=off -lineinfo -diag-suppress 177 kvm_exec.cu -o "$OUT/kvm_exec"
LOCK="flock -w ${KVM_LOCK_WAIT:-900} $HOME/.gpu.lock"; [ -z "${KVM_NO_LOCK:-}" ] || LOCK=""
$LOCK timeout 1200 bash -c "for n in $BUNDLES; do '$OUT/kvm_exec' --blocks 4 --quiet --dump-buffers '$OUT'/\$n/bundle > '$OUT'/\$n.dump 2> '$OUT'/\$n.err; echo \"\$n exit \$?\"; done" | tee "$OUT/runs.txt"
fail=0; grep -E "exit [1-9]" "$OUT/runs.txt" && fail=1
for n in $BUNDLES; do
  echo "== $n"; python3 -I tests/gptdiff/golden_check.py "$OUT/$n/workload.json" "$OUT/$n/bundle" "$OUT/$n.dump" --mode precise | tee "$OUT/$n.check.txt" || fail=1
done
echo "done (exit $fail)"; exit $fail
