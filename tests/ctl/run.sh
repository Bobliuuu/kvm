#!/usr/bin/env bash
# Device-side control on the GPU: build kvm_exec (one nvcc, not under the lock), generate bundles (same graph, different data), run each
# under a short GPU-lock hold with --golden (CPU VM semantics) and check the executor skipped exactly the expected tasks.
# Usage (repository root, on the GPU box): [KVM_ARCH=sm_80] [KVM_EXEC=prebuilt] tests/ctl/run.sh OUT_DIR
set -euo pipefail
OUT="${1:?out dir}"; mkdir -p "$OUT"
NVCC="$(command -v nvcc || ls /usr/local/cuda*/bin/nvcc | tail -1)"
EXE="${KVM_EXEC:-$OUT/kvm_exec}"
[ -x "$EXE" ] || "$NVCC" -O3 -arch="${KVM_ARCH:-sm_80}" -std=c++17 -Iinclude --fmad=false -Xcompiler -ffp-contract=off -lineinfo kvm_exec.cu -o "$EXE"
python3 -I tests/ctl/gen_bundles.py "$OUT/bundles" > "$OUT/gen.txt"
fail=0; : > "$OUT/summary.txt"
for d in "$OUT"/bundles/*/; do
  n=$(basename "$d"); tol=1e-6; case "$n" in gemmloop*) tol=1e-2;; esac
  rc=0
  flock -w 900 ~/.gpu.lock timeout 120 "$EXE" --blocks 4 --golden --golden-tol $tol --trace-csv "$OUT/$n.csv" "$d/bundle" > "$OUT/$n.log" 2>&1 || rc=$?
  # pred_skipped is the last CSV column; every unrolled iteration beyond the trip count must be skipped (all its predicated tasks)
  skipped=$(awk -F, 'NR>1 {s+=$NF} END {print s+0}' "$OUT/$n.csv" 2>/dev/null || echo -1)
  python3 -I - "$d/expect.json" "$skipped" "$rc" "$n" "$d/workload.json" <<'PY' | tee -a "$OUT/summary.txt" || fail=1
import json, sys
exp, skipped, rc, name, wl = json.load(open(sys.argv[1])), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
spec = json.load(open(wl))
per = {}
for g in spec['graphs']:
    for t in g['tasks']:
        if 'when' in t: per[t['when']['index']] = per.get(t['when']['index'], 0) + 1
want = 1 if name.startswith('branch') else sum(per.get(i, 0) for i in range(exp['trips'], exp['cap']))
ok = rc == 0 and skipped == want
print(f"{'PASS' if ok else 'FAIL'} {name}: exit {rc}, executor skipped {skipped} tasks, expected {want}", exp)
sys.exit(0 if ok else 1)
PY
done
[ "$fail" = 0 ] && echo "ALL CTL CASES PASS" | tee -a "$OUT/summary.txt"
exit $fail
