#!/usr/bin/env bash
# tests/generic/run.sh on the GPU box itself (no ssh, no rsync): same cases, builds, runs and checks, in one directory.
#   tests/generic/run_box.sh OUT          (KVM_ARCH=sm_80 default; used by tests/modal/run_suite.py --suite generic)
# Exit status 0 = every bundle and manifest check passed.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${1:?usage: run_box.sh OUT}"; mkdir -p "$OUT"
ARCH="${KVM_ARCH:-sm_80}"
PY="${PYTHON:-python3}"
cd "$ROOT"
CASES="$($PY -I tests/generic/gen_bundle.py --list | cut -d: -f1)"
MANIFESTS="reg_mlp cnn1d"
FL="-O3 -arch=$ARCH -std=c++17 -I. -Iinclude --fmad=false -Xcompiler -ffp-contract=off -lineinfo -diag-suppress 177"
nvcc --version | tail -1; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
( nvcc $FL kvm_exec.cu -o kvm_exec & nvcc $FL demo/kvm_run.cu -o demo/kvm_run & wait ) 2>&1 | grep -v 'warning\|Remark\|^$' | tee "$OUT/build.txt"
fail=0
for c in $CASES; do
  for tiling in cpu a100; do
    n="${c}_$tiling"; d="$OUT/$n"
    $PY -I tests/generic/gen_bundle.py "$d" "$c" --tiling "$tiling" --precise || { echo "$n gen FAILED"; fail=1; continue; }
    ./kvm_exec --blocks 4 --quiet --dump-buffers "$d/bundle" > "$d/dump.txt" 2> "$d/err.txt"; echo "$n exit $?"
    tail -n1 "$d/dump.txt" | sed "s/^/$n kvm_exec: /"
    echo "== $n"; $PY -I examples/models/check.py "$d/workload.json" "$d/bundle" "$d/dump.txt" --mode precise --tol 2e-4 \
      --json "$d/check.json" | tee "$d/check.txt"; [ "${PIPESTATUS[0]}" = 0 ] || fail=1
  done
done
for c in $MANIFESTS; do
  d="$OUT/man_$c"
  $PY -I tests/generic/gen_bundle.py "$d" "$c" --manifest --steps 6 --window 3 --tiling a100 --precise
  timeout 900 ./demo/kvm_run "$d" --out "$d/out" --observe '^loss\.' --compile-cmd "$PY runtime/compile_workload.py" > "$d/kvm_run.log" 2>&1
  echo "man_$c kvm_run exit $?"
  echo "== man_$c"; $PY -I demo/check_demo.py "$d" "$d/out" --observe '^loss\.' --golden "$d/golden.json" --loss-tol 2e-3 | tee "$d/check.txt"
  [ "${PIPESTATUS[0]}" = 0 ] || fail=1
done
echo "generic: exit $fail"; exit $fail
