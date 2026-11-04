#!/usr/bin/env bash
# The same golden check as run.sh, executed ON the GPU machine (no ssh): bundles were built there with prebuild.sh, so nothing big
# crosses the network. One hold of the GPU lock for all runs; compute-sanitizer (memcheck, racecheck, synccheck) runs on cases named san_*
# (small bundles: racecheck is ~1000x slower) or on all cases with --sanitize. Usage, from the repository root on the GPU machine:
#   examples/models/prebuild.sh logs/prebuilt && examples/models/run_box.sh logs/prebuilt logs/models-run [--sanitize] [--cases FILE]
# --check-only re-runs only check.py on the dumps of an earlier run (no GPU, no lock).
set -euo pipefail
PRE="${1:?usage: run_box.sh PREBUILT_DIR OUT_DIR [--sanitize] [--cases FILE]}"; OUT="${2:?}"; shift 2
SANITIZE=0; CHECK_ONLY=0; ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; CASES_FILE="$ROOT/examples/models/cases.txt"
while [ $# -gt 0 ]; do case "$1" in --sanitize) SANITIZE=1;; --check-only) CHECK_ONLY=1;; --cases) CASES_FILE="$2"; shift;; *) echo "unknown $1"; exit 2;; esac; shift; done
ARCH="${KVM_ARCH:-sm_80}"; mkdir -p "$OUT/bundles"; cd "$ROOT"
NVCC="$(command -v nvcc || ls /usr/local/cuda*/bin/nvcc | tail -1)"
[ "$CHECK_ONLY" = 1 ] || "$NVCC" -O3 -arch="$ARCH" -std=c++17 -Iinclude --fmad=false -Xcompiler -ffp-contract=off -lineinfo -diag-suppress 177 kvm_exec.cu -o "$OUT/kvm_exec"
trim() { sed 's/^[[:space:]]*//;s/[[:space:]]*$//'; }
declare -a NAMES MODES TOLS
while IFS='|' read -r name model steps mode tol flags; do
  name="$(echo "$name" | trim)"; case "$name" in ''|'#'*) continue;; esac
  [ -f "$PRE/$name/bundle/golden.json" ] || { echo "missing prebuilt $name"; exit 2; }
  [ "$CHECK_ONLY" = 1 ] || { rm -rf "$OUT/bundles/$name"; cp -R "$PRE/$name" "$OUT/bundles/$name"; }
  NAMES+=("$name"); MODES+=("$(echo "$mode" | trim)"); TOLS+=("$(echo "$tol" | trim)")
done < "$CASES_FILE"
run_all() {
  for c in "${NAMES[@]}"; do
    d="$OUT/bundles/$c"
    "$OUT/kvm_exec" --quiet --dump-buffers "$d/bundle" > "$d/dump.txt" 2> "$d/err.txt"; echo "$c exit $?"
    for i in 1 2 3; do "$OUT/kvm_exec" --quiet "$d/bundle" | tail -n1; done > "$d/times.txt"
    if [ "$SANITIZE" = 1 ] || [[ "$c" == san_* ]]; then for tool in memcheck racecheck synccheck; do
      echo "== compute-sanitizer $tool $c"; compute-sanitizer --tool $tool --error-exitcode 9 "$OUT/kvm_exec" --blocks 4 --quiet "$d/bundle" 2>&1 | tail -3; echo "sanitizer exit ${PIPESTATUS[0]}"
    done; fi
  done
}
export -f run_all; export NAMES_STR="${NAMES[*]}"
# shellcheck disable=SC2155
declare -f run_all > "$OUT/run_all.sh"
{ declare -p NAMES SANITIZE OUT; cat "$OUT/run_all.sh"; echo run_all; } > "$OUT/run_all_full.sh"
[ "$CHECK_ONLY" = 1 ] || flock -w 14400 ~/.gpu.lock timeout 3400 bash "$OUT/run_all_full.sh" | tee "$OUT/runs.txt"
fail=0; grep -E "exit [1-9]" "$OUT/runs.txt" && fail=1
for i in "${!NAMES[@]}"; do
  c="${NAMES[$i]}"; d="$OUT/bundles/$c"
  tail -n1 "$d/dump.txt" | sed "s/^/$c kvm_exec: /"
  python3 "$ROOT/examples/models/check.py" "$d/workload.json" "$d/bundle" "$d/dump.txt" --mode "${MODES[$i]}" --tol "${TOLS[$i]}" --json "$d/check.json" | tee "$d/check.txt" || fail=1
done
echo "out: $OUT"; exit $fail
