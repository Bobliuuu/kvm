#!/usr/bin/env bash
# Zoo-model golden check on a GPU box: bundles are generated here (no GPU needed), run remotely through kvm_exec under ONE hold of the
# GPU lock, and checked here against the compiler's CPU golden by check.py.
# Usage: KVM_HOST=user@host [KVM_KEY=key] [KVM_ARCH=sm_80] [KVM_NO_LOCK=1] examples/models/run.sh [--sanitize] [--cases FILE] [--prebuilt DIR]
# --prebuilt DIR: reuse DIR/<case>/ bundles made earlier by examples/models/prebuild.sh (generation is the slow, CPU-only part)
# A case line:  NAME | MODEL | STEPS | MODE(precise|tf32) | TOL | extra gen_bundle.py flags    (# comments, blank lines ignored)
set -euo pipefail
SANITIZE=0; CASES_FILE=""; OUT=""
PREBUILT=""
while [ $# -gt 0 ]; do case "$1" in --sanitize) SANITIZE=1;; --cases) CASES_FILE="$2"; shift;; --prebuilt) PREBUILT="$2"; shift;; *) echo "unknown $1"; exit 2;; esac; shift; done
HOST="${KVM_HOST:?set KVM_HOST=user@host (an environment variable, never a committed file)}"
ARCH="${KVM_ARCH:-sm_80}"
SSH="ssh -o StrictHostKeyChecking=no -o BatchMode=yes ${KVM_KEY:+-i $KVM_KEY}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CASES_FILE="${CASES_FILE:-$ROOT/examples/models/cases.txt}"
REMOTE='~/kvm-models-scratch'
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$ROOT/logs/$STAMP-models"; mkdir -p "$LOG"; WORK="$LOG/bundles"; mkdir -p "$WORK"
rsync -az -e "$SSH" --delete --exclude .git --exclude results --exclude logs --exclude docs "$ROOT/" "$HOST:$REMOTE/"
$SSH "$HOST" "cd $REMOTE && NVCC=\$(command -v nvcc || ls /usr/local/cuda*/bin/nvcc | tail -1) && \$NVCC -O3 -arch=$ARCH -std=c++17 -Iinclude --fmad=false -Xcompiler -ffp-contract=off -lineinfo -diag-suppress 177 kvm_exec.cu -o kvm_exec"
declare -a NAMES MODES TOLS
while IFS='|' read -r name model steps mode tol flags; do
  trim() { sed 's/^[[:space:]]*//;s/[[:space:]]*$//'; }
  name="$(echo "$name" | trim)"; case "$name" in ''|'#'*) continue;; esac
  model="$(echo "$model" | trim)"; steps="$(echo "$steps" | trim)"; mode="$(echo "$mode" | trim)"; tol="$(echo "$tol" | trim)"
  d="$WORK/$name"
  if [ -n "$PREBUILT" ] && [ -f "$PREBUILT/$name/bundle/golden.json" ]; then cp -R "$PREBUILT/$name" "$d"; echo "$name: prebuilt";
  else python3 "$ROOT/examples/models/gen_bundle.py" "$d" --model "$model" --steps "$steps" $([ "$mode" = precise ] && echo --precise) $flags | tail -1; fi
  $SSH "$HOST" "mkdir -p $REMOTE/mbundles/$name"
  rsync -az -e "$SSH" "$d/bundle/" "$HOST:$REMOTE/mbundles/$name/"
  NAMES+=("$name"); MODES+=("$mode"); TOLS+=("$tol")
done < "$CASES_FILE"
RUN="cd $REMOTE && for c in ${NAMES[*]}; do ./kvm_exec --quiet --dump-buffers mbundles/\$c > mbundles/\$c.dump 2> mbundles/\$c.err; echo \"\$c exit \$?\"; for i in 1 2 3; do ./kvm_exec --quiet mbundles/\$c | tail -n1; done > mbundles/\$c.times; done"
if [ "$SANITIZE" = 1 ]; then
  RUN="$RUN; for c in ${NAMES[*]}; do for tool in memcheck racecheck synccheck; do echo \"== compute-sanitizer \$tool \$c\"; compute-sanitizer --tool \$tool --error-exitcode 9 ./kvm_exec --blocks 4 --quiet mbundles/\$c 2>&1 | tail -3; echo \"sanitizer exit \${PIPESTATUS[0]}\"; done; done"
fi
FLK="flock -w 14400 ~/.gpu.lock"; [ -z "${KVM_NO_LOCK:-}" ] || FLK=""   # KVM_NO_LOCK=1: an outer hold exists
$SSH "$HOST" "$FLK timeout 3000 bash -c '$RUN'" | tee "$LOG/runs.txt"
fail=0; grep -E "exit [1-9]" "$LOG/runs.txt" && fail=1
for i in "${!NAMES[@]}"; do
  c="${NAMES[$i]}"; d="$WORK/$c"
  scp -q -o BatchMode=yes ${KVM_KEY:+-i $KVM_KEY} "$HOST:$REMOTE/mbundles/$c.dump" "$d/dump.txt"
  scp -q -o BatchMode=yes ${KVM_KEY:+-i $KVM_KEY} "$HOST:$REMOTE/mbundles/$c.times" "$d/times.txt"
  tail -n1 "$d/dump.txt" | sed "s/^/$c kvm_exec: /"; sed "s/^/$c repeat: /" "$d/times.txt"
  python3 "$ROOT/examples/models/check.py" "$d/workload.json" "$d/bundle" "$d/dump.txt" --mode "${MODES[$i]}" --tol "${TOLS[$i]}" --json "$d/check.json" | tee "$d/check.txt" || fail=1
done
echo "logs: $LOG"; exit $fail
