#!/usr/bin/env bash
# GPT golden check on a GPU box: gpt_tiny AdamW bundles (1, 2, 5 steps; 1xTF32 and 3xTF32 GEMMs) through kvm_exec vs the CPU golden.
# Usage: KVM_HOST=user@host [KVM_KEY=key] [KVM_ARCH=sm_80] tests/gptdiff/run.sh [--sanitize]
# Bundles are generated here (no GPU needed), run remotely under flock, and checked here by golden_check.py.
set -euo pipefail
SANITIZE=0; [ "${1:-}" = "--sanitize" ] && SANITIZE=1
HOST="${KVM_HOST:?set KVM_HOST=user@host (an environment variable, never a committed file)}"
ARCH="${KVM_ARCH:-sm_80}"
SSH="ssh -o StrictHostKeyChecking=no -o BatchMode=yes ${KVM_KEY:+-i $KVM_KEY}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
REMOTE='~/kvm-gptdiff-scratch'
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$ROOT/logs/$STAMP-gptdiff"; mkdir -p "$LOG"
WORK="$LOG/bundles"; mkdir -p "$WORK"
rsync -az -e "$SSH" --delete --exclude .git --exclude results --exclude logs --exclude docs "$ROOT/" "$HOST:$REMOTE/"
$SSH "$HOST" "cd $REMOTE && NVCC=\$(command -v nvcc || ls /usr/local/cuda*/bin/nvcc | tail -1) && \$NVCC -O3 -arch=$ARCH -std=c++17 -Iinclude --fmad=false -Xcompiler -ffp-contract=off -lineinfo -diag-suppress 177 kvm_exec.cu -o kvm_exec"
fail=0
CASES=""
# cpu tiling = the 544-task 2-step bundle of the original report; a100 tiling = fewer, larger tasks (different tile/edge structure).
for spec in "cpu 1" "cpu 2" "cpu 5" "a100 2"; do
  read -r tiling steps <<< "$spec"
  for mode in tf32 precise; do
    c="${tiling}_s${steps}_$mode"; d="$WORK/$c"
    python3 "$ROOT/tests/gptdiff/gen_bundle.py" "$d" --steps "$steps" --tiling "$tiling" $([ "$mode" = precise ] && echo --precise) >/dev/null
    $SSH "$HOST" "mkdir -p $REMOTE/gptdiff_bundles/$c"
    rsync -az -e "$SSH" "$d/bundle/" "$HOST:$REMOTE/gptdiff_bundles/$c/"
    CASES="$CASES $c"
  done
done
# ONE lock hold for everything (the box is shared; wait for the lock once, not once per case).
REMOTE_RUN="cd $REMOTE && for c in $CASES; do ./kvm_exec --blocks 4 --quiet --dump-buffers gptdiff_bundles/\$c > gptdiff_bundles/\$c.dump 2> gptdiff_bundles/\$c.err; echo \"\$c exit \$?\"; done"
if [ "$SANITIZE" = 1 ]; then
  REMOTE_RUN="$REMOTE_RUN; for c in cpu_s2_tf32 cpu_s5_tf32 cpu_s2_precise cpu_s5_precise; do for tool in memcheck racecheck synccheck; do echo \"== compute-sanitizer \$tool \$c\"; compute-sanitizer --tool \$tool --error-exitcode 9 ./kvm_exec --blocks 4 --quiet gptdiff_bundles/\$c 2>&1 | tail -3; echo \"sanitizer exit \${PIPESTATUS[0]}\"; done; done"
fi
$SSH "$HOST" "flock -w 14400 ~/.gpu.lock timeout 1200 bash -c '$REMOTE_RUN'" | tee "$LOG/runs.txt"
grep -E "exit [1-9]" "$LOG/runs.txt" && fail=1
for c in $CASES; do
  d="$WORK/$c"; mode="${c##*_}"
  scp -q -o BatchMode=yes ${KVM_KEY:+-i $KVM_KEY} "$HOST:$REMOTE/gptdiff_bundles/$c.dump" "$d/dump.txt"
  tail -n1 "$d/dump.txt" | sed "s/^/$c kvm_exec: /"
  python3 "$ROOT/tests/gptdiff/golden_check.py" "$d/workload.json" "$d/bundle" "$d/dump.txt" --mode "$mode" | tee "$d/check.txt" || fail=1
done
echo "logs: $LOG"; exit $fail
