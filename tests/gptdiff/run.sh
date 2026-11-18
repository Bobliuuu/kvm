#!/usr/bin/env bash
# GPT golden check on a GPU box: gpt_tiny AdamW bundles (1, 2, 5 steps; 1xTF32 and 3xTF32 GEMMs) through mvm_exec vs the CPU golden.
# Usage: MVM_HOST=user@host [MVM_KEY=key] [MVM_ARCH=sm_80] tests/gptdiff/run.sh [--sanitize]
# Bundles are generated here (no GPU needed), run remotely under flock, and checked here by golden_check.py.
# Remote scratch: private ~/mvm-scratch-gptdiff-$USER-$$ ($MVM_SCRATCH overrides; removed on success unless MVM_KEEP_SCRATCH=1).
# MVM_NO_LOCK=1 skips the GPU lock (an outer hold exists). See RUNNING_TESTS.md.
set -euo pipefail
SANITIZE=0; [ "${1:-}" = "--sanitize" ] && SANITIZE=1
HOST="${MVM_HOST:?set MVM_HOST=user@host (an environment variable, never a committed file)}"
ARCH="${MVM_ARCH:-sm_80}"
SSH="ssh -o StrictHostKeyChecking=no -o BatchMode=yes ${MVM_KEY:+-i $MVM_KEY}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
. "$ROOT/tests/remote_lib.sh"; mvm_scratch gptdiff; mvm_lock 14400
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$ROOT/logs/$STAMP-gptdiff"; mkdir -p "$LOG"
WORK="$LOG/bundles"; mkdir -p "$WORK"
rsync -az -e "$SSH" --exclude .git --exclude results --exclude logs --exclude docs "$ROOT/" "$HOST:$REMOTE/"
$SSH "$HOST" "cd $REMOTE && NVCC=\$(command -v nvcc || ls /usr/local/cuda*/bin/nvcc | tail -1) && \$NVCC -O3 -arch=$ARCH -std=c++17 -Iinclude --fmad=false -Xcompiler -ffp-contract=off -lineinfo -diag-suppress 177 mvm_exec.cu -o mvm_exec"
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
REMOTE_RUN="cd $REMOTE && for c in $CASES; do ./mvm_exec --blocks 4 --quiet --dump-buffers gptdiff_bundles/\$c > gptdiff_bundles/\$c.dump 2> gptdiff_bundles/\$c.err; echo \"\$c exit \$?\"; done"
if [ "$SANITIZE" = 1 ]; then
  REMOTE_RUN="$REMOTE_RUN; for c in cpu_s2_tf32 cpu_s5_tf32 cpu_s2_precise cpu_s5_precise; do for tool in memcheck racecheck synccheck; do echo \"== compute-sanitizer \$tool \$c\"; compute-sanitizer --tool \$tool --error-exitcode 9 ./mvm_exec --blocks 4 --quiet gptdiff_bundles/\$c 2>&1 | tail -3; echo \"sanitizer exit \${PIPESTATUS[0]}\"; done; done"
fi
$SSH "$HOST" "$FLK timeout 1200 bash -c '$REMOTE_RUN'" | tee "$LOG/runs.txt"
grep -E "exit [1-9]" "$LOG/runs.txt" && fail=1
for c in $CASES; do
  d="$WORK/$c"; mode="${c##*_}"
  scp -q -o BatchMode=yes ${MVM_KEY:+-i $MVM_KEY} "$HOST:$REMOTE/gptdiff_bundles/$c.dump" "$d/dump.txt"
  tail -n1 "$d/dump.txt" | sed "s/^/$c mvm_exec: /"
  python3 "$ROOT/tests/gptdiff/golden_check.py" "$d/workload.json" "$d/bundle" "$d/dump.txt" --mode "$mode" | tee "$d/check.txt" || fail=1
done
mvm_scratch_done
echo "logs: $LOG"; exit $fail
