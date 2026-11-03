#!/usr/bin/env bash
# Build and run the GPU executor tests on a GPU box (the host comes from the environment, never from a file).
# Usage: KVM_HOST=user@host [KVM_KEY=~/.ssh/key] [KVM_ARCH=sm_80] ./run_exec_tests.sh [mode] [suite...]
#   mode: (none) functional suites | --negative (build -DKVM_BREAK_FENCE, expect the version suite to FAIL)
#         --sanitize (compute-sanitizer memcheck + racecheck on the smoke suite) | --bench (dispatch cost)
# Raw output goes to logs/<UTC>-<what>/ (git-ignored). The remote scratch dir is ~/kvm-exec-scratch.
set -euo pipefail
HOST="${KVM_HOST:?set KVM_HOST=user@host (an environment variable, never a committed file)}"
ARCH="${KVM_ARCH:-sm_80}"
SSH="ssh -o StrictHostKeyChecking=no -o BatchMode=yes ${KVM_KEY:+-i $KVM_KEY}"
DIR="$(cd "$(dirname "$0")" && pwd)"
MODE=""; case "${1:-}" in --negative|--sanitize|--bench) MODE="$1"; shift;; esac
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"; WHAT="exec${MODE:+${MODE//--/-}}"
LOG="$DIR/logs/$STAMP-$WHAT"; mkdir -p "$LOG"
REMOTE='~/kvm-exec-scratch'
NVCC_BIN='$(command -v nvcc || ls /usr/local/cuda*/bin/nvcc | tail -1)'
# only the executor's own sources go to the box (the repo's .gitignore hides kvm_* files from rsync filters)
rsync -az -e "$SSH" --delete --include 'include/***' --include 'tests/' --include 'tests/gpu/***' --include 'kvm_exec.cu' --exclude '*' "$DIR/" "$HOST:$REMOTE/"
case "$MODE" in
  --negative) FLAGS="-DKVM_BREAK_FENCE"; RUN="./test_exec --expect-fail versions";;
  --sanitize) FLAGS=""; RUN="";;
  --bench)    FLAGS=""; RUN="./test_exec bench";;
  *)          FLAGS=""; RUN="./test_exec ${*:-}";;
esac
{
  echo "# $WHAT @ $STAMP  commit $(git -C "$DIR" rev-parse --short HEAD)$(git -C "$DIR" diff --quiet HEAD -- . 2>/dev/null || echo '+dirty')"
  $SSH "$HOST" "cd $REMOTE && $NVCC_BIN --version | tail -2 && nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv,noheader && B=\"-O3 -arch=$ARCH -std=c++17 -Iinclude --fmad=false -Xcompiler -ffp-contract=off -lineinfo -DKVM_EXEC_NO_MAIN $FLAGS\"; echo BUILD: nvcc \$B tests/gpu/test_exec.cu -o test_exec; $NVCC_BIN \$B tests/gpu/test_exec.cu -o test_exec 2>&1"
  if [ "$MODE" = "--sanitize" ]; then
    for tool in memcheck racecheck synccheck; do
      echo "== compute-sanitizer --tool $tool smoke"
      $SSH "$HOST" "cd $REMOTE && flock -w 900 ~/.gpu.lock timeout 900 compute-sanitizer --tool $tool --error-exitcode 9 ./test_exec smoke 2>&1 | tail -40; echo \"sanitizer exit: \${PIPESTATUS[0]}\""
    done
  else
    $SSH "$HOST" "cd $REMOTE && flock -w 900 ~/.gpu.lock timeout 600 $RUN; echo \"exit: \$?\""
  fi
} 2>&1 | tee "$LOG/run.txt"
echo "log: $LOG/run.txt"
