#!/usr/bin/env bash
# Build and run the scheduler tests on a GPU box (the host comes from the environment, never from a file).
# Usage: KVM_HOST=user@host [KVM_KEY=~/.ssh/key] [KVM_ARCH=sm_80] ./run_sched_tests.sh [mode] [suite...]
#   mode: (none) functional suites: cost share reserve edf inherit rand
#         --negative  starve_neg and inherit_neg with --expect-fail (the harness must DETECT starvation / deadlock)
#         --sanitize  compute-sanitizer memcheck + racecheck + synccheck on the `smoke` suite
#         --bench     measurements: calib dispatch arbiter rebalance baselines overload affinity mix
#                     (KVM_BUILD_ARGS='-DKVM_COST_LEGACY' builds with the pre-audit 20-entry cost table, the A/B baseline of `mix`)
#         --exec      the executor's own suites (tests/gpu/test_exec.cu) against this tree, scheduler off
# Raw output goes to logs/<UTC>-<what>/ (git-ignored). The remote scratch dir is ~/kvm-sched-scratch.
set -euo pipefail
FLK="flock -w 7200 ~/.gpu.lock"; [ -z "${KVM_NO_LOCK:-}" ] || FLK=""   # KVM_NO_LOCK=1: the caller already holds the GPU lock (tests/run_all_remote.sh)
HOST="${KVM_HOST:?set KVM_HOST=user@host (an environment variable, never a committed file)}"
ARCH="${KVM_ARCH:-sm_80}"
SSH="ssh -o StrictHostKeyChecking=no -o BatchMode=yes ${KVM_KEY:+-i $KVM_KEY}"
DIR="$(cd "$(dirname "$0")" && pwd)"
MODE=""; case "${1:-}" in --negative|--sanitize|--bench|--exec) MODE="$1"; shift;; esac
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"; WHAT="sched${MODE:+${MODE//--/-}}"
LOG="$DIR/logs/$STAMP-$WHAT"; mkdir -p "$LOG"
REMOTE="${KVM_REMOTE:-~/kvm-sched-scratch}"
NVCC_BIN='$(command -v nvcc || ls /usr/local/cuda*/bin/nvcc | tail -1)'
rsync -az -e "$SSH" --delete --include 'include/***' --include 'tests/' --include 'tests/gpu/***' --include 'runtime/' --include 'runtime/initgen.py' --include 'runtime/ref_types.py' --include 'kvm_exec.cu' --exclude '*' "$DIR/" "$HOST:$REMOTE/"
SRC=tests/gpu/test_sched.cu; BIN=test_sched
[ "$MODE" = "--exec" ] && { SRC=tests/gpu/test_exec.cu; BIN=test_exec; }
{
  echo "# $WHAT @ $STAMP  commit $(git -C "$DIR" rev-parse --short HEAD)$(git -C "$DIR" diff --quiet HEAD -- . 2>/dev/null || echo '+dirty')"
  BUILDER=build_sched.sh; [ "$MODE" = "--exec" ] && BUILDER=build.sh
  $SSH "$HOST" "cd $REMOTE && $NVCC_BIN --version | tail -2 && nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv,noheader && echo BUILD: tests/gpu/$BUILDER && KVM_ARCH=$ARCH bash tests/gpu/$BUILDER ${KVM_BUILD_ARGS:-} 2>&1 | grep -v 'warning\|^ *\^\|^\$\|Remark' || true"
  run() { $SSH "$HOST" "cd $REMOTE && echo '\$ ./$BIN $*' && $FLK timeout 900 ./$BIN $*; echo \"exit: \$?\""; }
  case "$MODE" in
    --negative) run --expect-fail starve_neg; run --expect-fail inherit_neg;;
    --sanitize)
      for tool in memcheck racecheck synccheck; do
        echo "== compute-sanitizer --tool $tool smoke"
        $SSH "$HOST" "cd $REMOTE && $FLK timeout 900 compute-sanitizer --tool $tool --error-exitcode 9 ./$BIN --quiet smoke 2>&1 | tail -30; echo \"sanitizer exit: \${PIPESTATUS[0]}\""
      done;;
    --bench) run --quiet ${*:-calib dispatch arbiter rebalance baselines overload affinity};;
    --exec)  run --quiet ${*:-};;
    *)       run ${*:---quiet cost share reserve edf inherit rand};;
  esac
} 2>&1 | tee "$LOG/run.txt"
echo "log: $LOG/run.txt"
