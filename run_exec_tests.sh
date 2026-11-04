#!/usr/bin/env bash
# Build and run the GPU executor tests on a GPU box (the host comes from the environment, never from a file).
# Usage: KVM_HOST=user@host [KVM_KEY=~/.ssh/key] [KVM_ARCH=sm_80] ./run_exec_tests.sh [mode] [suite...]
#   mode: --litmus (R's adversarial litmus harness: real device functions + 8 mutants that must FAIL, e2e scenarios, probes that must report no FINDING)
#         --gpt (frontend gpt_tiny bundle, 544 tasks: same bytes in every scheduler mode, sanitizers)
#         --fuzz (R's CLI fuzz corpus: 0 crashes/hangs; FC=cases FS=seed FZ=--targeted)
#   mode: (none) functional suites | --negative (build -DKVM_BREAK_FENCE, expect the version suite to FAIL)
#         --sanitize (compute-sanitizer memcheck + racecheck on the smoke suite) | --bench (dispatch cost)
# Raw output goes to logs/<UTC>-<what>/ (git-ignored). The remote scratch dir is ~/kvm-exec-scratch.
set -euo pipefail
FLK="flock -w 7200 ~/.gpu.lock"; [ -z "${KVM_NO_LOCK:-}" ] || FLK=""   # KVM_NO_LOCK=1: the caller already holds the GPU lock (tests/run_all_remote.sh)
HOST="${KVM_HOST:?set KVM_HOST=user@host (an environment variable, never a committed file)}"
ARCH="${KVM_ARCH:-sm_80}"
SSH="ssh -o StrictHostKeyChecking=no -o BatchMode=yes ${KVM_KEY:+-i $KVM_KEY}"
DIR="$(cd "$(dirname "$0")" && pwd)"
MODE=""; case "${1:-}" in --negative|--negative-park|--sanitize|--bench|--litmus|--fuzz|--gpt) MODE="$1"; shift;; esac
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"; WHAT="exec${MODE:+${MODE//--/-}}"
LOG="$DIR/logs/$STAMP-$WHAT"; mkdir -p "$LOG"
REMOTE="${KVM_REMOTE:-~/kvm-exec-scratch}"
NVCC_BIN='$(command -v nvcc || ls /usr/local/cuda*/bin/nvcc | tail -1)'
# only the executor's own sources go to the box (the repo's .gitignore hides kvm_* files from rsync filters)
rsync -az -e "$SSH" --delete --include 'include/***' --include 'tests/' --include 'tests/gpu/***' --include 'tests/litmus/***' --include 'runtime/' --include 'runtime/***' --include 'frontend/***' --include 'examples/***' --include 'kvm_exec.cu' --exclude '*' "$DIR/" "$HOST:$REMOTE/"
case "$MODE" in
  --negative) FLAGS="-DKVM_BREAK_FENCE"; RUN="./test_exec --expect-fail versions";;
  --negative-park) FLAGS="-DKVM_BREAK_PARK_RECHECK"; RUN="./test_exec --expect-fail litmus";;
  --sanitize) FLAGS=""; RUN="";;
  --bench)    FLAGS=""; RUN="./test_exec bench";;
  *)          FLAGS=""; RUN="./test_exec ${*:-}";;
esac
{
  echo "# $WHAT @ $STAMP  commit $(git -C "$DIR" rev-parse --short HEAD)$(git -C "$DIR" diff --quiet HEAD -- . 2>/dev/null || echo '+dirty')"
  $SSH "$HOST" "cd $REMOTE && $NVCC_BIN --version | tail -2 && nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv,noheader && echo BUILD: tests/gpu/build.sh $FLAGS && KVM_ARCH=$ARCH bash tests/gpu/build.sh $FLAGS 2>&1"
  if [ "$MODE" = "--litmus" ]; then
    $SSH "$HOST" "cd $REMOTE && KVM_ARCH=$ARCH KVM_NO_LOCK=${KVM_NO_LOCK:-} bash tests/litmus/run_litmus.sh"
  elif [ "$MODE" = "--gpt" ]; then
    $SSH "$HOST" "cd $REMOTE && KVM_ARCH=$ARCH KVM_NO_LOCK=${KVM_NO_LOCK:-} bash tests/gpu/run_gpt.sh"
  elif [ "$MODE" = "--fuzz" ]; then
    $SSH "$HOST" "cd $REMOTE && KVM_ARCH=$ARCH FC=${FC:-300} FS=${FS:-1} FZ='${FZ:-}' KVM_NO_LOCK=${KVM_NO_LOCK:-} bash tests/litmus/run_fuzz.sh"
  elif [ "$MODE" = "--sanitize" ]; then
    for tool in memcheck racecheck synccheck; do
      echo "== compute-sanitizer --tool $tool smoke"
      $SSH "$HOST" "cd $REMOTE && $FLK timeout 1800 compute-sanitizer --tool $tool --error-exitcode 9 ./test_exec smoke 2>&1 | tail -40; echo \"sanitizer exit: \${PIPESTATUS[0]}\""
    done
  else
    $SSH "$HOST" "cd $REMOTE && python3 tests/gpu/make_initgen_vectors.py --check && echo 'initgen vectors: regenerated from runtime/initgen.py, identical to the committed file'; $FLK timeout 2400 $RUN; echo \"exit: \$?\""
  fi
} 2>&1 | tee "$LOG/run.txt"
echo "log: $LOG/run.txt"
