#!/usr/bin/env bash
# Real Doom through the generic executor on a shared GPU box: compile the workload bundles, run them, diff every frame with the CPU oracle.
#   KVM_HOST=user@host doom/tests/run_exec.sh [frames=700] [chains=128] [rounds=24]
# Runs three scenarios on the same scheduler: doom alone; doom + matmul load at EQUAL priority; doom + matmul load with doom at HIGHER priority.
set -euo pipefail
. "$(dirname "$0")/../lib.sh"
: "${KVM_HOST:?set KVM_HOST=user@host}"
ROOT="$(cd "$DOOM_DIR/.." && pwd)"
TS="$(date -u +%Y%m%dT%H%M%SZ)"; LOGDIR="$ROOT/logs/$TS-doomexec"; mkdir -p "$LOGDIR"
info "logs: $LOGDIR"
rsync -az --delete --exclude .git --exclude 'doom/build' --exclude 'doom/third_party' --exclude logs "$ROOT/" "$KVM_HOST:kvm-doom-scratch/"
ssh "$KVM_HOST" "FRAMES=${1:-700} CHAINS=${2:-128} ROUNDS=${3:-24} bash -s" <<'REMOTE' 2>&1 | tee "$LOGDIR/run.log"
set -euo pipefail
cd ~/kvm-doom-scratch
export NVCC=$(ls /usr/local/cuda*/bin/nvcc | tail -1)
./doom/setup.sh && ./doom/build.sh && ./doom/cpu_build.sh
OUT=~/kvm-doom-scratch/logs/doomexec; rm -rf $OUT; mkdir -p $OUT
python3 doom/cpu_run.py --frames $FRAMES --out $OUT/cpu --log info
$NVCC -O3 -arch=sm_80 -std=c++17 -rdc=true -Iinclude --fmad=false -Xcompiler -ffp-contract=off doom/exec_main.cu doom/build/doom_dev.o -o doom/build/doom_exec 2>&1 | grep -v "cannot be statically determined" || true
run() {   # name, make_workload args
  local name="$1"; shift
  python3 doom/make_workload.py --frames $FRAMES "$@" --out $OUT/$name.json
  python3 runtime/compile_workload.py $OUT/$name.json $OUT/$name.bundle 2>&1 | grep -v "no reference semantics"
  echo "=== scenario $name"
  flock -w 900 ~/.gpu.lock timeout 600 doom/build/doom_exec $OUT/$name.bundle --wad doom/third_party/doom1.wad --out $OUT/$name
  python3 doom/compare.py $OUT/cpu $OUT/$name | head -3
  cat $OUT/$name/metrics.json
}
run alone
run load_equal --chains $CHAINS --rounds $ROUNDS --doom-priority 0 --compute-priority 0
run load_doom_first --chains $CHAINS --rounds $ROUNDS --doom-priority 10 --compute-priority 0
REMOTE
