#!/usr/bin/env bash
# Sync this worktree to a shared GPU box, build the engine object and the standalone test there, run it, and compare with the CPU oracle.
#   KVM_HOST=user@host doom/tests/run_gpu.sh [frames=200] [extra: SANITIZE=memcheck|racecheck|synccheck (1 = memcheck) runs under compute-sanitizer]
# The box is shared: everything GPU-side runs under flock; only this worktree is copied (to ~/kvm-doom-scratch).
set -euo pipefail
. "$(dirname "$0")/../lib.sh"
: "${KVM_HOST:?set KVM_HOST=user@host}"
FRAMES="${1:-200}"
ROOT="$(cd "$DOOM_DIR/.." && pwd)"
TS="$(date -u +%Y%m%dT%H%M%SZ)"; LOGDIR="$ROOT/logs/$TS-doomtest"; mkdir -p "$LOGDIR"
info "logs: $LOGDIR"
rsync -az --delete --exclude .git --exclude 'doom/build' --exclude 'doom/third_party' --exclude logs --exclude 'doom-gpu-v1-ref' "$ROOT/" "$KVM_HOST:kvm-doom-scratch/"
ssh "$KVM_HOST" "FRAMES=$FRAMES SAN='${SANITIZE:-}' bash -s" <<'REMOTE' 2>&1 | tee "$LOGDIR/run.log"
set -euo pipefail
cd ~/kvm-doom-scratch
export NVCC=$(ls /usr/local/cuda*/bin/nvcc | tail -1)
./doom/setup.sh
./doom/build.sh
./doom/cpu_build.sh
OUT=~/kvm-doom-scratch/logs/doomtest; rm -rf $OUT; mkdir -p $OUT
python3 doom/cpu_run.py --frames $FRAMES --out $OUT/cpu --log info
$NVCC -O2 -arch=sm_80 -rdc=true -Iinclude doom/tests/doomtest.cu doom/build/doom_dev.o -o doom/build/doomtest
RUN="doom/build/doomtest doom/third_party/doom1.wad $FRAMES $OUT/gpu"
if [ -n "$SAN" ]; then RUN="$(command -v compute-sanitizer || echo /usr/local/cuda/bin/compute-sanitizer) --tool ${SAN/#1/memcheck} --print-limit 20 $RUN"; fi
flock -w 900 ~/.gpu.lock timeout 600 $RUN
python3 doom/compare.py $OUT/cpu $OUT/gpu
REMOTE
