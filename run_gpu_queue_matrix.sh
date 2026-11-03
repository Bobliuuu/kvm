#!/usr/bin/env bash
# Validation matrix for all benchmark modes (incl. the ready queue) on a shared GPU box, smallest cases first.
# Usage: KVM_HOST=user@host [KVM_KEY=key] [KVM_ARCH=sm_80] [KVM_SCRATCH=kvm-queue-scratch] ./run_gpu_queue_matrix.sh
# Syncs kvm.cu to a scratch dir (never the box's main checkout), builds, then runs every case under flock + timeout.
# Output is also kept locally under logs/<UTC time>-queue-matrix/ (git-ignored).
set -euo pipefail
HOST="${KVM_HOST:?set KVM_HOST=user@host (an environment variable, never a committed file)}"
ARCH="${KVM_ARCH:-sm_80}"
SCRATCH="${KVM_SCRATCH:-kvm-queue-scratch}"
SSH="ssh -o StrictHostKeyChecking=no -o BatchMode=yes ${KVM_KEY:+-i $KVM_KEY}"
DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="$DIR/logs/$(date -u +%Y%m%dT%H%M%SZ)-queue-matrix"; mkdir -p "$LOG"
FLAGS="-O3 -arch=$ARCH -std=c++17 -DKTILE=32 -DKKC=64 -DKSTG=4"

$SSH "$HOST" "mkdir -p ~/$SCRATCH" && rsync -az -e "$SSH" "$DIR/kvm.cu" "$HOST:~/$SCRATCH/"
$SSH "$HOST" "cd ~/$SCRATCH && nvcc $FLAGS kvm.cu -o kvm && nvidia-smi --query-gpu=name,driver_version --format=csv,noheader && nvcc --version | tail -2" | tee "$LOG/build.txt"

# run <name> <timeout s> <command...>: one GPU job under the box lock; a failure stops the matrix.
run() {
  local name="$1" t="$2"; shift 2
  echo "=== $name: $* ===" | tee -a "$LOG/$name.txt"
  $SSH "$HOST" "cd ~/$SCRATCH && flock -w 900 ~/.gpu.lock timeout $t $*" 2>&1 | tee -a "$LOG/$name.txt"
  [[ ${PIPESTATUS[0]} -eq 0 ]] || { echo "FAILED: $name (log: $LOG/$name.txt)" >&2; exit 1; }
}
run selftest 120 ./kvm --selftest
run smoke 300 ./kvm --runs 1 --steps 2 --reps 2
run small 300 ./kvm --runs 1 --steps 10
run bench-r8 600 ./kvm --runs 8 --steps 300 --reps 7
run bench-r1 600 ./kvm --runs 1 --steps 300 --reps 7
run bench-r2 600 ./kvm --runs 2 --steps 300 --reps 7
run bench-r32 600 ./kvm --runs 32 --steps 100 --reps 5
run shapes 300 ./kvm --runs 4 --steps 50 --B 256 --D 256 --H 512 --nb 3 --reps 5
# Watchdog self-test: an unsatisfiable task must end the kernel with a diagnosis (exit 2) instead of hanging the GPU.
for m in claim queue; do
  echo "=== fault-$m ===" | tee -a "$LOG/fault.txt"
  out=$($SSH "$HOST" "cd ~/$SCRATCH && flock -w 900 ~/.gpu.lock timeout 120 ./kvm --runs 1 --steps 10 --reps 1 --modes $m --watchdog-ms 300 --inject-fault" 2>&1 || true)
  echo "$out" | tee -a "$LOG/fault.txt"
  grep -q "FAIL (watchdog)" <<<"$out" || { echo "watchdog did not fire for $m" >&2; exit 1; }
done
if $SSH "$HOST" 'command -v compute-sanitizer' >/dev/null; then
  for tool in memcheck racecheck; do
    run "$tool" 900 compute-sanitizer --tool "$tool" ./kvm --runs 2 --steps 10 --reps 1 --modes claim,queue --watchdog-ms 60000
  done
fi
echo "matrix passed; logs in $LOG"
