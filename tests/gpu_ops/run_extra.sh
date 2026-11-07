#!/usr/bin/env bash
# Build and run tests/gpu_ops/test_extra.cu (extra operator pack) on the GPU box; the host comes from the environment.
# Usage: KVM_HOST=user@host [KVM_ARCH=sm_80] tests/gpu_ops/run_extra.sh [--sanitize]
# ONE GPU lock hold covers all runs (plain, and with --sanitize: memcheck, racecheck, synccheck); KVM_NO_LOCK=1 skips it (outer hold).
# Remote scratch: private ~/kvm-scratch-extra-$USER-$$ (removed on success; KVM_KEEP_SCRATCH=1 keeps it) or $KVM_SCRATCH. See RUNNING_TESTS.md.
set -euo pipefail
HOST="${KVM_HOST:?set KVM_HOST=user@host (an environment variable, never a committed file)}"
ARCH="${KVM_ARCH:-sm_80}"
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=no ${KVM_KEY:+-i $KVM_KEY}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
. "$ROOT/tests/remote_lib.sh"; kvm_scratch extra; kvm_lock "${KVM_LOCK_WAIT:-900}"
rsync -az -e "$SSH" --exclude .git --exclude results --exclude logs --exclude docs "$ROOT/" "$HOST:$REMOTE/"
$SSH "$HOST" "cd $REMOTE/tests/gpu_ops && NVCC=\$(command -v nvcc || ls /usr/local/cuda*/bin/nvcc | tail -1) && \$NVCC -O2 -arch=$ARCH -std=c++17 --fmad=false -Xcompiler -ffp-contract=off -lineinfo -I../../include -I. test_extra.cu -o test_extra"
RUN="./test_extra"
if [ "${1:-}" = "--sanitize" ]; then
  RUN="./test_extra; for t in memcheck racecheck synccheck; do echo == \$t; compute-sanitizer --tool \$t --error-exitcode 9 ./test_extra | tail -4; echo exit \${PIPESTATUS[0]}; done"
fi
$SSH "$HOST" "cd $REMOTE/tests/gpu_ops && $FLK timeout 1500 bash -c '$RUN'"
kvm_scratch_done
