#!/usr/bin/env bash
# Usage: KVM_HOST=user@host ./run_gpu_ops.sh <test_name|all> [extra args]   (builds tests/gpu_ops/<name>.cu on the box and runs it)
# Syncs include/ and tests/ (no --delete) to a private remote scratch dir ~/kvm-scratch-gpu-ops-$USER-$$ ($KVM_SCRATCH overrides;
# removed on success unless KVM_KEEP_SCRATCH=1). KVM_NO_LOCK=1 skips the GPU lock (outer hold). See RUNNING_TESTS.md.
set -euo pipefail
HOST="${KVM_HOST:?set KVM_HOST=user@host}"
DIR="$(cd "$(dirname "$0")" && pwd)"
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=no"
. "$DIR/tests/remote_lib.sh"; kvm_scratch gpu-ops; kvm_lock 900
rsync -az -e "$SSH" --exclude .git --exclude 'results' "$DIR/include" "$DIR/tests" "$HOST:$REMOTE/"
T="${1:?test name}"; shift || true
$SSH "$HOST" "cd $REMOTE/tests/gpu_ops && export PATH=\$PATH:/usr/local/cuda/bin && nvcc -O2 -arch=sm_80 -std=c++17 -lineinfo -I../../include -I. $T.cu -o $T 2>&1 | grep -v -i 'remark\|^\s*\^\|^$' ; $FLK timeout 600 ${WRAP:-} ./$T $*"
kvm_scratch_done
