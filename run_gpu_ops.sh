#!/usr/bin/env bash
# Usage: MVM_HOST=user@host ./run_gpu_ops.sh <test_name|all> [extra args]   (builds tests/gpu_ops/<name>.cu on the box and runs it)
# Syncs include/ and tests/ (no --delete) to a private remote scratch dir ~/mvm-scratch-gpu-ops-$USER-$$ ($MVM_SCRATCH overrides;
# removed on success unless MVM_KEEP_SCRATCH=1). MVM_NO_LOCK=1 skips the GPU lock (outer hold). See RUNNING_TESTS.md.
set -euo pipefail
HOST="${MVM_HOST:?set MVM_HOST=user@host}"
DIR="$(cd "$(dirname "$0")" && pwd)"
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=no"
. "$DIR/tests/remote_lib.sh"; mvm_scratch gpu-ops; mvm_lock 900
rsync -az -e "$SSH" --exclude .git --exclude 'results' "$DIR/include" "$DIR/tests" "$HOST:$REMOTE/"
T="${1:?test name}"; shift || true
$SSH "$HOST" "cd $REMOTE/tests/gpu_ops && export PATH=\$PATH:/usr/local/cuda/bin && nvcc -O2 -arch=sm_80 -std=c++17 -lineinfo -I../../include -I. $T.cu -o $T 2>&1 | grep -v -i 'remark\|^\s*\^\|^$' ; $FLK timeout 600 ${WRAP:-} ./$T $*"
mvm_scratch_done
