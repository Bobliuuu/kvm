#!/usr/bin/env bash
# Usage: KVM_HOST=user@host ./run_gpu_ops.sh <test_name|all> [extra args]   (builds tests/gpu_ops/<name>.cu on the box and runs it)
# Syncs only this worktree (minus git-ignored files and .git, plus the ignored include/ sources) to a scratch dir.
set -euo pipefail
HOST="${KVM_HOST:?set KVM_HOST=user@host}"
DIR="$(cd "$(dirname "$0")" && pwd)"
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=no"
rsync -az -e "$SSH" --exclude .git --exclude 'results' "$DIR/include" "$DIR/tests" "$HOST:~/kvm-gptops-scratch/"
T="${1:?test name}"; shift || true
$SSH "$HOST" "cd ~/kvm-gptops-scratch/tests/gpu_ops && export PATH=\$PATH:/usr/local/cuda/bin && nvcc -O2 -arch=sm_80 -std=c++17 -lineinfo -I../../include -I. $T.cu -o $T 2>&1 | grep -v -i 'remark\|^\s*\^\|^$' ; flock -w 900 ~/.gpu.lock timeout 600 ${WRAP:-} ./$T $*"
