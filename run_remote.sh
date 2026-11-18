#!/usr/bin/env bash
# MVM_REMOTE_DIR=~/dir (default ~/magni-vm; use your own on a shared box), MVM_LOCK_WAIT/MVM_TIMEOUT seconds (default 900).
# Usage: MVM_HOST=user@host [MVM_KEY=~/.ssh/key] [MVM_ARCH=sm_80] [KTILE= KKC= KSTG=] ./run_remote.sh [mvm args]
set -euo pipefail
HOST="${MVM_HOST:?set MVM_HOST=user@host for the GPU box (an environment variable, never a committed file)}"
ARCH="${MVM_ARCH:-sm_80}"
SSH="ssh -o StrictHostKeyChecking=no -o BatchMode=yes ${MVM_KEY:+-i $MVM_KEY}"
DIR="$(cd "$(dirname "$0")" && pwd)"
TAG="${ARCH}_t${KTILE:-32}_${KKC:-64}_${KSTG:-4}"
rsync -az -e "$SSH" --filter=':- .gitignore' --exclude .git "$DIR/" "$HOST:${MVM_REMOTE_DIR:-~/magni-vm}/"
$SSH "$HOST" "cd ${MVM_REMOTE_DIR:-~/magni-vm} && nvcc -O3 -arch=$ARCH -std=c++17 -DKTILE=${KTILE:-32} -DKKC=${KKC:-64} -DKSTG=${KSTG:-4} ${MVM_NVCC:-} mvm.cu -o mvm_$TAG 2>&1 | grep -v -i 'warning\|remark\|^\s*\^\|^$' ; flock -w ${MVM_LOCK_WAIT:-900} ~/.gpu.lock timeout ${MVM_TIMEOUT:-900} ./mvm_$TAG $*"
