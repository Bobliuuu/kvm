#!/usr/bin/env bash
# KVM_REMOTE_DIR=~/dir (default ~/kittens-vm; use your own on a shared box), KVM_LOCK_WAIT/KVM_TIMEOUT seconds (default 900).
# Usage: KVM_HOST=user@host [KVM_KEY=~/.ssh/key] [KVM_ARCH=sm_80] [KTILE= KKC= KSTG=] ./run_remote.sh [kvm args]
set -euo pipefail
HOST="${KVM_HOST:?set KVM_HOST=user@host for the GPU box (an environment variable, never a committed file)}"
ARCH="${KVM_ARCH:-sm_80}"
SSH="ssh -o StrictHostKeyChecking=no -o BatchMode=yes ${KVM_KEY:+-i $KVM_KEY}"
DIR="$(cd "$(dirname "$0")" && pwd)"
TAG="${ARCH}_t${KTILE:-32}_${KKC:-64}_${KSTG:-4}"
rsync -az -e "$SSH" --filter=':- .gitignore' --exclude .git "$DIR/" "$HOST:${KVM_REMOTE_DIR:-~/kittens-vm}/"
$SSH "$HOST" "cd ${KVM_REMOTE_DIR:-~/kittens-vm} && nvcc -O3 -arch=$ARCH -std=c++17 -DKTILE=${KTILE:-32} -DKKC=${KKC:-64} -DKSTG=${KSTG:-4} ${KVM_NVCC:-} kvm.cu -o kvm_$TAG 2>&1 | grep -v -i 'warning\|remark\|^\s*\^\|^$' ; flock -w ${KVM_LOCK_WAIT:-900} ~/.gpu.lock timeout ${KVM_TIMEOUT:-900} ./kvm_$TAG $*"
