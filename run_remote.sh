#!/usr/bin/env bash
# Usage: [KTILE=32|64] ./run_remote.sh [kvm args]   (host via KVM_HOST, default ubuntu@129.146.78.123)
set -euo pipefail
HOST="${KVM_HOST:-ubuntu@129.146.78.123}"
DIR="$(cd "$(dirname "$0")" && pwd)"
rsync -az --exclude kvm --exclude kvm_dbg --exclude .git "$DIR/" "$HOST:~/kittens-vm/"
ssh "$HOST" "cd ~/kittens-vm && nvcc -O3 -arch=sm_80 -std=c++17 -DKTILE=${KTILE:-64} kvm.cu -o kvm_t${KTILE:-64} && flock -w 900 ~/.gpu.lock timeout 600 ./kvm_t${KTILE:-64} $*"
