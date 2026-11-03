#!/usr/bin/env bash
# Usage: ./run_remote.sh [kvm args]   (host via KVM_HOST, default ubuntu@129.146.78.123)
set -euo pipefail
HOST="${KVM_HOST:-ubuntu@129.146.78.123}"
DIR="$(cd "$(dirname "$0")" && pwd)"
rsync -az --exclude kvm --exclude .git "$DIR/" "$HOST:~/kittens-vm/"
ssh "$HOST" "cd ~/kittens-vm && nvcc -O3 -arch=sm_80 -std=c++17 kvm.cu -o kvm && flock -w 900 ~/.gpu.lock timeout 600 ./kvm $*"
