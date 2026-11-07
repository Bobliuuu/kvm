#!/usr/bin/env bash
# Local driver: rsync the repo to the GPU box, build kvm + kvm_lib.so there (CPU work, outside the lock), then run session.sh holding
# the shared GPU lock once.   KVM_HOST=<ssh host> [KVM_REMOTE_DIR=~/kvm-cu-perf] [KVM_ARCH=sm_80] [KTILE KKC KSTG] [KVM_LOCK_WAIT=7200]
#   bench/mlp_vs_torch/run_session.sh OUTNAME [RUNS...]      results are copied back to results/OUTNAME/
set -euo pipefail
HOST="${KVM_HOST:?set KVM_HOST}"; RD="${KVM_REMOTE_DIR:-~/kittens-vm}"; ARCH="${KVM_ARCH:-sm_80}"
FLAGS="-DKTILE=${KTILE:-32} -DKKC=${KKC:-64} -DKSTG=${KSTG:-4} ${KVM_NVCC:-}"
NAME="${1:?outname}"; shift
SSH="ssh -o BatchMode=yes ${KVM_KEY:+-i $KVM_KEY}"
DIR="$(cd "$(dirname "$0")/../.." && pwd)"
COMMIT="$(git -C "$DIR" rev-parse HEAD)$(git -C "$DIR" diff --quiet HEAD -- . && echo '' || echo '+uncommitted')"
rsync -az -e "$SSH" --filter=':- .gitignore' --exclude .git "$DIR/" "$HOST:$RD/"
$SSH "$HOST" "cd $RD && echo '$COMMIT' > COMMIT && nvcc -O3 -arch=$ARCH -std=c++17 $FLAGS kvm.cu -o kvm 2>&1 | grep -v -i 'warning\|remark\|^\s*\^\|^\$' ; \
  nvcc -O3 -arch=$ARCH -std=c++17 $FLAGS -shared -Xcompiler -fPIC -DKVM_LIB kvm.cu -o kvm_lib.so 2>&1 | grep -v -i 'warning\|remark\|^\s*\^\|^\$' ; \
  KVM_FLAGS='-O3 -arch=$ARCH -std=c++17 $FLAGS' flock -w ${KVM_LOCK_WAIT:-7200} ~/.gpu.lock timeout ${KVM_TIMEOUT:-3000} bench/mlp_vs_torch/session.sh out/$NAME $*"
mkdir -p "$DIR/results/$NAME"
rsync -az -e "$SSH" "$HOST:$RD/out/$NAME/" "$DIR/results/$NAME/"
echo "results in results/$NAME"
