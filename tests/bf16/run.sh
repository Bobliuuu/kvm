#!/usr/bin/env bash
# bf16-storage GPU check on a GPU box: gpt_tiny bundles with real bf16 buffers (and the fp32 control) through kvm_exec.
# Usage: KVM_HOST=user@host [KVM_KEY=key] [KVM_ARCH=sm_80] tests/bf16/run.sh OUT_DIR
# Bundles are generated here (no GPU), run remotely under flock, dumps are checked here. Remote scratch: $KVM_SCRATCH or a private dir.
set -euo pipefail
HOST="${KVM_HOST:?set KVM_HOST=user@host (an environment variable, never a committed file)}"
ARCH="${KVM_ARCH:-sm_80}"
OUT="${1:?output directory}"
SSH="ssh -o StrictHostKeyChecking=no -o BatchMode=yes ${KVM_KEY:+-i $KVM_KEY}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
. "$ROOT/tests/remote_lib.sh"; kvm_scratch bf16; kvm_lock 14400
mkdir -p "$OUT"; python3 "$ROOT/tests/bf16/gen.py" "$OUT" | tee "$OUT/gen.txt"
rsync -az -e "$SSH" --exclude .git --exclude results --exclude logs --exclude docs "$ROOT/" "$HOST:$REMOTE/"
$SSH "$HOST" "cd $REMOTE && NVCC=\$(command -v nvcc || ls /usr/local/cuda*/bin/nvcc | tail -1) && \$NVCC -O3 -arch=$ARCH -std=c++17 -Iinclude --fmad=false -Xcompiler -ffp-contract=off -lineinfo -diag-suppress 177 kvm_exec.cu -o kvm_exec"
for c in f32_s50 bf16s_s50 bf16s_s1 bf16s_s5; do
  $SSH "$HOST" "mkdir -p $REMOTE/b/$c"; rsync -az -e "$SSH" "$OUT/$c/bundle/" "$HOST:$REMOTE/b/$c/"
done
$SSH "$HOST" "cd $REMOTE && $FLK timeout 1200 bash -c 'for c in f32_s50 bf16s_s50 bf16s_s1 bf16s_s5; do ./kvm_exec --blocks 4 --quiet --dump-buffers b/\$c > b/\$c.dump 2> b/\$c.err; echo \"\$c exit \$?\"; nvidia-smi --query-gpu=memory.used --format=csv,noheader; done'" | tee "$OUT/runs.txt"
fail=0
for c in f32_s50 bf16s_s50 bf16s_s1 bf16s_s5; do
  scp -q -o BatchMode=yes ${KVM_KEY:+-i $KVM_KEY} "$HOST:$REMOTE/b/$c.dump" "$OUT/$c/dump.txt"
  g=""; case "$c" in bf16s_s1|bf16s_s5) g="--golden";; esac
  echo "== $c"; python3 "$ROOT/tests/bf16/check.py" "$OUT/$c" "$OUT/$c/dump.txt" $g | tee "$OUT/$c/check.txt" || fail=1
done
kvm_scratch_done
exit $fail
