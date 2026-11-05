#!/usr/bin/env bash
# Audit: run the generated new-model bundles on the GPU box (one lock hold) and compare with the CPU golden.
#   KVM_HOST=user@host ./run.sh WORKDIR case...      (WORKDIR gets gen.py output per case; kvm_exec is built from the repo root)
set -euo pipefail
HOST="${KVM_HOST:?set KVM_HOST}"; D="$(cd "$(dirname "$0")" && pwd)"; ROOT="$D/../../.."; W="$1"; shift
SSH="ssh -o BatchMode=yes"; R='~/kvm-audit-newmodel'
for c in "$@"; do python3 -I "$D/gen.py" "$W/$c" "$c" >"$W/$c.gen.log" 2>&1 || { echo "gen FAILED for $c"; tail -3 "$W/$c.gen.log"; exit 1; }; done
rsync -az -e "$SSH" --delete --exclude .git --exclude results --exclude logs --exclude docs "$ROOT/" "$HOST:$R/"
$SSH "$HOST" "mkdir -p $R/nm"
for c in "$@"; do rsync -az -e "$SSH" "$W/$c/bundle/" "$HOST:$R/nm/$c/"; done
$SSH "$HOST" "cd $R && NVCC=\$(command -v nvcc || ls /usr/local/cuda*/bin/nvcc | tail -1) && \$NVCC -O3 -arch=sm_80 -std=c++17 -Iinclude --fmad=false -Xcompiler -ffp-contract=off -diag-suppress 177 kvm_exec.cu -o kvm_exec 2>&1 | grep -i error || true
 flock -w 14400 ~/.gpu.lock bash -c 'for c in $*; do echo == \$c; timeout 600 ./kvm_exec --blocks 4 --quiet --dump-buffers nm/\$c > nm/\$c.dump 2> nm/\$c.err; echo exit \$?; tail -3 nm/\$c.err; done'"
for c in "$@"; do scp -q -o BatchMode=yes "$HOST:$R/nm/$c.dump" "$W/$c.dump"; python3 -I "$D/compare.py" "$W/$c" "$W/$c.dump" | tee "$W/$c.cmp"; done
