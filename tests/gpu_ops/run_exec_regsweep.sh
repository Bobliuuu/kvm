#!/usr/bin/env bash
# End-to-end register/occupancy sweep of the FULL executor kernel on a GPT bundle (task wall time reported by kvm_exec), one GPU lock hold.
# Usage: KVM_HOST=user@host BUNDLE_DIR=<remote dir of an existing bundle> tests/gpu_ops/run_exec_regsweep.sh <tag> [variant ...]
# A variant is extra nvcc flags with spaces written as commas, e.g. -maxrregcount=128 or -DKVM_GPT_BIG_BK=32,-maxrregcount=168; "base" = none.
# Builds into a private remote scratch dir (never a shared one); kvm_exec.cu is only compiled, not modified.
set -euo pipefail
HOST="${KVM_HOST:?set KVM_HOST}"; BUNDLE="${BUNDLE_DIR:?remote bundle dir}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; TAG="${1:?tag}"; shift
VARS="${*:-base}"
SSH="ssh -o BatchMode=yes -o ServerAliveInterval=30"
R='~/kvm-ops-regsweep'
rsync -az -e "$SSH" --exclude .git --exclude results --exclude logs --exclude docs "$ROOT/" "$HOST:$R/"
OUT="$ROOT/results/2026-11-04-a100-ops-perf/$TAG.txt"
cat > "$OUT.job" <<JOB
cd $R && export PATH=\$PATH:/usr/local/cuda/bin
for v in $VARS; do f=\${v//,/ }; [ "\$v" = base ] && f=""; nvcc -O3 -arch=sm_80 -std=c++17 -Iinclude --fmad=false -Xcompiler -ffp-contract=off -lineinfo -diag-suppress 177 \$f kvm_exec.cu -o kvm_exec_\$v 2>&1 | grep -E "error"; done
flock -w 28800 ~/.gpu.lock bash -c 'for v in $VARS; do echo VARIANT \$v; for i in 1 2 3 4 5 6 7; do ./kvm_exec_\$v --quiet $BUNDLE 2>&1 | grep -E "^OK|FAIL|mismatch" | tail -1; done; done'
echo JOB_DONE
JOB
scp -q "$OUT.job" "$HOST:$R/job_$TAG.sh"; rm -f "$OUT.job"
$SSH "$HOST" "cd $R && nohup bash job_$TAG.sh > job_$TAG.log 2>&1 < /dev/null &"
until $SSH "$HOST" "grep -q JOB_DONE $R/job_$TAG.log"; do sleep 30; done
$SSH "$HOST" "cat $R/job_$TAG.log" | tee "$OUT"
