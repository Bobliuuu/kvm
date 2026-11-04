#!/usr/bin/env bash
# Usage: KVM_HOST=user@host [TEST=1] [REF=1] [ONLY=mm|attn|mem] [GROUP=g] [VARIANTS="FLAG[,FLAG2] ..."] ./run_gpu_perf.sh <tag>
# Builds tests/gpu_ops/{test_ops,bench_ops} on the box (plus one bench_ops per VARIANTS entry, an nvcc define such as BENCH_MINB=2),
# then runs, under ONE hold of the GPU lock: test_ops (TEST=1), bench_ops (BENCH_REPS=7 repetitions each, median [min..max]), every
# variant, and the PyTorch/cuBLAS reference (REF=1, tests/gpu_ops/ref_torch.py). The remote job is detached (the lock queue can take
# hours and idle ssh sessions get dropped) and polled; output lands in results/2026-11-04-a100-ops-perf/<tag>.txt.
set -euo pipefail
HOST="${KVM_HOST:?set KVM_HOST=user@host}"
DIR="$(cd "$(dirname "$0")" && pwd)"
TAG="${1:?tag}"; OUT="$DIR/results/2026-11-04-a100-ops-perf"; mkdir -p "$OUT"
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ServerAliveInterval=30"
SCR="${SCRATCH:-kvm-ops-scratch}"; RD="~/$SCR/tests/gpu_ops"
rsync -az -e "$SSH" --exclude .git --exclude results "$DIR/include" "$DIR/tests" "$HOST:~/$SCR/"
cat > "$OUT/.job_$TAG.sh" <<JOB
cd $RD && export PATH=\$PATH:/usr/local/cuda/bin
NV='nvcc -O2 -arch=sm_80 -std=c++17 -lineinfo -I../../include -I.'
\$NV test_ops.cu -o test_ops 2>&1 | grep -E 'error'
\$NV -Xptxas -v bench_ops.cu -o bench_ops 2>&1 | grep -E 'error|spill' | sort | uniq -c
cuobjdump -res-usage bench_ops 2>&1 | grep -E 'Function|REG' | paste - - | sed -e 's/_Z[0-9]*//' | cut -c1-110
for v in ${VARIANTS:-}; do \$NV -D\${v//,/ -D} bench_ops.cu -o bench_\$v 2>&1 | grep error; done
\$NV --fmad=false -Xcompiler -ffp-contract=off test_extra.cu -o test_extra 2>&1 | grep error
flock -w 28800 ~/.gpu.lock bash -c '${TEST:+./test_extra > extra.log 2>&1; echo test_extra exit \$?; tail -2 extra.log; ./test_ops > test.log 2>&1; echo test_ops exit \$?; grep -E FAIL test.log | head -30; tail -1 test.log;}
  BENCH_ONLY=${ONLY:-} ./bench_ops; for v in ${VARIANTS:-}; do echo VARIANT \$v; BENCH_ONLY=${ONLY:-} ./bench_\$v; done; ${GROUP:+echo BENCH_GROUP=${GROUP}; BENCH_ONLY=mm BENCH_GROUP=${GROUP} ./bench_ops;} ${REF:+python3 -I ref_torch.py}'
echo JOB_DONE
JOB
scp -q -o BatchMode=yes "$OUT/.job_$TAG.sh" "$HOST:$RD/job_$TAG.sh"; rm -f "$OUT/.job_$TAG.sh"
$SSH "$HOST" "cd $RD && nohup bash job_$TAG.sh > job_$TAG.log 2>&1 < /dev/null &"
until $SSH "$HOST" "grep -q JOB_DONE $RD/job_$TAG.log"; do sleep 30; done
$SSH "$HOST" "cat $RD/job_$TAG.log" | tee "$OUT/$TAG.txt"
