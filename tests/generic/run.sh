#!/usr/bin/env bash
# Never-shipped models (tests/generic/zoo.py) on a GPU box vs the CPU golden, 3xTF32 GEMMs, plus two MANIFEST demos through demo/kvm_run.
#   KVM_HOST=user@host [KVM_KEY=key] [KVM_ARCH=sm_80] tests/generic/run.sh
# Bundles for every zoo case x {cpu, a100} tiling are generated here (no GPU needed), built and run remotely under ONE flock hold
# (kvm_exec --dump-buffers; kvm_run for the manifests), then checked here: golden_check.py --mode precise (every f32 buffer, activations
# through weights and optimizer state, within 2e-4 relative L2 of the golden) and demo/check_demo.py (loss falls, equals the CPU VM).
# Remote scratch: private ~/kvm-scratch-generic-$USER-$$ ($KVM_SCRATCH overrides; removed on success unless KVM_KEEP_SCRATCH=1).
# KVM_LOCK_WAIT=S sets the lock wait (default 900); KVM_PREBUILT=1 skips the nvcc builds (a kept $KVM_SCRATCH already has them).
# KVM_NO_LOCK=1 skips the GPU lock (an outer hold exists). See RUNNING_TESTS.md.
set -euo pipefail
HOST="${KVM_HOST:?set KVM_HOST=user@host (an environment variable, never a committed file)}"
ARCH="${KVM_ARCH:-sm_80}"
SSH="ssh -o StrictHostKeyChecking=no -o BatchMode=yes ${KVM_KEY:+-i $KVM_KEY}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
. "$ROOT/tests/remote_lib.sh"; kvm_scratch generic; kvm_lock "${KVM_LOCK_WAIT:-900}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$ROOT/logs/$STAMP-generic"; mkdir -p "$LOG"
WORK="$LOG/work"; mkdir -p "$WORK"
CASES="$(python3 -I "$ROOT/tests/generic/gen_bundle.py" --list | cut -d: -f1)"
MANIFESTS="reg_mlp cnn1d"
BUNDLES=""
for c in $CASES; do
  for tiling in cpu a100; do
    n="${c}_$tiling"; python3 -I "$ROOT/tests/generic/gen_bundle.py" "$WORK/$n" "$c" --tiling "$tiling" --precise | tee -a "$LOG/gen.txt"
    BUNDLES="$BUNDLES $n"
  done
done
for c in $MANIFESTS; do
  python3 -I "$ROOT/tests/generic/gen_bundle.py" "$WORK/man_$c" "$c" --manifest --steps 6 --window 3 --tiling a100 --precise | tee -a "$LOG/gen.txt"
done
rsync -az -e "$SSH" --exclude .git --exclude results --exclude logs --exclude docs "$ROOT/" "$HOST:$REMOTE/"
$SSH "$HOST" "mkdir -p $REMOTE/generic_bundles"
for n in $BUNDLES; do $SSH "$HOST" "mkdir -p $REMOTE/generic_bundles/$n"; rsync -az -e "$SSH" "$WORK/$n/bundle/" "$HOST:$REMOTE/generic_bundles/$n/"; done
for c in $MANIFESTS; do rsync -az -e "$SSH" "$WORK/man_$c/" "$HOST:$REMOTE/generic_bundles/man_$c/"; done
# build outside the lock hold (minutes of nvcc); the hold covers only the runs
[ -n "${KVM_PREBUILT:-}" ] || $SSH "$HOST" "cd $REMOTE && NVCC=\$(command -v nvcc || ls /usr/local/cuda*/bin/nvcc | tail -1) && \$NVCC --version | tail -1 && nvidia-smi --query-gpu=name,driver_version --format=csv,noheader && \
  FL='-O3 -arch=$ARCH -std=c++17 -I. -Iinclude --fmad=false -Xcompiler -ffp-contract=off -lineinfo -diag-suppress 177' && \
  \$NVCC \$FL kvm_exec.cu -o kvm_exec && \$NVCC \$FL demo/kvm_run.cu -o demo/kvm_run && ls -la kvm_exec demo/kvm_run" | tee "$LOG/build.txt"
RUN="cd $REMOTE && for n in $BUNDLES; do ./kvm_exec --blocks 4 --quiet --dump-buffers generic_bundles/\$n > generic_bundles/\$n.dump 2> generic_bundles/\$n.err; echo \"\$n exit \$?\"; done"
for c in $MANIFESTS; do
  RUN="$RUN; ./demo/kvm_run generic_bundles/man_$c --out generic_bundles/out_$c --observe '^loss\\.' --compile-cmd 'python3 runtime/compile_workload.py' > generic_bundles/out_$c.log 2>&1; echo \"man_$c kvm_run exit \$?\""
done
$SSH "$HOST" "$FLK timeout 1500 bash -c '$RUN'" | tee "$LOG/runs.txt"
fail=0
grep -E "exit [1-9]" "$LOG/runs.txt" && fail=1
for n in $BUNDLES; do
  d="$WORK/$n"
  scp -q -o BatchMode=yes ${KVM_KEY:+-i $KVM_KEY} "$HOST:$REMOTE/generic_bundles/$n.dump" "$d/dump.txt"
  tail -n1 "$d/dump.txt" | sed "s/^/$n kvm_exec: /"
  echo "== $n"; python3 -I "$ROOT/tests/gptdiff/golden_check.py" "$d/workload.json" "$d/bundle" "$d/dump.txt" --mode precise | tee "$d/check.txt" || fail=1
done
for c in $MANIFESTS; do
  d="$WORK/man_$c"; mkdir -p "$d/out"
  rsync -az -e "$SSH" --exclude buffers "$HOST:$REMOTE/generic_bundles/out_$c/" "$d/out/"
  scp -q -o BatchMode=yes ${KVM_KEY:+-i $KVM_KEY} "$HOST:$REMOTE/generic_bundles/out_$c.log" "$d/kvm_run.log" || true
  echo "== man_$c"; python3 -I "$ROOT/demo/check_demo.py" "$d" "$d/out" --observe '^loss\.' --golden "$d/golden.json" --loss-tol 2e-3 | tee "$d/check.txt" || fail=1
done
[ "$fail" = 0 ] && kvm_scratch_done
echo "logs: $LOG (exit $fail)"; exit $fail
