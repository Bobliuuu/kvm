#!/usr/bin/env bash
# Sync this worktree to a GPU box, build everything in tests/litmus and run a stage under the GPU lock.
# Usage: KVM_HOST=user@host tests/litmus/run_remote.sh <stage> [args...]
#   stages: build | litmus | mutants | e2e | probes | fuzz | sanitize | shell '<cmd>'
# Logs: remote ~/kvm-review-scratch/logs/<stage>.txt, copied to ./logs/litmus-<stage>.txt (git-ignored).
set -euo pipefail
HOST="${KVM_HOST:?set KVM_HOST=user@host (environment variable only)}"
HERE="$(cd "$(dirname "$0")/../.." && pwd)"; R='~/kvm-review-scratch'; STAGE="${1:?stage}"; shift || true
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=no"
rsync -az --delete --exclude .git --exclude results --exclude out --exclude logs --exclude fuzz_out -e "$SSH" "$HERE/" "$HOST:kvm-review-scratch/"
NV='$(command -v nvcc || ls /usr/local/cuda*/bin/nvcc | tail -1)'
LOCK='flock -w 900 ~/.gpu.lock timeout'; [ -z "${KVM_NO_LOCK:-}" ] || LOCK='timeout'   # KVM_NO_LOCK=1: an outer hold exists
ARCH="${KVM_ARCH:-sm_80}"; JOBS="${KVM_JOBS:-2}"
mkdir -p "$HERE/logs"
case "$STAGE" in
  build)
    $SSH "$HOST" "cd $R && mkdir -p out logs && KVM_ARCH=$ARCH KVM_JOBS=$JOBS tests/litmus/build.sh \$PWD/out && B='-O3 -arch=$ARCH -std=c++17 -Iinclude --fmad=false -lineinfo' &&
      $NV \$B kvm_exec.cu -o out/kvm_exec && $NV \$B -DKVM_EXEC_NO_MAIN tests/litmus/e2e.cu -o out/e2e && $NV \$B -DKVM_EXEC_NO_MAIN -DKVM_BREAK_FENCE tests/litmus/e2e.cu -o out/e2e_breakfence && echo BUILD OK" ;;
  buildnew)  # the slot-recycling executor, exported from a branch into ~/kvm-review-new (rsync it first: git archive <branch> include kvm_exec.cu)
    $SSH "$HOST" "ARCH=$ARCH bash -s" <<'REMOTE'
set -e
cd ~/kvm-review-scratch; mkdir -p out
NV=$(command -v nvcc || ls /usr/local/cuda*/bin/nvcc | tail -1); N=$HOME/kvm-review-new
B="-O3 -arch=$ARCH -std=c++17 -lineinfo -Wno-deprecated-gpu-targets"; X="-I$N/include"
$NV $B $X -DLIT_NEW tests/litmus/litmus.cu -o out/litmusN_prod &
$NV $B $X -DLIT_NEW -DKVM_BREAK_PARK_RECHECK -DLIT_MUTANT='"break_park_recheck"' tests/litmus/litmus.cu -o out/litmusN_norecheck &
$NV $B $X -DKVM_EXEC_NO_MAIN -DKVM_NEW -DKVM_EXEC_CU="\"$N/kvm_exec.cu\"" tests/litmus/e2e.cu -o out/e2eN &
$NV $B $X -DKVM_EXEC_NO_MAIN -DKVM_NEW -DKVM_BREAK_FENCE -DKVM_EXEC_CU="\"$N/kvm_exec.cu\"" tests/litmus/e2e.cu -o out/e2eN_breakfence &
$NV $B $X $N/kvm_exec.cu -o out/kvm_execN &
wait; ls out | grep N
REMOTE
    ;;
  shell) $SSH "$HOST" "cd $R/out && $*" ;;
  *)
    OUT="logs/$STAGE.txt"
    case "$STAGE" in
      litmus)  CMD="cd out && for t in sb dekker1 dekker3 mp; do $LOCK 900 ./litmus_prod \$t ${LK:-300} ${LL:-10}; done; $LOCK 1500 ./litmus_prod queue ${LQ:-10}" ;;
      mutants) CMD="cd out && for m in nodekkerfence norecheck exchbefore nofence; do $LOCK 300 ./litmus_\$m dekker1 100 5; $LOCK 300 ./litmus_\$m dekker3 50 5; done; $LOCK 300 ./litmus_weakpub mp 40 10; $LOCK 300 ./litmus_nofence mp 40 10; $LOCK 300 ./litmus_nofence sb 300 10; $LOCK 400 ./litmus_popplain queue 2; $LOCK 400 ./litmus_qrelax queue 5" ;;
      e2e)     CMD="cd out && for t in dag versions ring bigrecord probe_closerace; do $LOCK 600 ./e2e \$t ${ES:-3}; done; $LOCK 600 ./e2e_breakfence versions 1" ;;
      probes)  CMD="cd out && for t in probe_timeout probe_timeout_busy probe_deadlock probe_aging probe_spin; do $LOCK 200 ./e2e \$t; done" ;;
      fuzz)    CMD="$LOCK 3000 python3 tests/litmus/fuzz.py --exe out/kvm_exec --out out/fuzz_out --cases ${FC:-400} --seed ${FS:-1} ${FZ:-}" ;;
      devfuzz) CMD="cd out && $LOCK 1500 bash -c 'for i in \$(seq ${DS:-1} ${DN:-300}); do timeout 30 ./e2e devfuzz \$i 2>&1 | grep -E \"DEVFUZZ|EXCEPTION|rror\" | tail -2; echo rc=\${PIPESTATUS[0]} case=\$i; done'" ;;
      sanitize) CMD="for tool in memcheck initcheck synccheck racecheck; do for seed in live gen; do echo == \$tool \$seed; $LOCK 900 compute-sanitizer --tool \$tool --error-exitcode 9 out/kvm_exec --blocks 5 --max-tasks 4096 --live-cap 256 --ring-words 4096 --timeout-s 20 --golden out/fuzz_out/seed_\$seed/bundle 2>&1 | tail -12; done; done" ;;
      *) echo "unknown stage"; exit 2;;
    esac
    $SSH "$HOST" "cd $R && ($CMD) 2>&1 | tee logs/$STAGE.txt" | tee "$HERE/logs/litmus-$STAGE.txt" ;;
esac
