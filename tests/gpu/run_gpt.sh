#!/usr/bin/env bash
# On the GPU box, from the repository root: the frontend's gpt_tiny training bundle (544 tasks) through the CLI, with the scheduler off and in
# every mode; the final buffers must be byte-identical across all of them (B measured the result deterministic and schedule independent), then
# compute-sanitizer memcheck/racecheck/synccheck. Called by run_exec_tests.sh --gpt.
set -u
ARCH="${KVM_ARCH:-sm_80}"; NV="$(command -v nvcc || ls /usr/local/cuda*/bin/nvcc | tail -1)"; mkdir -p out
( cd frontend && rm -rf ../out/demo && PYTHONPATH=. python3 -m kvmfe.demo --all --out ../out/demo --train-steps 2 2>&1 | grep -E "demo:" )
$NV -O3 -arch=$ARCH -std=c++17 -Iinclude --fmad=false -Xcompiler -ffp-contract=off -lineinfo kvm_exec.cu -o out/kvm_exec_gpt 2>&1 | grep -i " error"
B=out/demo/bundles/train_w0; L="timeout 900"; [ -n "${KVM_NO_LOCK:-}" ] || L="flock -w 7200 $HOME/.gpu.lock timeout 900"; fail=0; ref=""
run() {  # name args...
  local n="$1"; shift; local o; o="$($L out/kvm_exec_gpt --quiet --dump-buffers "$@" $B 2>&1)"; local line; line="$(echo "$o" | grep -E '^(OK|FAIL):' | tail -1)"
  local h; h="$(echo "$o" | grep '^buffer' | md5sum | cut -c1-12)"; echo "$n: $line  buffers md5 $h"
  echo "$line" | grep -q "544/544" || { fail=$((fail+1)); echo "$o" | tail -3; }
  if [ -z "$ref" ]; then ref="$h"; elif [ "$h" != "$ref" ]; then fail=$((fail+1)); echo "  BUFFERS DIFFER from the first run"; fi; }
run "scheduler off, 16 blocks" --blocks 16
run "scheduler off, auto grid"
run "global, 2 classes" --sched global --classes 2 --blocks 16
run "static, 1 class" --sched static --classes 1 --blocks 16
run "dynamic, 2 classes" --sched dynamic --classes 2 --blocks 16 --weights 2,1 --min 1,0
run "dynamic, affinity" --sched dynamic --classes 2 --blocks 16 --affinity-us 50
for tool in memcheck racecheck synccheck; do
  o="$($L compute-sanitizer --tool $tool --error-exitcode 9 out/kvm_exec_gpt --quiet --blocks 4 --sched dynamic --classes 2 $B 2>&1 | tail -3)"; echo "sanitizer $tool: $(echo "$o" | tr '\n' ' ' | cut -c1-200)"
  echo "$o" | grep -qE "0 errors|0 hazards" || fail=$((fail+1))
done
[ $fail -eq 0 ] && echo "GPT PASS" || echo "GPT FAIL ($fail)"
exit $fail
