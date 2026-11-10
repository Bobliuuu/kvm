#!/usr/bin/env bash
# On the GPU box, from the repository root: build and run R's litmus harness against THIS tree's headers. Called by run_exec_tests.sh --litmus.
# Properties must report 0 violations; every mutant must report violations (its harness can fail); probes must print no FINDING.
set -u
ARCH="${KVM_ARCH:-sm_80}"; NV="$(command -v nvcc || ls /usr/local/cuda*/bin/nvcc | tail -1)"; B="-O3 -arch=$ARCH -std=c++17 -Iinclude --fmad=false -lineinfo"
mkdir -p out
# LIT_PREBUILT=1: use the binaries already in out/ (built elsewhere with LIT_BUILD_ONLY=1, same tree); LIT_BUILD_ONLY=1: build, then exit.
# The five executor builds below are independent and each needs a long ptxas run: they run in parallel (host RAM ~35 GB each).
if [ -z "${LIT_PREBUILT:-}" ]; then
echo "BUILD: tests/litmus/build.sh (prod + 10 mutants), e2e, e2e -DKVM_BREAK_FENCE|CHUNK_RELEASE|CONT, kvm_exec"
bash tests/litmus/build.sh "$PWD/out" 2>&1 | tail -2 &
$NV $B -DKVM_EXEC_NO_MAIN -DKVM_NEW tests/litmus/e2e.cu -o out/e2e 2>&1 | grep -i " error" &
$NV $B -DKVM_EXEC_NO_MAIN -DKVM_NEW -DKVM_BREAK_FENCE tests/litmus/e2e.cu -o out/e2e_breakfence 2>&1 | grep -i " error" &
$NV $B kvm_exec.cu -o out/kvm_exec 2>&1 | grep -i " error" &
$NV $B -DKVM_EXEC_NO_MAIN -DKVM_NEW -DKVM_BREAK_CHUNK_RELEASE tests/litmus/e2e.cu -o out/e2e_breakchunk 2>&1 | grep -i " error" &
$NV $B -DKVM_EXEC_NO_MAIN -DKVM_NEW -DKVM_BREAK_CONT tests/litmus/e2e.cu -o out/e2e_breakcont 2>&1 | grep -i " error" &
wait
ls out | tr '\n' ' '; echo
fi
[ -n "${LIT_BUILD_ONLY:-}" ] && exit 0
L="timeout"; [ -n "${KVM_NO_LOCK:-}" ] || L="flock -w 7200 $HOME/.gpu.lock timeout"; cd out; fail=0; ok_props=0; ok_mut=0
prop() {  # name, timeout, command...
  local n="$1" t="$2"; shift 2; local o; o="$($L "$t" "$@" 2>&1)"; local rc=$?; local last; last="$(echo "$o" | grep -E "^TOTAL" | tail -1)"
  echo "property $n: $last (exit $rc)"; if [ $rc -eq 0 ]; then ok_props=$((ok_props+1)); else fail=$((fail+1)); echo "$o" | tail -5; fi; }
mut() {
  local n="$1" t="$2"; shift 2; local o; o="$($L "$t" "$@" 2>&1)"; local rc=$?; local last; last="$(echo "$o" | grep -E "^TOTAL" | tail -1)"
  echo "mutant $n: $last (exit $rc)"; if [ $rc -eq 1 ]; then ok_mut=$((ok_mut+1)); elif [ $rc -eq 124 ]; then echo "  (hung: the broken claim livelocks the queue; counted as detected)"; ok_mut=$((ok_mut+1)); else echo "  EXPECTED violations"; fail=$((fail+1)); fi; }
echo "== properties on the production headers (must be 0 violations)"
prop "sb(controls included)" 900 ./litmus_prod sb "${LK:-100}" "${LL:-5}" 2>/dev/null || true
prop dekker1 900 ./litmus_prod dekker1 "${LK:-100}" "${LL:-5}"
prop dekker3 900 ./litmus_prod dekker3 "${LK:-100}" "${LL:-5}"
prop mp 900 ./litmus_prod mp "${LK:-100}" "${LL:-5}"
prop queue 1500 ./litmus_prod queue "${LQ:-5}"
prop "queue(fifo)" 1500 ./litmus_prod queuefifo "${LQ:-5}"
echo "== mutants (each must report violations)"
for m in exchbefore norecheck; do mut "$m/dekker1" 300 ./litmus_$m dekker1 100 5; mut "$m/dekker3" 300 ./litmus_$m dekker3 50 5; done
mut weakpub/mp 300 ./litmus_weakpub mp 40 10
# R's documented finding S4/row 4: on the A100 the Dekker fences and the MP fence are not observably needed (acq_rel RMWs behave sequentially
# consistent there); the model requires them. Reported, not required to fail.
for c in "nodekkerfence dekker1 100 5" "nodekkerfence dekker3 50 5" "nofence mp 40 10"; do set -- $c; o="$($L 300 ./litmus_$1 $2 $3 $4 2>&1)"; echo "mutant $1/$2 (informational, expected NOT detected on A100): $(echo "$o" | grep ^TOTAL | tail -1)"; done
mut popplain/queue 120 ./litmus_popplain queue 2
mut popticket/queue 120 ./litmus_popticket queue 2
mut fifoticket/queuefifo 120 ./litmus_fifoticket queuefifo 2
mut fifotail/queuefifo 120 ./litmus_fifotail queuefifo 2
o="$($L 400 ./litmus_qrelax queue 5 2>&1)"; echo "mutant qrelax/queue (S4, expected NOT detected on A100: the release store is required by the model, not observable here): $(echo "$o" | grep ^TOTAL | tail -1)"
echo "== e2e scenarios with slot recycling and windows (each must pass), and the KVM_BREAK_FENCE control (must report stale reads)"
for t in dag versions ring bigrecord lateadmit contver probe_closerace; do o="$($L 900 ./e2e $t "${ES:-2}" 2>&1)"; rc=$?; echo "e2e $t: $(echo "$o" | tail -1) (exit $rc)"; [ $rc -eq 0 ] && ok_props=$((ok_props+1)) || { fail=$((fail+1)); echo "$o" | tail -5; }; done
o="$($L 900 ./e2e_breakfence versions 1 2>&1)"; rc=$?; echo "e2e_breakfence versions: $(echo "$o" | tail -1) (exit $rc)"; [ $rc -ne 0 ] && ok_mut=$((ok_mut+1)) || { fail=$((fail+1)); echo "  EXPECTED stale reads"; }
# chunked release (indegree add-then-check) and continuation (version wait) controls, each on the scenario that exercises it: lateadmit
# makes predecessors finish before their successors are admitted, contver offers the continuation a successor whose version wait is unmet
for c in "breakchunk lateadmit" "breakcont contver"; do set -- $c; o="$($L 300 ./e2e_$1 $2 "${ES:-2}" 2>&1)"; rc=$?; echo "e2e_$1 $2 (control): $(echo "$o" | tail -1) (exit $rc)"; if [ $rc -ne 0 ]; then echo "  detected"; ok_mut=$((ok_mut+1)); else echo "  NOT DETECTED (the scenario no longer reaches the broken step)"; fail=$((fail+1)); fi; done
echo "== probes (R's reproductions of C1, C2, C5, C6: each must print ok)"
for t in probe_timeout probe_timeout_busy probe_deadlock probe_aging probe_spin; do $L 200 ./e2e $t 2>&1 | grep "E2E probe" | tee /tmp/probe_$$.txt; if grep -q FINDING /tmp/probe_$$.txt || ! grep -q "ok" /tmp/probe_$$.txt; then fail=$((fail+1)); else ok_props=$((ok_props+1)); fi; done
echo "SUMMARY: $ok_props properties/scenarios/probes passed, $ok_mut mutants/controls detected, $fail failures"
[ $fail -eq 0 ] && echo "LITMUS PASS" || echo "LITMUS FAIL"
exit $fail
