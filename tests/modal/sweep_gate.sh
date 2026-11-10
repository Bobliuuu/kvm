#!/usr/bin/env bash
# GPU gate for a tree that changes executor device code, operator packs or control flow, in ONE container:
#   modal run --detach tests/modal/run_suite.py --suite custom --tag NAME --cmd 'bash tests/modal/sweep_gate.sh'
# Stages (each prints "STAGE <name> exit <rc>"): exec suite, extra op pack, control bundles, operator-pack flow, generic zoo, and a
# short litmus check (prod binary only, every property under timeout -k; the full tests/litmus/run_litmus.sh can hang past any timeout
# on Modal, see ROADMAP.md Phase A). Optional: a tree at tests/modal/_main_tree (include/ + tests/litmus/) runs the same check as a baseline.
set -u
OUT=/logs/$TAG; mkdir -p "$OUT"
export KVM_NO_LOCK=1
stage() { local n=$1; shift; echo; echo "===== STAGE $n: $*"; local t0=$SECONDS; ( "$@" ) > "$OUT/$n.log" 2>&1; local rc=$?
  tail -25 "$OUT/$n.log"; echo "STAGE $n exit $rc ($((SECONDS - t0)) s)"; }

exec_suite() {
  python3 tests/gpu/make_initgen_vectors.py --check && echo "initgen vectors identical"
  bash tests/gpu/build.sh 2>&1 | grep -v "warning #177\|TILE\|\^\|^$\|Remark"
  timeout 2400 ./test_exec
}
extra() {
  cd tests/gpu_ops && nvcc -O2 -arch=sm_80 -std=c++17 --fmad=false -Xcompiler -ffp-contract=off -lineinfo -I../../include -I. test_extra.cu -o test_extra \
    && timeout 1500 ./test_extra
}
stage exec exec_suite
stage extra extra
stage ctl bash tests/ctl/run.sh "$OUT/ctl"
stage packs bash tests/packs/run.sh "$OUT/packs"
stage generic bash tests/generic/run_box.sh "$OUT/generic"
litmus_quick() {  # tree: build litmus_prod from that tree's headers, run each property with a hard kill
  local t=$1 o="$OUT/litmus_$2"; mkdir -p "$o"
  KVM_ONLY=prod KVM_JOBS=1 bash "$t/tests/litmus/build.sh" "$o" || return 1
  local f=0
  for p in "sb 100 5" "dekker1 100 5" "dekker3 50 5" "mp 100 5" "queue 5"; do
    timeout -k 10 600 "$o/litmus_prod" $p > "$o/${p%% *}.txt" 2>&1; rc=$?
    echo "litmus $2 ${p%% *}: exit $rc $(grep ^TOTAL "$o/${p%% *}.txt" | tail -1)"; [ $rc = 0 ] || f=1
  done; return $f
}
stage litmus_sweep litmus_quick . sweep
[ -d tests/modal/_main_tree ] && stage litmus_main litmus_quick tests/modal/_main_tree main
