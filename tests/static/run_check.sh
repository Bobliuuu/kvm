#!/usr/bin/env bash
# Static-path check on a GPU box (run it under the GPU lock), on small GPT and MLP bundles at several worker counts:
#   1. the static schedule with fusion off (--no-fuse) leaves every final buffer bitwise equal to the dynamic path;
#   2. the fusion pass leaves a bundle it does not fire on exactly as it was: the GPT bundle's default static run is bitwise equal too;
#   3. the fused MLP step (the MLP bundle's default static run) fires, repeats bitwise, and is judged as the MLP bench judges a run: final buffers within
#      1e-3 relative L2 of the CPU golden (the executor's --golden, the MLP bench's gate 1), and its loss and weights against float64 and
#      eager PyTorch (fused_vs_fp64.py: the bench's gate 3 and its float64 rule). It adds in its own order, so it is never compared
#      bitwise with the unfused path;
#   4. the plan's CPU replay (every dependency honoured, no deadlock).
#   KVM_EXE=./kvm_exec KVM_PY=python3 tests/static/run_check.sh [workers ...]   (from the repo root; KVM_PY needs numpy and torch; scratch in a temp dir)
#   KVM_CHECK_MLP_HIDDEN and KVM_CHECK_MLP_DEPTH set the MLP bundle's width and depth (default 72 and 2, the panel's narrow rows):
#   * The one-block fused step must fit one block's shared memory. 72 needs 99.3 KB at 16 rows, which an A100 has (163 KB) and an RTX 3090
#     (99 KB) has not; use 48 there.
#   * A width one block cannot hold takes the split form (gangs of 6 x S fmlp_split members), which fires only where a gang fits the
#     workers, and a worker count where the pass does not fire fails the check: KVM_CHECK_MLP_HIDDEN=256 tests/static/run_check.sh 16 48 107;
#     the panel's mlp256 shape needs 16 shares or more (96 members), so KVM_CHECK_MLP_HIDDEN=256 KVM_CHECK_MLP_DEPTH=4 tests/static/run_check.sh
#     107 108; and KVM_CHECK_MLP_DEPTH=4 with 128 at 48 107 108 for the deepest.
#   * A width the split form cannot hold either takes the grid form (one gang of every worker a step, fmlp_grid), sized for the run's workers:
#     KVM_CHECK_MLP_HIDDEN=1024 tests/static/run_check.sh 107 108 (the panel's mlp1024 shape).
set -euo pipefail
EXE="${KVM_EXE:-./kvm_exec}"; [ -x "$EXE" ] || { echo "no executor at $EXE (set KVM_EXE)"; exit 2; }
WORKERS=("${@:-1 4 16 107}"); read -r -a WORKERS <<< "${WORKERS[*]}"
PY="${KVM_PY:-python3}"
MLP_H="${KVM_CHECK_MLP_HIDDEN:-72}"; [[ "$MLP_H" =~ ^[0-9]+$ ]] || { echo "KVM_CHECK_MLP_HIDDEN must be a width, got $MLP_H"; exit 2; }
MLP_D="${KVM_CHECK_MLP_DEPTH:-2}"; [[ "$MLP_D" =~ ^[1-9]$ ]] || { echo "KVM_CHECK_MLP_DEPTH must be a depth 1-9, got $MLP_D"; exit 2; }
"$PY" -c 'import numpy, torch' 2> /dev/null || { echo "no numpy and torch in $PY (set KVM_PY)"; exit 2; }
TMP="$(mktemp -d "${TMPDIR:-/tmp}/kvm-static-check.XXXXXX")"; trap 'rm -rf "$TMP"' EXIT
"$PY" bench/vs_pytorch/gen.py "$TMP/gpt" --model gpt_tiny --steps 3 --tiling a100 > /dev/null
echo "MLP bundle: hidden $MLP_H, depth $MLP_D, 3 steps"
"$PY" -I bench/exec_vs_torch/gen_mlp.py "$TMP/mlp" --steps 3 --hidden "$MLP_H" --depth "$MLP_D" > /dev/null
if command -v g++ > /dev/null; then g++ -std=c++17 -O2 -Iinclude tests/static/test_static_plan.cpp -o "$TMP/plan_test"; "$TMP/plan_test" "$TMP/gpt/bundle"; "$TMP/plan_test" "$TMP/mlp/bundle"; fi
fails=0
fail() { echo "FAIL: $*"; fails=$((fails + 1)); }
# run NAME BUNDLE FLAGS...: one executor run, final buffers to $TMP/NAME.buf, its log to $TMP/NAME.log; a failed run stops the check
run() {
  local name=$1 b=$2; shift 2
  "$EXE" "$@" --dump-buffers "$TMP/$b/bundle" > "$TMP/$name.out" 2> "$TMP/$name.log" || { echo "FAIL: run $name ($*)"; tail -20 "$TMP/$name.log"; exit 1; }
  grep '^buffer' "$TMP/$name.out" > "$TMP/$name.buf"
}
same() { cmp -s "$TMP/$1.buf" "$TMP/$2.buf"; }
fused_runs() { sed -nE 's/.*fuse: ([0-9]+) run\(s\) fused.*/\1/p' "$TMP/$1.log" | tail -1; }
for b in gpt mlp; do
  run "$b.dyn" "$b" --dynamic
  for W in "${WORKERS[@]}"; do
    run "$b.st$W" "$b" --static --no-fuse --workers "$W"
    if same "$b.dyn" "$b.st$W"; then echo "ok: $b W=$W static schedule (--no-fuse) bitwise equal to dynamic ($(wc -l < "$TMP/$b.st$W.buf") buffers)"; else fail "$b W=$W static schedule (--no-fuse) differs from dynamic"; fi
  done
done
for W in "${WORKERS[@]}"; do
  run "gpt.def$W" gpt --static --workers "$W"
  n=$(fused_runs "gpt.def$W")
  if [ "${n:-0}" -ne 0 ]; then fail "gpt W=$W default static: the pass fused $n run(s) of the GPT bundle"
  elif same gpt.dyn "gpt.def$W"; then echo "ok: gpt W=$W: the pass does not fire and the run stays bitwise equal to dynamic"
  else fail "gpt W=$W default static (the pass did not fire) differs from dynamic"; fi
  run "mlp.fu$W" mlp --static --workers "$W"
  run "mlp.fu$W.b" mlp --static --workers "$W"
  n=$(fused_runs "mlp.fu$W")
  if [ "${n:-0}" -ge 1 ]; then echo "ok: mlp W=$W: $n run(s) fused"; else fail "mlp W=$W: the pass did not fire (log: $(grep -m1 'fuse:' "$TMP/mlp.fu$W.log" || echo none))"; fi
  if same "mlp.fu$W" "mlp.fu$W.b"; then echo "ok: mlp W=$W fused run repeats bitwise"; else fail "mlp W=$W two fused runs differ"; fi
  if "$EXE" --static --workers "$W" --golden --golden-tol 1e-3 "$TMP/mlp/bundle" > "$TMP/mlp.g$W.out" 2> "$TMP/mlp.g$W.log"; then
    echo "ok: mlp W=$W fused run within 1e-3 of the CPU golden ($(sed -nE 's/.*worst buffer relative L2 error ([^ ]+).*/worst \1/p' "$TMP/mlp.g$W.log"))"
  else fail "mlp W=$W fused run against the CPU golden: $(grep -m3 'golden' "$TMP/mlp.g$W.log" | tr '\n' ' ')"; fi
done
if "$PY" tests/static/fused_vs_fp64.py "$TMP/mlp" $(for W in "${WORKERS[@]}"; do echo "$TMP/mlp.fu$W.out"; done); then :; else fail "fused runs against float64 and eager (above)"; fi
[ "$fails" -eq 0 ] && echo "PASS: static schedule bitwise equal to the dynamic path; fused MLP step within its gates" || { echo "FAIL: $fails checks failed"; exit 1; }
