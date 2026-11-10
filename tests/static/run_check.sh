#!/usr/bin/env bash
# Static-schedule check on a GPU box (run it under the GPU lock): final buffers of the static path bitwise equal to the dynamic path, at several worker counts,
# on small GPT and MLP bundles, and the plan's CPU replay (every dependency honoured, no deadlock).
#   KVM_EXE=./kvm_exec tests/static/run_check.sh [workers ...]        (from the repo root; needs numpy; writes to a scratch directory)
set -euo pipefail
EXE="${KVM_EXE:-./kvm_exec}"; [ -x "$EXE" ] || { echo "no executor at $EXE (set KVM_EXE)"; exit 2; }
WORKERS=("${@:-1 4 16 107}"); read -r -a WORKERS <<< "${WORKERS[*]}"
PY="${KVM_PY:-python3}"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/kvm-static-check.XXXXXX")"; trap 'rm -rf "$TMP"' EXIT
"$PY" bench/vs_pytorch/gen.py "$TMP/gpt" --model gpt_tiny --steps 3 --tiling a100 > /dev/null
"$PY" -I bench/exec_vs_torch/gen_mlp.py "$TMP/mlp" --steps 3 --hidden 72 --depth 2 --no-golden > /dev/null
if command -v g++ > /dev/null; then g++ -std=c++17 -O2 -Iinclude tests/static/test_static_plan.cpp -o "$TMP/plan_test"; "$TMP/plan_test" "$TMP/gpt/bundle"; "$TMP/plan_test" "$TMP/mlp/bundle"; fi
fails=0
for b in gpt mlp; do
  "$EXE" --quiet --dynamic --dump-buffers "$TMP/$b/bundle" > "$TMP/$b.dyn" 2> "$TMP/$b.err" || { echo "FAIL: dynamic run of $b"; cat "$TMP/$b.err"; exit 1; }
  grep -c '^buffer' "$TMP/$b.dyn" > /dev/null
  for W in "${WORKERS[@]}"; do
    "$EXE" --quiet --static --workers "$W" --dump-buffers "$TMP/$b/bundle" > "$TMP/$b.st" 2>> "$TMP/$b.err" || { echo "FAIL: static run of $b W=$W"; cat "$TMP/$b.err"; exit 1; }
    if cmp -s <(grep '^buffer' "$TMP/$b.dyn") <(grep '^buffer' "$TMP/$b.st"); then echo "ok: $b W=$W bitwise equal to dynamic ($(grep -c '^buffer' "$TMP/$b.st") buffers)"; else echo "FAIL: $b W=$W differs from dynamic"; fails=$((fails+1)); fi
  done
done
[ "$fails" -eq 0 ] && echo "PASS: static schedule bitwise equal to the dynamic path" || { echo "FAIL: $fails configurations differ"; exit 1; }
