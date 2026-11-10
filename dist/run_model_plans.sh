#!/usr/bin/env bash
# Frontend models partitioned over the GPUs of THIS machine (run on the GPU box): data parallelism and pipeline stages.
#   dist/run_model_plans.sh TEST_DIST OUT_DIR [correctness|scaling|all]
# correctness: gpt_tiny / llama_tiny / gpt_small, dp x2 (and x4 with >= 4 GPUs) and 2-stage pipelines, vs the compiler's CPU golden
#              (expect lines in dist.txt: dp vs the accum=P golden, pipeline vs the unpartitioned golden).
# scaling:     a mid-size GPT (no CPU golden): dp xP vs the same model with accum=P on ONE GPU through the same harness, state dumped
#              and compared (tests/dist/compare_dumps.py), kernel time of both; P = 2, 4, 8 up to the GPU count.
# Every run prints the harness's own lines ("rank r: N tasks, X ms kernel", "values compared ... worst"); a summary is OUT_DIR/summary.txt.
set -u -o pipefail
TD="${1:?test_dist binary}"; OUT="${2:?out dir}"; WHAT="${3:-all}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; PY="python3 -I"
NGPU="$(nvidia-smi -L | grep -c '^GPU')"; SMS="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
BLOCKS="${KVM_BLOCKS:-$(python3 - <<'PY'
import ctypes
try:
    cu = ctypes.CDLL('libcuda.so.1'); cu.cuInit(0); v = ctypes.c_int(); cu.cuDeviceGetAttribute(ctypes.byref(v), 16, 0); print(v.value)
except Exception:
    print(80)
PY
)}"
GTOL="${KVM_GOLDEN_TOL:-2e-3}"   # GPU (3xTF32) vs CPU golden after AdamW steps: the 1-GPU reference run measures the same noise floor
mkdir -p "$OUT"; SUM="$OUT/summary.txt"; : > "$SUM"; FAIL=0
echo "# $(date -u +%FT%TZ) GPUs: $NGPU x $SMS, executor blocks per rank: $BLOCKS" | tee -a "$SUM"
run() {  # name plan_dir [extra test_dist args]
  local n="$1" d="$2"; shift 2
  # persistent executors must be co-resident: ranks sharing a GPU split its SMs (1 block per SM with the GPT pack)
  local nr; nr="$(ls -d "$d"/rank* | wc -l)"; local per=$(( (nr + NGPU - 1) / NGPU )); local blocks=$(( BLOCKS / per ))
  KVM_DIST_MAX_TASKS=${KVM_DIST_MAX_TASKS:-1048576} KVM_DIST_LIVE_CAP=${KVM_DIST_LIVE_CAP:-262144} KVM_DIST_RING_WORDS=${KVM_DIST_RING_WORDS:-67108864} KVM_DIST_SUCC_CAP=${KVM_DIST_SUCC_CAP:-16777216} timeout "${KVM_PLAN_TIMEOUT:-900}" "$TD" plan "$d" --blocks "$blocks" "$@" > "$OUT/$n.log" 2>&1
  local rc=$?
  local ms; ms="$(grep -o 'tasks, [0-9.]* ms kernel' "$OUT/$n.log" | awk '{print $2}' | sort -g | tail -1)"
  local wr; wr="$(grep -o '[0-9]* values compared.*' "$OUT/$n.log" | tail -1)"
  if [ -n "${NEED_EXPECT:-}" ] && ! grep -q '^expect ' "$d"/rank0/dist.txt; then echo "$n: plan has no expect lines (golden missing)"; rc=3; fi
  printf '%-34s rc=%d  ranks %d x %d blocks  max kernel ms over ranks %-9s %s\n' "$n" "$rc" "$nr" "$blocks" "${ms:-?}" "$wr" | tee -a "$SUM"
  [ $rc -eq 0 ] || { FAIL=1; tail -8 "$OUT/$n.log" | sed 's/^/    | /'; }
}
gen() {  # a failed generation is a FAIL, not a skipped run
  local o; o="$( (cd "$ROOT" && $PY tests/dist/gpt_dp.py "$@") 2>&1 )"; local rc=$?
  echo "$o" | grep -v Warning | tail -1 | tee -a "$SUM"
  [ $rc -eq 0 ] || { echo "plan generation FAILED (exit $rc): $*" | tee -a "$SUM"; FAIL=1; return 1; }
}
if [ "$WHAT" = correctness ] || [ "$WHAT" = all ]; then
  NEED_EXPECT=1
  for m in gpt_tiny llama_tiny gpt_small; do
    for P in 2 4; do
      [ "$P" -gt 2 ] && [ "$m" != gpt_tiny ] && continue
      gen "$OUT/plans/dp_${m}_x$P" --model $m --ranks $P --steps 2 --precise --golden || continue
      run "dp_${m}_x${P}" "$OUT/plans/dp_${m}_x$P/plan" --tol "$GTOL"
      run "dp_${m}_x${P}_ref1gpu" "$OUT/plans/dp_${m}_x$P/ref_plan" --tol "$GTOL"
    done
    gen "$OUT/plans/pipe_${m}_x2" --model $m --ranks 2 --steps 2 --precise --golden --pipeline || continue
    run "pipe_${m}_x2" "$OUT/plans/pipe_${m}_x2/plan" --tol "$GTOL"
    run "pipe_${m}_x2_ref1gpu" "$OUT/plans/pipe_${m}_x2/ref_plan" --tol "$GTOL"
  done
fi
if [ "$WHAT" = scaling ] || [ "$WHAT" = all ]; then
  NEED_EXPECT=
  # 3xTF32 GEMMs (--precise): with 1xTF32 the AdamW step turns gradient rounding noise into weight differences of order lr (see
  # tests/gptdiff/golden_check.py), so the dp-vs-reference state check would only bound noise, not test the partition
  MID="${KVM_SCALE_MODEL:-{\"name\": \"gpt_dp_mid\", \"vocab\": 1024, \"d_model\": 512, \"layers\": 6, \"heads\": 8, \"seq\": 256, \"batch\": 4}}"
  for P in 2 4 8; do
    [ "$P" -gt "$NGPU" ] && continue
    gen "$OUT/plans/scale_x$P" --model "$MID" --ranks $P --steps "${KVM_SCALE_STEPS:-3}" --tiling a100 --precise --symbolic || continue
    mkdir -p "$OUT/dump_x$P" "$OUT/dump_ref_x$P"
    run "scale_dp_x$P" "$OUT/plans/scale_x$P/plan" --dump "$OUT/dump_x$P"
    run "scale_ref1gpu_accum$P" "$OUT/plans/scale_x$P/ref_plan" --dump "$OUT/dump_ref_x$P"
    (cd "$ROOT" && $PY tests/dist/compare_dumps.py "$OUT/dump_x$P" "$OUT/dump_ref_x$P" --tol "${KVM_SCALE_TOL:-1e-3}" --json "$OUT/compare_x$P.json") 2>&1 | tee -a "$SUM" | grep -q '^PASS' || FAIL=1
  done
fi
echo "RESULT: $([ $FAIL = 0 ] && echo PASS || echo FAIL)" | tee -a "$SUM"
exit $FAIL
