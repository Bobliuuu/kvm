#!/usr/bin/env bash
# The GPU measurements of results/2026-11-06-a100-bf16, on the GPU box itself, from the repository root, after the builds exist:
#   nvcc -O3 -arch=sm_80 -std=c++17 -Iinclude --fmad=false -Xcompiler -ffp-contract=off -lineinfo -diag-suppress 177 kvm_exec.cu -o kvm_exec
#   (cd tests/gpu_ops && nvcc -O2 -arch=sm_80 -std=c++17 [--fmad=false -Xcompiler -ffp-contract=off] -lineinfo -I../../include -I. X.cu [-lcublas] -o X)
#     for X in test_bf16 test_ops test_extra bench_mma_bf16
#   python3 bench/vs_pytorch/gen.py DATA/NAME --model JSON --steps N [--reuse none] --mode tf32 --tiling a100    (CPU; bundles used below)
#
#   bash bench/bf16/run_box.sh correctness OUT_DIR DATA_DIR    # op tests (bf16 + regression of the fp32 paths), sanitizers, model zoo, step-1 numerics
#   bash bench/bf16/run_box.sh perf        OUT_DIR DATA_DIR    # GEMM TFLOP/s vs cuBLAS, executor step time, PyTorch baselines, 3-step losses
#   bash bench/bf16/run_box.sh profile     OUT_DIR DATA_DIR    # traced executor launches (f32, bf16s): per-op task counts and run time
#
# Each mode takes the GPU lock once (flock -w 7200 ~/.gpu.lock; KVM_NO_LOCK=1 skips it) and holds it for its stages only; the stages are
# independent, a failing stage does not stop the next, and every stage logs to OUT_DIR/<stage>.log. DATA_DIR holds the gen.py bundles:
#   f32_1 b16_1 (1 step, buffer reuse 'shape': timing), f32_n1 bf16s_n1 (1 step, reuse none: gradients survive), bf16s_3 (3 steps: losses, PyTorch).
set -uo pipefail
ROOT="$(pwd)"; EXEC="${KVM_EXEC:-./kvm_exec}"; DUMP_EXEC="$EXEC"   # KVM_EXEC: the executor build to measure (needs --dump-bin)
MODE="${1:?correctness|perf}"; OUT="$(realpath -m "${2:?OUT_DIR}")"; DATA="$(realpath -m "${3:?DATA_DIR}")"
mkdir -p "$OUT"
# ONE lock hold for the whole mode (re-exec under flock); every command below runs while it is held
if [ -z "${KVM_LOCK_HELD:-}" ] && [ "${KVM_NO_LOCK:-}" = "" ]; then
  exec flock -w "${KVM_LOCK_WAIT:-7200}" "$HOME/.gpu.lock" env KVM_LOCK_HELD=1 bash "$0" "$@"
fi

stage() {   # stage NAME TIMEOUT_S CMD... : log to OUT/NAME.log, record the exit code
  local name="$1" t="$2"; shift 2
  echo "== $name $(date -u +%T)" | tee -a "$OUT/stages.txt"
  timeout "$t" "$@" > "$OUT/$name.log" 2>&1
  local rc=$?
  echo "   $name exit $rc $(date -u +%T)" | tee -a "$OUT/stages.txt"
  return 0
}
env_info() {
  { date -u +%FT%TZ; nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv,noheader; nvcc --version | tail -n 1
    python3 -c "import torch;p=torch.cuda.get_device_properties(0);print('torch',torch.__version__,'cuda',torch.version.cuda,p.name,'SMs',p.multi_processor_count)"
    echo "commit: ${KVM_COMMIT:-unknown}"; } > "$OUT/env.txt" 2>&1
}
env_info

case "$MODE" in
correctness)
  stage test_bf16 1500 bash -c "cd tests/gpu_ops && ./test_bf16"
  stage test_ops 1500 bash -c "cd tests/gpu_ops && ./test_ops"
  stage test_extra 1500 bash -c "cd tests/gpu_ops && ./test_extra"
  stage test_bf16_memcheck 2400 bash -c "cd tests/gpu_ops && TEST_BF16_SMALL=1 compute-sanitizer --tool memcheck --error-exitcode 9 ./test_bf16 | tail -6"
  stage test_bf16_racecheck 2400 bash -c "cd tests/gpu_ops && TEST_BF16_SMALL=1 compute-sanitizer --tool racecheck --error-exitcode 9 ./test_bf16 | tail -6"
  stage test_bf16_synccheck 2400 bash -c "cd tests/gpu_ops && TEST_BF16_SMALL=1 compute-sanitizer --tool synccheck --error-exitcode 9 ./test_bf16 | tail -6"
  stage model_zoo_bf16s 3400 env KVM_NO_LOCK=1 KVM_EXEC_BIN="$EXEC" bash examples/models/run_box.sh "$DATA/zoo" "$OUT/zoo" --cases examples/models/cases_bf16s.txt
  stage grads_f32 1500 python3 bench/bf16/exec_grads.py "$DATA/f32_n1" "$DUMP_EXEC" --out "$OUT/grads_f32"
  stage grads_bf16s 1500 python3 bench/bf16/exec_grads.py "$DATA/bf16s_n1" "$DUMP_EXEC" --out "$OUT/grads_bf16s"
  ;;
perf)
  stage bench_mma_bf16 1500 bash -c "cd tests/gpu_ops && BENCH_REPS=7 ./bench_mma_bf16"
  stage bench_mma_bf16_nouter 1500 bash -c "cd tests/gpu_ops && MMA_NOUTER=1 BENCH_REPS=7 MMA_SET=gpt ./bench_mma_bf16"   # n-outer tile order (what the frontend emits for wide GEMMs)
  stage bench_mma_bf16_f32out 1500 bash -c "cd tests/gpu_ops && MMA_OUT=f32 BENCH_REPS=7 MMA_SET=gpt ./bench_mma_bf16"
  stage exec_f32 2400 python3 bench/vs_pytorch/exec_bench.py "$DATA/f32_1" "$EXEC" --reps 5 --out "$OUT/exec_f32.json"
  stage exec_bf16s 2400 python3 bench/vs_pytorch/exec_bench.py "$DATA/b16_1" "$EXEC" --reps 5 --out "$OUT/exec_bf16s.json"
  stage torch_tf32 3600 python3 bench/vs_pytorch/torch_bench.py "$DATA/bf16s_3" --modes eager,graph,compile_graph --precision tf32 --reps 5 --out "$OUT/torch_tf32.json"
  stage torch_bf16 3600 python3 bench/vs_pytorch/torch_bench.py "$DATA/bf16s_3" --modes eager,graph,compile_graph --precision bf16 --reps 5 --out "$OUT/torch_bf16.json"
  stage exec_bf16s_3step 2400 python3 bench/vs_pytorch/exec_bench.py "$DATA/bf16s_3" "$EXEC" --reps 3 --out "$OUT/exec_bf16s_3step.json"
  stage losses_vs_bf16 1500 python3 bench/bf16/exec_losses_bin.py "$DATA/bf16s_3" "$DUMP_EXEC" --torch "$OUT/torch_bf16.json" --mode eager --out "$OUT/losses_vs_torch_bf16.json"
  stage losses_vs_tf32 1500 python3 bench/bf16/exec_losses_bin.py "$DATA/bf16s_3" "$DUMP_EXEC" --torch "$OUT/torch_tf32.json" --mode eager --out "$OUT/losses_vs_torch_tf32.json"
  ;;
profile)   # what the executor's step is made of: per-op task counts and run time from a traced launch (f32 and bf16s)
  for v in f32:f32_1 bf16s:b16_1; do
    stage "trace_${v%%:*}" 1500 bash -c "$EXEC --quiet --trace-csv /tmp/kvm_bf16_trace.csv '$DATA/${v##*:}/bundle' && python3 bench/vs_pytorch/trace_summary.py /tmp/kvm_bf16_trace.csv 1; rm -f /tmp/kvm_bf16_trace.csv"
  done
  ;;
*) echo "unknown mode $MODE" >&2; exit 2;;
esac
echo "== done $(date -u +%T)" | tee -a "$OUT/stages.txt"
