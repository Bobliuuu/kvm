#!/usr/bin/env bash
# sm_90a (H100) builds and tests of the Hopper TMA + wgmma tiles, run ON the GPU box from the repository root (no ssh here; reach the box
# through your own ~/.ssh/config alias and rsync the tree):
#   bash tests/gpu_ops/run_hopper_box.sh build OUT     # all sm_90a binaries in parallel (test_exec full needs ~35 GB RAM for ptxas)
#   bash tests/gpu_ops/run_hopper_box.sh test  OUT     # test_hopper (both launch sizes), test_ops, test_bf16, test_extra, test_exec, sanitizers
#   bash tests/gpu_ops/run_hopper_box.sh bench OUT     # test_hopper --bench (TFLOP/s: Hopper tile, sm_80 tile, cuBLAS)
# Every stage logs to OUT/<stage>.log and appends "<stage> exit N" to OUT/stages.txt; a failing stage does not stop the next.
set -uo pipefail
export PATH=$PATH:/usr/local/cuda/bin
MODE="${1:?build|test|bench}"; OUT="$(realpath -m "${2:?OUT}")"; mkdir -p "$OUT" bin_h100
ARCH="${KVM_ARCH:-sm_90a}"
OPF="-O2 -arch=$ARCH -std=c++17 --fmad=false -Xcompiler -ffp-contract=off -lineinfo -I../../include -I."
EXF="-O3 -arch=$ARCH -std=c++17 -I. -Iinclude --fmad=false -Xcompiler -ffp-contract=off -lineinfo -diag-suppress 177"
stage() { local n="$1" t="$2"; shift 2; echo "== $n $(date -u +%T)" >> "$OUT/stages.txt"; timeout "$t" "$@" > "$OUT/$n.log" 2>&1; echo "$n exit $? $(date -u +%T)" >> "$OUT/stages.txt"; }
case "$MODE" in
build)
  { nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv,noheader; nvcc --version | tail -1; nproc; free -g | head -2; } > "$OUT/env.txt" 2>&1
  stage b_test_hopper 3600 bash -c "cd tests/gpu_ops && nvcc $OPF test_hopper.cu -lcublas -o ../../bin_h100/test_hopper" &
  stage b_test_hopper_default 3600 bash -c "cd tests/gpu_ops && nvcc $OPF -DKVM_HOPPER_DEFAULT_POLICY test_hopper.cu -lcublas -o ../../bin_h100/test_hopper_default" &
  stage b_test_ops 5400 bash -c "cd tests/gpu_ops && nvcc $OPF test_ops.cu -o ../../bin_h100/test_ops" &
  stage b_test_bf16 5400 bash -c "cd tests/gpu_ops && nvcc $OPF test_bf16.cu -o ../../bin_h100/test_bf16" &
  stage b_test_extra 7200 bash -c "cd tests/gpu_ops && nvcc $OPF test_extra.cu -o ../../bin_h100/test_extra" &
  stage b_test_exec 7200 bash -c "KVM_ARCH=$ARCH bash tests/gpu/build.sh && mv test_exec bin_h100/test_exec" &
  stage b_kvm_exec 7200 bash -c "nvcc $EXF kvm_exec.cu -o bin_h100/kvm_exec" &
  wait
  ls -la bin_h100 >> "$OUT/stages.txt"
  ;;
test)
  stage test_hopper 3600 bash -c "cd tests/gpu_ops && ../../bin_h100/test_hopper --quiet"
  stage test_hopper_default 3600 bash -c "cd tests/gpu_ops && ../../bin_h100/test_hopper_default --quiet"
  stage test_ops 1800 bash -c "cd tests/gpu_ops && ../../bin_h100/test_ops"
  stage test_bf16 1800 bash -c "cd tests/gpu_ops && ../../bin_h100/test_bf16"
  stage test_extra 1800 bash -c "cd tests/gpu_ops && ../../bin_h100/test_extra"
  stage test_exec 3000 bash -c "python3 tests/gpu/make_initgen_vectors.py --check && ./bin_h100/test_exec"
  for t in memcheck racecheck synccheck; do
    stage "hopper_$t" 3600 bash -c "cd tests/gpu_ops && compute-sanitizer --tool $t --error-exitcode 9 ../../bin_h100/test_hopper --quiet | tail -8"
  done
  ;;
bench)
  stage bench_hopper 3600 bash -c "cd tests/gpu_ops && ../../bin_h100/test_hopper --bench"
  stage bench_hopper_96k 3600 bash -c "cd tests/gpu_ops && ../../bin_h100/test_hopper --bench 96"
  ;;
step)   # gpt_162m, 1 step: executor with the Hopper tiles (default) vs KVM_NO_HOPPER=1 (sm_80 tiles, same binary), interleaved; then PyTorch
  D="${KVM_DATA:-/tmp/kvm_data}"; mkdir -p "$D"
  M='{"name":"gpt_162m","vocab":50304,"d_model":768,"layers":12,"heads":12,"seq":1024,"batch":4'
  for v in "f32_1 f32 1" "b16_1 bf16s 1" "b16_3 bf16s 3"; do set -- $v
    python3 bench/vs_pytorch/gen.py "$D/$1" --model "$M,\"precision\":\"$2\"}" --steps "$3" --mode tf32 --tiling a100 > "$OUT/gen_$1.log" 2>&1 &
  done; wait
  for r in 1 2 3 4 5; do
    for v in hop:f32_1: sm80:f32_1:1 hop:b16_1: sm80:b16_1:1; do IFS=: read n b off <<< "$v"
      o=$(KVM_NO_HOPPER=$off timeout 900 ./bin_h100/kvm_exec --quiet "$D/$b/bundle" 2>&1 | grep -E '^(OK|FAIL):' | tail -1)
      echo "rep $r $n $b $o" | tee -a "$OUT/ab_runs.txt"
    done
  done
  KVM_LOG=2 ./bin_h100/kvm_exec "$D/f32_1/bundle" 2>&1 | grep -iE 'hopper|dynamic smem' | head -3 > "$OUT/hopper_armed.txt"
  stage torch_tf32 3600 python3 bench/vs_pytorch/torch_bench.py "$D/b16_3" --modes eager,graph,compile_graph --precision tf32 --reps 5 --out "$OUT/torch_tf32.json"
  stage torch_bf16 3600 python3 bench/vs_pytorch/torch_bench.py "$D/b16_3" --modes eager,graph,compile_graph --precision bf16 --reps 5 --out "$OUT/torch_bf16.json"
  ;;
esac
cat "$OUT/stages.txt"
