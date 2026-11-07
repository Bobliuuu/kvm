#!/usr/bin/env bash
# gpt_162m step-1 numerics check on the GPU box: the executor's loss and every parameter gradient against PyTorch fp64 (truth), 1xTF32 and fp32.
# Run from the repository root with demo/kvm_run built (nvcc -O3 -arch=sm_80 -std=c++17 -I. -Iinclude --fmad=false -Xcompiler -ffp-contract=off
# -lineinfo -diag-suppress 177 demo/kvm_run.cu -o demo/kvm_run), in one hold of the GPU lock:
#   flock -w 7200 ~/.gpu.lock bash bench/vs_pytorch/grads162.sh WORK_DIR OUT_DIR
# 1 step, reuse none (every gradient buffer survives), lr 3e-4 with 2 warmup steps (the demo's `large` preset). WORK_DIR needs ~3 GB.
set -uo pipefail
WORK="$(realpath -m "${1:?WORK_DIR}")"; OUT="$(realpath -m "${2:?OUT_DIR}")"; mkdir -p "$WORK" "$OUT"
PREC="${PRECISION:-f32}"   # f32 (TF32 matmuls, the default) | bf16 | bf16s (real bf16 storage, bf16 tensor-core tiles)
M='{"name":"gpt_162m","vocab":50304,"d_model":768,"layers":12,"heads":12,"seq":1024,"batch":4,"precision":"'"$PREC"'"}'
{ date -u +%FT%TZ; nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv,noheader; nvcc --version | tail -n 1
  python3 -c "import torch;p=torch.cuda.get_device_properties(0);print('torch',torch.__version__,'cuda',torch.version.cuda,p.name,'SMs',p.multi_processor_count)"
  echo "commit: ${KVM_COMMIT:-unknown}"; } | tee "$OUT/env.txt"
( cd frontend && python3 -m kvmfe.demo --out "$WORK/g1" --model "$M" --train-steps 1 --window 1 --serve-every 0 --game-frames 0 --tiling a100 \
    --lr 3e-4 --warmup 2 --reuse none --max-bundles 8 ) 2>&1 | tail -n 3
DUMP=$(python3 - "$WORK/g1" <<'PY'
import json, re, sys
s = json.load(open(sys.argv[1] + '/workloads/train.json'))
g = sorted({t['args']['g'] for gr in s['graphs'] for t in gr['tasks'] if t['op'] in ('adamw', 'adamw_lr')})
print('^(' + '|'.join(re.escape(x) for x in g) + r'|loss\.0|clip_coef|m\..*|w\..*)$')
PY
)
mkdir -p "$WORK/g1out"
timeout 3000 ./demo/kvm_run "$WORK/g1" --out "$WORK/g1out" --observe '^loss\.' --compile-cmd "python3 runtime/compile_workload.py" --dump "$DUMP" > "$WORK/g1out.log" 2>&1
echo "kvm_run exit $?"; tail -n 5 "$WORK/g1out.log"
cp "$WORK/g1out.log" "$OUT/kvm_run.log"; cp "$WORK/g1out/observations.jsonl" "$OUT/observations.jsonl"; cp "$WORK/g1out/run.json" "$OUT/run.json" 2>/dev/null
python3 tests/gpt162/torch_ref.py "$WORK/g1" --model "$M" --steps 1 --lr 3e-4 --warmup 2 --grads "$WORK/g1out/buffers" --out "$OUT/grads.json" 2>&1 | tee "$OUT/grads.log"
rm -rf "$WORK/g1out/buffers"
