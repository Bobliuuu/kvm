#!/usr/bin/env bash
# gpt_162m learning run: the executor and PyTorch (1xTF32 and fp32-highest AdamW) on the SAME token stream, then a hold-out evaluation.
#   bash tests/gpt162/learning.sh OUT_DIR MODEL_JSON STEPS LR WARMUP [HOLDOUT_BATCHES]      (repo root; demo/kvm_run built, no game pack)
# Generation (CPU) runs outside the GPU lock; the executor run and the torch curves share ONE `flock ~/.gpu.lock` hold.
# Writes OUT_DIR/{exec_kvm_run.log,exec_observations.jsonl,exec_run.json,curve.json,curve.log,nvidia.txt}; deletes the scratch (1 GB+) afterwards.
set -euo pipefail
OUT="${1:?OUT_DIR}"; MODEL="${2:?MODEL_JSON}"; STEPS="${3:?STEPS}"; LR="${4:?LR}"; WARMUP="${5:?WARMUP}"; HOLD="${6:-8}"
SCR="${KVM_SCRATCH:-$PWD/.scratch-learning}"; mkdir -p "$OUT" "$SCR"
echo "== gen $(date -u +%T)"
( cd frontend && python3 -m kvmfe.demo --out "$SCR/demo" --model "$MODEL" --train-steps "$STEPS" --window 1 --serve-every 0 --game-frames 0 \
    --tiling a100 --lr "$LR" --warmup "$WARMUP" --reuse shape --max-bundles 2 ) 2>&1 | tail -3
run() {
  echo "== lock held $(date -u +%T)"
  { nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; nvcc --version | tail -1
    python3 -c "import torch;p=torch.cuda.get_device_properties(0);print('torch',torch.__version__,'cuda',torch.version.cuda,p.name,'SMs',p.multi_processor_count)"; } | tee "$OUT/nvidia.txt"
  mkdir -p "$SCR/out"
  echo "== executor $(date -u +%T)"
  /usr/bin/time -v ./demo/kvm_run "$SCR/demo" --out "$SCR/out" --observe '^loss\.' --dump '^w\.' --compile-cmd "python3 runtime/compile_workload.py" \
      > "$OUT/exec_kvm_run.log" 2> "$OUT/exec_time.log" || echo "kvm_run exit $?"
  tail -4 "$OUT/exec_kvm_run.log"; grep -E "Elapsed|Maximum resident" "$OUT/exec_time.log" || true
  cp "$SCR/out/observations.jsonl" "$OUT/exec_observations.jsonl"; cp "$SCR/out/run.json" "$OUT/exec_run.json" 2>/dev/null || true
  echo "== torch curves + hold-out $(date -u +%T)"
  python3 tests/gpt162/torch_ref.py "$SCR/demo" --model "$MODEL" --steps "$STEPS" --lr "$LR" --warmup "$WARMUP" --curve --obs "$OUT/exec_observations.jsonl" \
      --holdout "$HOLD" --exec-weights "$SCR/out/buffers" --out "$OUT/curve.json" 2>&1 | tee "$OUT/curve.log"
  echo "== done $(date -u +%T)"
}
if [ -n "${KVM_NO_LOCK:-}" ]; then run   # a private GPU (Modal container): no shared-box lock
else flock -w 7200 ~/.gpu.lock bash -c "$(declare -f run); OUT='$OUT' SCR='$SCR' MODEL='$MODEL' STEPS='$STEPS' LR='$LR' WARMUP='$WARMUP' HOLD='$HOLD' run"
fi
rm -rf "$SCR"
