#!/usr/bin/env bash
# Executor on the 3-step gpt_162m bundle: default configuration (5 launches), the demo's admission configuration (3 launches), and the
# per-step losses against the PyTorch eager run of the same bundle. One lock hold:
#   flock -w 14400 ~/.gpu.lock bash bench/vs_pytorch/executor162.sh OUT_DIR TORCH_EAGER_JSON
set -uo pipefail
OUT="${1:?OUT_DIR}"; TORCH="${2:?torch_eager.json}"; mkdir -p "$OUT"
echo "== default config $(date -u +%T)"
python3 bench/vs_pytorch/exec_bench.py w_162m ./kvm_exec --reps 5 --out "$OUT/exec3_default.json" 2>&1 | tee "$OUT/exec3_default.log"
echo "== demo config (live-cap 65536, ring 4M words, 2M slots) $(date -u +%T)"
python3 bench/vs_pytorch/exec_bench.py w_162m ./kvm_exec --reps 3 --out "$OUT/exec3_demo.json" -- --live-cap 65536 --ring-words 4194304 --max-tasks 2097152 2>&1 | tee "$OUT/exec3_demo.log"
echo "== losses vs torch eager $(date -u +%T)"
python3 bench/vs_pytorch/exec_losses.py w_162m ./kvm_exec --torch "$TORCH" --mode eager --out "$OUT/losses.json" 2>&1 | tee "$OUT/losses.log"
echo "== done $(date -u +%T)"
