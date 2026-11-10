#!/usr/bin/env bash
# Runs ON the GPU box, inside the GPU lock, from the repo root: the whole kvm.cu measurement session in one lock hold.
#   bench/mlp_vs_torch/session.sh OUTDIR [RUNS...]      (needs ./kvm and ./kvm_lib.so built by run_session.sh)
# Per run count: torch_mlp.py (torch variants + the VM programs in ONE process, so ratios come from one process), with the VM's
# per-instruction timeline for a mid-run step window. A fused-program smoke test (hang-guarded) decides whether the fused program joins
# the comparison. Failures are logged and the session continues.
set -u
OUT=${1:?outdir}; shift
RUNS=${*:-1 8 32}
STEPS=${STEPS:-300}; REPS=${REPS:-5}; ROUNDS=${ROUNDS:-3}
mkdir -p "$OUT"
{
  date -u +"%Y-%m-%dT%H:%M:%SZ"
  nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,clocks.sm --format=csv
  nvcc --version | tail -2
  python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).multi_processor_count, 'SMs')"
  echo "flags: ${KVM_FLAGS:-}"
  cat COMMIT 2>/dev/null
} > "$OUT/env.txt" 2>&1
./kvm --selftest > "$OUT/selftest.txt" 2>&1; echo "selftest rc=$?" | tee -a "$OUT/selftest.txt"
PROGS=tiles
if [ "${NOFUSED:-0}" != 1 ]; then
  for prog in fused fused16; do   # smoke test each fused program (hang-guarded); only passing ones join the comparison
    ok=1
    for r in 1 8 32; do
      timeout 120 ./kvm --prog $prog --runs $r --steps 40 --reps 2 --ref-steps 10 > "$OUT/smoke_${prog}_R$r.txt" 2>&1; rc=$?
      echo "$prog smoke R=$r rc=$rc: $(tail -1 "$OUT/smoke_${prog}_R$r.txt")"
      [ $rc -eq 0 ] || ok=0
    done
    [ $ok -eq 1 ] && PROGS=$PROGS,$prog
  done
fi
echo "progs: $PROGS"
for R in $RUNS; do
  echo "=== runs=$R $(date -u +%T)"
  timeout 2400 python3 bench/mlp_vs_torch/torch_mlp.py --lib ./kvm_lib.so --runs "$R" --steps "$STEPS" --reps "$REPS" --rounds "$ROUNDS" --progs "$PROGS" \
      --timeline-prefix "$OUT/timeline_R$R" --out "$OUT/result_R$R.json" ${TORCH_ARGS:-} > "$OUT/log_R$R.txt" 2>&1
  echo "rc=$?"; tail -26 "$OUT/log_R$R.txt"
done
