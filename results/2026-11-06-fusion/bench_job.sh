#!/usr/bin/env bash
# Runs ON the GPU box from the repo copy root:  nohup bash results/2026-11-06-fusion/bench_job.sh OUTDIR "VARIANTS" [REPS] > OUTDIR/job.log 2>&1 &
# VARIANTS: space separated TILING:FUSE, e.g. "cpu:0 a100:0 a100:1". Builds bundles (CPU), then ONE short flock hold for all timing.
set -uo pipefail
O="$1"; VARS="$2"; REPS="${3:-5}"; R="$(pwd)"; mkdir -p "$O"; export TILEFLAGS=1
NVCC=$(command -v nvcc || ls /usr/local/cuda*/bin/nvcc | tail -1)
[ -x ./kvm_exec ] || $NVCC -O3 -arch=sm_80 -std=c++17 -Iinclude --fmad=false -Xcompiler -ffp-contract=off -diag-suppress 177 kvm_exec.cu -o kvm_exec 2>&1 | grep -i error
for v in $VARS; do t=${v%%:*}; f=${v##*:}
  python3 -I results/2026-11-06-fusion/count.py mlp_ce 200 $t $f $O/mlp_${t}_$f | tee -a $O/counts.txt
  python3 -I results/2026-11-06-fusion/count.py xfmr_odd 50 $t $f $O/xfmr_${t}_$f | tee -a $O/counts.txt
done
flock -w 14400 ~/.gpu.lock bash -c '
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
  for v in '"$VARS"'; do t=${v%%:*}; f=${v##*:}
    for c in mlp xfmr; do for r in $(seq '"$REPS"'); do echo "kvm $c $t fuse=$f rep $r: $(timeout 300 ./kvm_exec '"$O"'/${c}_${t}_$f/bundle 2>&1 | grep "kernel done")"; done; done
  done
  PY=$HOME/venv-baseline/bin/python
  $PY results/2026-11-04-claims-audit/newmodel/torch_bench.py mlp_ce 200 '"$REPS"' 2>&1 | tail -2; $PY results/2026-11-04-claims-audit/newmodel/torch_bench.py xfmr_odd 50 '"$REPS"' 2>&1 | tail -2' > $O/time.txt 2>&1
echo done >> $O/job.done
