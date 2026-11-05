#!/usr/bin/env bash
# Timing: one kvm_exec launch of the whole N-step run vs PyTorch (eager / whole-step CUDA graph / torch.compile), one GPU-lock hold.
#   KVM_HOST=user@host [PYBIN=path/to/python-with-torch] ./bench.sh WORKDIR     -> raw log on stdout
set -euo pipefail
HOST="${KVM_HOST:?set KVM_HOST}"; D="$(cd "$(dirname "$0")" && pwd)"; ROOT="$D/../../.."; W="$1"; PYBIN="${PYBIN:-python3}"
SSH="ssh -o BatchMode=yes"; R='~/kvm-audit-newmodel'
python3 -I "$D/gen.py" "$W/t_mlp" mlp_ce --steps 200 --no-golden >/dev/null 2>&1
python3 -I "$D/gen.py" "$W/t_xfmr" xfmr_odd --steps 50 --no-golden >/dev/null 2>&1
rsync -az -e "$SSH" --delete --exclude .git --exclude results --exclude logs --exclude docs "$ROOT/" "$HOST:$R/"
$SSH "$HOST" "mkdir -p $R/nm $R/tb"; rsync -az -e "$SSH" "$W/t_mlp/bundle/" "$HOST:$R/nm/t_mlp/"; rsync -az -e "$SSH" "$W/t_xfmr/bundle/" "$HOST:$R/nm/t_xfmr/"
scp -q -o BatchMode=yes "$D/libref.py" "$D/torch_bench.py" "$HOST:$R/tb/"
$SSH "$HOST" "cd $R && NVCC=\$(command -v nvcc || ls /usr/local/cuda*/bin/nvcc | tail -1) && \$NVCC -O3 -arch=sm_80 -std=c++17 -Iinclude --fmad=false -Xcompiler -ffp-contract=off -diag-suppress 177 kvm_exec.cu -o kvm_exec 2>&1 | grep -i error || true
 nvidia-smi --query-gpu=name,clocks.max.sm --format=csv,noheader; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
 flock -w 14400 ~/.gpu.lock bash -c 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
  for r in 1 2 3 4 5; do echo \"kvm_exec mlp_ce 200 steps rep \$r\"; ./kvm_exec nm/t_mlp 2>&1 | grep \"kernel done\"; done
  for r in 1 2 3 4 5; do echo \"kvm_exec xfmr_odd 50 steps rep \$r\"; ./kvm_exec nm/t_xfmr 2>&1 | grep \"kernel done\"; done
  $PYBIN tb/torch_bench.py mlp_ce 200 5 2>&1 | tail -3; $PYBIN tb/torch_bench.py xfmr_odd 50 5 2>&1 | tail -3; $PYBIN tb/libref.py 2>&1 | tail -3'"
