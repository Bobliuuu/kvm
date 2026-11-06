#!/usr/bin/env bash
# ON THE GPU BOX, from the repo root (rsynced). Builds outside the lock, generates bundles on the CPU, then ONE short lock hold runs the GPU part.
#   bench/regimes/box_job.sh OUT_DIR "R list" [STEPS]      env: VARIANTS="name=-DFLAG ..." extra kvm_exec builds
set -euo pipefail
OUT="$1"; RUNS="${2:-1,2,4,8,16}"; STEPS="${3:-50}"; ARCH="${KVM_ARCH:-sm_80}"
export PATH=$PATH:/usr/local/cuda/bin
mkdir -p "$OUT"; L="$OUT/log"; mkdir -p "$L"
FL="-O3 -arch=$ARCH -std=c++17 -Iinclude --fmad=false -Xcompiler -ffp-contract=off -diag-suppress 177"
VM="--vm base=./kvm_exec"
nvcc $FL kvm_exec.cu -o kvm_exec > "$L/build_base.txt" 2>&1
for v in ${VARIANTS:-}; do n="${v%%=*}"; f="${v#*=}"; nvcc $FL $f kvm_exec.cu -o "kvm_exec_$n" > "$L/build_$n.txt" 2>&1; VM="$VM --vm $n=./kvm_exec_$n"; done
TILING="${TILING:-a100}"
python3 -I bench/regimes/gen_mlp.py "$OUT/gate" --steps 3 --tiling "$TILING" > "$L/gen_gate.txt" 2>&1
python3 -I bench/regimes/gen_mlp.py "$OUT/time" --steps "$STEPS" --tiling "$TILING" --no-golden > "$L/gen_time.txt" 2>&1
{ nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv,noheader; nvcc --version | tail -1; python3 -c "import torch;print('torch',torch.__version__)"; git rev-parse --short HEAD 2>/dev/null || cat COMMIT; echo "flags: $FL ${VARIANTS:-}"; nvidia-smi -q | grep -m1 "Multiprocessor" || true; } > "$L/env.txt" 2>&1
flock -w "${LOCK_WAIT:-3600}" ~/.gpu.lock timeout "${RUN_TIMEOUT:-900}" python3 bench/regimes/run_runs.py "$OUT" --steps "$STEPS" --runs "$RUNS" --reps 5 $VM ${EXTRA:-} > "$L/runs.txt" 2>&1
echo "exit $?" >> "$L/runs.txt"
