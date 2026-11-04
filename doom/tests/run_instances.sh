#!/usr/bin/env bash
# N independent Doom engines in one process: every instance must be byte-identical to the CPU oracle, alone and beside a matmul load.
#   KVM_HOST=user@host doom/tests/run_instances.sh [frames=300] [maxN=8]
# Builds (no GPU lock) DOOM_INSTANCES=maxN objects into a separate build dir, then runs everything under ONE lock hold.
# Engine sources/WAD are reused from ~/kvm-doom-scratch/doom/third_party on the box.
set -euo pipefail
: "${KVM_HOST:?set KVM_HOST=user@host}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; FR="${1:-300}"; MAXN="${2:-8}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"; LOGDIR="$ROOT/logs/$TS-doominst"; mkdir -p "$LOGDIR"
SSH="ssh -o StrictHostKeyChecking=no -o BatchMode=yes"; REMOTE="${KVM_REMOTE:-kvm-adaptive-doom-scratch}"
rsync -az -e "$SSH" --delete --include 'include/***' --include 'runtime/***' --include 'doom/' --include 'doom/*.py' --include 'doom/*.cu' --include 'doom/*.sh' --include 'doom/glue/***' --include 'doom/player/***' --include 'doom/tests/***' --include 'kvm_exec.cu' --exclude '*' "$ROOT/" "$KVM_HOST:$REMOTE/"
$SSH "$KVM_HOST" "FR=$FR MAXN=$MAXN REMOTE=$REMOTE bash -s" <<'REMOTE_SCRIPT' 2>&1 | tee "$LOGDIR/run.log"
set -euo pipefail
cd ~/$REMOTE; ln -sfn ~/kvm-doom-scratch/doom/third_party doom/third_party
NVCC=$(ls /usr/local/cuda*/bin/nvcc | tail -1); export NVCC
export BUILD=$PWD/doom/build_n$MAXN
echo "# $(date -u +%FT%TZ) $($NVCC --version | tail -1) $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
T0=$(date +%s); DOOM_INSTANCES=$MAXN bash doom/build.sh 2>&1 | grep -E "INFO|WARN|ERROR|prefix" | tail -12; echo "engine build: $(( $(date +%s) - T0 )) s"
ls -l $BUILD/doom_dev*.o | awk '{print $5, $9}'
cp ~/kvm-doom-scratch/doom/build/libdoom_cpu.so $BUILD/ 2>/dev/null || true
OBJS="$BUILD/doom_dev.o"; for k in $(seq 1 $((MAXN-1))); do OBJS="$OBJS $BUILD/doom_dev_i$k.o"; done
OUT=$PWD/logs/doominst; rm -rf $OUT; mkdir -p $OUT
python3 doom/cpu_run.py --frames $FR --lib $BUILD/libdoom_cpu.so --out $OUT/cpu --log info 2>&1 | tail -1
build() { # N
  $NVCC -O3 -arch=sm_80 -std=c++17 -rdc=true -DKVM_WITH_GAME -DKVM_DOOM_INSTANCES=$1 -I. -Iinclude --fmad=false -Xcompiler -ffp-contract=off -Xptxas -v doom/exec_main.cu $(echo $OBJS | tr ' ' '\n' | head -$1 | tr '\n' ' ') -o $BUILD/doom_exec_n$1 2> $OUT/build_n$1.txt || { tail -20 $OUT/build_n$1.txt; exit 1; }
  echo "build N=$1: $(grep -A1 'kvm_exec_kernel' $OUT/build_n$1.txt | grep -E 'registers|stack' | tail -1 | sed 's/^ *//') size $(stat -c %s $BUILD/doom_exec_n$1)"
}
for n in 1 2 4 8; do [ $n -le $MAXN ] && build $n; done
# the GPU phase, one script so that the lock is taken exactly once
cat > $OUT/hold.sh <<HOLD
set -uo pipefail
cd ~/$REMOTE
wl() { python3 doom/make_workload.py --frames $FR --doom-priority 0 --compute-priority 0 "\$@" >/dev/null; }
for n in 1 2 4 8; do
  [ \$n -le $MAXN ] || continue
  for load in alone load; do
    name=n\${n}_\$load; a="--instances \$n"; [ \$load = load ] && a="\$a --chains 300 --rounds 12 --burst"
    wl \$a --out $OUT/\$name.json
    python3 runtime/compile_workload.py $OUT/\$name.json $OUT/\$name.bundle --allow-opaque 2>&1 | tail -1
    echo "=== \$name"
    $BUILD/doom_exec_n\$n $OUT/\$name.bundle --wad doom/third_party/doom1.wad --out $OUT/\$name --timeout-s 300 --sched dynamic --classes 2 --min 1,0 --urgent-us 20000,0 2>&1 | grep -E "instance [0-9]+:|35 Hz|frames [0-9]+, achieved|compute check|OK:|FAIL|ERROR|error|DEVICE" || true
    if [ \$n = 1 ]; then python3 doom/compare.py $OUT/cpu $OUT/\$name | head -2
    else for k in \$(seq 0 \$((n-1))); do echo -n "inst\$k: "; python3 doom/compare.py $OUT/cpu $OUT/\$name/inst\$k | head -1; done; fi
  done
done
echo "=== memcheck N=2, 12 frames"
python3 doom/make_workload.py --frames 12 --instances 2 --doom-priority 0 --out $OUT/mc.json >/dev/null
python3 runtime/compile_workload.py $OUT/mc.json $OUT/mc.bundle --allow-opaque 2>&1 | tail -1
compute-sanitizer --tool memcheck --error-exitcode 9 $BUILD/doom_exec_n2 $OUT/mc.bundle --wad doom/third_party/doom1.wad --out $OUT/mc --timeout-s 300 2>&1 | grep -E "ERROR SUMMARY|Error|OK:|FAIL" | head
echo "sanitizer exit: \${PIPESTATUS[0]}"
HOLD
flock -w 900 $HOME/.gpu.lock timeout 3000 bash $OUT/hold.sh 2>&1 || echo "run failed"
REMOTE_SCRIPT
echo "log: $LOGDIR/run.log"
