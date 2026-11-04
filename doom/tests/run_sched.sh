#!/usr/bin/env bash
# Real Doom beside a saturating bursty compute load under the three schedulers, plus the task-affinity measurement.
#   KVM_HOST=user@host doom/tests/run_sched.sh [frames=700] [chains=300] [rounds=40]
# The engine objects (doom/build, doom/third_party) are reused from ~/kvm-doom-scratch on the box (built by doom/build.sh); the CPU oracle is regenerated.
# Doom and the load are separate workload ids (graph 0 = doom, graph 1 = compute); NO hand-set priorities in any scenario (all 0), Doom tasks carry deadline_us only.
set -euo pipefail
: "${KVM_HOST:?set KVM_HOST=user@host}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TS="$(date -u +%Y%m%dT%H%M%SZ)"; LOGDIR="$ROOT/logs/$TS-doomsched"; mkdir -p "$LOGDIR"
SSH="ssh -o StrictHostKeyChecking=no -o BatchMode=yes"
rsync -az -e "$SSH" --delete --include 'include/***' --include 'runtime/***' --include 'doom/' --include 'doom/*.py' --include 'doom/*.cu' --include 'doom/*.sh' --include 'doom/glue/***' --include 'doom/tests/***' --include 'kvm_exec.cu' --exclude '*' "$ROOT/" "$KVM_HOST:${KVM_REMOTE:-kvm-sched-scratch}/"
$SSH "$KVM_HOST" "FRAMES=${1:-700} CHAINS=${2:-300} ROUNDS=${3:-40} KVM_REMOTE='${KVM_REMOTE:-~/kvm-sched-scratch}' ONLY='${ONLY:-}' KVM_NO_LOCK=${KVM_NO_LOCK:-} bash -s" <<'REMOTE' 2>&1 | tee "$LOGDIR/run.log"
set -euo pipefail
cd "${KVM_REMOTE/#\~/$HOME}"
for d in build third_party; do ln -sfn ~/kvm-doom-scratch/doom/$d doom/$d; done
NVCC=$(ls /usr/local/cuda*/bin/nvcc | tail -1)
echo "# $(date -u +%FT%TZ)  $($NVCC --version | tail -1)  $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
echo "BUILD: nvcc -O3 -arch=sm_80 -std=c++17 -rdc=true -DKVM_WITH_GAME -I. -Iinclude --fmad=false -Xcompiler -ffp-contract=off doom/exec_main.cu doom/build/doom_dev.o -o doom/build/doom_exec_sched"
$NVCC -O3 -arch=sm_80 -std=c++17 -rdc=true -DKVM_WITH_GAME -I. -Iinclude --fmad=false -Xcompiler -ffp-contract=off doom/exec_main.cu doom/build/doom_dev.o -o doom/build/doom_exec_sched 2>&1 | grep -v "cannot be statically determined" | grep -i "error" || true
OUT=$PWD/logs/doomsched; rm -rf $OUT; mkdir -p $OUT
python3 doom/cpu_run.py --frames $FRAMES --out $OUT/cpu --log info 2>&1 | tail -2
L="timeout 900"; [ -n "${KVM_NO_LOCK:-}" ] || L="flock -w 7200 $HOME/.gpu.lock timeout 900"
wl() { python3 doom/make_workload.py --frames $FRAMES "$@" ; }
# bundles: doom alone, and doom + bursty load, both with ALL priorities 0
wl --doom-priority 0 --out $OUT/alone.json >/dev/null
wl --doom-priority 0 --compute-priority 0 --chains $CHAINS --rounds $ROUNDS --burst --out $OUT/load.json >/dev/null
for b in alone load; do python3 runtime/compile_workload.py $OUT/$b.json $OUT/$b.bundle 2>&1 | grep -v "no reference semantics" | tail -1; done
run() {  # name, bundle, doom_exec args...
  local name="$1" bundle="$2"; shift 2
  [ -z "${ONLY:-}" ] || [[ " $ONLY " == *" $name "* ]] || return 0
  echo "=== scenario $name   doom_exec $bundle $*"
  $L doom/build/doom_exec_sched $OUT/$bundle.bundle --wad doom/third_party/doom1.wad --out $OUT/$name --timeout-s 600 "$@" 2>&1 | grep -E "workload [0-9]:|35 Hz|frames [0-9]+, achieved|compute check|OK:|FAIL|ERROR|error" || true
  python3 doom/compare.py $OUT/cpu $OUT/$name | head -2
  python3 - $OUT/$name/metrics.json <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
print("METRICS fps %.1f | tick gap p50 %.2f p99 %.2f max %.2f ms | gaps over 28.57 ms: %d | frame latency p99 %.2f ms | tick queue wait p99 %.0f us | doom on %d SMs | block util %.0f%%" % (
  m["achieved_fps"], m["tick_gap_us"]["p50"] / 1e3, m["tick_gap_us"]["p99"] / 1e3, m["tick_gap_us"]["max"] / 1e3, m["tick_gap_over_28571us"], m["frame_latency_us"]["p99"] / 1e3, m["tick_queue_wait_us"]["p99"], m["sms_used_by_doom"], 100 * m["mean_block_utilization"]))
PY
}
# -- Doom alone: affinity effect on the real chain (215 workers: ticks migrate over SMs; 1 worker: every task on one SM)
run alone_orig            alone
run alone_one_worker      alone --blocks 2
run alone_dyn             alone --sched dynamic --classes 1
run alone_dyn_chainaff    alone --sched dynamic --classes 1 --affinity-us 2000
run alone_dyn_opaff       alone --sched dynamic --classes 1 --op-affinity 1 --affinity-us 2000
# -- Doom beside the saturating bursty load: (i) one global queue, no priorities  (ii) static SM split  (iii) dynamic: reservation + deadlines only
run load_orig             load
run load_global           load --sched global  --classes 2
run load_static_10        load --sched static  --classes 2 --shares 1,9
run load_static_25        load --sched static  --classes 2 --shares 1,3
run load_dyn              load --sched dynamic --classes 2 --min 1,0 --urgent-us 20000,0
run load_dyn_noresv       load --sched dynamic --classes 2 --min 0,0 --urgent-us 0,0
run load_dyn_opaff        load --sched dynamic --classes 2 --min 1,0 --urgent-us 20000,0 --op-affinity 1,0 --affinity-us 2000
REMOTE
echo "log: $LOGDIR/run.log"
