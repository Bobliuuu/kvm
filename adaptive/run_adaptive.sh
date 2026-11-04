#!/usr/bin/env bash
# Sync, build and run a batch of adaptive_run experiments on the GPU box under ONE lock hold, then fetch the raw logs.
#   KVM_HOST=user@host adaptive/run_adaptive.sh GROUP [SCENARIO_DIR]     GROUP: smoke | main | multi | bw | suites (see adaptive/batch.sh)
# Env: REPS DUR OUT (remote dir name under the scratch, default runs-<group>) KVM_REMOTE (default ~/kvm-adaptive-scratch) KVM_KEY
# The scenario (adaptive/make_scenario.py) is generated locally and synced once. Logs: logs/<UTC>-adaptive-<group>/ (git-ignored).
set -euo pipefail
HOST="${KVM_HOST:?set KVM_HOST=user@host (an environment variable, never a committed file)}"
GROUP="${1:?group}"; SCN="${2:-}"
DIR="$(cd "$(dirname "$0")/.." && pwd)"; REMOTE="${KVM_REMOTE:-kvm-adaptive-scratch}"
SSH="ssh -o StrictHostKeyChecking=no -o BatchMode=yes ${KVM_KEY:+-i $KVM_KEY}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"; LOG="$DIR/logs/$STAMP-adaptive-$GROUP"; mkdir -p "$LOG"
OUT="${OUT:-runs-$GROUP}"
$SSH "$HOST" "mkdir -p $REMOTE"
rsync -az -e "$SSH" --delete --include 'include/***' --include 'runtime/***' --include 'tests/' --include 'tests/gpu/***' --include 'tests/host/***' --include 'adaptive/***' \
  --include 'doom/' --include 'doom/*.py' --include 'doom/*.sh' --include 'doom/glue/***' --include 'kvm_exec.cu' --include 'frontend/***' --exclude '*' "$DIR/" "$HOST:$REMOTE/"
[ -z "$SCN" ] || rsync -az -e "$SSH" --delete "$SCN/" "$HOST:$REMOTE/scn/"
COMMIT="$(git -C "$DIR" rev-parse --short HEAD)$(git -C "$DIR" diff --quiet HEAD -- . 2>/dev/null || echo +dirty)"
# two binaries: adaptive_run (1 Doom instance, the engine of every ordinary run) and adaptive_run_n8 (8 instances, for the multi-tenant group)
$SSH "$HOST" "cd $REMOTE && for d in doom/build doom/build_n8; do [ ! -L \$d ] || rm \$d; done; mkdir -p doom/build doom/build_n8; for f in ~/kvm-doom-scratch/doom/build/*; do ln -sfn \$f doom/build/; ln -sfn \$f doom/build_n8/; done; \
  ln -sfn ~/kvm-doom-scratch/doom/third_party doom/third_party; rm -f doom/build/doom_dev_i*.o; \
  cp -n \${KVM_DOOM_INST_DIR:-~/kvm-adaptive-doom-scratch/doom/build_n8}/doom_dev_i*.o doom/build_n8/ 2>/dev/null; \
  bash adaptive/build.sh 2>&1 | grep -E 'error' -A3; DOOM_BUILD_DIR=doom/build_n8 OUT_BIN=adaptive_run_n8 bash adaptive/build.sh 2>&1 | grep -E 'error' -A3; \
  bash tests/gpu/build_adaptive.sh 2>&1 | grep -E 'error' -A3; true"
echo "commit $COMMIT, group $GROUP, remote $REMOTE/$OUT" | tee "$LOG/run.log"
# one hold for the whole batch; the batch runs detached so a dropped ssh session does not abort it
$SSH "$HOST" "cd $REMOTE && rm -rf $OUT && mkdir -p $OUT && nohup flock -w ${LOCK_WAIT:-14400} ~/.gpu.lock env OUT=$OUT REPS=${REPS:-5} DUR=${DUR:-24} BATCH_COMMIT=$COMMIT bash adaptive/batch.sh $GROUP > $OUT/batch.log 2>&1 &"
sleep 5
until $SSH "$HOST" "cd $REMOTE && grep -q '^# batch $GROUP done' $OUT/batch.log"; do sleep 20; done
rsync -az -e "$SSH" "$HOST:$REMOTE/$OUT/" "$LOG/$OUT/"
echo "fetched to $LOG/$OUT" | tee -a "$LOG/run.log"
