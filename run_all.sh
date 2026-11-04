#!/usr/bin/env bash
# Run a group of the test entry points while holding the GPU box's lock ONCE (the box is shared; queueing per stage can take hours).
# Usage: KVM_HOST=user@host KVM_REMOTE='~/kvm-sched-scratch' ./run_all.sh <group>
#   correctness : run_exec_tests.sh (functional, --negative, --negative-park, --sanitize, --litmus, --gpt) + run_sched_tests.sh (functional, --negative, --sanitize)
#   fuzz        : run_exec_tests.sh --fuzz (FC cases, FS seed) and --fuzz with FZ=--targeted
#   final       : functional exec + litmus + sched suites, both fuzz runs, then bench and the Doom scenario (one lock hold)
#   bench       : run_sched_tests.sh --bench, then doom/tests/run_sched.sh
# The lock is held by one background ssh for the duration; the stages run with KVM_NO_LOCK=1. Logs: logs/<UTC>-<what>/run.txt per stage.
set -euo pipefail
HOST="${KVM_HOST:?set KVM_HOST=user@host}"; DIR="$(cd "$(dirname "$0")" && pwd)"; GROUP="${1:?group: correctness|fuzz|bench}"
SSH="ssh -o StrictHostKeyChecking=no -o BatchMode=yes ${KVM_KEY:+-i $KVM_KEY}"
mkfifo "$DIR/.lockpipe.$$"; trap 'rm -f "$DIR/.lockpipe.$$"' EXIT
# the remote shell takes the lock, prints LOCKED, then waits until its stdin closes
( $SSH "$HOST" 'flock -w 14400 ~/.gpu.lock bash -c "echo LOCKED; cat > /dev/null"' < "$DIR/.lockpipe.$$" > "$DIR/.lockout.$$" 2>&1 ) &
HOLD=$!; exec 9> "$DIR/.lockpipe.$$"
for i in $(seq 1 1800); do grep -q LOCKED "$DIR/.lockout.$$" 2>/dev/null && break; sleep 5; done
grep -q LOCKED "$DIR/.lockout.$$" || { echo "could not take the GPU lock"; exit 1; }
echo "GPU lock held; running group $GROUP"; export KVM_NO_LOCK=1
case "$GROUP" in
  correctness) for m in "" --negative --negative-park --sanitize --litmus --gpt; do "$DIR/run_exec_tests.sh" $m || true; done
               for m in "" --negative --sanitize; do "$DIR/run_sched_tests.sh" $m || true; done;;
  fuzz)        "$DIR/run_exec_tests.sh" --fuzz || true; FZ=--targeted "$DIR/run_exec_tests.sh" --fuzz || true;;
  final)       "$DIR/run_exec_tests.sh" || true; "$DIR/run_exec_tests.sh" --litmus || true; "$DIR/run_sched_tests.sh" || true; "$DIR/run_exec_tests.sh" --fuzz || true
               FZ=--targeted "$DIR/run_exec_tests.sh" --fuzz || true; "$DIR/run_sched_tests.sh" --bench || true; "$DIR/doom/tests/run_sched.sh" || true;;
  fuzz2)       "$DIR/run_exec_tests.sh" --fuzz || true; FZ=--targeted "$DIR/run_exec_tests.sh" --fuzz || true; "$DIR/run_sched_tests.sh" || true;;
  bench)       "$DIR/run_sched_tests.sh" --bench || true; "$DIR/doom/tests/run_sched.sh" || true;;
esac
exec 9>&-; wait $HOLD || true; rm -f "$DIR/.lockout.$$"
echo "lock released"
