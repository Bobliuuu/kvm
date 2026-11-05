#!/usr/bin/env bash
# The whole demo on a GPU box, one command: GPT training in windows + live inference on versioned weights + REAL Doom, concurrently,
# on one persistent kernel, driven only by MANIFEST.json (demo/kvm_run.cu). Checks loss, version coherence, Doom bytes vs the CPU engine,
# and the 35 Hz cadence. The GPU lock is taken ONCE (flock) for the run (and the sanitizer runs).
#
#   KVM_HOST=user@host [KVM_KEY=key] demo/run_demo.sh [tiny|large|custom] [options] [-- kvm_run options]
#     --model M --steps N --window W --serve-every K --frames F --tiling T   (override the preset)
#     --tf32           1xTF32 GEMMs instead of the exact 3xTF32 (the preset 'large' uses it)
#     --sanitize       also run compute-sanitizer memcheck/racecheck/synccheck on a small demo, inside the same lock hold
#     --tag NAME       run name (default: preset-UTC)     --lock-wait S (default 900)
# Output: logs/<UTC>-demo-<tag>/ (git-ignored): run.json, trace.csv, observations.jsonl, check.log, report.json, kvm_run.log.
set -euo pipefail
HOST="${KVM_HOST:?set KVM_HOST=user@host (an environment variable, never a committed file)}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PRESET=tiny; MODEL=""; STEPS=""; WINDOW=""; SERVE_EVERY=""; FRAMES=""; TILING=""; PRECISE=""; SANITIZE=0; TAG=""; LOCK_WAIT=900; RUN_ARGS=""
if [ $# -gt 0 ] && [[ "$1" != -* ]]; then PRESET="$1"; shift; fi
while [ $# -gt 0 ]; do
  case "$1" in
    --model) MODEL="$2"; shift 2;; --steps) STEPS="$2"; shift 2;; --window) WINDOW="$2"; shift 2;; --serve-every) SERVE_EVERY="$2"; shift 2;;
    --frames) FRAMES="$2"; shift 2;; --tiling) TILING="$2"; shift 2;; --tf32) PRECISE=0; shift;; --sanitize) SANITIZE=1; shift;;
    --tag) TAG="$2"; shift 2;; --lock-wait) LOCK_WAIT="$2"; shift 2;; --) shift; RUN_ARGS="$*"; break;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done
case "$PRESET" in
  tiny)  : "${MODEL:=gpt_tiny}"; : "${STEPS:=200}"; : "${WINDOW:=10}"; : "${SERVE_EVERY:=50}"; : "${FRAMES:=700}"; : "${TILING:=a100}"; : "${PRECISE:=1}"; MAXB=2;;
  large) : "${MODEL:=gpt_162m}"; : "${STEPS:=6}"; : "${WINDOW:=1}"; : "${SERVE_EVERY:=2}"; : "${FRAMES:=700}"; : "${TILING:=a100}"; : "${PRECISE:=0}"; MAXB=8;;
  custom) : "${MODEL:?--model}"; : "${STEPS:?--steps}"; : "${WINDOW:?--window}"; : "${SERVE_EVERY:?--serve-every}"; : "${FRAMES:?--frames}"; : "${TILING:=a100}"; : "${PRECISE:=1}"; MAXB=2;;
  *) echo "unknown preset $PRESET (tiny|large|custom)" >&2; exit 2;;
esac
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"; TAG="${TAG:-$PRESET-$STAMP}"
LOG="$ROOT/logs/$STAMP-demo-$TAG"; mkdir -p "$LOG"
SSH="ssh -o StrictHostKeyChecking=no -o BatchMode=yes ${KVM_KEY:+-i $KVM_KEY}"
REMOTE='kvm-demo-scratch'
{
echo "# demo $TAG  host=<KVM_HOST>  commit=$(git -C "$ROOT" rev-parse --short HEAD)$(git -C "$ROOT" diff --quiet || echo +dirty)  preset=$PRESET model=$MODEL steps=$STEPS window=$WINDOW serve_every=$SERVE_EVERY frames=$FRAMES tiling=$TILING precise=$PRECISE sanitize=$SANITIZE"
rsync -az -e "$SSH" --delete --exclude .git --exclude logs --exclude docs --exclude results --exclude 'doom/build' --exclude 'doom/third_party' --exclude runs --exclude 'demo/kvm_run' "$ROOT/" "$HOST:$REMOTE/"
# Detached on the box (nohup): a long wait in the GPU-lock queue must not depend on this ssh session. We poll the remote log.
$SSH "$HOST" "cd $REMOTE && mkdir -p runs && rm -f runs/$TAG.done && (KVM_COMMIT='$(git -C "$ROOT" rev-parse --short HEAD)' DEMO_TAG='$TAG' MODEL='$MODEL' TILING='$TILING' STEPS='$STEPS' WINDOW='$WINDOW' SERVE_EVERY='$SERVE_EVERY' FRAMES='$FRAMES' PRECISE='$PRECISE' MAXB='$MAXB' SANITIZE='$SANITIZE' RUN_ARGS='$RUN_ARGS' LOCK_WAIT='$LOCK_WAIT' nohup bash -c 'bash demo/remote_demo.sh > runs/$TAG.log 2>&1; echo \$? > runs/$TAG.done' > /dev/null 2>&1 &)"
off=0
while :; do
  sleep 20
  chunk="$($SSH "$HOST" "tail -c +$((off + 1)) $REMOTE/runs/$TAG.log 2>/dev/null | head -c 200000" || true)"
  if [ -n "$chunk" ]; then printf '%s\n' "$chunk"; off=$((off + ${#chunk} + 1)); fi
  rc="$($SSH "$HOST" "cat $REMOTE/runs/$TAG.done 2>/dev/null" || true)"
  if [ -n "$rc" ]; then
    rest="$($SSH "$HOST" "tail -c +$((off + 1)) $REMOTE/runs/$TAG.log" || true)"; [ -n "$rest" ] && printf '%s\n' "$rest"
    exit "$rc"
  fi
done
} 2>&1 | tee "$LOG/run.log"
status=${PIPESTATUS[0]}
rsync -az -e "$SSH" --exclude 'buffers' --exclude 'demo' --exclude 'demo_small' --exclude 'oracle' --exclude 'doom_candidate' "$HOST:$REMOTE/runs/$TAG/" "$LOG/remote/" || true
echo "logs: $LOG (exit $status)"
exit "$status"
