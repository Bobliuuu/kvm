#!/usr/bin/env bash
# Run the demo on the GPU box once per scheduler (real trace, loss, serve tokens, Doom frames) and copy the raw outputs under logs/showcase-<mode>/.
#   KVM_HOST=user@host [KVM_KEY=key] [KVM_REMOTE=my-scratch-dir] [KVM_LOCK_WAIT=7200] demo/viz/collect.sh [modes...]     default: dynamic static global
# One demo/run_demo.sh run per mode (preset tiny: 200 AdamW steps, 5 serve requests, 700 Doom frames; one GPU-lock hold of about a minute each).
# 'static' splits the SMs 70/10/20 between train/serve/game (the strawman the dynamic scheduler is compared with); 'global' is one shared queue.
# The modes run concurrently from here (each waits for the lock on its own); the first run on a fresh scratch dir must not race the compile,
# so on a new box run `demo/run_demo.sh tiny` once first. Then: demo/viz/make_all.sh --package
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MODES=("$@"); [ ${#MODES[@]} -gt 0 ] || MODES=(dynamic static global)
export KVM_REMOTE="${KVM_REMOTE:-kvm-showcase-scratch}"
one() {
  local m="$1" extra
  case "$m" in
    dynamic) extra="--sched dynamic";; global) extra="--sched global";; static) extra="--sched static --shares 0.7,0.1,0.2";;
    *) echo "unknown mode $m (dynamic|static|global)" >&2; return 2;;
  esac
  "$ROOT/demo/run_demo.sh" tiny --tag "showcase-$m" --fetch-frames --reuse-build --lock-wait "${KVM_LOCK_WAIT:-7200}" -- $extra
  local last; last="$(ls -dt "$ROOT"/logs/*-demo-showcase-"$m" | head -1)"
  rm -rf "$ROOT/logs/showcase-$m"; mv "$last" "$ROOT/logs/showcase-$m"
  echo "collected $m -> logs/showcase-$m"
}
pids=()
for m in "${MODES[@]}"; do one "$m" > "$ROOT/logs/collect-$m.out" 2>&1 & pids+=($!); done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
for m in "${MODES[@]}"; do echo "== $m"; tail -3 "$ROOT/logs/collect-$m.out"; done
exit $rc
