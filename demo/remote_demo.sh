#!/usr/bin/env bash
# Box side of demo/run_demo.sh (run it through that script). Builds, generates the demo from the frontend, checks the manifest on the CPU,
# then runs the GPU part under ONE hold of the GPU lock, then verifies. Configuration arrives as environment variables:
#   [SKIP_BUILD=1 reuse the existing demo/kvm_run]  DEMO_TAG MODEL TILING STEPS WINDOW SERVE_EVERY FRAMES PRECISE(0/1) MAXB SANITIZE(0/1) RUN_ARGS NVCC_ARCH LOCK_WAIT
set -euo pipefail
cd "$(dirname "$0")/.."
RUN="runs/$DEMO_TAG"; rm -rf "$RUN"; mkdir -p "$RUN"
NVCC="$(command -v nvcc || ls /usr/local/cuda*/bin/nvcc | tail -1)"
say() { printf '\n=== %s  %s\n' "$(date -u +%FT%TZ)" "$*"; }

say "environment"
echo "$($NVCC --version | tail -1)"; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
git -C . rev-parse --short HEAD 2>/dev/null || echo "commit: ${KVM_COMMIT:-unknown}"

say "Doom engine objects + WAD"
if [ ! -f doom/build/doom_dev.o ]; then
  DS="${KVM_DOOM_SCRATCH:-$HOME/kvm-doom-scratch}"
  if [ -f "$DS/doom/build/doom_dev.o" ]; then for d in build third_party; do ln -sfn "$DS/doom/$d" "doom/$d"; done
  else bash doom/setup.sh && bash doom/build.sh && bash doom/cpu_build.sh; fi
fi
ls -la doom/build/doom_dev.o doom/third_party/doom1.wad

say "build kvm_run"
BUILD="$NVCC -O3 -arch=${NVCC_ARCH:-sm_80} -std=c++17 -rdc=true -DKVM_WITH_GAME -I. -Iinclude --fmad=false -Xcompiler -ffp-contract=off -lineinfo -diag-suppress 177"
if [ "${SKIP_BUILD:-0}" = 1 ] && [ -x demo/kvm_run ]; then
  echo "reusing demo/kvm_run from an earlier run (SKIP_BUILD=1; the sources are the same commit)"
else
echo "$BUILD demo/kvm_run.cu doom/build/doom_dev.o -o demo/kvm_run"
$BUILD demo/kvm_run.cu doom/build/doom_dev.o -o demo/kvm_run 2>&1 | grep -v "cannot be statically determined" || true
fi
test -x demo/kvm_run

gen() {  # dir model steps window serve_every frames
  local d="$1"; shift
  ( cd frontend && python3 -m kvmfe.demo --all --out "../$d" --model "$1" --train-steps "$2" --window "$3" --serve-every "$4" --serve-policy exact \
      --game-frames "$5" --game-fb-slots 0 --tiling "$TILING" --max-bundles "$MAXB" $( [ "$PRECISE" = 1 ] && echo --precise ) ) 2>&1 | tail -4
  # the engine's data file travels with the demo (the manifest names it; the runner resolves it in --assets)
}
ASSETS="$PWD/doom/third_party"
COMPILE="python3 runtime/compile_workload.py"

say "CPU oracle for Doom ($FRAMES frames)"
python3 doom/cpu_run.py --frames "$FRAMES" --out "$RUN/oracle" --log warning 2>&1 | tail -2

say "generate demo: $MODEL, $STEPS steps in windows of $WINDOW, a request every $SERVE_EVERY, $FRAMES frames, tiling $TILING, precise=$PRECISE"
gen "$RUN/demo" "$MODEL" "$STEPS" "$WINDOW" "$SERVE_EVERY" "$FRAMES"
OBS='^(loss\.|.*\.cur[0-9]+$)'
DUMP='^(doom_state|doom_frame\.)'

say "CPU-only manifest check (parse, version flow, every bundle loads/remaps/validates)"
./demo/kvm_run "$RUN/demo" --check-only --assets "$ASSETS" --compile-cmd "$COMPILE" 2>&1 | tee "$RUN/check_only.log" | tail -4

if [ "$SANITIZE" = 1 ]; then
  say "generate the small demo for compute-sanitizer (4 steps, 8 frames)"
  gen "$RUN/demo_small" "$MODEL" 4 2 2 8
fi

GOLDEN_ARGS="--golden $RUN/golden.json"
if [ "${GOLDEN:-1}" = 1 ]; then
say "golden: the CPU reference VM runs the same MANIFEST"
python3 - <<PY 2>&1 | tail -3
import sys, json; sys.path.insert(0, 'demo'); sys.argv = ['x']
import check_demo
g = check_demo.golden_observations(__import__('pathlib').Path("$RUN/demo"), r'$OBS', __import__('pathlib').Path("$RUN/golden.json"))
print('golden admissions:', len(g))
PY
else GOLDEN_ARGS="--no-golden"; say "golden skipped (pure-Python CPU VM cannot run this model)"; fi

say "GPU: one lock hold ($LOCK_WAIT s max wait)"
cat > "$RUN/gpu.sh" <<GPU
set -uo pipefail
echo "lock acquired \$(date -u +%FT%TZ)"
timeout 3600 ./demo/kvm_run "$RUN/demo" --out "$RUN/out" --assets "$ASSETS" --observe '$OBS' --dump '$DUMP' --compile-cmd "$COMPILE" ${RUN_ARGS:-} > "$RUN/kvm_run.log" 2>&1
echo "kvm_run exit \$?"
if [ "$SANITIZE" = 1 ]; then
  for tool in memcheck racecheck synccheck; do
    echo "== compute-sanitizer \$tool on the small demo"
    timeout 3000 compute-sanitizer --tool \$tool --error-exitcode 9 ./demo/kvm_run "$RUN/demo_small" --out "$RUN/san_\$tool" --assets "$ASSETS" --compile-cmd "$COMPILE" --timeout-s 600 --stall-s 600 > "$RUN/sanitizer_\$tool.log" 2>&1
    echo "sanitizer \$tool exit \$?"; tail -4 "$RUN/sanitizer_\$tool.log"
  done
fi
echo "lock released \$(date -u +%FT%TZ)"
GPU
flock -w "$LOCK_WAIT" ~/.gpu.lock bash "$RUN/gpu.sh" || { echo "could not take the GPU lock within $LOCK_WAIT s"; exit 75; }
tail -8 "$RUN/kvm_run.log"

say "verify"
python3 demo/check_demo.py "$RUN/demo" "$RUN/out" --doom-oracle "$RUN/oracle" $GOLDEN_ARGS --observe "$OBS" --report "$RUN/report.json" 2>&1 | tee "$RUN/check.log"
