#!/usr/bin/env bash
# Build the real Doom engine (doomgeneric) + libc + glue + scripted player into a relocatable device object build/doom_dev.o with
# clang's NVPTX backend; link it into a CUDA program with nvcc -rdc=true. Idempotent: skipped when no input changed.
#   ARCH=sm_80 (default; sm_90, sm_100 ...)  NVCC=/path/to/nvcc  FORCE=1  LOG_LEVEL=debug|info|warn
set -euo pipefail
. "$(dirname "$0")/lib.sh"
ARCH="${ARCH:-sm_80}"
DOOM_INSTANCES="${DOOM_INSTANCES:-1}"   # N > 1 also builds doom_dev_i<k>.o (k = 1..N-1): the same engine with every symbol prefixed i<k>_ (independent globals)
for t in clang llvm-link llvm-dis llc python3; do need "$t" "install clang+llvm with the NVPTX target (apt install clang llvm)"; done
NVCC="${NVCC:-$(command -v nvcc || ls /usr/local/cuda*/bin/nvcc 2>/dev/null | tail -1 || true)}"
[ -x "${NVCC:-}" ] || die "nvcc not found: set NVCC=/path/to/nvcc"
SRC="$TP/doomgeneric/doomgeneric"
[ -d "$SRC" ] && [ -s "$TP/doom1.wad" ] || die "engine/WAD missing: run doom/setup.sh"
clang --print-targets | grep -q nvptx64 || die "this clang has no nvptx64 target ($(clang --version | head -1))"

mkdir -p "$BUILD"
STAMP="$(cat "$DOOM_DIR"/patch_engine.py "$DOOM_DIR"/build.sh "$DOOM_DIR"/lib.sh "$DOOM_DIR"/glue/*.[ch] "$DOOM_DIR"/glue/*.py "$DOOM_DIR"/glue/libc/*.h "$DOOM_DIR"/glue/libc/sys/*.h \
  "$DOOM_DIR"/player/player.c "$DOOM_DIR"/player/route_e1m1.h | md5sum | cut -d' ' -f1)-$(git -C "$TP/doomgeneric" rev-parse HEAD)-$ARCH-n$DOOM_INSTANCES-p${DOOM_PROF:-0}"
if [ -s "$BUILD/doom_dev.o" ] && [ "$(cat "$BUILD/stamp" 2>/dev/null)" = "$STAMP" ] && [ -z "${FORCE:-}" ]; then
  info "doom_dev.o up to date ($ARCH)"; exit 0
fi
info "building doom_dev.o for $ARCH with $(clang --version | head -1), $("$NVCC" --version | tail -1)"

RES="$(clang -print-resource-dir)"
CF=(--target=nvptx64-nvidia-cuda -march=sm_80 -O2 -ffreestanding -fno-stack-protector -fno-jump-tables -w -nostdinc
    -isystem "$DOOM_DIR/glue/libc" -isystem "$RES/include" -I"$BUILD/src" -I"$DOOM_DIR/glue" ${DOOM_PROF:+-DDOOM_PROF} -DNORMALUNIX -DLINUX -D_DEFAULT_SOURCE
    -DDOOMGENERIC_RESX=320 -DDOOMGENERIC_RESY=200)
python3 "$DOOM_DIR/patch_engine.py" "$SRC" "$BUILD/src" gpu
OBJ="$BUILD/obj"; rm -rf "$OBJ"; mkdir -p "$OBJ"
cc1() {   # name source [extra flags]
  local n="$1" s="$2"; shift 2
  clang "${CF[@]}" "$@" -c -emit-llvm "$s" -o "$OBJ/$n.bc" 2> "$OBJ/$n.err" || { warn "FAIL $n"; head -15 "$OBJ/$n.err" >&2; return 1; }
}
debug "compiling $(echo $ENGINE_FILES | wc -w) engine files + glue with $(nproc 2>/dev/null || sysctl -n hw.ncpu) jobs"
pids=()
for f in $ENGINE_FILES; do
  opt=-O2; [ "$f" = m_menu ] && opt=-O1
  cc1 "$f" "$BUILD/src/$f.c" "$opt" & pids+=($!)
done
cc1 libc "$DOOM_DIR/glue/libc.c" & pids+=($!)
cc1 gpu_par "$DOOM_DIR/glue/gpu_par.c" & pids+=($!)
cc1 gpu_glue "$DOOM_DIR/glue/gpu_glue.c" & pids+=($!)
cc1 doom_glue "$DOOM_DIR/glue/doom_glue.c" & pids+=($!)
cc1 player "$DOOM_DIR/player/player.c" -I"$DOOM_DIR/player" & pids+=($!)
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ $fail = 0 ] || die "compile failed (logs in $OBJ/*.err)"
debug "linking $(ls "$OBJ"/*.bc | wc -l) bitcode modules"
llvm-link "$OBJ"/*.bc -o "$BUILD/doom.bc"
llvm-dis "$BUILD/doom.bc" -o "$BUILD/doom.ll"
python3 "$DOOM_DIR/glue/widen_i1_globals.py" "$BUILD/doom.ll" "$BUILD/doom_i8.ll"
python3 "$DOOM_DIR/glue/fix_varargs.py" "$BUILD/doom_i8.ll" "$BUILD/doom_nv.ll"
llc -march=nvptx64 -mcpu=sm_80 -mattr=+ptx75 -O2 "$BUILD/doom_nv.ll" -o "$BUILD/doom.ptx"
# the PTX is always sm_80; ptxas compiles it forward for newer GPUs. Registers are capped so the engine's CTA leaves room for other work.
"$NVCC" -gencode "arch=compute_80,code=$ARCH" -dc -maxrregcount=112 "$BUILD/doom.ptx" -o "$BUILD/doom_dev.o" 2>&1 | grep -v 'Potential out of bound' || true
[ -s "$BUILD/doom_dev.o" ] || die "nvcc produced no doom_dev.o"
inst() {   # k: the same engine with every symbol prefixed i<k>_
  local k=$1
  python3 "$DOOM_DIR/glue/prefix_symbols.py" "$BUILD/doom_nv.ll" "$BUILD/doom_i$k.ll" "i${k}_"
  llc -march=nvptx64 -mcpu=sm_80 -mattr=+ptx75 -O2 "$BUILD/doom_i$k.ll" -o "$BUILD/doom_i$k.ptx"
  "$NVCC" -gencode "arch=compute_80,code=$ARCH" -dc -maxrregcount=112 "$BUILD/doom_i$k.ptx" -o "$BUILD/doom_dev_i$k.o" 2>&1 | grep -v 'Potential out of bound' || true
  [ -s "$BUILD/doom_dev_i$k.o" ] || die "nvcc produced no doom_dev_i$k.o"
}
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"; running=0
for k in $(seq 1 $((DOOM_INSTANCES - 1))); do
  inst "$k" & running=$((running + 1))
  if [ "$running" -ge "$JOBS" ]; then wait -n || die "instance build failed"; running=$((running - 1)); fi
done
wait || die "instance build failed"
for k in $(seq 1 $((DOOM_INSTANCES - 1))); do [ -s "$BUILD/doom_dev_i$k.o" ] || die "missing doom_dev_i$k.o"; done
echo "$STAMP" > "$BUILD/stamp"
info "built $BUILD/doom_dev.o ($(wc -c < "$BUILD/doom_dev.o") bytes, $ARCH)"
