#!/usr/bin/env bash
# Build the same engine, glue and scripted player for the CPU (build/libdoom_cpu.so): the correctness oracle for the GPU.
set -euo pipefail
. "$(dirname "$0")/lib.sh"
need gcc "a C compiler is needed for the CPU oracle"
SRC="$TP/doomgeneric/doomgeneric"
[ -d "$SRC" ] && [ -s "$TP/doom1.wad" ] || die "engine/WAD missing: run doom/setup.sh"
mkdir -p "$BUILD"
python3 "$DOOM_DIR/patch_engine.py" "$SRC" "$BUILD/src_cpu" cpu
info "building libdoom_cpu.so"
gcc -O2 -fPIC -shared -w -DNORMALUNIX -DLINUX -D_DEFAULT_SOURCE -DDOOMGENERIC_RESX=320 -DDOOMGENERIC_RESY=200 \
  -I"$BUILD/src_cpu" -I"$DOOM_DIR/glue" -I"$DOOM_DIR/player" \
  $(for f in $ENGINE_FILES; do echo "$BUILD/src_cpu/$f.c"; done) "$DOOM_DIR/glue/doom_glue.c" "$DOOM_DIR/player/player.c" \
  -o "$BUILD/libdoom_cpu.so" -lm
info "built $BUILD/libdoom_cpu.so"
