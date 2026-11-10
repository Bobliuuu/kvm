#!/usr/bin/env bash
# Build an executor (kvm_exec.cu, demo/kvm_run.cu, ...) with pluggable operator packs. FRAMEWORK.md "Operator packs".
#
#   eval "$(tools/kvm_build.sh --print-env --pack packs/my_pack)"            # python side: KVM_EXTRA_DEFS / KVM_EXTRA_REFS
#   tools/kvm_build.sh --pack packs/my_pack -o kvm_exec kvm_exec.cu          # executor with my_op
#   tools/kvm_build.sh --pack game --pack packs/my_pack -o kvm_run demo/kvm_run.cu
#
# A pack is a directory with ops.def (X(name, id, "args") lines), ops.cuh (one KVM_OP per op), ref.py (CPU reference, OPS dict) and an
# optional pack.conf: rdc=1 (needs relocatable device code), link=<object relative to the pack>, define=NAME[=VALUE] (repeatable).
# `game` is built in: -rdc=true, -DKVM_WITH_GAME and doom/build/doom_dev.o are added by the flag (doom/build.sh makes the object).
# Several packs combine into one generated .def/.cuh passed through the KVM_EXTRA_OPS_DEF/CUH hooks; tools/check_packs.py refuses
# duplicate names or ids before nvcc starts. Extra arguments after the source go to nvcc. NVCC and KVM_ARCH (default sm_80) are honored.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PACKS=(); OUT=""; PRINT=0; SRC=""; PASS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --pack) PACKS+=("${2:?--pack needs a directory or 'game'}"); shift 2;;
    -o) OUT="${2:?-o needs a path}"; shift 2;;
    --print-env) PRINT=1; shift;;
    -h|--help) sed -n '2,13p' "$0"; exit 0;;
    *) if [ -z "$SRC" ]; then SRC="$1"; else PASS+=("$1"); fi; shift;;
  esac
done

RDC=0; NV_EXTRA=(); LINK=(); DEFS=(); REFS=(); CUHS=()
for p in ${PACKS[@]+"${PACKS[@]}"}; do
  if [ "$p" = game ]; then
    RDC=1; NV_EXTRA+=(-DKVM_WITH_GAME)
    LINK+=("$ROOT/doom/build/doom_dev.o")
    continue
  fi
  p="${p%/}"
  [ -f "$p/ops.def" ] || { echo "kvm_build: $p/ops.def not found" >&2; exit 2; }
  DEFS+=("$p/ops.def")
  [ -f "$p/ref.py" ] && REFS+=("$p/ref.py")
  [ -f "$p/ops.cuh" ] && CUHS+=("$(cd "$p" && pwd -P)/ops.cuh")
  if [ -f "$p/pack.conf" ]; then
    while IFS='=' read -r k v; do
      k="${k%%#*}"; k="${k// /}"; [ -z "$k" ] && continue
      case "$k" in
        rdc) [ "$v" = 1 ] && RDC=1;;
        link) LINK+=("$(cd "$p" && pwd -P)/$v");;
        define) NV_EXTRA+=("-D$v");;
        *) echo "kvm_build: $p/pack.conf: unknown key '$k'" >&2; exit 2;;
      esac
    done < "$p/pack.conf"
  fi
done

join() { local IFS=':'; echo "$*"; }
if [ "$PRINT" = 1 ]; then
  printf "export KVM_EXTRA_DEFS='%s'\n" "$(join ${DEFS[@]+"${DEFS[@]}"})"
  printf "export KVM_EXTRA_REFS='%s'\n" "$(join ${REFS[@]+"${REFS[@]}"})"
  exit 0
fi

[ -n "$SRC" ] || { echo "usage: tools/kvm_build.sh [--pack DIR|game]... -o OUT SOURCE.cu [nvcc args]" >&2; exit 2; }
[ -n "$OUT" ] || OUT="$(basename "${SRC%.cu}")"
if [ ${#DEFS[@]} -gt 0 ]; then
  python3 -I "$ROOT/tools/check_packs.py" "${DEFS[@]}"
  GEN="$(mktemp -d)"; trap 'rm -rf "$GEN"' EXIT
  for d in "${DEFS[@]}"; do echo "#include \"$(cd "$(dirname "$d")" && pwd -P)/ops.def\""; done > "$GEN/packs.def"
  for c in ${CUHS[@]+"${CUHS[@]}"}; do echo "#include \"$c\""; done > "$GEN/packs.cuh"
  NV_EXTRA+=("-DKVM_EXTRA_OPS_DEF=\"$GEN/packs.def\"" "-DKVM_EXTRA_OPS_CUH=\"$GEN/packs.cuh\"")
fi
[ "$RDC" = 1 ] && NV_EXTRA+=(-rdc=true)
set -x
"${NVCC:-nvcc}" -O3 -arch="${KVM_ARCH:-sm_80}" -std=c++17 -I"$ROOT" -I"$ROOT/include" --fmad=false -Xcompiler -ffp-contract=off -lineinfo \
  -diag-suppress 177 ${NV_EXTRA[@]+"${NV_EXTRA[@]}"} ${PASS[@]+"${PASS[@]}"} "$SRC" ${LINK[@]+"${LINK[@]}"} -o "$OUT"
