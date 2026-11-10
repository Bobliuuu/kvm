#!/usr/bin/env bash
# Core-only build of a bench program that includes kvm_exec.cu (no gpt/extra/comm/game packs): about a minute.
#   bench/hop/build_core.sh SRC OUT [nvcc flags...]   (run from the repository root, on the GPU box)
set -euo pipefail
SRC="$1"; OUT="$2"; shift 2
NVCC="$(command -v nvcc || ls /usr/local/cuda*/bin/nvcc | tail -1)"
INC="$(mktemp -d)"; trap 'rm -rf "$INC"' EXIT
for f in include/*; do case "$(basename "$f")" in kvm_ops_gpt.*|kvm_ops_extra.*|kvm_ops_comm.*|kvm_ops_game.*) ;; *) cp "$f" "$INC/";; esac; done
exec "$NVCC" -O3 -arch="${KVM_ARCH:-sm_80}" -std=c++17 -I"$INC" -I. --fmad=false -Xcompiler -ffp-contract=off -diag-suppress 177 -DKVM_EXEC_NO_MAIN "$@" "$SRC" -o "$OUT"
