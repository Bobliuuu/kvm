#!/usr/bin/env bash
# ON the GPU box (repo copy in $SRC, default ~/kvm-scratch-general-bf16/src): bf16 storage checks on the A100, one short lock hold per run.
#   gen (CPU only, slow pure-Python golden) -> build kvm_exec (not under the lock, one nvcc at a time) -> run each precision under flock.
# Output: $OUT/*.txt. Compares kvm_exec --golden for every bundle and the dumped weights/losses of bf16 vs bf16-real (must be identical).
set -uo pipefail
SRC="${SRC:-$HOME/kvm-scratch-general-bf16/src}"; W="${W:-$HOME/kvm-scratch-general-bf16}"; OUT="$W/out"; mkdir -p "$OUT"
STEPS="${STEPS:-3}"; WIDE="${WIDE:---wide}"
cd "$SRC"
for p in f32 bf16 bf16-real; do
  [ -f "$W/b_$p/bundle/golden.json" ] || python3 -I tests/bf16/gen_bundle.py "$W/b_$p" "$p" --steps "$STEPS" --tiling a100 $WIDE 2>&1 | grep -v INFO | tee -a "$OUT/gen.txt"
done
[ -x "$W/kvm_exec" ] || nvcc -O3 -arch=sm_80 -std=c++17 -I. -Iinclude --fmad=false -Xcompiler -ffp-contract=off -lineinfo -diag-suppress 177 kvm_exec.cu -o "$W/kvm_exec" > "$OUT/build_exec.txt" 2>&1
for p in f32 bf16 bf16-real; do
  flock -w 900 ~/.gpu.lock timeout 300 "$W/kvm_exec" --blocks 4 --dump-buffers --golden --golden-tol 5e-3 "$W/b_$p/bundle" > "$OUT/run_$p.txt" 2>&1
  echo "$p exit $?" | tee -a "$OUT/summary.txt"
done
