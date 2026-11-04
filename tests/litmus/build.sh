#!/usr/bin/env bash
# Build the litmus binaries on the GPU box: the production header plus MUTANTS (each removes exactly one ordering step).
# Usage (in a worktree copy): tests/litmus/build.sh <outdir>      -> <outdir>/litmus_prod, litmus_<mutant>...
set -euo pipefail
OUT="${1:?outdir}"; HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(cd "$HERE/../.." && pwd)"
NVCC="$(command -v nvcc || ls /usr/local/cuda*/bin/nvcc | tail -1)"
mkdir -p "$OUT"
B="-O3 -arch=sm_80 -std=c++17 -lineinfo"
$NVCC $B -I"$ROOT/include" "$HERE/litmus.cu" -o "$OUT/litmus_prod" &
$NVCC $B -DLIT_NOFENCE -DLIT_MUTANT='"nofence(all __threadfence)"' -I"$ROOT/include" "$HERE/litmus.cu" -o "$OUT/litmus_nofence" &
mut() {  # name, sed-expression
  local d="$OUT/hdr_$1"; rm -rf "$d"; mkdir -p "$d"; cp "$ROOT"/include/* "$d/"
  sed -E "$2" "$ROOT/include/kvm_exec.cuh" > "$d/kvm_exec.cuh"
  if cmp -s "$d/kvm_exec.cuh" "$ROOT/include/kvm_exec.cuh"; then echo "mutant $1: sed matched nothing" >&2; exit 1; fi
  $NVCC $B -DLIT_MUTANT="\"$1\"" -I"$d" "$HERE/litmus.cu" -o "$OUT/litmus_$1" &
}
# Dekker: drop the two fences that implement the argument (waiter: push->fence->load; publisher: store->fence->exchange)
mut nodekkerfence 's#^(\s*)__threadfence\(\);  // seq_cst fence: push.*#\1/* mutant */#; s#^(\s*)__threadfence\(\);( +)// version store -> fence.*#\1/* mutant */#'
# Dekker: waiter never re-checks the version after pushing (classic lost wakeup)
mut norecheck 's#if \(ld_acq\(S.version \+ var\) >= need && atom_cas\(S.state \+ slot, 1, 2\) == 1\) kvm_enqueue#if (0 \&\& atom_cas(S.state + slot, 1, 2) == 1) kvm_enqueue#'
# Dekker: publisher takes the list BEFORE storing the version
mut exchbefore 's#^(\s*)int old = atom_cas\(S.version \+ var, ver - 1, ver\);.*#\1int n0 = atom_exch(S.park_head + var, -1); int old = atom_cas(S.version + var, ver - 1, ver);#; s#^(\s*)int n = atom_exch\(S.park_head \+ var, -1\);#\1int n = n0;#'
# MP: publisher fence removed and release CAS demoted to a relaxed atomicCAS
mut weakpub 's#^(\s*)__threadfence\(\);( +)// fence.sc.gpu: weights before version.*#\1/* mutant */#; s#int old = atom_cas\(S.version \+ var, ver - 1, ver\);#int old = atomicCAS(S.version + var, ver - 1, ver);#'
# Queue: popper claims the head with a plain store instead of CAS (double pop under contention)
mut popplain 's#if \(atom_cas64\(&q->head, h, h \+ 1\) == h\) \{#if ((*(volatile unsigned long long*)\&q->head = h + 1), true) {#'
# Queue: producer publishes the cell with a relaxed store (no release): payload may be stale
mut qrelax 's#st_rel64\(&cell->seq, pos \+ 1\);#*(volatile unsigned long long*)\&cell->seq = pos + 1;#'
wait
echo built in "$OUT"
