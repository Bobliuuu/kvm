#!/usr/bin/env bash
# Build the litmus binaries on the GPU box: the production headers plus MUTANTS (each removes exactly one ordering step).
# Usage (repository root): tests/litmus/build.sh <outdir>      -> <outdir>/litmus_prod, litmus_<mutant>...
set -euo pipefail
OUT="${1:?outdir}"; HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(cd "$HERE/../.." && pwd)"
NVCC="$(command -v nvcc || ls /usr/local/cuda*/bin/nvcc | tail -1)"
mkdir -p "$OUT"
B="-O3 -arch=sm_80 -std=c++17 -lineinfo -Wno-deprecated-gpu-targets"
$NVCC $B -I"$ROOT/include" "$HERE/litmus.cu" -o "$OUT/litmus_prod" &
$NVCC $B -DLIT_NOFENCE -DLIT_MUTANT='"nofence(all __threadfence)"' -I"$ROOT/include" "$HERE/litmus.cu" -o "$OUT/litmus_nofence" &
# the executor's own mutant: a waiter that never re-checks the version after parking (classic lost wakeup)
$NVCC $B -DKVM_BREAK_PARK_RECHECK -DLIT_MUTANT='"norecheck(KVM_BREAK_PARK_RECHECK)"' -I"$ROOT/include" "$HERE/litmus.cu" -o "$OUT/litmus_norecheck" &
mut() {  # name, header file, sed-expression: a header copy with exactly one step removed
  local d="$OUT/hdr_$1"; rm -rf "$d"; mkdir -p "$d"; cp "$ROOT"/include/* "$d/"
  sed -E "$3" "$ROOT/include/$2" > "$d/$2"
  if cmp -s "$d/$2" "$ROOT/include/$2"; then echo "mutant $1: sed matched nothing" >&2; exit 1; fi
  $NVCC $B -DLIT_MUTANT="\"$1\"" -I"$d" "$HERE/litmus.cu" -o "$OUT/litmus_$1" &
}
# Dekker: drop the two fences that implement the argument (waiter: push->fence->load; publisher: store->fence->detach)
mut nodekkerfence kvm_exec.cuh 's#^(\s*)__threadfence\(\);  // seq_cst fence: push.*#\1/* mutant */#; s#^(\s*)__threadfence\(\);( +)// version store -> fence.*#\1/* mutant */#'
# Dekker: publisher detaches the list BEFORE storing the version (and not after)
mut exchbefore kvm_exec.cuh 's#^(\s*)int old = atom_cas\(S.version \+ var, ver - 1, ver\);#\1kvm_scan_parked(S, var); int old = atom_cas(S.version + var, ver - 1, ver);#; s#^  kvm_scan_parked\(S, var\);$#  /* mutant */#'
# MP: publisher fence removed and release CAS demoted to a relaxed atomicCAS
mut weakpub kvm_exec.cuh 's#^(\s*)__threadfence\(\);( +)// fence.sc.gpu: weights before version.*#\1/* mutant */#; s#int old = atom_cas\(S.version \+ var, ver - 1, ver\);#int old = atomicCAS(S.version + var, ver - 1, ver);#'
# Queue: popper claims the head with a plain store instead of CAS (double pop under contention)
mut popplain kvm_sched.cuh 's#if \(atom_cas64\(&q->head, cd.head, cd.head \+ 1\) == cd.head\) \{#if ((*(volatile unsigned long long*)\&q->head = cd.head + 1), true) {#'
# Queue: producer publishes the cell with a relaxed store (no release): payload may be stale
mut qrelax kvm_sched.cuh 's#st_rel64\(&cell->seq, pos \+ 1\);#*(volatile unsigned long long*)\&cell->seq = pos + 1;#'
wait
echo built in "$OUT"
