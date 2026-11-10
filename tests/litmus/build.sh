#!/usr/bin/env bash
# Build the litmus binaries on the GPU box: the production headers plus MUTANTS (each removes exactly one ordering step).
# Usage (repository root): [KVM_ARCH=sm_80] [KVM_JOBS=2] tests/litmus/build.sh <outdir>      -> <outdir>/litmus_prod, litmus_<mutant>...
# KVM_ARCH: target architecture (default sm_80). KVM_JOBS: concurrent nvcc processes (default 2; each can need several GB, 8 at once
# was OOM-killed on a loaded box). Every build failure is reported by name and the script exits non-zero after all builds finished.
set -euo pipefail
OUT="${1:?outdir}"; HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(cd "$HERE/../.." && pwd)"
NVCC="$(command -v nvcc || ls /usr/local/cuda*/bin/nvcc | tail -1)"
mkdir -p "$OUT"
ARCH="${KVM_ARCH:-sm_80}"; JOBS="${KVM_JOBS:-2}"
B="-O3 -arch=$ARCH -std=c++17 -lineinfo -Wno-deprecated-gpu-targets"
PIDS=(); NAMES=(); RC=()
nv() {  # name, nvcc args...: run in the background; build number i first waits for build i-JOBS, so at most JOBS run at once
  local name="$1" i=${#PIDS[@]}; shift
  if [ "$i" -ge "$JOBS" ]; then wait "${PIDS[$((i - JOBS))]}" && RC[$((i - JOBS))]=0 || RC[$((i - JOBS))]=1; fi
  "$NVCC" "$@" & PIDS+=($!); NAMES+=("$name"); RC+=(-)
}
mut() {  # name, header file, sed-expression: a header copy with exactly one step removed
  local d="$OUT/hdr_$1"; rm -rf "$d"; mkdir -p "$d"; cp "$ROOT"/include/* "$d/"
  sed -E "$3" "$ROOT/include/$2" > "$d/$2"
  if cmp -s "$d/$2" "$ROOT/include/$2"; then echo "mutant $1: sed matched nothing" >&2; exit 1; fi
  nv "$1" $B -DLIT_MUTANT="\"$1\"" -I"$d" "$HERE/litmus.cu" -o "$OUT/litmus_$1"
}
nv prod $B -I"$ROOT/include" "$HERE/litmus.cu" -o "$OUT/litmus_prod"
nv nofence $B -DLIT_NOFENCE -DLIT_MUTANT='"nofence(all __threadfence)"' -I"$ROOT/include" "$HERE/litmus.cu" -o "$OUT/litmus_nofence"
# the executor's own mutant: a waiter that never re-checks the version after parking (classic lost wakeup)
nv norecheck $B -DKVM_BREAK_PARK_RECHECK -DLIT_MUTANT='"norecheck(KVM_BREAK_PARK_RECHECK)"' -I"$ROOT/include" "$HERE/litmus.cu" -o "$OUT/litmus_norecheck"
# Dekker: drop the two fences that implement the argument (waiter: push->fence->load; publisher: store->fence->detach)
mut nodekkerfence kvm_exec.cuh 's#^(\s*)__threadfence\(\);  // seq_cst fence: push.*#\1/* mutant */#; s#^(\s*)__threadfence\(\);( +)// version store -> fence.*#\1/* mutant */#'
# Dekker: publisher detaches the list BEFORE storing the version (and not after)
mut exchbefore kvm_exec.cuh 's#^(\s*)int old = atom_cas\(S.version \+ var, ver - 1, ver\);#\1kvm_scan_parked(S, var); int old = atom_cas(S.version + var, ver - 1, ver);#; s#^  kvm_scan_parked\(S, var\);$#  /* mutant */#'
# MP: publisher fence removed and release CAS demoted to a relaxed atomicCAS
mut weakpub kvm_exec.cuh 's#^(\s*)__threadfence\(\);( +)// fence.sc.gpu: weights before version.*#\1/* mutant */#; s#int old = atom_cas\(S.version \+ var, ver - 1, ver\);#int old = atomicCAS(S.version + var, ver - 1, ver);#'
# Queue, ticket claim: the popper takes its head ticket with a plain read and store instead of the fetch-add (two poppers take one ticket: double pop)
mut popplain kvm_sched.cuh 's#const unsigned long long h = atom_add64\(\&q->head, 1ull\);#const unsigned long long h = *(volatile unsigned long long*)\&q->head; *(volatile unsigned long long*)\&q->head = h + 1;#'
# Queue, ticket claim: the popper reserves an item with a plain read-modify-write instead of the atomic fetch-add (two poppers reserve the same item: double pop)
mut popticket kvm_sched.cuh 's#\(long long\)atom_add64\(&q->avail, ~0ull\) > 0#(long long)((*(volatile unsigned long long*)\&q->avail)--) > 0#'
# Queue: producer publishes the cell with a relaxed store (no release): payload may be stale
mut qrelax kvm_sched.cuh 's#st_rel64\(&cell->seq, pos \+ 1\);#*(volatile unsigned long long*)\&cell->seq = pos + 1;#'
bad=0
for i in "${!PIDS[@]}"; do
  [ "${RC[$i]}" = - ] && { wait "${PIDS[$i]}" && RC[$i]=0 || RC[$i]=1; }
  [ "${RC[$i]}" = 0 ] || { echo "BUILD FAILED: litmus_${NAMES[$i]}" >&2; bad=1; }
done
[ "$bad" = 0 ] || exit 1
echo built in "$OUT" "($ARCH)"
