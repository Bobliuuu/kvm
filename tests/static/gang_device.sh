#!/usr/bin/env bash
# The gang barriers' device test (test_gang_device.cu) built with every arrival-fence form and with the mutants it must catch, then run
# and judged. A mutant is a copy of include/kvm_gang.cuh with exactly one step removed (the litmus suite's convention): a test that cannot
# fail proves nothing.
#   prod_f0, prod_f1, prod_f2   the header as it ships, KVM_GANG_FENCE = 0, 1, 2: must pass (no wrong value, clean control words, the
#                               abort path ends);
#   nowait                      both barriers' polls removed, so a member never waits: must read wrong values, from L2 and through L1;
#   relaxedpoll_f0, _f1, _f2    the waiters' acquire loads made relaxed, so no L1 invalidate follows a poll. Form 1's arrival invalidates
#                               nothing either, so its reads through L1 must see the round before. Forms 0 and 2 keep an invalidate at
#                               the arrival (sm_80: the CCTL.IVALL of MEMBAR.SC's fence and of fence.acq_rel), so theirs are reported,
#                               not judged: the PTX model does not promise what that hardware step gives.
# Usage (repository root): [KVM_ARCH=sm_80] [NVCC=nvcc] [NVCC_FLAGS=...] tests/static/gang_device.sh build|run|all OUTDIR [ROUNDS=2000]
# `run` exits 0 when every judged binary behaved as it must, 1 otherwise; every run's output is OUTDIR/<name>.log.
set -euo pipefail
STEP="${1:?build, run or all}"; OUT="${2:?outdir}"; ROUNDS="${3:-2000}"
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(cd "$HERE/../.." && pwd)"
ARCH="${KVM_ARCH:-sm_80}"
read -r -a EXTRA <<< "${NVCC_FLAGS:-}"
NAMES=(prod_f0 prod_f1 prod_f2 nowait relaxedpoll_f0 relaxedpoll_f1 relaxedpoll_f2)
mkdir -p "$OUT"

find_nvcc() {  # $NVCC, else the one on PATH, else the newest /usr/local/cuda*/bin/nvcc; only `build` needs it
  local c found="${NVCC:-}"
  [ -n "$found" ] || found="$(command -v nvcc || true)"
  if [ -z "$found" ]; then for c in /usr/local/cuda*/bin/nvcc; do [ -x "$c" ] && found="$c"; done; fi
  [ -n "$found" ] || { echo "no nvcc: set NVCC, or put nvcc on PATH" >&2; exit 2; }
  NVCC="$found"
}
compile() {  # name, include dir, fence form
  "$NVCC" -O3 -arch="$ARCH" -std=c++17 -lineinfo "${EXTRA[@]}" -DKVM_GANG_FENCE="$3" -I"$2" "$HERE/test_gang_device.cu" -o "$OUT/$1"
  echo "built $1 ($ARCH, KVM_GANG_FENCE=$3)"
}
mutant() {  # name, sed expression, lines it must change: prints the include dir that holds the mutated header
  local d="$OUT/hdr_$1"
  rm -rf -- "${d:?}"; mkdir -p "$d"; cp "$ROOT"/include/* "$d/"
  sed -E "$2" "$ROOT/include/kvm_gang.cuh" > "$d/kvm_gang.cuh"
  local n; n=$(diff "$ROOT/include/kvm_gang.cuh" "$d/kvm_gang.cuh" | grep -c '^>' || true)
  if [ "$n" != "$3" ]; then echo "mutant $1: the sed changed $n line(s), not $3: the header moved under it" >&2; exit 2; fi
  echo "$d"
}

build() {
  find_nvcc
  for f in 0 1 2; do compile "prod_f$f" "$ROOT/include" "$f"; done
  local d
  # nowait: the flag-line poll leaves at once, and the counted poll's loop never runs
  d=$(mutant nowait 's#if \(__all_sync\(0xffffffffu, \(int\)\(v - epoch\) >= 0\)\) break;#(void)v; break;  /* mutant: no wait */#; s#\(long long\)\(ld_acquire\(count\) - target\) < 0#false /* mutant: no wait */#' 2)
  compile nowait "$d" 0
  # relaxedpoll: both acquire loads (u32 and u64) become relaxed ones
  d=$(mutant relaxedpoll 's#ld\.acquire\.gpu\.global#ld.relaxed.gpu.global#' 2)
  for f in 0 1 2; do compile "relaxedpoll_f$f" "$d" "$f"; done
}

# name -> "rc wrong_from_L2 wrong_through_L1", from the run's log
run_one() {
  local log="$OUT/$1.log" rc=0
  timeout 900 "$OUT/$1" "$ROUNDS" > "$log" 2>&1 || rc=$?
  awk -v rc="$rc" '/ barrier, .*reads from L2/ { m = 0 } / barrier, .*reads through L1/ { m = 1 }
                   / wrong values read/ { for (i = 1; i < NF; ++i) if ($(i + 1) == "wrong") w[m] += $i }
                   END { printf "%d %d %d\n", rc, w[0], w[1] }' "$log"
}

run() {
  local bad=0 name rc l2 l1 verdict
  printf '%-15s %4s %14s %14s  %s\n' binary rc "wrong from L2" "wrong thru L1" verdict
  for name in "${NAMES[@]}"; do
    read -r rc l2 l1 < <(run_one "$name")
    case "$name" in
      prod_*)         [ "$rc" = 0 ] && verdict="ok: passes as it must" || verdict="BAD: the shipped form fails";;
      nowait)         [ "$rc" != 124 ] && [ "$l2" -gt 0 ] && [ "$l1" -gt 0 ] && verdict="ok: caught from L2 and through L1" \
                        || verdict="BAD: the test cannot see a barrier that does not wait";;
      relaxedpoll_f1) [ "$rc" != 124 ] && [ "$l1" -gt 0 ] && verdict="ok: caught through L1" \
                        || verdict="BAD: the L1 reads cannot see a missing invalidate";;
      *)              verdict="reported (rc $rc)";;
    esac
    [ "$rc" = 124 ] && verdict="BAD: hung (timeout)"
    case "$verdict" in BAD*) bad=1;; esac
    printf '%-15s %4s %14s %14s  %s\n' "$name" "$rc" "$l2" "$l1" "$verdict"
  done
  [ "$bad" = 0 ] && echo "gang_device: every judged binary behaved as it must" || echo "gang_device: FAIL"
  return "$bad"
}

case "$STEP" in
  build) build;;
  run) run;;
  all) build && run;;
  *) echo "usage: $0 build|run|all OUTDIR [ROUNDS]" >&2; exit 2;;
esac
