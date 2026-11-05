#!/usr/bin/env bash
# One command for everything distributed: CPU stages, single-GPU stages, and (when the hardware exists) multi-GPU / multi-node.
# Self-checking, every stage bounded by a timeout, one PASS/FAIL/SKIP line per stage, exit status 1 if any stage FAILED.
#
#   dist/run_multigpu_tests.sh                run on THIS machine (needs nvcc + a GPU for the GPU stages)
#   dist/run_multigpu_tests.sh --dry-run      print every stage and command, execute nothing
#   dist/run_multigpu_tests.sh --cpu-only     stages that need no GPU (laptop)
#   KVM_HOST=user@gpu-box dist/run_multigpu_tests.sh --remote     rsync the sources to the box and run it there
#   KVM_NODES=nodeA,nodeB dist/run_multigpu_tests.sh              also run the multi-node TCP stage
#                      (nodes: [user@]host[:ranks]; ssh key login; same CUDA arch; the sources are rsynced to ~/kvm-dist-run)
# Env: KVM_ARCH (default: detected, else sm_80), KVM_GPU_LOCK (flock file, taken once for the whole run; default ~/.gpu.lock if it
#      exists), KVM_LOCK_WAIT (seconds to wait for it, default 3000), KVM_STAGE_TIMEOUT
#      (seconds per stage, default 600), KVM_ONLY="ops pingpong" (run just those stages by name substring).
# Stages that need more than one physical GPU print "SKIP: needs 2 GPUs" on a smaller machine and run automatically otherwise.
# Logs: logs/<UTC>-dist/<stage>.log (git-ignored).
set -o pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DRY=0; CPU_ONLY=0; REMOTE=0
for a in "$@"; do case "$a" in --dry-run) DRY=1;; --cpu-only) CPU_ONLY=1;; --remote) REMOTE=1;; -h|--help) sed -n '2,16p' "$0"; exit 0;; *) echo "unknown option $a" >&2; exit 2;; esac; done

if [ "$REMOTE" = 1 ]; then
  HOST="${KVM_HOST:?set KVM_HOST=user@host (an environment variable, never a committed file)}"
  SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new ${KVM_KEY:+-i $KVM_KEY}"
  RDIR='~/kvm-dist-run'
  fwd=(); [ "$DRY" = 1 ] && fwd+=(--dry-run); [ "$CPU_ONLY" = 1 ] && fwd+=(--cpu-only)
  COMMIT="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo '?')$(git -C "$ROOT" diff --quiet HEAD 2>/dev/null || echo '+dirty')"
  echo "[remote] rsync -> $HOST:$RDIR"
  [ "$DRY" = 1 ] || rsync -az --delete -e "$SSH" --include 'include/***' --include 'dist/***' --include 'runtime/***' --include 'examples/***' \
      --include 'tests/' --include 'tests/dist/***' --include 'tests/gpu/***' --include 'kvm_exec.cu' --exclude '*' "$ROOT/" "$HOST:$RDIR/"
  exec $SSH "$HOST" "cd $RDIR && KVM_LOCK_HELD='${KVM_LOCK_HELD:-}' KVM_COMMIT='$COMMIT' KVM_NODES='${KVM_NODES:-}' KVM_ARCH='${KVM_ARCH:-}' KVM_ONLY='${KVM_ONLY:-}' KVM_STAGE_TIMEOUT='${KVM_STAGE_TIMEOUT:-600}' bash dist/run_multigpu_tests.sh ${fwd[*]:-}"
fi

cd "$ROOT"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"; LOG="$ROOT/logs/$STAMP-dist"; B="$ROOT/logs/dist-build"
mkdir -p "$LOG" "$B"
TMO="${KVM_STAGE_TIMEOUT:-600}"
PY="python3 -I"
NGPU=0; if command -v nvidia-smi >/dev/null 2>&1; then NGPU="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU')"; fi
NVCC="$(command -v nvcc || ls /usr/local/cuda*/bin/nvcc 2>/dev/null | tail -1 || true)"; HAVE_NVCC=1; [ -z "$NVCC" ] && { HAVE_NVCC=0; NVCC=nvcc; }
ARCH="${KVM_ARCH:-}"
if [ -z "$ARCH" ] && [ "$NGPU" -gt 0 ]; then cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')"; [ -n "$cc" ] && ARCH="sm_$cc"; fi
ARCH="${ARCH:-sm_80}"
LOCKF="${KVM_GPU_LOCK:-$HOME/.gpu.lock}"; LOCKWAIT="${KVM_LOCK_WAIT:-3000}"     # a shared box serialises GPU users with flock
# One acquisition for the whole run: the stages (and the two-executor tests in particular, which need both kernels resident at
# once) must not be interleaved with other users' jobs, and queueing per stage can take far longer than the run itself.
if [ "$DRY" = 0 ] && [ "$CPU_ONLY" = 0 ] && [ "$NGPU" -gt 0 ] && [ -e "$LOCKF" ] && command -v flock >/dev/null 2>&1 && [ -z "${KVM_LOCK_HELD:-}" ]; then
  echo "== waiting for $LOCKF (up to ${LOCKWAIT}s); held for the whole run"
  exec flock -w "$LOCKWAIT" "$LOCKF" env KVM_LOCK_HELD=1 bash "$0" "$@"
fi
NVFLAGS="-O3 -arch=$ARCH -std=c++17 -I$ROOT/include --fmad=false -Xcompiler -ffp-contract=off -lineinfo -DKVM_EXEC_NO_MAIN"

echo "== KVM distributed test run $STAMP  commit ${KVM_COMMIT:-$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo '?')}  GPUs: $NGPU  arch: $ARCH  nvcc: $([ "$HAVE_NVCC" = 1 ] && echo "$NVCC" || echo none)  dry-run: $DRY"
declare -a NAMES RESULTS NOTES
FAILED=0

# stage NAME NEEDS CMD   NEEDS: cpu | gpu | gpu2 (>= 2 GPUs) | nodes (KVM_NODES with >= 2 nodes)
stage() {
  local name="$1" needs="$2" cmd="$3" reason=""
  if [ -n "${KVM_ONLY:-}" ]; then local hit=0; for k in $KVM_ONLY; do [[ "$name" == *"$k"* ]] && hit=1; done; [ "$hit" = 0 ] && return; fi
  case "$needs" in
    gpu)   [ "$CPU_ONLY" = 1 ] && reason="--cpu-only"; [ -z "$reason" ] && [ "$NGPU" -lt 1 ] && reason="no GPU"; [ -z "$reason" ] && [ "$HAVE_NVCC" = 0 ] && reason="no nvcc";;
    gpu2)  [ "$CPU_ONLY" = 1 ] && reason="--cpu-only"; [ -z "$reason" ] && [ "$NGPU" -lt 2 ] && reason="needs 2 GPUs (found $NGPU)"; [ -z "$reason" ] && [ "$HAVE_NVCC" = 0 ] && reason="no nvcc";;
    nodes) [ "$CPU_ONLY" = 1 ] && reason="--cpu-only"; [ -z "$reason" ] && [ "$(echo "${KVM_NODES:-}" | tr ',' '\n' | grep -c .)" -lt 2 ] && reason="needs KVM_NODES with 2+ nodes"; [ -z "$reason" ] && [ "$NGPU" -lt 1 ] && reason="no GPU on the launch node";;
  esac
  if [ "$DRY" = 1 ]; then
    echo "DRY  [$needs] $name${reason:+  (would SKIP: $reason)}"; echo "       \$ $cmd"
    NAMES+=("$name"); RESULTS+=("DRY"); NOTES+=("$reason"); return
  fi
  if [ -n "$reason" ]; then
    echo "SKIP $name: $reason"; NAMES+=("$name"); RESULTS+=("SKIP"); NOTES+=("$reason"); return
  fi
  echo "---- RUN  $name  (timeout ${TMO}s)"
  local t0=$SECONDS rc
  timeout "$TMO" bash -c "$cmd" > "$LOG/$name.log" 2>&1; rc=$?
  local dt=$((SECONDS - t0))
  if [ $rc -eq 0 ]; then echo "PASS $name (${dt}s)"; RESULTS+=("PASS"); NOTES+=("${dt}s")
  else echo "FAIL $name (exit $rc, ${dt}s)  log: $LOG/$name.log"; tail -15 "$LOG/$name.log" | sed 's/^/     | /'; RESULTS+=("FAIL"); NOTES+=("exit $rc"); FAILED=1; fi
  NAMES+=("$name")
}

TT="$B/test_transport"; TD="$B/test_dist"; TDN="$B/test_dist_neg"; TDB="$B/test_dist_nobp"; PLANS="$B/plans"

# ---- CPU stages (laptop-safe) -------------------------------------------------------------------------------------------------
stage cpu-partitioner-sim cpu "cd $ROOT && $PY -m unittest discover -s tests/dist -p 'test_partition.py' 2>&1 | tail -15; exit \${PIPESTATUS[0]}"
stage cpu-transport-proxy cpu "g++ -O2 -std=c++17 -pthread -I$ROOT/dist $ROOT/tests/dist/test_transport.cpp -o $TT && KVM_DIST_LOG=1 $TT"
stage cpu-rdma-shim cpu "g++ -O2 -std=c++17 -pthread -I$ROOT/dist $ROOT/tests/dist/test_rdma.cpp -o $B/test_rdma && $B/test_rdma"
stage cpu-xdc-protocol cpu "g++ -O2 -std=c++17 -pthread -I$ROOT/dist $ROOT/tests/dist/test_xdc.cpp -o $B/test_xdc && $B/test_xdc"
stage cpu-scaleout-sim cpu "cd $ROOT && $PY -m unittest discover -s tests/dist -p 'test_scaleout*.py' 2>&1 | tail -15; exit \${PIPESTATUS[0]}"
stage cpu-fault-recovery cpu "cd $ROOT && $PY -m unittest discover -s tests/dist -p 'test_fault*.py' 2>&1 | tail -15; exit \${PIPESTATUS[0]}"
stage cpu-launcher-dry-run cpu "cd $ROOT && $PY dist/launch.py --dry-run --nodes a@h1:2,h2 -- ./prog --rank {rank} --peers {peers} | grep -c '^DRY rank' | grep -qx 3"
stage cpu-multiprocess-tcp cpu "cd $ROOT && PORTS=\$($PY -c \"import socket;s=[socket.socket() for _ in range(3)];[x.bind(('127.0.0.1',0)) for x in s];print(','.join(str(x.getsockname()[1]) for x in s))\") && \
  $PY dist/launch.py --local 3 --timeout 120 -- $TT proc {rank} {world} \"\$PORTS,drop=0.1,dup=0.1,delay=0.2\""

# ---- single-GPU stages (everything here is real device code; remote memory = another allocation) ------------------------------
stage gpu-build gpu "cd $ROOT && $NVCC $NVFLAGS tests/dist/test_dist.cu -o $TD -lpthread && $NVCC $NVFLAGS -DKVM_DIST_BREAK_ORDER tests/dist/test_dist.cu -o $TDN -lpthread && $NVCC $NVFLAGS -DKVM_DIST_NO_BP_POLL tests/dist/test_dist.cu -o $TDB -lpthread && $NVCC $NVFLAGS tests/gpu/test_exec.cu -o $B/test_exec -lpthread"
stage gpu-executor-regression gpu "$B/test_exec smoke"
stage gpu-ops gpu "$TD ops"
stage gpu-pingpong-versions gpu "$TD pingpong"
stage gpu-errors-no-hang gpu "$TD errors && $TD backpressure"
stage gpu-negative-break-order gpu "$TDN --expect-fail=stale pingpong && $TDB --expect-fail=timeout backpressure"
# (no two-executor suite under the sanitizer: it serialises kernel launches, and two persistent kernels waiting on each other would deadlock by construction)
stage gpu-sanitizer gpu "export KVM_DIST_QUICK=1; compute-sanitizer --error-exitcode 9 --tool memcheck $TD ops && compute-sanitizer --error-exitcode 9 --tool racecheck $TD ops && compute-sanitizer --error-exitcode 9 --tool synccheck $TD ops"
stage gpu-plans-in-process gpu "cd $ROOT && $PY tests/dist/make_plans.py $PLANS --devices $((NGPU>1?NGPU:2)) && for p in tp_direct tp_ring tp_tree zero_mixed zero_net zero_bf16_net dp_fp_net pipe_net pipe_p2p; do \
  tol=0; [ \$p = dp_fp_net ] && tol=1e-5; echo \"== plan \$p\"; $TD plan $PLANS/\$p --tol \$tol || exit 1; done"
stage gpu-plans-tcp-faults gpu "cd $ROOT && for p in zero_net tp_ring_net pipe_net zero_mixed; do echo \"== plan \$p over TCP with drop/dup/delay\"; $TD plan $PLANS/\$p --net tcp --fault drop=0.05,dup=0.05,delay=0.1 || exit 1; done"
stage gpu-multiprocess-tcp gpu "cd $ROOT && for p in zero_net tp_ring_net pipe_net; do echo \"== plan \$p: one process per rank\"; PORTS=\$($PY -c \"import socket;s=[socket.socket() for _ in range(4)];[x.bind(('127.0.0.1',0)) for x in s];print(','.join('127.0.0.1:'+str(x.getsockname()[1]) for x in s))\"); \
  $PY dist/launch.py --local 4 --timeout 300 -- $TD plan $PLANS/\$p --rank {rank} --peers \"\$PORTS\" || exit 1; done"

# ---- needs >= 2 physical GPUs -----------------------------------------------------------------------------------------------------
stage multigpu-p2p-bandwidth gpu2 "$TD bw"
stage multigpu-p2p-pingpong gpu2 "$TD pingpong"
stage multigpu-plans-one-rank-per-gpu gpu2 "cd $ROOT && $PY tests/dist/make_plans.py $B/plans_gpus --devices $NGPU && for p in tp_gpus zero_gpus pipe_p2p tp_ring zero_mixed; do echo \"== plan \$p, ranks spread over $NGPU GPUs\"; $TD plan $B/plans_gpus/\$p || exit 1; done"

# ---- needs several hosts ------------------------------------------------------------------------------------------------------------
stage multinode-tcp-plans nodes "cd $ROOT && rm -rf $B/src && mkdir -p $B/src && cp -R include dist runtime examples tests kvm_exec.cu $B/src/ && WORLD=\$(echo \"\$KVM_NODES\" | tr ',' '\n' | awk -F: 'NF>1 && \$NF ~ /^[0-9]+\$/ {s+=\$NF; next} {s+=1} END {print s}') && \
  $PY tests/dist/make_plans.py $B/plans_nodes --devices \$WORLD >/dev/null && PLIST=\$($PY tests/dist/make_plans.py $B/plans_nodes --devices \$WORLD --print-proxy-only \$WORLD) && [ -n \"\$PLIST\" ] && \
  for p in \$PLIST; do echo \"== plan \$p across \$WORLD ranks on the KVM_NODES hosts\"; \
    $PY dist/launch.py --timeout 400 --workdir '~/kvm-dist-run' --rsync $B/src/ '~/kvm-dist-run/' --rsync $B/plans_nodes/ '~/kvm-dist-run/plans_nodes/' \
      --setup \"$NVCC $NVFLAGS tests/dist/test_dist.cu -o test_dist -lpthread\" -- ./test_dist plan plans_nodes/\$p --rank {rank} --peers {peers} || exit 1; done"

# ---- summary --------------------------------------------------------------------------------------------------------------------------
echo; echo "================ SUMMARY ($STAMP) ================"
np=0; nf=0; ns=0
for i in "${!NAMES[@]}"; do
  printf '%-6s %-36s %s\n' "${RESULTS[$i]}" "${NAMES[$i]}" "${NOTES[$i]}"
  case "${RESULTS[$i]}" in PASS) np=$((np+1));; FAIL) nf=$((nf+1));; SKIP) ns=$((ns+1));; esac
done
if [ "$DRY" = 1 ]; then echo "dry run: ${#NAMES[@]} stages listed"; exit 0; fi
echo "passed $np, failed $nf, skipped $ns   (GPUs: $NGPU; logs: $LOG)"
[ "$ns" -gt 0 ] && echo "skipped stages need hardware this machine lacks: see the SKIP lines above (e.g. 'needs 2 GPUs'); rerun this script where it exists"
[ "$FAILED" = 0 ] && echo "RESULT: PASS" || echo "RESULT: FAIL"
exit "$FAILED"
