#!/usr/bin/env bash
# gpt_162m step time (baseline tree vs this tree, f32 and bf16s) and step-1 numerics, in ONE Modal container (run via
#   modal run --detach tests/modal/run_suite.py --suite custom --tag NAME --cmd 'bash tests/modal/step162.sh'
# The baseline tree is expected at /src/basetree (a `git archive` of the reference commit, copied into the checkout before the upload).
# Output: /logs/$TAG/. Builds run two at a time (a kvm_exec ptxas link needs up to ~35 GB of the 96 GB).
set -u
export PATH=$PATH:/usr/local/cuda/bin
OUT=/logs/$TAG/step162; mkdir -p "$OUT" /work
FL="-O3 -arch=sm_80 -std=c++17 -Iinclude --fmad=false -Xcompiler -ffp-contract=off -lineinfo -diag-suppress 177"
B=/src/basetree
ls $B > /dev/null || { echo "no basetree"; exit 3; }
cp bench/vs_pytorch/gen.py bench/vs_pytorch/exec_bench.py bench/vs_pytorch/trace_summary.py $B/bench/vs_pytorch/ 2>/dev/null || { mkdir -p $B/bench/vs_pytorch; cp bench/vs_pytorch/{gen,exec_bench,trace_summary}.py $B/bench/vs_pytorch/; }
echo "== build exec (base, merged) $(date -u +%T)"
( cd $B && nvcc $FL kvm_exec.cu -o /work/exec_base > $OUT/build_base.log 2>&1; echo "base rc=$?" ) &
( nvcc $FL kvm_exec.cu -o /work/exec_new > $OUT/build_new.log 2>&1; echo "new rc=$?" ) &
pip install -q torch numpy > $OUT/pip.log 2>&1 &
wait
ls -la /work/exec_base /work/exec_new
echo "== bundles $(date -u +%T)"
gen() { # tree dir tiling precision (the baseline tree has no precision field: f32 only)
  P=""; [ "$4" = f32 ] && [ "$1" = "$B" ] || P="--precision $4"
  ( cd $1 && python3 bench/vs_pytorch/gen.py /work/$2 --model gpt_162m --steps 1 --mode tf32 --tiling $3 $P > $OUT/gen_$2.log 2>&1; echo "gen $2 rc=$?; tasks: $(python3 -c "import json;print(json.load(open('/work/$2/info.json'))['tasks'])" 2>/dev/null)" )
}
gen $B base_f32 a100 f32
gen /src new_f32 a100 f32
gen /src new_trunc a100_trunc f32
gen /src new_bf16s a100 bf16s
echo "== kvm_run build (merged) in background $(date -u +%T)"
( nvcc -O3 -arch=sm_80 -std=c++17 -I. -Iinclude --fmad=false -Xcompiler -ffp-contract=off -lineinfo -diag-suppress 177 demo/kvm_run.cu -o /work/kvm_run > $OUT/build_kvm_run.log 2>&1; echo "kvm_run rc=$?" ) &
KR=$!
echo "== interleaved step times (best of 5, 3 rounds) $(date -u +%T)"
pairs="base_f32:exec_base new_f32:exec_new new_trunc:exec_new new_bf16s:exec_new"
for round in 1 2 3; do
  for p in $pairs; do
    b=${p%%:*}; e=${p##*:}
    [ -d /work/$b/bundle ] || continue
    echo "-- round $round $b $e"
    python3 bench/vs_pytorch/exec_bench.py /work/$b /work/$e --reps 5 --out $OUT/${b}__${e}.r$round.json 2>&1 | tail -n 2
  done
done
echo "== traces (workers busy) $(date -u +%T)"
for p in $pairs; do
  b=${p%%:*}; e=${p##*:}
  [ -d /work/$b/bundle ] || continue
  /work/$e --quiet --trace-csv /work/$b.csv /work/$b/bundle 2>&1 | tail -n 1
  python3 bench/vs_pytorch/trace_summary.py /work/$b.csv 1 > $OUT/trace_$b.txt 2>&1; head -n 3 $OUT/trace_$b.txt
  rm -f /work/$b.csv
done
timeout 2400 tail --pid=$KR -f /dev/null; ls -la /work/kvm_run || exit 4
echo "== numerics vs float64 $(date -u +%T)"
cp /work/kvm_run demo/kvm_run
for cfg in "a100 f32" "a100_trunc f32" "a100 bf16s"; do
  set -- $cfg
  TILING=$1 PRECISION=$2 bash bench/vs_pytorch/grads162.sh /work/num_$1_$2 $OUT/num_$1_$2 2>&1 | tail -n 25
  rm -rf /work/num_$1_$2
done
echo "== done $(date -u +%T)"
