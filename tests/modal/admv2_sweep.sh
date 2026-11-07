#!/usr/bin/env bash
# window-size sweep: do the host-side window drains (live_cap 16384 tasks / ring_words 65536 cut the bundle into ~65 windows,
# each submitted only after its producers' windows completed) cause the idle workers? Builds are saved to the volume (/logs/bin) for reuse.
set -u
export PATH=$PATH:/usr/local/cuda/bin
O=/logs/$TAG; mkdir -p $O /w /logs/bin
F="-O3 -arch=sm_80 -std=c++17 -Iinclude --fmad=false -Xcompiler -ffp-contract=off -lineinfo -diag-suppress 177 ${EXTRA:-}"
if [ ! -x /logs/bin/ex_new_${BINTAG:-a} ]; then
  nvcc $F kvm_exec.cu -o /logs/bin/ex_new_${BINTAG:-a} > $O/build_new.log 2>&1 &
  (cd _main && nvcc $F kvm_exec.cu -o /logs/bin/ex_main_${BINTAG:-a} > $O/build_main.log 2>&1) &
fi
M='{"name":"gpt_162m","vocab":50304,"d_model":768,"layers":12,"heads":12,"seq":1024,"batch":4'
g() { python3 bench/vs_pytorch/gen.py /w/$1 --model "$M,\"precision\":\"$2\"}" --steps 1 --mode tf32 --tiling a100 > $O/gen_$1.log 2>&1; echo "gen $1 $?"; }
g f32 f32 & g b16 bf16s & wait
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv,noheader > $O/env.txt; echo built; ls -la /logs/bin
N=/logs/bin/ex_new_${BINTAG:-a}; Mn=/logs/bin/ex_main_${BINTAG:-a}
run() { echo "$1 | $(timeout 900 $2 --quiet ${@:4} /w/$3/bundle 2>&1 | grep -E '^(OK|FAIL)|rror' | tail -1)" | tee -a $O/sweep.txt; }
for rep in 1 2 3; do
  run "new b16 default" $N b16
  run "new b16 live64k ring4M" $N b16 --live-cap 65536 --ring-words 4194304
  run "new b16 live64k ring16M" $N b16 --live-cap 65536 --ring-words 16777216
  run "main b16 default" $Mn b16
  run "main b16 live64k ring16M" $Mn b16 --live-cap 65536 --ring-words 16777216
  run "new f32 default" $N f32
  run "new f32 live256k ring32M maxtasks512k" $N f32 --live-cap 262144 --ring-words 33554432 --max-tasks 524288
  run "main f32 live256k ring32M maxtasks512k" $Mn f32 --live-cap 262144 --ring-words 33554432 --max-tasks 524288
done
$N --quiet --live-cap 65536 --ring-words 16777216 --trace-csv $O/trace_b16_big.csv /w/b16/bundle 2>&1 | tail -1
$N --quiet --trace-csv $O/trace_b16_def.csv /w/b16/bundle 2>&1 | tail -1
for t in big def; do python3 bench/vs_pytorch/trace_summary.py $O/trace_b16_$t.csv 1 2>&1 | head -n 3 | tee -a $O/sweep.txt; done
gzip -f $O/trace_b16_*.csv
