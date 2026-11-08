#!/usr/bin/env bash
# chunked release round 2: gradients vs float64 (kvm_run built from this tree) + task-coarsening sweep on the prebuilt ex_chunk_a (no executor rebuild).
set -u
export PATH=$PATH:/usr/local/cuda/bin
O=/logs/$TAG; mkdir -p $O /w
F="-O3 -arch=sm_80 -std=c++17 -Iinclude --fmad=false -Xcompiler -ffp-contract=off -lineinfo -diag-suppress 177"
nvcc $F -I. demo/kvm_run.cu -o /w/kvm_run > $O/build_kvm_run.log 2>&1 &
pip install -q torch numpy > $O/pip.log 2>&1 &
M='{"name":"gpt_162m","vocab":50304,"d_model":768,"layers":12,"heads":12,"seq":1024,"batch":4'
BIG="rows=256,ew_rows=512,cols=512,vocab=4096,opt_elems=1048576"
MID="rows=128,ew_rows=512,cols=256,vocab=2048,opt_elems=1048576"
g() { n=$1; p=$2; shift 2; python3 bench/vs_pytorch/gen.py /w/$n --model "$M,\"precision\":\"$p\"}" --steps 1 --mode tf32 --tiling a100 "$@" > $O/gen_$n.log 2>&1; echo "gen $n $(grep -o '"tasks": [0-9]*' $O/gen_$n.log)"; }
g b16 bf16s & g b16big bf16s --tile $BIG & g b16mid bf16s --tile $MID & g f32 f32 & g f32big f32 --tile $BIG & wait
for rep in 1 2 3; do for b in b16 b16big b16mid f32 f32big; do
  echo "rep $rep $b | $(timeout 900 /logs/bin/ex_chunk_a --quiet /w/$b/bundle 2>&1 | grep -E '^(OK|FAIL)|rror' | tail -1)" | tee -a $O/sweep.txt
done; done
wait
cp /w/kvm_run demo/kvm_run
for P in bf16s f32; do PRECISION=$P KVM_COMMIT=chunked-release bash bench/vs_pytorch/grads162.sh /w/num_$P $O/num_$P 2>&1 | tail -n 25; rm -rf /w/num_$P; done
echo done
