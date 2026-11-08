#!/usr/bin/env bash
# round 3: successor chaining (--chain 0|1|2) and critical-path priorities (KVM_CRIT_PRIO=1 at bundle compile) vs the chunked-release binary
# (/logs/bin/ex_chunk_a), interleaved in one container; then gradients vs float64 and a 300-step loss curve vs torch on the new binary.
set -u
export PATH=$PATH:/usr/local/cuda/bin
O=/logs/$TAG; mkdir -p $O /w /logs/bin
F="-O3 -arch=sm_80 -std=c++17 -Iinclude --fmad=false -Xcompiler -ffp-contract=off -lineinfo -diag-suppress 177"
nvcc $F kvm_exec.cu -o /w/ex_chain > $O/build_chain.log 2>&1 &
nvcc $F -I. demo/kvm_run.cu -o /w/kvm_run > $O/build_kvm_run.log 2>&1 &
pip install -q torch numpy > $O/pip.log 2>&1 &
M='{"name":"gpt_162m","vocab":50304,"d_model":768,"layers":12,"heads":12,"seq":1024,"batch":4'
g() { n=$1; p=$2; c=$3; KVM_CRIT_PRIO=$c python3 bench/vs_pytorch/gen.py /w/$n --model "$M,\"precision\":\"$p\"}" --steps 1 --mode tf32 --tiling a100 > $O/gen_$n.log 2>&1; echo "gen $n $?"; }
g b16 bf16s 0 & g b16c bf16s 1 & g f32 f32 0 & g f32c f32 1 & wait
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv,noheader | tee $O/env.txt
cp /w/ex_chain /logs/bin/ex_chain_a; ls -la /w/ex_chain /w/kvm_run
r() { echo "rep $1 $2 $3 | $(timeout 900 $4 --quiet $5 /w/$3/bundle 2>&1 | grep -E '^(OK|FAIL)|rror' | tail -1)" | tee -a $O/ab.txt; }
for rep in 1 2 3 4 5; do for b in b16 f32; do
  r $rep chunked $b /logs/bin/ex_chunk_a ""
  for c in 0 1 2; do r $rep chain$c $b /w/ex_chain "--chain $c"; done
  r $rep crit_chunked ${b}c /logs/bin/ex_chunk_a ""
  for c in 0 1 2; do r $rep crit_chain$c ${b}c /w/ex_chain "--chain $c"; done
done; done
/w/ex_chain --quiet --chain 1 --trace-csv /w/t.csv /w/b16c/bundle | tail -1; python3 bench/vs_pytorch/trace_summary.py /w/t.csv 1 2>&1 | head -n 8 | tee $O/trace_b16c_chain1.txt
cp /w/kvm_run demo/kvm_run
for P in bf16s f32; do PRECISION=$P KVM_COMMIT=chain3 bash bench/vs_pytorch/grads162.sh /w/num_$P $O/num_$P 2>&1 | grep -E "loss vs|worst tensor|kvm_run exit|^OK|FAIL"; rm -rf /w/num_$P; done
# 300-step learning curve, gpt_tiny, executor (new binary, defaults) vs torch eager
python3 bench/vs_pytorch/gen.py /w/lc --model gpt_tiny --steps 300 --mode tf32 --tiling a100 > $O/gen_lc.log 2>&1
python3 bench/vs_pytorch/torch_bench.py /w/lc --reps 1 --precision tf32 --modes eager --out /w/lc_torch.json 2>&1 | tail -3
python3 bench/vs_pytorch/exec_losses.py /w/lc /w/ex_chain --torch /w/lc_torch.json --mode eager --out $O/losses.json 2>&1 | tail -12 | tee $O/losses.txt
echo done
