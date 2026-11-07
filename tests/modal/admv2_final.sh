#!/usr/bin/env bash
# final: build kvm_exec (auto-sized windows) + kvm_run in parallel, step times vs main (binary reused from the volume), step-1 gradients vs float64.
set -u
export PATH=$PATH:/usr/local/cuda/bin
O=/logs/$TAG; mkdir -p $O /w
F="-O3 -arch=sm_80 -std=c++17 -Iinclude --fmad=false -Xcompiler -ffp-contract=off -lineinfo -diag-suppress 177"
nvcc $F kvm_exec.cu -o /w/ex_fin > $O/build_exec.log 2>&1 &
nvcc $F -I. demo/kvm_run.cu -o /w/kvm_run > $O/build_kvm_run.log 2>&1 &
pip install -q torch numpy > $O/pip.log 2>&1 &
M='{"name":"gpt_162m","vocab":50304,"d_model":768,"layers":12,"heads":12,"seq":1024,"batch":4'
g() { python3 bench/vs_pytorch/gen.py /w/$1 --model "$M,\"precision\":\"$2\"}" --steps 1 --mode tf32 --tiling a100 > $O/gen_$1.log 2>&1; }
g f32 f32 & g b16 bf16s & wait
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv,noheader | tee $O/env.txt; ls -la /w/ex_fin /w/kvm_run /logs/bin
MB=/logs/bin/ex_main_a
for rep in 1 2 3 4 5; do for b in b16 f32; do
  [ -x $MB ] && echo "rep $rep main $b | $(timeout 900 $MB --quiet /w/$b/bundle 2>&1 | grep -E '^(OK|FAIL)' | tail -1)" | tee -a $O/ab.txt
  echo "rep $rep final $b | $(timeout 900 /w/ex_fin --quiet /w/$b/bundle 2>&1 | grep -E '^(OK|FAIL)|rror' | tail -1)" | tee -a $O/ab.txt
done; done
/w/ex_fin --quiet --trace-csv /w/t.csv /w/b16/bundle | tail -1; python3 bench/vs_pytorch/trace_summary.py /w/t.csv 1 2>&1 | head -n 16 | tee $O/trace_b16_final.txt; gzip -c /w/t.csv > $O/trace_b16_final.csv.gz
cp /w/kvm_run demo/kvm_run
for P in bf16s f32; do PRECISION=$P KVM_COMMIT=$(cat COMMIT 2>/dev/null) bash bench/vs_pytorch/grads162.sh /w/num_$P $O/num_$P 2>&1 | tail -n 25; rm -rf /w/num_$P; done
echo done
