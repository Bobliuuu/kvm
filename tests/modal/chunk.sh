#!/usr/bin/env bash
# chunked-release A/B: this tree's kvm_exec (plain + KVM_ADMIT_PROF) vs the admission-v2 binary on the volume (/logs/bin/ex_new_a) and main (ex_main_a), interleaved, same container.
# Run: modal run --detach tests/modal/run_suite.py --suite custom --tag NAME --cmd 'bash tests/modal/chunk.sh'
set -u
export PATH=$PATH:/usr/local/cuda/bin
O=/logs/$TAG; mkdir -p $O /w /logs/bin
F="-O3 -arch=sm_80 -std=c++17 -Iinclude --fmad=false -Xcompiler -ffp-contract=off -lineinfo -diag-suppress 177"
nvcc $F kvm_exec.cu -o /w/ex_chunk > $O/build_chunk.log 2>&1 &
nvcc $F -DKVM_ADMIT_PROF kvm_exec.cu -o /w/ex_prof > $O/build_prof.log 2>&1 &
M='{"name":"gpt_162m","vocab":50304,"d_model":768,"layers":12,"heads":12,"seq":1024,"batch":4'
g() { python3 bench/vs_pytorch/gen.py /w/$1 --model "$M,\"precision\":\"$2\"}" --steps 1 --mode tf32 --tiling a100 > $O/gen_$1.log 2>&1; }
g f32 f32 & g b16 bf16s & wait
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv,noheader | tee $O/env.txt
cp /w/ex_chunk /logs/bin/ex_chunk_a; ls -la /w/ex_chunk /w/ex_prof /logs/bin
for rep in 1 2 3 4 5; do for b in b16 f32; do
  for v in ex_main_a ex_new_a; do echo "rep $rep $v $b | $(timeout 900 /logs/bin/$v --quiet /w/$b/bundle 2>&1 | grep -E '^(OK|FAIL)' | tail -1)" | tee -a $O/ab.txt; done
  echo "rep $rep chunk $b | $(timeout 900 /w/ex_chunk --quiet /w/$b/bundle 2>&1 | grep -E '^(OK|FAIL)|rror' | tail -1)" | tee -a $O/ab.txt
done; done
for b in b16 f32; do /w/ex_prof /w/$b/bundle 2>&1 | tail -n 12 | tee $O/prof_$b.txt; done
/w/ex_chunk --quiet --trace-csv /w/t.csv /w/b16/bundle | tail -1; python3 bench/vs_pytorch/trace_summary.py /w/t.csv 1 2>&1 | head -n 40 | tee $O/trace_b16.txt; gzip -c /w/t.csv > $O/trace_b16.csv.gz
echo done
