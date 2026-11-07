export PATH=$PATH:/usr/local/cuda/bin; O=/logs/combined-bisect; mkdir -p $O /w
F="-O3 -arch=sm_80 -std=c++17 -Iinclude --fmad=false -Xcompiler -ffp-contract=off -lineinfo -diag-suppress 177"
(nvcc $F kvm_exec.cu -o /w/e_comb > $O/build_comb.log 2>&1; echo "comb $?" >> $O/builds.txt) &
for t in main attn bf16; do (cd _$t && nvcc $F kvm_exec.cu -o /w/e_$t > $O/build_$t.log 2>&1; echo "$t $?" >> $O/builds.txt) & done
M='{"name":"gpt_162m","vocab":50304,"d_model":768,"layers":12,"heads":12,"seq":1024,"batch":4}'
(cd _main && python3 bench/vs_pytorch/gen.py /w/m_f32 --model "$M" --steps 1 --mode tf32 --tiling a100) > $O/gen.log 2>&1
wait; cat $O/builds.txt
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv,noheader > $O/env.txt; nvcc --version | tail -1 >> $O/env.txt
for r in 1 2 3 4 5 6 7; do for e in main attn bf16 comb; do
  o=$(timeout 900 /w/e_$e --quiet /w/m_f32/bundle 2>&1 | grep -E '^(OK|FAIL):' | tail -1); echo "rep $r $e $o" | tee -a $O/ab_runs.txt
done; done
echo finished >> $O/ab_runs.txt
