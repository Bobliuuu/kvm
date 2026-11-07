export PATH=$PATH:/usr/local/cuda/bin; O=/logs/combined-steps; mkdir -p $O /w
F="-O3 -arch=sm_80 -std=c++17 -Iinclude --fmad=false -Xcompiler -ffp-contract=off -lineinfo -diag-suppress 177"
(nvcc $F kvm_exec.cu -o /w/kvm_exec_comb > $O/build_comb.log 2>&1; echo "comb $?" >> $O/builds.txt) &
(cd _main && nvcc $F kvm_exec.cu -o /w/kvm_exec_main > $O/build_main.log 2>&1; echo "main $?" >> $O/builds.txt) &
M='{"name":"gpt_162m","vocab":50304,"d_model":768,"layers":12,"heads":12,"seq":1024,"batch":4'
g() { (cd $1 && python3 bench/vs_pytorch/gen.py /w/$2 --model "$M${3:+,\"precision\":\"$3\"}}" --steps 1 --mode tf32 --tiling a100) > $O/gen_$2.log 2>&1; cp /w/$2/info.json $O/info_$2.json; }
g . c_f32 f32 & g . c_b16 bf16s & g _main m_f32 "" & wait
cd /src
# trace stats: tasks/edges from the compiler's metadata
for b in c_f32 c_b16 m_f32; do python3 - $b <<PY >> $O/bundle_stats.txt
import json,sys
m=json.load(open('/w/%s/bundle/metadata.json'%sys.argv[1]))
print(sys.argv[1], {k:v for k,v in m.items() if not isinstance(v,(list,dict))})
PY
done
wait
cat $O/builds.txt
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv,noheader > $O/env.txt; nvcc --version | tail -1 >> $O/env.txt
for r in 1 2 3 4 5 6 7; do
  for v in main:kvm_exec_main:m_f32 comb_f32:kvm_exec_comb:c_f32 comb_bf16s:kvm_exec_comb:c_b16 combexec_on_main_bundle:kvm_exec_comb:m_f32; do
    IFS=: read n e b <<< "$v"
    o=$(timeout 900 /w/$e --quiet /w/$b/bundle 2>&1 | grep -E '^(OK|FAIL):' | tail -1)
    echo "rep $r $n $o" | tee -a $O/ab_runs.txt
  done
done
echo finished >> $O/ab_runs.txt
