#!/usr/bin/env bash
# A/B: baseline vs variant(s). VARIANTS env: space list "name:-Dflags" ; default runnext.  Interleaved reps, bitwise buffer compare, hop analysis, then test_exec for each variant.
cd /src; D=results/2026-11-06-admission; W=/tmp/w; mkdir -p $W
VARIANTS="${VARIANTS:-rn:-DKVM_RUN_NEXT}"
python3 -I results/2026-11-04-claims-audit/newmodel/gen.py $W/t_mlp mlp_ce --steps 200 --no-golden > $OUT/gen_mlp.log 2>&1 &
python3 -I results/2026-11-04-claims-audit/newmodel/gen.py $W/t_xfmr xfmr_odd --steps 50 --no-golden > $OUT/gen_xfmr.log 2>&1 &
NV=$(command -v nvcc)
FL="-O3 -arch=sm_80 -std=c++17 -Iinclude --fmad=false -Xcompiler -ffp-contract=off -diag-suppress 177"
$NV $FL kvm_exec.cu -o /tmp/ke_base 2>&1 | grep -i error &
for v in $VARIANTS; do n=${v%%:*}; f=${v#*:}; $NV $FL $f kvm_exec.cu -o /tmp/ke_$n 2>&1 | grep -i error & done
wait
nvidia-smi --query-gpu=name,clocks.max.sm --format=csv,noheader | tee $OUT/gpu.txt
names="base $(for v in $VARIANTS; do echo -n "${v%%:*} "; done)"
for c in mlp xfmr; do
  for r in 1 2 3 4 5; do for n in $names; do
    echo "$c $n rep $r $(/tmp/ke_$n $W/t_$c/bundle 2>&1 | grep 'kernel done')"; done; done
done | tee $OUT/timing.txt
# bitwise-identical buffers (small step counts, same bundles): dump must match the baseline
for c in mlp xfmr; do for n in $names; do /tmp/ke_$n --quiet --dump-buffers $W/t_$c/bundle 2>&1 | md5sum | sed "s/^/$c $n dump md5 /"; done; done | tee $OUT/dump_md5.txt
for n in $names; do for c in mlp xfmr; do /tmp/ke_$n --trace-csv $W/${c}_$n.csv $W/t_$c/bundle >/dev/null 2>&1; python3 -I $D/hopan.py $W/t_$c/bundle $W/${c}_$n.csv > $OUT/hopan_${c}_$n.txt; echo "== $c $n"; cat $OUT/hopan_${c}_$n.txt; done; done
if [ -z "${NOTEST:-}" ]; then
  for v in $VARIANTS; do n=${v%%:*}; f=${v#*:}
    KVM_ARCH=sm_80 bash tests/gpu/build.sh $f 2>&1 | grep -i " error" ; mv test_exec /tmp/te_$n
    (timeout 3000 /tmp/te_$n --quiet > $OUT/test_exec_$n.txt 2>&1; echo "exit $?" >> $OUT/test_exec_$n.txt) &
  done; wait; tail -n 3 $OUT/test_exec_*.txt
fi
