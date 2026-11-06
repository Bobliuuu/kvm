#!/usr/bin/env bash
# Round 2: autosized windows (kvm_exec main) as base; variants: run-next continuation, poll back-off. Interleaved A/B, md5 of buffers, test_exec for rn.
cd /src; D=results/2026-11-06-admission; . $D/jobs/common.sh; mkdir -p /tmp/w
genbundles
build_cached base
build_cached rn -DKVM_RUN_NEXT
build_cached p64 -DKVM_BACKOFF_CAP=64
build_cached rnp64 -DKVM_RUN_NEXT -DKVM_BACKOFF_CAP=64
( KVM_ARCH=sm_80 bash tests/gpu/build.sh -DKVM_RUN_NEXT 2>&1 | grep -i " error"; mv test_exec /tmp/te_rn; echo built te_rn ) &
( cp -r /src /tmp/src2 && cd /tmp/src2 && KVM_ARCH=sm_80 bash tests/gpu/build.sh -DKVM_RUN_NEXT -DKVM_BACKOFF_CAP=64 2>&1 | grep -i " error"; mv test_exec /tmp/te_rnp64; echo built te_rnp64 ) &
wait
nvidia-smi --query-gpu=name,clocks.max.sm --format=csv,noheader | tee $OUT/gpu.txt
names="base rn p64 rnp64"
for c in mlp xfmr; do
  for r in 1 2 3 4 5; do for n in $names; do
    echo "$c $n rep $r $(/tmp/ke_$n /tmp/w/t_$c/bundle 2>&1 | grep -E 'kernel done|FAIL|ERROR' | head -1)"; done; done
done | tee $OUT/timing.txt
for c in mlp xfmr; do for n in $names; do /tmp/ke_$n --quiet --dump-buffers /tmp/w/t_$c/bundle 2>&1 | md5sum | sed "s/^/$c $n dump md5 /"; done; done | tee $OUT/dump_md5.txt
for n in base rn rnp64; do for c in mlp xfmr; do
  /tmp/ke_$n --trace-csv /tmp/w/${c}_$n.csv /tmp/w/t_$c/bundle >/dev/null 2>&1; echo "== $c $n"; python3 -I $D/hopan.py /tmp/w/t_$c/bundle /tmp/w/${c}_$n.csv | tee $OUT/hopan_${c}_$n.txt; done; done
for n in rn rnp64; do (timeout 2400 /tmp/te_$n --quiet > $OUT/test_exec_$n.txt 2>&1; echo "exit $?" >> $OUT/test_exec_$n.txt); tail -n 4 $OUT/test_exec_$n.txt; done
