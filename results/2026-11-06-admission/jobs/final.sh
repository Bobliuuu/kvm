#!/usr/bin/env bash
# Final validation: new default build (autosize + parallel post_task) vs the pre-change binary (cached, original window defaults).
cd /src; D=results/2026-11-06-admission; . $D/jobs/common.sh; mkdir -p /tmp/w
genbundles
build_cached new
build_cached newrn -DKVM_RUN_NEXT
( KVM_ARCH=sm_80 bash tests/gpu/build.sh 2>&1 | grep -i " error"; mv test_exec /tmp/te_new; echo built te_new ) &
wait
cp /out/bin/ke_b21157fc80c7-68b329 /tmp/ke_old
nvidia-smi --query-gpu=name,clocks.max.sm --format=csv,noheader | tee $OUT/gpu.txt
OLDARGS="--live-cap 16384 --ring-words 65536"
for c in mlp xfmr; do
  for r in 1 2 3 4 5; do
    echo "$c old rep $r $(/tmp/ke_old $OLDARGS /tmp/w/t_$c/bundle 2>&1 | grep -E 'kernel done|FAIL|ERROR' | head -1)"
    echo "$c new rep $r $(/tmp/ke_new /tmp/w/t_$c/bundle 2>&1 | grep -E 'kernel done|FAIL|ERROR' | head -1)"
    echo "$c newrn rep $r $(/tmp/ke_newrn /tmp/w/t_$c/bundle 2>&1 | grep -E 'kernel done|FAIL|ERROR' | head -1)"
  done
done | tee $OUT/timing.txt
# bitwise: all buffers identical to the old binary (drop the timing line)
for c in mlp xfmr; do
  /tmp/ke_old $OLDARGS --quiet --dump-buffers /tmp/w/t_$c/bundle 2>&1 | grep '^buffer' | md5sum | sed "s/^/$c old /"
  /tmp/ke_new --quiet --dump-buffers /tmp/w/t_$c/bundle 2>&1 | grep '^buffer' | md5sum | sed "s/^/$c new /"
  /tmp/ke_newrn --quiet --dump-buffers /tmp/w/t_$c/bundle 2>&1 | grep '^buffer' | md5sum | sed "s/^/$c newrn /"
done | tee $OUT/dump_md5.txt
for n in new; do for c in mlp xfmr; do /tmp/ke_$n --trace-csv /tmp/w/${c}_$n.csv /tmp/w/t_$c/bundle >/dev/null 2>&1; echo "== $c $n"; python3 -I $D/hopan.py /tmp/w/t_$c/bundle /tmp/w/${c}_$n.csv | tee $OUT/hopan_${c}_$n.txt; done; done
(timeout 2400 /tmp/te_new --quiet > $OUT/test_exec_new.txt 2>&1; echo "exit $?" >> $OUT/test_exec_new.txt); grep -E "FAIL|checks|PASS" $OUT/test_exec_new.txt | tail -n 40
(timeout 900 compute-sanitizer --tool racecheck --error-exitcode 9 /tmp/te_new smoke > $OUT/racecheck_smoke.txt 2>&1; echo "exit $?" >> $OUT/racecheck_smoke.txt); tail -n 5 $OUT/racecheck_smoke.txt
