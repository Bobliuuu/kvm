#!/bin/bash
chk() { local arm=$1 name=$1-$2 wl=$3 steps=$4 gangs=$5; shift 5
  mkdir -p /root/work/gangcnt/lines/run-20261111T134732Z/work/$name; ln -sf /root/work/gangcnt/lines/run-20261111T134732Z/vm/vm-*.json /root/work/gangcnt/lines/run-20261111T134732Z/work/$name/
  /root/py/bin/python3 -I /root/work/tree/bench/fused_gpt/fgpt_bed.py /root/work/fgsbed/run-20261111T074845Z/bundles/$wl --bed /root/work/gangcnt/lines/run-20261111T134732Z/bed-$arm --steps $steps --gangs $gangs --work /root/work/gangcnt/lines/run-20261111T134732Z/work/$name "$@" > /root/work/gangcnt/lines/run-20261111T134732Z/check-$name.log 2>&1
  local rc=$?
  echo "CHECK $name: $([ $rc = 0 ] && echo ok || echo "FAIL (rc $rc)"); $(grep -cE ': ok$' /root/work/gangcnt/lines/run-20261111T134732Z/check-$name.log) ok, $(grep -cE ': FAIL' /root/work/gangcnt/lines/run-20261111T134732Z/check-$name.log) FAIL"
  grep -E ': FAIL|no row chunk|exited [0-9]|Traceback|error' /root/work/gangcnt/lines/run-20261111T134732Z/check-$name.log | head -3
}
for a in L0 L64; do
  chk $a tiny-p-k2-g16 w_tiny_p 2 16; chk $a small-p-k2-g32 w_small_p 2 32
  chk $a d128-p-k1-g64 w_d128_precise 1 64; chk $a d128-p-resync-g64 w_d128_precise 1 64 --resync 1
  chk $a ho128-p-k1-g64 w_ho128_precise 1 64; chk $a ho128-p-resync-g64 w_ho128_precise 1 64 --resync 1
done
for r in 1 2 3 4 5; do
  for wg in w_tiny:16 w_small:32 w_d128_tf32_k30:64 w_ho128_tf32_k30:64; do
    w=${wg%%:*}; g=${wg#*:}
    case $((r % 3)) in 1) order="L32 L0 L64";; 2) order="L0 L64 L32";; 0) order="L64 L32 L0";; esac
    for arm in $order; do
      wd=/root/work/gangcnt/lines/run-20261111T134732Z/time-$arm-$w; lg=/root/work/gangcnt/lines/run-20261111T134732Z/phase-$arm-$w-r$r.log
      /root/py/bin/python3 -I /root/work/tree/bench/fused_gpt/fgpt_bed.py /root/work/fgsbed/run-20261111T074845Z/bundles/$w --bed /root/work/gangcnt/lines/run-20261111T134732Z/bed-$arm --time-only --bed-args clock --steps 20 --gangs $g --work $wd > $lg 2>&1
      echo "$w round $r $arm $(grep -oE '[0-9.]+ us a step in all; the SMs ran at [0-9]+ MHz|error word [1-9][0-9]*|Traceback' $lg | head -1)"
      echo "HASH $w g$g r$r $arm $(cat $wd/out-G$g-clock/*.bin 2>/dev/null | sha256sum | cut -c1-16)"
    done
  done
done
echo AW_GPU_END
