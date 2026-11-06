#!/usr/bin/env bash
# Window-size sweep on the baseline binary: record size (live_cap / ring_words) vs time.
cd /src; D=results/2026-11-06-admission; . $D/jobs/common.sh; mkdir -p /tmp/w
genbundles; build_cached base; wait
nvidia-smi --query-gpu=name,clocks.max.sm --format=csv,noheader | tee $OUT/gpu.txt
run() { c=$1; shift; echo "$c $* : $(/tmp/ke_base "$@" /tmp/w/t_$c/bundle 2>&1 | grep -E 'kernel done|FAIL|ERROR' | head -1)"; }
for rep in 1 2 3; do
  run mlp
  for lc in 2048 4096 8192 16384 32768; do rw=$((lc*64)); run mlp --live-cap $lc --ring-words $rw; done
  run xfmr
  for lc in 4096 8192 16384 32768 65536; do rw=$((lc*64)); run xfmr --live-cap $lc --ring-words $rw; done
done | tee $OUT/win.txt
