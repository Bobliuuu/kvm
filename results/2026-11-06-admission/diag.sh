#!/usr/bin/env bash
# One lock hold: baseline timing + traces. usage: diag.sh BIN OUTDIR   (bundles in ~/kvm-scratch-admit/t_mlp t_xfmr)
BIN=$1; O=$2; S=${S:-$HOME/kvm-scratch-admit}; mkdir -p $O
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader > $O/gpu.txt
for r in 1 2 3; do echo "mlp default rep $r"; $BIN $S/t_mlp/bundle 2>&1 | grep "kernel done"; done > $O/mlp_default.txt
for b in 4 8 16 32 64; do for r in 1 2; do echo "mlp blocks $b rep $r"; $BIN --blocks $b $S/t_mlp/bundle 2>&1 | grep "kernel done"; done; done > $O/mlp_blocks.txt
$BIN --trace-csv $O/mlp_trace.csv $S/t_mlp/bundle 2>&1 | grep "kernel done" > $O/mlp_trace.txt
for r in 1 2 3; do echo "xfmr default rep $r"; $BIN $S/t_xfmr/bundle 2>&1 | grep "kernel done"; done > $O/xfmr_default.txt
$BIN --trace-csv $O/xfmr_trace.csv $S/t_xfmr/bundle 2>&1 | grep "kernel done" > $O/xfmr_trace.txt
