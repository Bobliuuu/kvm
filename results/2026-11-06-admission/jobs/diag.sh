#!/usr/bin/env bash
# Baseline diagnosis on Modal: build, gen bundles, timing, traces, hop analysis. $OUT = result dir.
set -x
cd /src; D=results/2026-11-06-admission; W=/tmp/w
python3 -I results/2026-11-04-claims-audit/newmodel/gen.py $W/t_mlp mlp_ce --steps 200 --no-golden > $OUT/gen_mlp.log 2>&1
python3 -I results/2026-11-04-claims-audit/newmodel/gen.py $W/t_xfmr xfmr_odd --steps 50 --no-golden > $OUT/gen_xfmr.log 2>&1
nvidia-smi -q -d CLOCK | grep -A4 "Max Clocks" > $OUT/clocks.txt
bash $D/build.sh /src /tmp/kvm_base
S=$W bash $D/diag.sh /tmp/kvm_base $OUT
for c in mlp xfmr; do python3 -I $D/hopan.py $W/t_$c/bundle $OUT/${c}_trace.csv > $OUT/${c}_hopan.txt; cat $OUT/${c}_hopan.txt; done
rm -f $OUT/*_trace.csv.big; ls -la $OUT; cat $OUT/*default.txt $OUT/mlp_blocks.txt
