#!/usr/bin/env bash
# ON the box, repo copy root, kvm_exec already built: GPU-vs-CPU-golden check of 3-step bundles per variant (ONE lock hold) and the executor
# functional suite (test_exec built outside the lock).   nohup bash gate_job.sh OUTDIR "tiling:fuse ..." > OUTDIR/job.log 2>&1 &
set -uo pipefail
O="$1"; VARS="$2"; mkdir -p "$O"; NVCC=$(command -v nvcc || ls /usr/local/cuda*/bin/nvcc | tail -1)
KVM_ARCH=sm_80 bash tests/gpu/build.sh > $O/build_exec.log 2>&1
for v in $VARS; do t=${v%%:*}; f=${v##*:}
  for c in mlp_ce xfmr_odd; do GOLDEN=1 python3 -I results/2026-11-06-fusion/count.py $c 3 $t $f $O/${c}_${t}_$f | head -1; done
done
flock -w 7200 ~/.gpu.lock bash -c '
  for v in '"$VARS"'; do t=${v%%:*}; f=${v##*:}; for c in mlp_ce xfmr_odd; do d='"$O"'/${c}_${t}_$f
    timeout 300 ./kvm_exec --blocks 4 --quiet --dump-buffers $d/bundle > $d.dump 2> $d.err; echo exit $?; done; done
  timeout 1500 ./test_exec > '"$O"'/exec.txt 2>&1; echo exit $? >> '"$O"'/exec.txt'
for v in $VARS; do t=${v%%:*}; f=${v##*:}; for c in mlp_ce xfmr_odd; do d=$O/${c}_${t}_$f; echo "== $c $t fuse=$f"; python3 -I results/2026-11-04-claims-audit/newmodel/compare.py $d $d.dump; done; done > $O/compare.txt 2>&1
echo done > $O/job.done
