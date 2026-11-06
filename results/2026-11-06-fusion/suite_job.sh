#!/usr/bin/env bash
# Runs ON the GPU box from the repo copy root: builds test_extra (op pack, incl. wrappers with the 128x128 tile) and test_exec (executor
# functional suites) outside the lock, then runs both in ONE short lock hold.   nohup bash suite_job.sh OUTDIR > OUTDIR/job.log 2>&1 &
set -uo pipefail
O="$1"; mkdir -p "$O"; R="$(pwd)"; NVCC=$(command -v nvcc || ls /usr/local/cuda*/bin/nvcc | tail -1)
( cd tests/gpu_ops && $NVCC -O2 -arch=sm_80 -std=c++17 --fmad=false -Xcompiler -ffp-contract=off -lineinfo -I../../include -I. test_extra.cu -o "$O/test_extra" > "$O/build_extra.log" 2>&1 ) &
( KVM_ARCH=sm_80 bash tests/gpu/build.sh > "$O/build_exec.log" 2>&1 ) &
wait
flock -w 7200 ~/.gpu.lock bash -c "cd $R/tests/gpu_ops && timeout 900 $O/test_extra > $O/extra.txt 2>&1; echo exit \$? >> $O/extra.txt; cd $R && timeout 1500 ./test_exec > $O/exec.txt 2>&1; echo exit \$? >> $O/exec.txt"
echo done > "$O/job.done"
