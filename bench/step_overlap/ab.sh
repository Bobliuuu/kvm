#!/bin/bash
# one GPU-lock hold: interleaved rounds over overlap levels, two tile configs
cd $WORK; mkdir -p out
nvidia-smi --query-gpu=name,driver_version,clocks.sm --format=csv > out/env.txt
nvcc --version | tail -2 >> out/env.txt
for round in 1 2 3; do
 for bin in kvm kvm32; do
  for L in 0 1 2; do
   ./$bin --steps 1000 --reps 15 --overlap $L --json out/${bin}_L${L}_r${round}.json > out/${bin}_L${L}_r${round}.log 2>&1
  done
 done
done
echo done > out/DONE
