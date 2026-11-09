#!/bin/bash
cd ~/stepov; mkdir -p tl
for L in 0 1 2; do
  for bin in kvm kvm32; do
    ./$bin --steps 200 --reps 3 --overlap $L --timeline tl/${bin}_L$L.csv --tl-steps 100 106 --json tl/${bin}_L$L.json > tl/${bin}_L$L.log 2>&1
  done
done
echo done > tl/DONE
