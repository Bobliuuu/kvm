#!/bin/bash
cd $WORK/exec; mkdir -p out
KR=$KVM_RUN
echo "$(sha256sum $KR | cut -c1-16) kvm_run" > out/env.txt; nvidia-smi --query-gpu=name,driver_version --format=csv >> out/env.txt
for rep in 1 2 3 4 5; do
  for v in w40 w40bar w1; do
    $KR $v --out out/${v}_$rep --compile-cmd "python3 /nonexistent" --dump '^w\.' > out/${v}_$rep.log 2>&1
  done
done
echo done > out/DONE
