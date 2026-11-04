#!/usr/bin/env bash
# On the GPU box, from the repository root: build kvm_exec and run R's CLI fuzz corpus. Must end with 0 crashes and 0 hangs.
set -u
NV="$(command -v nvcc || ls /usr/local/cuda*/bin/nvcc | tail -1)"; mkdir -p out
$NV -O3 -arch="${KVM_ARCH:-sm_80}" -std=c++17 -Iinclude --fmad=false -lineinfo kvm_exec.cu -o out/kvm_exec 2>&1 | grep -i " error"
L="timeout 3000"; [ -n "${KVM_NO_LOCK:-}" ] || L="flock -w 7200 $HOME/.gpu.lock timeout 3000"
$L python3 tests/litmus/fuzz.py --exe out/kvm_exec --out out/fuzz_out --cases "${FC:-300}" --seed "${FS:-1}" ${FZ:-}
echo "exit: $?"
