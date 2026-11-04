# N independent real-Doom engines in one process (A100-SXM4-40GB, 108 SMs, driver 580.105.08, nvcc 12.8)
Command: `KVM_HOST=... doom/tests/run_instances.sh 300 8` (build without the lock, then `hold.sh` in one lock hold). Raw: hold.txt, ptxas_n*.txt, build_and_launch.txt.
- Every instance's 300 frames are raw-byte, fingerprint and state identical to the CPU oracle for N = 1, 2, 4, 8, alone and beside a 300-chain matmul+spin load (all 12 scenarios, 0 mismatches). compute-sanitizer memcheck, N=2, 12 frames: 0 errors.
- kvm_exec_kernel: 94 registers, 4744 B static smem, 0 stack frame for N=1 and N=8 (unchanged); binary 9.6 MB (N=1) to 21.4 MB (N=8); each instance object is 1.31 MB. Engine build (8 objects) 155 s on the box.
- Boot tick (frame 0): 218 ms (N=1), 192 ms (N=8). Unpaced (ticks back to back), per-instance cadence does not degrade: tick gap p50 6.3-6.5 ms, p99 10.3-10.9 ms, 163-166 fps per instance at N=1..8, with or without the load (a tick needs one CTA, 8 instances use 8 of 105 workers).
- Not measured: per-instance device memory (engine globals + zone heap are in the 1.31 MB objects' .bss; not queried), N > 8 (static limit KVM_DOOM_INSTANCES <= 8).
- 35 Hz pacing is not enforced by the executor (frames run back to back), so these numbers are capacity, not paced behavior.
