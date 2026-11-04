# GPU executor on an A100 (2026-11-03)

- GPU: NVIDIA A100-SXM4-40GB, 108 SMs, sm_80, driver 580.105.08 (CUDA 13.0), nvcc/runtime 12.8, SM clock max 1410 MHz.
- Commit: d84d618 (branch feat/executor), clean tree.
- Build: `nvcc -O3 -arch=sm_80 -std=c++17 -Iinclude --fmad=false -Xcompiler -ffp-contract=off -lineinfo -DKVM_EXEC_NO_MAIN tests/gpu/test_exec.cu`
  (`--fmad=false` makes the device arithmetic per-op IEEE so fp results are compared bitwise with the CPU reference; with FMA
  the fp ops would need a tolerance, which is NOT what was run). Negative run adds `-DKVM_BREAK_FENCE`.
- Commands: `KVM_HOST=... ./run_exec_tests.sh` / `--negative` / `--bench` / `--sanitize` (each under `flock ~/.gpu.lock`).
- Files: functional.txt, negative-break-fence.txt, bench.txt, sanitizer.txt (raw output of those runs).

## Results (all from the files here)

| Claim | Evidence |
|---|---|
| Every op matches the CPU reference | 48074 integer-valued elements bitwise equal; 48074 fp elements, max relative error 0 (bitwise under `--fmad=false`); matmul at 12 shapes incl. 1x1x1, 17x15x33, 2x64x129; spin(300us) ran 302 us |
| Device errors, not UB | 13 injected faults (bad op, arity, buffer index, bounds, priority, var, version gap, bad successor, slot/pool/live-cap exhaustion, unsatisfiable wait) each end the kernel with the right decoded code |
| Random DAGs | 267,600 tasks in 96 seed x grid runs (1, 2, 15 workers, auto grid) plus 6,000 tasks through 64-cell rings (93 wraps): exactly-once (per-task counter), every edge ordered by device timestamps, node values bitwise vs CPU; 1,570 duplicate edges removed by the builder |
| Priority / FIFO / aging | one worker, 340 queued tasks: 0 priority inversions, 0 FIFO violations. Chain of 400 priority-10 tasks vs one priority-0 task: without aging it ran last (400/400); with 2000 us/level aging it ran after 189, after waiting 20.1 ms |
| Dynamic admission | 300 bundles through a 512-word ring: 218 wraps, 258 backpressure waits, bitwise correct; 4 host threads x 60 bundles concurrently; close() drains; oversized record -> TOO_BIG; stalled live window -> BACKPRESSURE_TIMEOUT then device TIMEOUT |
| Versions | 180 runs (vector weights of 65,536 floats, and the predict->mse_grad->sgd graph), random submission times, random spin ops, 2/4/24 blocks: 3225 exact + 653 at_least readers, 0 violations |
| The checker can fail | `-DKVM_BREAK_FENCE` (publish before the op's stores): 1555 reader violations detected, suite FAIL, `--expect-fail` returns 0. Hand-made bad observations: 7/7 classified, injected trace faults 2/2 flagged |
| Sanitizers | compute-sanitizer memcheck, racecheck, synccheck on the `smoke` suite (DAG, versions, ring wrap, all op kinds, 2-4 blocks): 0 errors / 0 hazards. racecheck sees shared memory only; global-memory ordering rests on the PTX argument in `include/kvm_exec.cuh` |
| Compiler bundle | `examples/live_training_inference.json` compiled by runtime/compile_workload.py runs on the A100 and `kvm_exec --golden` matches golden.json: 10/10 buffer values bitwise, versions, observed versions (not in these logs; run with `kvm_exec --golden <bundle>`) |

## Dispatch cost (bench.txt; times from %globaltimer, resolution ~1 us)

| Case | blocks | us/task | tasks/s |
|---|---|---|---|
| independent noop, 65,536 tasks | 2 (1 worker) | 4.7 | 2.1e5 |
| | 32 | 1.40 | 7.2e5 |
| | 108 | 1.34 | 7.5e5 |
| | 540 (auto) | 1.53 | 6.5e5 |
| dependent chain of noops (admit -> pop -> run -> wake) | 2 / 32 | 5.9 / 5.6 | 1.7e5 / 1.8e5 |
| | 540 | 10.4 | 9.6e4 |

A dependent hop costs about 5.6 us (publish/decrement, enqueue, warp scan, head CAS, 4 barriers), the same order as the
original VM's ~5 us hop in global memory; throughput saturates near 0.75M tasks/s (one admitter block releases roots; all workers
contend on the same bucket heads). Idle polling by 540 blocks slows a chain down 2x: size the grid to the work (`--blocks`).
Not compared against kvm.cu's Instr VM in the same process; no claim about it.

## Limits
- Slot pool is bounded per launch (`max_tasks`, no recycling); admission ring and ready rings recycle.
- A bundle record must fit in the admission ring, and a bundle has at most `live_cap` tasks.
- Parked (version-waiting) tasks count toward `live_cap`; if the live window is full of waiters whose publisher is still in the ring, admission stalls (reported as TIMEOUT).
- Removing only the `__threadfence` (keeping barrier + release CAS) was not tried; the negative test hoists the publish.
