# kvm ready-queue mode on an A100, 2026-11-03

`kvm.cu` at 88c26a4 (the commit adding the ready-queue mode). NVIDIA A100-SXM4-40GB (108 SMs, 1410 MHz max SM clock), driver
580.105.08 (CUDA 13.0), nvcc 12.8.93. The box is shared with other jobs, so every ratio below is measured inside one process
with the modes interleaved (see "Method").

```bash
nvcc -O3 -arch=sm_80 -std=c++17 -DKTILE=32 -DKKC=64 -DKSTG=4 kvm.cu -o kvm
./kvm --selftest
./kvm --runs R --steps 300 --reps 7        # MLP B=128 D=128 H=256 O=64; all five modes
./run_gpu_queue_matrix.sh                  # the whole set below, under flock + timeout
```

## Correctness

- `--selftest`: 16 of 16 cases ok (`selftest.txt`).
- Every benchmark run ends in `PASS` (`smoke`, `small`, `bench-r1/r2/r8/r32`, `shapes`). In each, the final weights of all five
  modes (sequential, sequential graph, batched graph, claim-and-wait VM, ready-queue VM) are **bitwise identical on every pass**
  (0 differing words across 6-8 passes each; the old `1e-3` tolerance is gone). Loss vs CPU fp32: at most 3.8e-5 relative.
- `compute-sanitizer` memcheck and racecheck (2 runs, 10 steps, claim and queue modes): 0 errors, 0 hazards.
- Watchdog self-test (`fault.txt`): with an unsatisfiable task injected, each VM ends the kernel after 300 ms with a diagnosis
  and the run reports `FAIL (watchdog)`, instead of hanging the GPU.

## Speed (min of the passes; ratio = median over passes of time(baseline)/time(mode), [min, max])

| runs | claim VM vs batched graph | ready queue vs batched graph | ready queue vs claim VM | claim VM / queue / batched, us per step per run |
|---|---|---|---|---|
| 1  | 1.11x [1.10, 1.11] | 0.86x [0.86, 0.87] | 0.78x [0.77, 0.79] | 25.8 / 32.9 / 28.7 |
| 2  | 1.15x [1.14, 1.16] | 0.88x [0.88, 0.90] | 0.77x [0.76, 0.78] | 13.0 / 17.0 / 14.9 |
| 8  | 1.51x [1.51, 1.51] | 1.42x [1.41, 1.43] | 0.94x [0.94, 0.95] | 4.29 / 4.53 / 6.46 |
| 32 (100 steps) | 1.03x [1.03, 1.03] | 0.97x [0.97, 0.98] | 0.94x [0.94, 0.94] | 3.53 / 3.74 / 3.65 |

`shapes.txt` (4 runs, 50 steps, B=256 D=256 H=512, 3 data batches) ran while another job was on the GPU for its first four passes:
claim 1.25x [1.24, 1.37], queue 1.19x [1.19, 1.50] vs batched; queue vs claim 0.96x [0.95, 1.10]. Not used above.

**The ready queue is slower than claim-and-wait everywhere: 0.77-0.78x on a dependent chain, 0.94-0.95x once there is enough
independent work (8+ runs).** It beats the batched graph at 8 runs (1.42x) but only because claim-and-wait does (1.51x).

## Why (costs from AGENTS.md, "Where a VM can win")

The tape's dependencies are phase barriers, and the lowered DAG encodes exactly the same barriers the counters do, so the queue
has no extra parallelism to find (the claim VM already hands tasks out in tape order, which is phase-major). It only pays for the
same hop differently. The dependency hop through global memory is about 5 us here: 1 run costs 25.8 us per step over 5 dependent
phases, about 5.2 us per hop. The queue's hop is about 6.6 us: the last predecessor must decrement, fence, reserve a slot,
write the task id, fence again, set the ready flag, and the consumer then reads the flag and, dependently, the task id, where
the claim VM polls one counter. With 1 run the chain is the whole step, so the 1.4 us extra per hop shows in full (0.78x); with
8 or more runs other blocks hide it, leaving the cost of the ~29 atomics per task (0.94x).

Two bugs found on the way (first A100 run of the earlier patch, 247 ms against 10.3 ms for the claim VM at 8 runs, 0.04x):
every idle block polled the same head, tail and flag lines and raced a CAS on the head, and one thread did about 29 dependent
atomic round trips per finished task. Fixed by taking a ticket with one `atomicAdd` and spinning on the ticket's own slot, and
by releasing successors with all 128 threads of the block. The ready-scan mode (scan the whole tape per claim) was removed.

## Method and limits

`--reps 7`: one warm-up pass, then 7 passes, each running all five modes back to back from a reset state, so a ratio compares
neighbours in time. The spread is across passes of one process; absolute times still move with other jobs on the box (the first
two passes of `bench-r2` were slower for that reason, and ratios barely changed).
Not shown: other GPUs, other model shapes beyond `shapes.txt`, or the claim VM with a controller warp (see AGENTS.md).
Raw output: `*.txt` here (`[pass N]` lines are per-pass times and weight comparisons).
