# KVM — a "kittens" GPU virtual machine (version 2 of [open-flapping-airplanes](https://github.com/Bobliuuu/open-flapping-airplanes))

Attempt to reproduce the idea from Flapping Airplanes' "kittens" VM: one persistent kernel takes over
the whole GPU and interprets an **instruction tape**, so an *entire training run* (every step: forward,
loss, backward, SGD) is a single kernel launch, with no kernel boundaries between ops or steps.

`kvm.cu` is a self-contained prototype (CUDA, sm_80, one file).

## How it works

- **Tape**: 80-byte instructions (`GEMM` tile with epilogue flags, `SGD`). Tile GEMM is TF32 `wmma`
  with a `cp.async` smem ring; epilogues fuse ReLU, ReLU-mask, MSE loss/gradient and in-place SGD.
- **Scheduling**: persistent grid (SMs x occupancy). Blocks claim the next tape entry with one
  `atomicAdd`; the tape is topologically ordered, so waiting never deadlocks.
- **Dependencies**: monotonic global counters (never reset), per-row-block where possible
  (`H1[rb]` -> `Y[rb]` -> `dH1[rb]`) so step s+1 can start before step s is fully finished.
  No grid barriers.
- **Multi-run**: `--runs R` interleaves R independent training runs (an LR sweep) in one tape; the VM
  fills the GPU with whichever run's tiles are ready.
- **Baselines** run the *same tape* the classic way: one launch per op-phase (sequential, one model at a
  time), the same launches in a CUDA graph per run, and "batched" (one launch per phase for all runs,
  vmap-like, captured in a graph).

## Build / run

```bash
nvcc -O3 -arch=sm_80 -std=c++17 -DKTILE=32 -DKKC=64 -DKSTG=4 kvm.cu -o kvm
./kvm --selftest                       # GEMM, all 4 transpose combos vs CPU
./kvm --runs 8 --steps 300             # 8 fused LR-sweep runs vs baselines
```

Options: `--reps N` (timed passes), `--modes seq,seqgraph,batched,claim,queue`, `--watchdog-ms` (every VM spin is bounded; a
stuck run ends in `FAIL (watchdog)` with the blocked instruction or queue state), `--quiet`, `--inject-fault` (watchdog test).
`./run_gpu_queue_matrix.sh` runs the whole validation matrix on a GPU box.

Knobs: `-DKTILE` (32|64), `-DKKC` (K chunk, 32|64), `-DKSTG` (cp.async stages). Dims must be multiples
of max(KTILE, KKC). `run_remote.sh` rsyncs to a GPU box (`KVM_HOST`), builds and runs under `flock`.

## Results (A100-SXM4-40GB, MLP B=128 D=128 H=256 O=64, KTILE=32 KKC=64 KSTG=4)

Measured at 88c26a4 on a shared A100 (108 SMs, driver 580.105.08, nvcc 12.8); the full set, with raw output, is in
[results/2026-11-03-a100-queue](results/2026-11-03-a100-queue/README.md). Five modes run the same tape; `--modes` picks a subset.
Every run ends in `PASS`: final weights bitwise identical across the modes on every pass, loss within 4e-5 of CPU fp32.
Ratios are the median of 5-7 interleaved passes in one process, [min, max] over passes.

| runs | claim VM vs batched graph | ready queue vs batched graph | ready queue vs claim VM |
|---|---|---|---|
| 1  | 1.11x [1.10, 1.11] | 0.86x [0.86, 0.87] | 0.78x [0.77, 0.79] |
| 2  | 1.15x [1.14, 1.16] | 0.88x [0.88, 0.90] | 0.77x [0.76, 0.78] |
| 8  | 1.51x [1.51, 1.51] | 1.42x [1.41, 1.43] | 0.94x [0.94, 0.95] |
| 32 (100 steps) | 1.03x [1.03, 1.03] | 0.97x [0.97, 0.98] | 0.94x [0.94, 0.94] |

The ready queue (`queue`: dependencies of the whole tape lowered to a DAG on the host, successors released on the device) is
slower than claim-and-wait at every size: its dependency hop is about 6.6 us against 5.2 us. The earlier README numbers
(1.50x, 1.63x, 1.13x, measured at acb3bf4 before the loop opcode) are superseded.

## Findings / caveats

- A single small dense run is latency-bound: each dependent hop costs ~5 us (fence + counter + poll +
  first `cp.async` round trip + wmma chain), so the VM is only 1.11x over the best graph there (25.8 against 28.7 us per step).
- Fusing many runs is where the VM pays off most (1.51x at 8 runs); at 32 runs it is compute-bound (3.5 us/run-step) and the
  batched graph is within 3%.
- The GPU box is shared, so absolute numbers move run to run (baselines varied by up to ~1.5x between
  builds); ratios within one run are the meaningful figure.
- Tile GEMM is far from peak (TF32 wmma, 4 warps, ~10 TFLOPs at the largest size tried). Tape is
  host-generated (88 MB at 32 runs x 300 steps); a loop opcode would shrink it.
- Not done: on-chip weight residency across steps, bf16 path, transformer ops, H100 features
  (TMA/wgmma). That is what a real kittens VM would add.
