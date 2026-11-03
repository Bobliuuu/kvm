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

Knobs: `-DKTILE` (32|64), `-DKKC` (K chunk, 32|64), `-DKSTG` (cp.async stages). Dims must be multiples
of max(KTILE, KKC). `run_remote.sh` rsyncs to a GPU box (`KVM_HOST`), builds and runs under `flock`.

## Results (A100-SXM4-40GB, MLP B=128 D=128 H=256 O=64, KTILE=32 KKC=64 KSTG=4)

Every mode is checked: loss matches a CPU fp32 reference (~3e-5 rel), final weights are identical
across all four modes.

| runs | sequential launches | sequential graphs | batched graph (vmap-like) | **KVM** | KVM vs best baseline |
|---|---|---|---|---|---|
| 1  | 45.1 us/step | 38.0 | 37.9 | **25.3** | 1.50x |
| 8  | 37.0 us/step/run | 31.1 | 6.83 | **4.18** | 1.63x (8.9x vs sequential) |
| 32 | 36.9 us/step/run | 31.1 | 3.88 | **3.45** | 1.13x (10.7x vs sequential) |

## Findings / caveats

- A single small dense run is latency-bound: each dependent hop costs ~5 us (fence + counter + poll +
  first `cp.async` round trip + wmma chain), so the VM is only ~1.2-1.5x over graphs there.
- Fusing many runs is where the VM pays off; at 32 runs it becomes compute-bound (~3.5 us/run-step).
- The GPU box is shared, so absolute numbers move run to run (baselines varied by up to ~1.5x between
  builds); ratios within one run are the meaningful figure.
- Tile GEMM is far from peak (TF32 wmma, 4 warps, ~10 TFLOPs at the largest size tried). Tape is
  host-generated (88 MB at 32 runs x 300 steps); a loop opcode would shrink it.
- Not done: on-chip weight residency across steps, bf16 path, transformer ops, H100 features
  (TMA/wgmma). That is what a real kittens VM would add.
