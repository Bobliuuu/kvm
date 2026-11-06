# KVM — a "kittens" GPU virtual machine (version 2 of [open-flapping-airplanes](https://github.com/Bobliuuu/open-flapping-airplanes))

Attempt to reproduce the idea from Flapping Airplanes' "kittens" VM: one persistent kernel takes over
the whole GPU and interprets work itself, so an *entire training run* (every step: forward,
loss, backward, optimizer) needs no kernel boundaries between ops or steps.

This tree has two layers:

1. `kvm.cu`: the original single-file prototype (80-byte instruction tape, MLP training, sm_80). Sections "How it works" to "Findings" below.
2. The integrated framework around a generic executor, `kvm_exec.cu` + `include/kvm_exec.cuh`: a model description goes in, a task DAG runs on
   the GPU, nothing in it is specific to one model. Map and status in the next two sections.

## The integrated framework

```
 frontend/kvmfe  model config -> autodiff -> tiles -> dataflow edges     FRONTEND.md
        |  workload JSON
 runtime/compile_workload.py   ABI v2 bundle (tasks, edges, args, versions) + CPU golden     FRAMEWORK.md
        |  bundle
 kvm_exec.cu / kvm_exec.cuh    persistent kernel: admission ring, slot recycling, ready rings, version waits
   + kvm_sched.*               per-workload scheduler (reservations, weights, EDF, inheritance, bandwidth arbiter), host controller kvm_adapt.h
   + operator registry packs   kvm_ops.def (core, ids 0-31), kvm_ops_gpt (32-63), kvm_ops_comm (64-95), kvm_ops_extra (96-121), kvm_ops_game (Doom)
   + dist/, runtime/partition.py   multi-GPU / multi-node code without NCCL                  DISTRIBUTED.md
```

| Part | Where |
|---|---|
| ABI v2 (64-byte task descriptor, typed args, weight versions) | `include/kvm_task_abi.h`, `runtime/abi.py`, FRAMEWORK.md |
| Operator registry packs (one `.def` per pack, sparse unique ids) | `include/kvm_ops*.def`, `.cuh` bodies, CPU references `runtime/ref_*.py` |
| Executor (dynamic admission, slot recycling, streaming, litmus-tested ordering) | `kvm_exec.cu`, `include/kvm_exec.cuh`, `results/2026-11-03-a100-exec`, `results/2026-11-04-a100-exec-recycling` |
| Scheduler and adaptive controller | `include/kvm_sched.*`, `include/kvm_adapt.h`, `adaptive/`, `results/2026-11-03-a100-sched`, `results/2026-11-04-a100-sched-integrated`, `results/2026-11-04-a100-adaptive` |
| Frontend and model zoo (GPT, Llama/GQA, MoE, MLP, ViT, bf16, fusion, remat) | `frontend/kvmfe`, FRONTEND.md, `results/2026-11-04-a100-models` |
| Real Doom (doomgeneric compiled for the device) | `doom/`, `results/2026-11-03-a100-doom` |
| Multi-GPU code | `dist/`, `runtime/partition.py`, DISTRIBUTED.md, `results/2026-11-03-dist-singlebox` |

### Status (A100-SXM4-40GB unless stated; each row cites the result that measured it)

| Status | What |
|---|---|
| **Implemented and measured** | `kvm.cu` MLP training: 1.50x over the best baseline at 1 run, 1.13x at 32 (table below; predates the loop opcode). |
| | Executor: 8582 checks, 0 failed, in two recorded runs (`2026-11-04-a100-sched-integrated`, `2026-11-04-a100-ops-perf`); 2.1M-task DAG through 65,536 slots; both negative builds detected; sanitizers clean; litmus 15 properties pass, 7 mutants detected. Dispatch 1.38 us/task at 108 blocks, dependent hop 7.0 us (`2026-11-04-a100-exec-recycling`). |
| | Scheduler: weighted shares, reservations, EDF, priority inheritance, overload isolation (train 8-11x better than a priority queue under a serve flood) (`2026-11-03-a100-sched`, `2026-11-04-a100-sched-integrated`). |
| | Real Doom: every frame equals the CPU oracle in all recorded runs (2000 standalone, 700 per executor scenario, 12/12 scheduler scenarios, 162 adaptive runs with up to 8 instances); 35 Hz held beside a saturating load under the dynamic scheduler (0 of 698 gaps over 28.57 ms). |
| | GPT/extra op packs: `test_ops` 1396 and `test_extra` 2089 checks in the recorded round-2 run, sanitizers clean (`2026-11-04-a100-ops-perf`); 2-layer GQA GPT, 200 AdamW steps vs a double CPU model, loss gap 2.0e-4 (`2026-11-03-a100-gptops`). |
| | Model zoo: 24 bundles (5 families, bf16, fusion, remat, accumulation, LR schedule) match the CPU golden within the stated per-mode tolerances; 24 sanitizer runs clean (`2026-11-04-a100-models`). GPT golden mismatch is 1xTF32 gradient noise through AdamW, not a bug (`2026-11-04-a100-gpt-golden`). |
| | Multi-GPU code on ONE GPU: 14 stages pass, 4 skipped (`2026-11-03-dist-singlebox`). |
| **Code only (no result measures the claim)** | Multi-GPU and multi-node execution (needs 2+ GPUs / hosts; the stages SKIP). RDMA transport is a stub. 162M-parameter GPT: compiled and inspected, never executed on the executor. The 1T-parameter section of DISTRIBUTED.md is arithmetic. Affinity hint (implemented, tested, no benefit measured, off by default). bf16 matmul tile speed (correct, speed not benchmarked). |
| **Not built** | Class-aware or sliced admission (the single admission FIFO is the tail-latency limit, below). Device-side task yield/preemption. Dynamic shapes, data-dependent control flow. Real bf16/fp16 storage (bf16 is emulated in fp32 buffers). Hopper/Blackwell paths (TMA, wgmma, tcgen05; the existing sm_80 code builds and passes unchanged on an H100 as sm_90a: 8586 exec checks, test_ops 1396, test_extra 2089, `results/2026-11-06-h100-sm90`). A rendered timeline figure from the executor's traces. A comparison of the executor against PyTorch eager / CUDA graphs / `torch.compile` on the same model (no result has one, so "faster than the conventional stack" is shown only for the `kvm.cu` MLP). |

### Known failures and honest negatives

| Item | State | Evidence |
|---|---|---|
| `recycle` long-running admission flake | 3M tasks through 4096 slots failed 2 of 3 runs on the adaptive tree (producers gave up after the 60 s window wait); the base commit shows the same flake; passes in 6.2-6.3 s otherwise. Not root-caused; scheduler off in that test. | `2026-11-04-a100-adaptive` failure list 1 |
| `test_ops` T=1 attention-backward tolerance | At T=1, dq and dk are exactly 0 mathematically and only rounding noise remains, so the bound is absolute (`tests/gpu_ops/test_ops.cu`, line 264). A failure of this case was reported against the integrated tree; no result README records it. Treat as open. | code comment only |
| Matmul at about 0.5x cuBLAS | Best 51.0 TFLOP/s (256x128 tile) against cuBLAS TF32 105.6 at 4096^3 (0.48x); 0.34-0.51x on other shapes. The 70% target was not reached. Attention is 15-30% below torch fp32 SDPA. | `2026-11-04-a100-ops-perf` |
| No multi-GPU hardware run | Nothing ran on more than one GPU, NVLink, or a network. | `2026-11-03-dist-singlebox` |
| Bandwidth arbiter | No latency win: 7% p99 in a synthetic chain; none with real Doom/serve tenants (they are not bandwidth-sensitive). It is a throttle that trades stream throughput for little latency. | `2026-11-03-a100-sched`, `2026-11-04-a100-adaptive` |
| Admission FIFO tail latency | The admitter ingests one record at a time at 1.9 us/task (0.52M tasks/s): a latency bundle waits for all K tasks ahead of it. A 47k-task training record puts every Doom frame behind 100-300 ms; no policy fixes it, host-side per-class budgets help at the price of shedding load (38% of serve requests dropped in the tuned run). | `2026-11-04-a100-adaptive` |
| Adaptive controller | Did not beat a hand-set fixed priority configuration (Doom frames late 470 vs 353 of 4030; serve p99 70.7 vs 60.8 ms). Its value is robustness (starvation floor, fast attack), not better numbers. | `2026-11-04-a100-adaptive` |
| Dynamic scheduler vs plain priority | Serve p99 1.7-1.8x worse than a global queue giving serve a higher priority, at equal throughput; adds 3-4 us per task dispatch. Wins only against static splits and under overload. | `2026-11-04-a100-sched-integrated` |
| Dispatch ceiling | About 0.7M tasks/s; the GPT-pack kernel uses 255 registers, so 1 block per SM (grid 108). Tasks under about 100 us per CTA are dispatch-bound. | `2026-11-04-a100-exec-recycling`, `2026-11-04-a100-ops-perf` |
| RTX 5070 Laptop (sm_120) | `kvm.cu` passes but never clearly beats the batched graph (0.73-1.12x). | `2026-11-03-rtx5070-laptop-sm120` |

Tests and how to run them: [RUNNING_TESTS.md](RUNNING_TESTS.md). Quick check without a GPU:
`python3 -I -m unittest discover -s tests && python3 -I -m unittest discover -s tests/frontend`.

## The `kvm.cu` prototype

`kvm.cu` is a self-contained prototype (CUDA, sm_80, one file). The numbers below predate its loop opcode: rerun before quoting.

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
