# KVM — a "kittens" GPU virtual machine (version 2 of [open-flapping-airplanes](https://github.com/Bobliuuu/open-flapping-airplanes))

One persistent kernel takes over the whole GPU and interprets a task graph itself, so GPT training, live inference on the weights being
trained, and real Doom run at the same time on the same SMs with no kernel boundary between any of their ops. A frontend compiles a model
to a bundle, the kernel admits it and schedules it, and every task leaves a timestamp. This is an attempt to reproduce the capability behind
Flapping Airplanes' "kittens" VM ([their post](https://x.com/flappyairplanes/status/2052107357585649810)); not affiliated with them. They have
published no code or numbers, so each claim here needs a result file in `results/`.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/timeline-dark.png">
  <img alt="Executor timeline: 106 SM rows over 4.2 s, coloured by op class: training windows, serve requests and Doom frames interleaved in one kernel" src="assets/timeline-light.png">
</picture>

*Executor timeline of the tiny demo, from the kernel's own per-task timestamps (`results/2026-11-06-demo-tiny/trace.csv.gz`, all 26,565 tasks; script
`demo/viz/timeline.py`). The 20 vertical bursts are the 20 training windows; the sparse tail is Doom at 174 frames/s. Most workers are idle: this
demo is latency-bound, not throughput-bound (see "What is not shown").*

![Demo replay: Doom frames, loss curve, timeline filling in](assets/demo-tiny.gif)

*Replay of the same run, recorded data only ([MP4](assets/demo-tiny.mp4), `demo/viz/make_video.py`): Doom frames are the device's output, raw-identical
to the CPU engine; loss values are from `observations.jsonl`; playback is slowed (23x until 345 ms, 3x after) and the clock on screen says so.*

## What runs, what was measured, what was not

| Item | Result | Evidence |
|---|---|---|
| Training, serving and Doom in ONE kernel launch, from `MANIFEST.json` only (`gpt_tiny`, 200 AdamW steps in 20 windows, 5 requests, 700 frames) | all checks pass on A100-SXM4-40GB: 26,565/26,565 tasks in 4.23 s, 0 version violations over 184 version-carrying tasks, 20/20 token buffers equal the golden decode of the same weights version, 700/700 Doom frames identical to the CPU engine | `results/2026-11-06-demo-tiny` |
| Training correctness at tiny scale | loss 4.146 -> 0.0042; max relative deviation from the CPU golden 3.8e-6 (tolerance 2e-3) | same |
| Doom cadence beside the load | 174 frames/s against 35 needed, 0 of 698 gaps over 28.57 ms | same; `results/2026-11-03-a100-doom` |
| 162M GPT (`gpt_162m`) in the same demo | runs: 1.95M tasks for 6 steps, err 0, 22,935 version tasks exact, Doom still 700/700; **1.81 s/step = 2,260 tokens/s, 30x the 59.97 ms critical-path bound** | `results/2026-11-06-demo-large` |
| 162M GPT learns | **not shown**: over 40 steps the loss only drifts (10.94 -> 10.89 mean of first/last 10, +-0.3 step noise), there is no golden at this size | `results/2026-11-06-demo-large` |
| Why 162M is slow | task throughput, not compute: 318k tasks/step through one admitter (1.9 us/task, 0.52M tasks/s); 9 of 106 workers busy on average | `results/2026-11-06-demo-large`, `2026-11-04-a100-adaptive` |
| Matmul and attention speed | matmul about 0.5x cuBLAS TF32 (51.0 vs 105.6 TFLOP/s at 4096^3); attention 15-30% below torch fp32 SDPA | `results/2026-11-04-a100-ops-perf` |
| Executor vs PyTorch eager / CUDA graph / `torch.compile` on the same model | **PLACEHOLDER: being measured on `bench/exec-vs-torch`; no number yet. Nothing here claims the demo beats PyTorch.** | pending |
| Admission change (class-aware / sliced) | **PLACEHOLDER: being changed on `perf/admission-bottleneck`; the numbers on this page are the single-FIFO baseline.** | pending |
| Runs on more than one GPU, NVLink or a network | not done (no hardware); the timeline above therefore has no comm ops | `results/2026-11-03-dist-singlebox` |

Beyond the demo, the one place the VM beats a conventional baseline is the `kvm.cu` MLP prototype (1.50x at 1 run, 1.13x at 32; table at the bottom).
Known failures and negatives are listed in full under "Known failures and honest negatives" below.

## Run the demo (one command, needs an sm_80 GPU box over ssh)

```
export KVM_HOST=user@host KVM_KEY=~/.ssh/key KVM_REMOTE=my-private-scratch   # KVM_REMOTE: a scratch dir on the box (default kvm-demo-scratch)
demo/run_demo.sh tiny  --lock-wait 14400    # gpt_tiny, 200 steps: all checks pass (about 4 s of GPU), results/2026-11-06-demo-tiny
demo/run_demo.sh large --lock-wait 14400    # gpt_162m, 6 steps: runs, 1.8 s/step = 2,260 tokens/s, results/2026-11-06-demo-large
```

`demo/run_demo.sh` rsyncs, builds, takes the GPU lock, runs the one launch and verifies it (loss, version coherence, Doom bytes against the CPU engine,
35 Hz cadence). The first build on a busy box takes about 15 minutes (nvcc); later runs reuse `demo/kvm_run` while it is newer than its sources.
Raw output lands in `logs/<UTC>-demo-<tag>/` (git-ignored): `run.json`, `trace.csv`, `observations.jsonl`, `check.log`, `report.json`. Options: the header of `demo/run_demo.sh`.

### Reproduce the figures

```
python3 demo/viz/timeline.py                                  # assets/timeline-{light,dark}.png from results/2026-11-06-demo-tiny/trace.csv.gz
python3 demo/viz/make_video.py --frames logs/<run>/out/doom_candidate/frames.bin --gif assets/demo-tiny.gif    # assets/demo-tiny.mp4
python3 demo/viz/cost_figure.py                               # assets/vm-costs.svg
```

The figures need numpy, matplotlib and (video) ffmpeg. To redraw from a new run, pass `--trace`/`--run-json` (timeline) or `--trace`/`--obs` (video) of that run;
copy its `trace.csv` (gzip it) next to the other results. `frames.bin` is 45 MB and stays out of git; the MP4 is the committed artifact.

## How it fits together

```mermaid
flowchart LR
  F["frontend/kvmfe<br/>model config -> autodiff -> tiles -> dataflow edges"] -->|workload JSON| C["runtime/compile_workload.py<br/>ABI v2 bundle: tasks, edges, args, versions<br/>+ CPU golden"]
  C -->|bundle records| H["host runner<br/>demo/kvm_run.cu"]
  H -->|"admission ring (records)"| A
  subgraph K["ONE persistent kernel (grid = 1 admitter + 1 policy + 106 workers)"]
    A["admitter<br/>slots, edge counters"] -->|task ready| R["ready rings"]
    R --> W["workers (1 per SM)<br/>op registry: core, GPT, comm, extra, Doom"]
    W -->|"done: decrement successors,<br/>bump variable version"| A
    V["version waits<br/>exact / at-least"] -.-> W
  end
  W -->|"per-task timestamps"| T[("trace.csv")]
  T --> P["demo/viz/timeline.py"]
```

Admission is the single serialized stage; everything else is parallel across SMs. Source: `kvm_exec.cu`, `include/kvm_exec.cuh`, `include/kvm_sched.*`.

### Three workloads, one weight-version flow

```mermaid
sequenceDiagram
  participant TR as train (windows of 10 steps)
  participant WV as weights (versioned variable)
  participant SV as serve (request every 50 steps)
  participant DM as Doom (700 frames)
  TR->>WV: window 0 writes version v1
  SV->>WV: wait for exact version, read it
  SV-->>SV: decode tokens (checked against golden decode of that version)
  TR->>WV: window 1 writes v2 (next window reuses the compiled record, versions shifted)
  Note over TR,DM: all three share the 106 workers - Doom is a tick + 4 present + status task per frame
  DM->>DM: frames run in order (35 Hz cadence checked from the trace)
```

Serving does not stop training: the version tag on each read is recorded in the trace, and `check_demo.py` verifies that all 184
version-carrying tasks saw the version they asked for.

### Where a VM wins and loses

![Per-event costs: launch costs a VM removes, hop and admission costs it adds](assets/vm-costs.svg)

```mermaid
flowchart TD
  Q{"what dominates the step?"} -->|"launch gaps, last-wave tails, cold starts between small dependent ops"| W1["VM can win<br/>kvm.cu MLP: 1.50x at 1 run (25.3 vs 37.9 us)"]
  Q -->|"many independent runs / workloads to interleave"| W2["VM can win<br/>Doom + serve + train share one GPU, Doom held at 35 Hz"]
  Q -->|"big GEMMs, compute or bandwidth bound"| L1["VM has little to win<br/>matmul 0.5x cuBLAS in the registry; graphs lose nothing"]
  Q -->|"hundreds of thousands of tiny tasks"| L2["VM loses to its own admission and dependency path<br/>162M GPT: 5.5 us/task, 30x off the critical path"]
```

Every cost the VM adds (a dependent hop through global memory, one admitter, 255-register kernel with 1 block per SM) must be paid back by a cost it removes.
Evidence points both ways, including against us: open-flapping-airplanes' own per-op kernels beat its megakernel by 13-25% on a 56 us step, and Megatron + CUDA
graphs beat it on a 162M GPT (43.1 s vs 48.1 s, B200).

## What is not shown (read before quoting)

- The tiny demo is far from its limit: 106 workers are mostly idle, the 4.2 s span is Doom ticks (700 frames, one tick task of about 5.9 ms each, plus a 208 ms boot tick), and training plus serving finish in 345 ms.
- Nothing here says the VM is faster than PyTorch, a CUDA graph or `torch.compile` on this model: the comparison does not exist yet (placeholder above).
- 162M: it runs and stays finite, nothing more. It is task-throughput bound (30x off its critical path), and its loss has no independent reference.
- A run with more tasks than slots recycles slots, and the version and cadence checks then skip by design (the 40-step 162M run did; the 6-step one did not).

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
| **Implemented and measured** | Rendered executor timeline from the per-task trace (`demo/viz/timeline.py`, `results/2026-11-06-demo-tiny`); the full demo (training windows + serving on versioned weights + Doom in one launch), `results/2026-11-06-demo-tiny`. |
| | `kvm.cu` MLP training: 1.50x over the best baseline at 1 run, 1.13x at 32 (table below; predates the loop opcode). |
| | Executor: 8582 checks, 0 failed, in two recorded runs (`2026-11-04-a100-sched-integrated`, `2026-11-04-a100-ops-perf`); 2.1M-task DAG through 65,536 slots; both negative builds detected; sanitizers clean; litmus 15 properties pass, 7 mutants detected. Dispatch 1.38 us/task at 108 blocks, dependent hop 7.0 us (`2026-11-04-a100-exec-recycling`). |
| | Scheduler: weighted shares, reservations, EDF, priority inheritance, overload isolation (train 8-11x better than a priority queue under a serve flood) (`2026-11-03-a100-sched`, `2026-11-04-a100-sched-integrated`). |
| | Real Doom: every frame equals the CPU oracle in all recorded runs (2000 standalone, 700 per executor scenario, 12/12 scheduler scenarios, 162 adaptive runs with up to 8 instances); 35 Hz held beside a saturating load under the dynamic scheduler (0 of 698 gaps over 28.57 ms). |
| | GPT/extra op packs: `test_ops` 1396 and `test_extra` 2089 checks in the recorded round-2 run, sanitizers clean (`2026-11-04-a100-ops-perf`); 2-layer GQA GPT, 200 AdamW steps vs a double CPU model, loss gap 2.0e-4 (`2026-11-03-a100-gptops`). |
| | Model zoo: 24 bundles (5 families, bf16, fusion, remat, accumulation, LR schedule) match the CPU golden within the stated per-mode tolerances; 24 sanitizer runs clean (`2026-11-04-a100-models`). GPT golden mismatch is 1xTF32 gradient noise through AdamW, not a bug (`2026-11-04-a100-gpt-golden`). |
| | Multi-GPU code on ONE GPU: 14 stages pass, 4 skipped (`2026-11-03-dist-singlebox`). |
| **Code only (no result measures the claim)** | Multi-GPU and multi-node execution (needs 2+ GPUs / hosts; the stages SKIP). RDMA transport is a stub. 162M-parameter GPT correctness (it runs, `results/2026-11-06-demo-large`, but has no golden and its loss only drifts). The 1T-parameter section of DISTRIBUTED.md is arithmetic. Affinity hint (implemented, tested, no benefit measured, off by default). bf16 matmul tile speed (correct, speed not benchmarked). |
| **Not built** | Class-aware or sliced admission (the single admission FIFO is the tail-latency limit, below). Device-side task yield/preemption. Dynamic shapes, data-dependent control flow. Real bf16/fp16 storage (bf16 is emulated in fp32 buffers). Hopper/Blackwell paths (TMA, wgmma, tcgen05). A comparison of the executor against PyTorch eager / CUDA graphs / `torch.compile` on the same model (no result has one, so "faster than the conventional stack" is shown only for the `kvm.cu` MLP). |

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
