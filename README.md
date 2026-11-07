# 🐱⚡ KVM — a "kittens" GPU virtual machine 🖥️🔥

### 🧠 one kernel · 🎮 real Doom · 🤖 162M GPT training · 🔁 live inference · 🚀 whole GPU

*Version 2 of [open-flapping-airplanes](https://github.com/Bobliuuu/open-flapping-airplanes) 🛩️*

![GPU](https://img.shields.io/badge/GPU-A100%20(sm__80)-76b900) ![CUDA](https://img.shields.io/badge/CUDA-12.x-76b900) ![demo](https://img.shields.io/badge/demo-all%20checks%20pass-brightgreen) ![Doom](https://img.shields.io/badge/Doom-700%2F700%20frames%20identical-blue) ![vs PyTorch](https://img.shields.io/badge/vs%20PyTorch-15--18x%20slower%20(162M)-red) ![multi-GPU](https://img.shields.io/badge/multi--GPU-not%20run-lightgrey)

Legend used below: ✅ measured and on main, 🧪 code only (no result measures it), ❌ target not met, 🚧 in progress or on an unmerged branch.

## 🚀 Demo ready

One persistent kernel runs a **162M-parameter GPT training**, **live inference on the weights being trained** and **real Doom** at the same time, then verifies itself.

```
export KVM_HOST=user@host                      # an sm_80 GPU box reachable over ssh (optional: KVM_KEY, KVM_REMOTE=<scratch dir on the box>)
demo/run_demo.sh large --lock-wait 14400       # 162M GPT, 6 steps, serving, 700 Doom frames
demo/run_demo.sh tiny  --lock-wait 14400       # gpt_tiny, 200 steps: the quick one (~4 s of GPU)
```

| | |
|---|---|
| 👀 What you see | the executor timeline (hero below) and a replay GIF of Doom + loss + timeline, drawn from the run's own `trace.csv`; `check.log` with PASS/SKIP lines |
| ✅ Expected checks | executor: every task done, err 0; weight versions coherent (22,935 of 22,935 exact at 162M); Doom 700/700 frames identical to the CPU engine; 35 Hz cadence held (172.7 frames/s); training loss finite |
| ⏱ Runtime | about 14 s of kernel at 162M (4 s at tiny); about 1 minute of wall time once built. The very first build on a busy box takes about 15 min (nvcc); later runs reuse the binary |
| 🧰 Requirements | an sm_80 GPU (tested: A100-SXM4-40GB), nvcc 12.x, python3 with numpy; matplotlib and ffmpeg only for the figures |
| ⚠️ Honest limits | it is **15-18x slower than PyTorch** on the same 162M step (1042 ms vs 59-70 ms): task admission, not compute, is the bottleneck (workers are busy under a fifth of the time). The 162M loss only drifts in 6-40 steps (no learning claimed). Single GPU only |

One persistent kernel takes over the whole GPU and interprets a task graph itself, so GPT training, live inference on the weights being
trained, and real Doom run at the same time on the same SMs with no kernel boundary between any of their ops. A frontend compiles a model
to a bundle, the kernel admits it and schedules it, and every task leaves a timestamp. This is an attempt to reproduce the capability behind
Flapping Airplanes' "kittens" VM ([their post](https://x.com/flappyairplanes/status/2052107357585649810)); not affiliated with them. They have
published no code or numbers, so each claim here needs a result file in `results/`.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/timeline-162m-dark.png">
  <img alt="Executor timeline of the 162M GPT demo: 106 SM rows over 13.7 s, coloured by op class; Doom frames and six training steps with serve requests in one kernel" src="assets/timeline-162m-light.png">
</picture>

*Hero: the full demo with a **162M-parameter GPT** (6 AdamW steps), serving and 700 Doom frames in one kernel launch, from the kernel's own per-task timestamps
(`results/2026-11-06-demo-large-trace`, all 1,948,168 tasks drawn; script `demo/viz/timeline.py`). Each blue block is one training step; the white gaps
between blocks are idle workers: this run is admission/dependency bound, not compute bound, and it is 15-18x slower than PyTorch (table below).
Doom (yellow/red, left) holds 172.7 frames/s beside it.*

![Replay of the 162M demo: Doom frames, loss, timeline filling in](assets/demo-162m.gif)

*Replay of that run, recorded data only ([MP4](assets/demo-162m.mp4), `demo/viz/make_video.py`): Doom frames are the device's output, raw-identical to the CPU engine;
the six loss values are from `observations.jsonl` (they only drift: no learning is claimed); playback is 1.5x slower than real time.*

### The quick demo: `gpt_tiny` (4 s of GPU, every check passes)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/timeline-dark.png">
  <img alt="Executor timeline of the tiny demo: 106 SM rows, 20 training windows, serve requests and Doom frames interleaved in one kernel" src="assets/timeline-light.png">
</picture>

*Tiny demo (`results/2026-11-06-demo-tiny/trace.csv.gz`, 26,565 tasks). The 20 vertical bursts are the 20 training windows; the sparse tail is Doom at 174
frames/s; workers are mostly idle, so it is latency-bound. The axis is split at 345 ms because training and serving end there.*

![Tiny demo replay: Doom frames, loss curve, timeline filling in](assets/demo-tiny.gif)

*Replay of the tiny run ([MP4](assets/demo-tiny.mp4)): Doom frames are device output, loss from `observations.jsonl`; playback is slowed (23x until 345 ms, 3x after) and the clock on screen says so.*

## 📊 What runs, what was measured, what was not

| Item | Result (✅ measured, 🧪 code only, ❌ not met, 🚧 in progress) | Evidence |
|---|---|---|
| ✅ Training, serving and Doom in ONE kernel launch, from `MANIFEST.json` only (`gpt_tiny`, 200 AdamW steps in 20 windows, 5 requests, 700 frames) | all checks pass on A100-SXM4-40GB: 26,565/26,565 tasks in 4.23 s, 0 version violations over 184 version-carrying tasks, 20/20 token buffers equal the golden decode of the same weights version, 700/700 Doom frames identical to the CPU engine | `results/2026-11-06-demo-tiny` |
| ✅ Training correctness at tiny scale | loss 4.146 -> 0.0042; max relative deviation from the CPU golden 3.8e-6 (tolerance 2e-3) | same |
| ✅ Doom cadence beside the load | 174 frames/s against 35 needed, 0 of 698 gaps over 28.57 ms | same; `results/2026-11-03-a100-doom` |
| ✅ 162M GPT (`gpt_162m`) in the same demo | **passes** (re-run with full trace): 1,948,168/1,948,168 tasks in 13.66 s, err 0, 22,935 version tasks exact, Doom 700/700 identical, 172.7 frames/s, 0/698 gaps over 28.57 ms; 1.8 s/step = 2,260 tokens/s, 30x the 59.97 ms critical-path bound; serving has no golden decode at this size | `results/2026-11-06-demo-large-trace`, `results/2026-11-06-demo-large` |
| ✅ 162M numerics | step-1 loss and all 196 gradient tensors match float64 as well as PyTorch's own TF32 does (worst 1.44e-3 vs 1.46e-3 relative L2) | `results/2026-11-06-gpt162-vs-torch` |
| ❌ 162M GPT learns | **not shown**: six steps give 10.979 -> 10.924, and over 40 steps the loss only drifts (10.94 -> 10.89 mean of first/last 10, +-0.3 step noise), there is no golden at this size | `results/2026-11-06-demo-large` |
| ✅ Why 162M is slow | task throughput, not compute: 318k tasks/step through one admitter (1.9 us/task, 0.52M tasks/s); 9 of 106 workers busy on average (27% inside a step) | `results/2026-11-06-demo-large`, `2026-11-04-a100-adaptive`, `2026-11-06-gpt162-vs-torch` |
| ❌ Matmul and attention speed | matmul about 0.5x cuBLAS TF32 (51.0 vs 105.6 TFLOP/s at 4096^3); attention 15-30% below torch fp32 SDPA | `results/2026-11-04-a100-ops-perf` |
| ❌ Executor vs PyTorch on the same 162M model and AdamW | **15-18x slower**: executor 1042 ms/step (1-step tape) against torch eager 69.6, whole-step CUDA graph 68.2, `torch.compile` 59-60 ms. Why: admission is 604 ms of the 1042 (318k tasks/step at 1.9 us, single FIFO); the workers are only 27% busy; tile work alone would be about 190 ms/step at perfect admission (matmul tiles at 0.5x cuBLAS); the dependency-hop term is 0.5% of the step. At tiny/small sizes it beats eager 1.6-2.3x but never a CUDA graph (3.2-12.6x slower). The demo does not beat PyTorch. | `results/2026-11-06-gpt162-vs-torch`, `results/2026-11-06-vs-pytorch` |
| 🚧 Admission change (class-aware / sliced) | **PLACEHOLDER: being changed on `perf/admission-bottleneck`; the numbers on this page are the single-FIFO baseline.** | pending |
| 🧪 Runs on more than one GPU, NVLink or a network | not done (no hardware); the timeline above therefore has no comm ops | `results/2026-11-03-dist-singlebox` |

Beyond the demo, the one place the VM beats a conventional baseline is the `kvm.cu` MLP prototype (1.50x at 1 run, 1.13x at 32; table at the bottom).
Known failures and negatives are listed in full under "Known failures and honest negatives" below.

## ▶️ Run the demo in detail (needs an sm_80 GPU box over ssh)

```
export KVM_HOST=user@host KVM_KEY=~/.ssh/key KVM_REMOTE=my-private-scratch   # KVM_REMOTE: a scratch dir on the box (default kvm-demo-scratch)
demo/run_demo.sh tiny  --lock-wait 14400    # gpt_tiny, 200 steps: all checks pass (about 4 s of GPU), results/2026-11-06-demo-tiny
demo/run_demo.sh large --lock-wait 14400    # gpt_162m, 6 steps: all checks pass, 1.8 s/step = 2,260 tokens/s, results/2026-11-06-demo-large-trace
```

`demo/run_demo.sh` rsyncs, builds, takes the GPU lock, runs the one launch and verifies it (loss, version coherence, Doom bytes against the CPU engine,
35 Hz cadence). The first build on a busy box takes about 15 minutes (nvcc); later runs reuse `demo/kvm_run` while it is newer than its sources.
Raw output lands in `logs/<UTC>-demo-<tag>/` (git-ignored): `run.json`, `trace.csv`, `observations.jsonl`, `check.log`, `report.json`. Options: the header of `demo/run_demo.sh`.

### 🔁 Reproduce the figures

```
python3 demo/viz/timeline.py                                  # assets/timeline-{light,dark}.png from results/2026-11-06-demo-tiny/trace.csv.gz
python3 demo/viz/make_video.py --frames logs/<run>/out/doom_candidate/frames.bin --gif assets/demo-tiny.gif    # assets/demo-tiny.mp4
python3 demo/viz/compact_trace.py logs/<run>/out/trace.csv OUT.npz   # 229 MB CSV -> 6.9 MB npz the 162M figure reads
python3 demo/viz/cost_figure.py                               # assets/vm-costs.svg
```

The figures need numpy, matplotlib and (video) ffmpeg. To redraw from a new run, pass `--trace`/`--run-json` (timeline) or `--trace`/`--obs` (video) of that run;
copy its `trace.csv` (gzip it) next to the other results. `frames.bin` is 45 MB and stays out of git; the MP4 is the committed artifact.

## 📉 Optimization journey: the 162M step, 1047 ms on main to PyTorch's 30-70 ms

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/step-time-journey-dark.png">
  <img alt="Bar chart of the 162M GPT step time: executor on main 1047 ms, four unmerged branches measured alone (fusion 868, attention 905, completion path no change, bf16 484), a reserved bar for the combined branch, and PyTorch 31-70 ms" src="assets/step-time-journey-light.png">
</picture>

🚧 Each orange bar is a **branch, unmerged**, measured alone against its own baseline in its own lock hold; none of these speedups is main's, and they have not been combined
(the `combined` bar is reserved until `perf/combined` is measured). Data: [`demo/viz/data/step_time_journey.json`](demo/viz/data/step_time_journey.json), redraw with `python3 demo/viz/journey.py`.

| Change | State | Result (162M, 1 step, A100) | Results dir (on that branch) |
|---|---|---|---|
| Executor on main | ✅ on main | 1047 ms median (1042 best); PyTorch eager 69.6, CUDA graph 68.2, compile 59-60 | `results/2026-11-06-gpt162-vs-torch` |
| Frontend fusion: half the dependency edges | 🚧 branch `perf/fusion-162m`, unmerged | 868 ms median (-17%), final weights bitwise identical; ~36 ns/edge, ~4 us/task | `results/2026-11-06-a100-fusion` |
| Attention in 128-row tensor-core tiles | 🚧 branch `perf/attention-162m`, unmerged | 905 ms (1.16x); attn tasks 55,680 -> 14,208; faster than torch fp32 SDPA | `results/2026-11-06-a100-attention` |
| Cheaper worker completion path | 🚧 branch `perf/completion-path`, unmerged | **no end-to-end change** (1042-1091 ms either way); the step is bound by the ready-queue pop | `results/2026-11-06-a100-completion` |
| Real bf16 storage + bf16 tensor-core tile | 🚧 branch `perf/bf16-a100`, unmerged | 484 ms vs 1104 f32 in the same hold (2.28x); still 15.7x slower than autocast-bf16 + CUDA graph (30.9 ms) | `results/2026-11-06-a100-bf16` |
| Admission / ready-queue pop | 🚧 branch `perf/admission-bottleneck`, in progress | **PLACEHOLDER**: no number yet | pending |
| All of the above combined | 🚧 branch `perf/combined`, in progress | **PLACEHOLDER**: reserved bar in the figure | pending |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/time-breakdown-dark.png">
  <img alt="Stacked worker-seconds per step: matmul, attention, other tasks, and idle time, for the TF32 executor and the bf16 executor" src="assets/time-breakdown-light.png">
</picture>

Where the time goes: workers are busy 17-19% of the step in both runs. bf16 shrinks the matmul slice (15 to 3.5 CTA-s) and the step with it, but the idle slice (waiting for
admission, the ready-queue pop and dependencies; the traces do not separate them) stays the bulk. The TF32 bar is approximate (see the JSON `source`). That is why the
priority is admission and task count, not faster kernels.

## 🧩 How it fits together

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

### ⚖️ Where a VM wins and loses

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

## ⚠️ What is not shown (read before quoting)

- The tiny demo is far from its limit: 106 workers are mostly idle, the 4.2 s span is Doom ticks (700 frames, one tick task of about 5.9 ms each, plus a 208 ms boot tick), and training plus serving finish in 345 ms.
- The VM is not faster than PyTorch at 162M (15-18x slower) and does not beat a CUDA graph at any size measured; its win region is launch-bound eager execution on tiny models.
- 162M: it runs and stays finite, nothing more. It is task-throughput bound (30x off its critical path), and its loss has no independent reference.
- A run with more tasks than slots recycles slots, and the version and cadence checks then skip by design (the 40-step 162M run did; the 6-step one did not).

## 🏗️ The integrated framework

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

### 📋 Status (A100-SXM4-40GB unless stated; each row cites the result that measured it)

| Status | What |
|---|---|
| ✅ **Implemented and measured** | Rendered executor timeline from the per-task trace (`demo/viz/timeline.py`, `results/2026-11-06-demo-tiny`); the full demo (training windows + serving on versioned weights + Doom in one launch), `results/2026-11-06-demo-tiny`. |
| | `kvm.cu` MLP training: 1.50x over the best baseline at 1 run, 1.13x at 32 (table below; predates the loop opcode). |
| | Executor: 8582 checks, 0 failed, in two recorded runs (`2026-11-04-a100-sched-integrated`, `2026-11-04-a100-ops-perf`); 2.1M-task DAG through 65,536 slots; both negative builds detected; sanitizers clean; litmus 15 properties pass, 7 mutants detected. Dispatch 1.38 us/task at 108 blocks, dependent hop 7.0 us (`2026-11-04-a100-exec-recycling`). |
| | Scheduler: weighted shares, reservations, EDF, priority inheritance, overload isolation (train 8-11x better than a priority queue under a serve flood) (`2026-11-03-a100-sched`, `2026-11-04-a100-sched-integrated`). |
| | Real Doom: every frame equals the CPU oracle in all recorded runs (2000 standalone, 700 per executor scenario, 12/12 scheduler scenarios, 162 adaptive runs with up to 8 instances); 35 Hz held beside a saturating load under the dynamic scheduler (0 of 698 gaps over 28.57 ms). |
| | GPT/extra op packs: `test_ops` 1396 and `test_extra` 2089 checks in the recorded round-2 run, sanitizers clean (`2026-11-04-a100-ops-perf`); 2-layer GQA GPT, 200 AdamW steps vs a double CPU model, loss gap 2.0e-4 (`2026-11-03-a100-gptops`). |
| | Model zoo: 24 bundles (5 families, bf16, fusion, remat, accumulation, LR schedule) match the CPU golden within the stated per-mode tolerances; 24 sanitizer runs clean (`2026-11-04-a100-models`). GPT golden mismatch is 1xTF32 gradient noise through AdamW, not a bug (`2026-11-04-a100-gpt-golden`). |
| | Multi-GPU code on ONE GPU: 14 stages pass, 4 skipped (`2026-11-03-dist-singlebox`). |
| | Executor vs PyTorch on the same GPT (tiny/small/mid, TF32 and fp32): correct, but 3.2x-12.6x **slower** than a whole-step CUDA graph and 3.5x slower than eager at 1024 tokens/step; 1.6-2.3x faster than eager only at 32-128 tokens (`2026-11-06-vs-pytorch`). Goal 3 is not met for transformer training. |
| | Executor vs PyTorch at 162M (`2026-11-06-gpt162-vs-torch`): gradients match float64 as well as torch-tf32, but 1042 ms/step against 59-70 ms (15-18x slower). |
| 🧪 **Code only (no result measures the claim)** | Multi-GPU and multi-node execution (needs 2+ GPUs / hosts; the stages SKIP). RDMA transport is a stub. 162M-parameter GPT correctness (it runs, `results/2026-11-06-demo-large`, but has no golden and its loss only drifts). The 1T-parameter section of DISTRIBUTED.md is arithmetic. Affinity hint (implemented, tested, no benefit measured, off by default). bf16 matmul tile speed (correct, speed not benchmarked). |
| ❌ **Not built** | Class-aware or sliced admission (the single admission FIFO is the tail-latency limit, below). Device-side task yield/preemption. Dynamic shapes, data-dependent control flow. Real bf16/fp16 storage (bf16 is emulated in fp32 buffers). Hopper/Blackwell paths (TMA, wgmma, tcgen05). |

### ❌ Known failures and honest negatives

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

## 🐱 The `kvm.cu` prototype

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
