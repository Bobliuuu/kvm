# Executor (`kvm_exec`) vs PyTorch eager / whole-step CUDA graph / `torch.compile`, small dependent workloads and the 162M GPT (A100, 2026-11-09)

**Verdict: AGENTS.md goal 3 is NOT met for the generic executor.** On the workloads where the cost model says a VM should win (small MLP, one run
or many runs, 10 to 1000 steps) the executor is correct and **beats only PyTorch eager (1.1x to 3.9x) and `torch.compile` without CUDA graphs (1.2-1.9x at R=1)**.
It **loses to a whole-step CUDA graph by 1.9x (R=32) to 4.8x (R=1), to R graphs on R streams by 5.7x to 7.2x, and to a batched graph by 5.5x to 33x and a compiled
batched graph by 9x to 69x**; at hidden 256-1024 it loses to eager too. At 162M it is 17.7x slower than eager and 21x slower than `torch.compile` + graph.
The one VM result that stands is `kvm.cu`'s MLP (README). Where the time goes (measured, "Why"): the executor admits the whole tape before the first task can
run (2.4-3.1 us per task), runs one resident block per SM (98,304 B dynamic smem) and pays about 28 us per dependency level; a graph runs the same step at about 8 us per level.

## Setup

| | |
|---|---|
| GPU | NVIDIA A100-SXM4-40GB, 108 SMs, sm_80, driver 580.105.08, CUDA 12.8 (nvcc 12.8), max SM clock 1410 MHz; shared box, every stage inside one `flock ~/.gpu.lock` hold |
| PyTorch | 2.7.0+cu128, Triton 3.3.0; TF32 matmuls (`allow_tf32`); `torch.compile(mode='max-autotune-no-cudagraphs', fullgraph=True)`; SDPA attention and fused AdamW for the GPT |
| Executor | `nvcc -O3 -arch=sm_80 -std=c++17 -Iinclude --fmad=false -Xcompiler -ffp-contract=off -diag-suppress 177 kvm_exec.cu` (`raw/env.txt`) on `origin/main` 1c2111b; default grid, 1xTF32 GEMMs, frontend tiling `a100`. MLP bundles are run with the whole tape as ONE admission record (`--max-tasks/--live-cap/--ring-words` sized to the tape): measured 2.0-2.2x faster than the default windowing at R >= 8 ("Why", item 2) |
| Code / commit | `bench/exec_vs_torch/` (driver commit ad9ec50 on branch `bench/exec-vs-torch-v2`; `tables.py`, `dag_depth.py`, `ab_occupancy.sh` were added after the runs and touch no measured path); GPT side reuses `bench/vs_pytorch/torch_bench.py` (one fix: the graphed warm-up reset the step counter, needed for 1-3 step tapes) |
| Command | `bash bench/exec_vs_torch/box_job.sh OUT` on the box (builds outside the lock, one lock hold per stage; `REPS=7`); tables: `python3 bench/exec_vs_torch/tables.py raw > tables.md`; raw output in `raw/` (JSON per stage, `stage_logs/` per stage, `env.txt`) |

**Same process.** `kvm_exec` has no in-process API, so "same process" is: ONE driver process owns the GPU and every PyTorch variant (built, compiled, captured
first), and per rep it times each variant once and then launches the executor as a child, all inside one lock hold, so every ratio in a row comes from one time
window (spread: in `tables.md`, `[median .. worst]`). The executor time is the kernel's own cudaEvent `OK: n/n tasks, X ms` of one launch running all N steps
of all R runs (persistent-kernel start, host admission of the tape included; process start and bundle load excluded). PyTorch time is CUDA events around N
steps in steady state (graphs captured and torch.compile warmed outside the timer, state reset before every rep). Best of 7 is the headline.

## Workloads (identical bytes on both sides)

- **MLP** (the claims-audit `mlp_ce`: embedding -> depth x [linear + tanh-GELU] -> linear -> mean CE, 96 tokens, vocab 100, d 48, hidden 72, depth 2, SGD
  lr 0.05; 51 tasks/step/run with tiling `a100`, longest dependency chain about 15 tasks per step, `dag_depth.py`). `gen_mlp.py` builds the bundle; initial weights and
  tokens are read from its `workload.json` by `mlp_torch.py`. **The bundle trains on batch 0 at every step** (verified against its CPU golden, 6 steps equal to 1e-7;
  the other batches in the stream are never read); PyTorch does the same. R independent runs = `replicate.py` copies of the bundle (no shared edge) vs R
  separate graphs (1 stream, R streams), one batched `bmm` graph (vmap-like, needs identical runs), and the same batched model compiled. Every torch variant
  executes the same step body including the loss log.
- **GPT**: `gen.py` workloads `gpt_tiny` (16.9k params, 32 tokens, 50 steps), `gpt_small` (116k, 128 tokens, 50 steps), `gpt_162m` (162M, 4096 tokens, 3 steps), AdamW lr 3e-3, clip 1.0 (same
  setup as `2026-11-06-vs-pytorch`, now interleaved with the executor in one driver).

## Correctness first (all gates passed before any timing; a failure aborts the stage)

1. executor 3-step bundle within 1e-3 relative L2 of the CPU golden (hidden 72 only: the pure-Python golden hangs on larger models); 4 replicated copies bitwise equal to the single run (79-119 buffers).
2. executor N-step loss and final weights vs PyTorch eager from the same bytes, and both vs a float64 PyTorch run (independent reference):

| MLP | executor vs eager, loss (max rel over steps) | weights (rel L2, max buffer) | executor vs fp64 | eager vs fp64 | loss start -> end |
|---|---|---|---|---|---|
| h72, 50 steps | 2.4e-4 | 5.1e-5 | 3.1e-4 | 1.9e-4 | 4.537 -> 0.343 |
| h72, 10 / 200 / 1000 steps | 1.2e-5 / 3.2e-4 / 9.5e-4 | 9.9e-5 / 4.5e-5 / 4.1e-5 | 1.3e-5 / 3.5e-4 / 9.2e-4 | 1.7e-5 / 3.5e-4 / 8.3e-4 | 4.54 -> 0.0046 at 1000 |
| h256 depth 4, 50 steps | 1.0e-3 | 4.6e-4 | 8.2e-4 | 6.7e-4 | 4.80 -> 0.010 |
| h1024 depth 2, 50 steps | 6.0e-4 | 7.5e-5 | 8.7e-4 | 6.9e-4 | 6.10 -> 0.007 |

   The relative loss difference grows with the 500-1000x loss decay (the tail is small), so the gate is 3e-3; the executor is as close to float64 as eager is.
3. every other PyTorch variant (all R runs) vs eager's trajectory: eager, graphs, batched graph within 2.1e-4 (batched: bmm rounding); **`torch.compile` variants differ by
   1.0e-2 to 1.4e-2 from both eager and float64** (Triton TF32 matmuls round differently; same on `gpt_small`, 3.1e-3), so the compile gate is 3e-2 (MLP) / 1e-2 (GPT) and the number is reported.
   GPT: executor vs eager loss 1.3e-4 (tiny), 7.3e-4 (small), 2.3e-5 (162M, 3 steps).
4. A negative control the gate caught: `multi_graph` (R graphs replayed on R streams) first returned wrong results at hidden >= 256 because PyTorch keys the cuBLAS workspace by (handle,
   stream) and graphs captured on one stream share it; the fix (one capture stream per graph, `mlp_torch.py`) is what is timed.

## Results

Full tables with `[median .. worst]` and every ratio: `tables.md`. us per step (all R runs advance one step), best of 7. Ratio = variant / executor (above 1: executor faster).

### 1. Many runs (R independent runs, hidden 72, 50 steps, 51 tasks/step/run)

| R | eager | compile (no graph) | compile + graph | graphs, 1 stream | graphs, R streams | batched graph | batched compile + graph | **executor** |
|---|---|---|---|---|---|---|---|---|
| 1 | 804 (1.45x) | 957 (1.73x) | 62.6 (0.11x) | 115 (0.21x) | - | - | 60.8 (0.11x) | **554** |
| 2 | 1595 (2.21x) | - | - | 216 (0.30x) | 127 (0.18x) | 132 (0.18x) | 61.6 (0.09x) | **720** |
| 4 | 2996 (2.74x) | - | - | 431 (0.39x) | 152 (0.14x) | 137 (0.13x) | 65.2 (0.06x) | **1091** |
| 8 | 5645 (3.01x) | - | - | 862 (0.46x) | 291 (0.16x) | 146 (0.08x) | 71.2 (0.04x) | **1873** |
| 16 | - | - | - | 1740 (0.49x) | 574 (0.16x) | 153 (0.04x) | 79.3 (0.02x) | **3524** |
| 32 | - | - | - | 3565 (0.54x) | 1128 (0.17x) | 198 (0.03x) | 96.3 (0.01x) | **6615** |

Spread of the executor over 7 reps is 2-15% (median within 3% of best except R=1 and R=2); of the graph variants 0-25% (the 1-stream graph at R=1 flips between 115 and 140 us with GPU clocks). A repeat of the whole table
in another lock hold agreed within 3% (first pass: executor 573 / 722 / 1091 / 1885 / 3535 / 6615).

### 2. Many steps (hidden 72; N steps per launch; R=1 and R=8)

| N | R | eager | compile + graph | graphs, 1 stream | graphs, R streams | batched compile + graph | **executor** |
|---|---|---|---|---|---|---|---|
| 10 | 1 | 751 (1.09x) | 77.2 (0.11x) | 141 (0.20x) | - | 75.2 (0.11x) | **691** |
| 50 | 1 | 804 (1.45x) | 62.6 (0.11x) | 115 (0.21x) | - | 60.8 (0.11x) | **554** |
| 200 | 1 | 725 (1.41x) | 58.9 (0.11x) | 109 (0.21x) | - | 57.4 (0.11x) | **516** |
| 1000 | 1 | 771 (1.58x) | 75.8 (0.16x) | 121 (0.25x) | - | 57.2 (0.12x) | **489** |
| 10 | 8 | 5664 (2.43x) | - | 1071 (0.46x) | 310 (0.13x) | 89.9 (0.04x) | **2327** |
| 1000 | 8 | 5816 (3.91x) | - | 884 (0.59x) | 284 (0.19x) | 71.7 (0.05x) | **1486** |

More steps do not rescue it: the executor improves 29% from N=10 to N=1000 (launch and ramp amortize), the graph 14%, so the ratio to the graph moves only from 0.20x to 0.25x at R=1 and from
0.46x to 0.59x at R=8: the remaining gap is per step (and per task), not per launch.

### 3. Size (50 steps)

| model | R | eager | compile + graph | graphs, 1 stream | graphs, R streams | batched compile + graph | **executor** |
|---|---|---|---|---|---|---|---|
| h72 d2, 51 tasks/step | 1 | 804 (1.45x) | 62.6 (0.11x) | 115 (0.21x) | - | 60.8 (0.11x) | **554** |
| h256 d4, 157 tasks/step | 1 | 1051 (0.91x) | 105.6 (0.09x) | 168 (0.15x) | - | 103 (0.09x) | **1155** |
| h256 d4 | 8 | 8884 (1.62x) | - | 1365 (0.25x) | 303 (0.06x) | 137 (0.02x) | **5501** |
| h1024 d2, 465 tasks/step | 1 | 852 (0.27x) | 104 (0.03x) | 129 (0.04x) | - | 104 (0.03x) | **3153** |
| h1024 d2 | 8 | 6098 (0.43x) | - | 1116 (0.08x) | 440 (0.03x) | 316 (0.02x) | **14029** |

As the GEMMs grow the executor loses even to eager (h256 R=1 0.91x, h1024 0.27x): matmul tiles run at about 0.5x cuBLAS (`2026-11-04-a100-ops-perf`) and a graph spends the same wall time on fewer, bigger kernels.

### 4. GPT (same bytes, interleaved with the executor; us per optimizer step, best of 7 (162M: 5), `[median .. worst]` in `tables.md`)

| model | tokens/step | eager | whole-step graph | compile (no graph) | compile + graph | **executor** | executor vs eager / graph / compile+graph |
|---|---|---|---|---|---|---|---|
| gpt_tiny | 32 | 3011 | 364 | 2047 | 173 | **1031** | 2.9x faster / 2.8x slower / 6.0x slower |
| gpt_small | 128 | 4816 | 646 | 3089 | 332 | **2429** | 2.0x faster / 3.8x slower / 7.3x slower |
| gpt_162m | 4096 | 70,602 | 68,245 | - | 59,097 | **1,251,526** | 17.7x slower / 18.3x slower / 21.2x slower |

(`gpt_tiny` the graph's best 364 us sits next to a 428 us median: two clock states; the compile + graph step 173/203 us; ratio on medians 2.7x / 5.7x.) The 2026-11-06 campaign
(separate processes) gave 1166 / 2818 us for the executor on tiny / small: this run is 12-14% faster on the same bundles. At 162M the executor needs 1.25 s per optimizer step on a 3-step tape
(829k tasks; 1.04 s on the 1-step tape of 2026-11-06): admission alone is 276k tasks x about 2 us = 0.55 s per step.

## Why (the AGENTS.md cost model, measured)

`kvm_exec`'s two costs are the dependency hop through global memory and the per-task admission/dispatch; the graph's are one launch edge per dependent kernel.

1. **Admission precedes execution.** One MLP run is 2550 tasks for 50 steps; the trace (`--trace-csv`) shows the first task becomes ready 8.0 ms after kernel start (3.1 us per task) and, at
   R=32 (81,600 tasks), 199.7 ms after (2.45 us per task) while the last task ends at 319 ms and the kernel at 344 ms. So 58% of the executor's time at R=32 (200 of 344 ms) is the submit/admit of the record before any
   worker starts; the cost scales with tasks, not with steps: it is why many steps do not amortize it (section 2) and why R runs cost the executor a full run each. Marginal cost of one more run: 195 us/step = 51 tasks at 3.8 us
   (admission 2.45 + execution 1.4), against 33 us/step for a graph on its own stream and 2.2 us/step in the batched graph (same kernels, a wider grid).
2. **Windowing makes it worse.** With the executor's default admission window (about 1000 tasks per record) the same R=32 run takes 727 ms instead of 330 ms (R=8: 183-197 vs 94-95 ms; R=1: no difference,
   28-29 ms): `raw/stage_logs/flags_*.txt`. The tape is sized to one record in every timed run, i.e. the executor is given its best known configuration.
3. **The dependency hop is about 25 us per level, not 5.** R=1: 2550 tasks, chain of 15 tasks per step; `trace_r1`: mean task 13.8 us, mean ready-to-start wait 12.5 us, workers busy 1.5%, span 21.6 ms = 432 us per step
   (+8.0 ms admission, +2 ms other = 31.8 ms kernel time) = 15 levels x 29 us. A graph runs the same step in 109-121 us (about 8 us per chain level).
   The README's 5 us hop (`kvm.cu`) is for one tiny dependent op on a persistent block; here the wait includes the ready-queue pickup, and the 1-CTA matmul tiles alone are 15.3 us each (69% of task time).
4. **Fewer workers than the hardware allows.** `kvm_exec` reports `1 resident blocks/SM with 98304 B dynamic smem -> grid 108 (1 admitter + 107 workers)`. The 96 KiB is `KVM_GPT_MMA_SMEM` (the bf16
   matmul's stage buffers, merged since the 2026-11-06 vm-wins measurement, whose kernel had 72 KiB and 2 blocks/SM); the f32 bundles measured here never run that op. At R=32 the utilisation is 6.9%
   with a mean ready-to-start wait of 217 us: tasks are admission-limited and queue behind each other. A/B below ("Occupancy A/B").
5. **Big GEMMs.** From h256 up the 1-CTA tiles at 0.5x cuBLAS decide it (section 3), as for the 162M GPT: 276k tasks/step, 21.2 ms floor in `2026-11-06-gpt162-vs-torch`, 1.25 s measured.
6. **Why the graphs win R runs so cheaply.** A batched graph issues the same kernels for any R (kernel cost grows with the grid, 61 -> 96 us for R=1 -> 32); R graphs on R streams overlap their
   nodes (33 us marginal per run); the executor pays per task. The README's "where a VM can win" list (small dependent steps, many runs) is exactly the region where a CUDA graph is already nearly free of the costs the VM removes.

## Honest limits

- The executor ratio compares a child process's kernel time with in-process CUDA events (same lock hold, interleaved reps); `kvm_exec` has no in-process API. Fixed costs inside the executor's time (admission of the tape,
  persistent-kernel start) are real and are counted; a hypothetical in-process API would not remove them.
- `torch.compile` here is `max-autotune-no-cudagraphs` (+ our own whole-step graph for `compile + graph`); `reduce-overhead` and a `--fuse` executor bundle were not run (the 2026-11-06 campaign had `reduce-overhead`: it was
  slower than `compile + graph`). The executor runs unfused bundles and 1xTF32 GEMMs only.
- The MLP bundle trains on batch 0 at every step (finding above; a defect of the audit generator's token stream, not corrected: both sides do the same work). Heterogeneous runs (different shapes, where the
  batched graph does not apply), data-dependent control flow, an H100/B200 and the `KVM_CHAIN` / predicated-task branch (`bench/vm-wins`, not on main) were not measured.
- `--no-golden` at hidden 256/1024: no CPU golden gate (hangs in the pure-Python CPU VM, `worker did not stop`); gates 2-4 stand. 162M uses a 3-step tape (loss trajectory of 3 steps, not a learning curve; learning at 162M
  is `2026-11-06-a100-learning-162m`).
- Shared box: other agents' jobs hold the lock in turn; the first pass and the repeat of section 1 agree within 3%, the box was never contended while a stage held the lock.
