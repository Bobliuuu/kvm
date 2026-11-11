# Ceiling analysis: the 162M GPT step and the small-MLP wins (A100)

**Question.** Is 293 ms/step (bf16s, 162M GPT) the most the VM can be pushed to? **No.** It is 22x above the cheapest term that bounds
it and about 2.3-2.7x above what the *same* tasks would take with a free scheduler. Almost all of the gap is the scheduler (one start every
1.6-2.2 us for 58.6k tasks, plus a 4,096-task look-ahead window), not the GPU. But the ceiling is also not "beat the CUDA graph": the
graph's 31 ms is within ~3x of the compute floor, so the VM can win at most a few ms (about 1.1x) on this model, and only after matching
cuBLAS-class tiles.

**Two things to know before using the 293 ms figure.**

1. It is the `perf/combined` tree (Nov 6-7). `origin/main` (#23, `results/2026-11-10-a100-merged`) has a different executor (one-record tape,
   ticket-claim ready queue, bigger tiles) and reads **118.0 ms/step on the 162M f32/TF32 bundle** (53.9k tasks, PyTorch whole-step graph
   70.7 ms, `torch.compile` graph 63.1 ms: 1.7-1.9x, not 12x). The README's "845 ms f32" row is stale. The bf16s 162M step has **not been
   measured on the merged executor** (its Phase B row did not run), so no number here claims it.
2. The breakdown below is measured on the tree that produced 293 ms (binary `kvm_exec` sha256 `64f69cca…`, bundle `gpt_162m` bf16s, 58,634
   tasks, 1 step, `--trace-csv`): 267 ms kernel time in my three reads (267.16, 267.63, 267.16; the repo's 293.3 ms is the median of 7 on
   Modal). Merged-tree facts are taken from its committed files and one static look at its bundle (below), not re-timed.

Machine: A100-SXM4-40GB (108 SMs, 1410 MHz, 400 W), driver 580.105.08, CUDA 12.8.93, shared box, GPU lock held.

## 1. Floors, with the arithmetic

Step = 4,096 tokens, 12 layers, d768, ffn 3072, vocab 50,304, 162.3M parameters (`info.json`: 3.307 TFLOP, 14.67 GB minimum traffic,
777 logical ops, depth 326 for the bf16s tape; the task DAG is 390 levels deep).

| term | arithmetic | floor |
|---|---|---|
| Matmul FLOPs, bf16 tensor cores | 6 x 123.5M matmul params (12 x (4d² + 2 x d x 3072) = 84.9M, plus the 38.6M head) x 4,096 = **3.035 TFLOP**; / 312 TFLOP/s | **9.7 ms** |
| the same at the realized cuBLAS rate | measured below (`floors_bench.py`) | see 1b |
| Attention FLOPs, causal | 3.307 - 3.035 = **0.272 TFLOP** (fwd 2 + bwd 5 matmuls of T²·hd/2 per head, 12 layers, B4 H12 T1024 hd64: 77 + 193 GFLOP) | bf16 MMA: 0.9 ms; the executor's 3xTF32 attention: 3 x 0.272 / 156 = **5.2 ms** |
| All FLOPs at TF32 (what the f32 bundle uses) | 3.307 / 156 | 21.2 ms (the repo's earlier floor) |
| HBM bytes | 14.67 GB / 1,555 GB/s spec | **9.4 ms**; at the measured copy rate see 1b |
| Dependency chain | 326 logical levels (390 task levels) x hop | 28 us (executor, measured by earlier results): 9.1 (10.9) ms; 8 us (graph edge, earlier results): 2.6 ms; raw global-memory hop `hopbench` 0.66 us one-way, idle chip, 1 pair: **0.2 ms** |
| Largest of the three (valid because compute-bound GEMM tiles and memory-bound tiles can sit on different SMs at once) | | **~10 ms** (bf16 FLOPs), ~13 ms at cuBLAS rates |

So the bf16 step has a ~10-13 ms floor; torch's CUDA graph (31 ms) is 2.4-3x above it, eager (47 ms) 3.6-4.7x, the executor 267-293 ms is 22-29x
above. The dependency chain is never the binding term at this size.

### 1b. Measured floors on this A100

PENDING-1B

## 2. Where the 267 ms goes (from `trace_162m_bf16s_1step.csv.gz` and the bundle; `analyze_trace.py`, output `analysis_trace.txt`)

Kernel time 267 ms. The trace covers first admit to last end = 240.3 ms; the other 27 ms (10%) is outside it (launch, setup, exit; not
broken down further).

**Worker-time inside the 240 ms (107 workers):**

| share | what | equals |
|---|---|---|
| 34.4% | running a task (sum of task run times 8,782 CTA-ms) | 82 ms if perfectly packed |
| 19.5% | idle **while a task was ready and waiting** (ready-queue pop is the limit) | 47 ms |
| 46.1% | idle with **nothing ready** (admission look-ahead and dependencies) | 110 ms |

**Why "ready but idle" exists.** Whenever 100 or more tasks were ready (47 ms of the run) only 58 of 107 workers were running, at **458 starts
per ms = 2.18 us per start**; the best 1 ms window held 636 starts (1.6 us). Mean task is 150 us, so keeping 107 workers busy needs one start per
1.4 us. 58,634 tasks x 1.6-2.2 us = **94-128 ms** just to dispatch. This is the serialized stage the README names (the ready-queue pop). The
merged tree's f32 row says the same thing from the other side: 117.9 ms / 53,772 tasks per step = 2.19 us per task.

**Why "nothing ready" exists.** Admission tracks completion with a lag of about 1.6 ms (the mean admit-to-ready wait is 1.6 ms) and at most 4,302
tasks are ever admitted-and-unfinished (mean 559): the executor sees a window of about 4,096 tasks of a 58,634-task tape, so levels that are
wider than the window cannot be fed. The merged tree admits the tape as one record; the bf16s step on it is the number nobody has.

**Per op class (CTA-ms, share of the 8,782):**

| class | tasks | CTA-ms | mean us | note |
|---|---|---|---|---|
| matmul_mma_bf16 | 30,910 | 3,513 (40%) | 114 | 3.035 TFLOP / 3.513 CTA-s = 0.86 TFLOP/s per SM = **30% of the 2.89 TFLOP/s per-SM bf16 peak** while 107 CTAs share L2/HBM |
| attention (fwd 404, bwd dkv 958, dq 608, delta 117) | 14,208 | 2,087 (24%) | | 0.272 TFLOP in 3xTF32 = 0.82 TFLOP of TF32 work in 2.09 CTA-s = 0.39 TFLOP/s per SM = 27% of the 1.44 TFLOP/s TF32 per-SM peak |
| round_bf16 | 7,214 | 775 (9%) | 107 | casts and weight shadows (the other 6,560 are not touched by the storage change) |
| adamw | 881 | 543 (6%) | 616 | |
| rope fwd + bwd | 1,536 | 515 (6%) | 330 | |
| colsum, gelu_bwd, layernorm bwd, norm_bwd_params, CE fwd/bwd, embedding | 2,500 | ~1,400 (16%) | 237-4,700 | CE bwd 4.7 ms per task x 32, embedding bwd 4.4 ms x 25 |

**Critical path.** Longest path through the DAG with the measured run times and zero hop cost = **77 ms**. 16 ms of it is `round_bf16`, 15 ms
matmul tiles, 7 ms `gelu_bwd`, 6 ms layernorm bwd, 4.7 ms CE bwd, 4.7 ms embedding bwd, 4.5 ms attention bwd, 4.2 ms rope bwd: long single
tasks on a chain, not hops.

**Replay bounds on the same tasks** (greedy list schedule over the measured DAG and measured run times, 107 workers, tape order):

| scheduler | makespan |
|---|---|
| free scheduler, no hop | **108 ms** (work/107 = 82 ms; critical path 77 ms) |
| + 7 us hop on every edge | 109 ms |
| + 28 us hop on every edge | 116 ms |
| one start per 1.6 us | 131 ms |
| one start per 2.2 us | 160 ms |
| free scheduler, matmul tiles at cuBLAS-class speed (x0.45) | 93 ms |
| ... and attention x0.5 | 81 ms |
| measured | 240 ms (trace) / 267 ms (kernel) |

The model is crude (run times are the in-situ ones, which already include contention; no look-ahead window; priority = tape order) so read it as
"what the DAG allows", not a prediction. It says: the hop is worth 1-8 ms, the pop is worth 20-50 ms, and the part the replay cannot explain
(160 -> 240 ms) is the look-ahead window plus cases where tasks wait on a pop that is itself contended.

PENDING-LOOKAHEAD

## 3. Ranked attacks (bf16s 162M, relative to 267 ms on this tree)

Gain is what the replay above and the trace allow; none is implemented here.

| # | attack | est. gain | what bounds it | kind |
|---|---|---|---|---|
| 1 | Take the serial pop and the look-ahead window out: one-record tape and static / pre-assigned queues (the merged tree already has both for f32; the static path declines bf16 GEMMs and casts because its cost model does not price them) | 267 -> ~110-130 ms (2.0-2.4x): removes the 19.5% + most of the 46.1% idle | the DAG's own width and critical path (77 ms); then N tasks x pop cost | engineering |
| 2 | Fewer, fatter tasks: 58.6k -> ~15-25k (fuse the 7.2k rounds into producers, rope into attention loads, split only the long chain tasks) | at 1.6 us/start: 94 -> 24-40 ms of dispatch floor; with #1 done, small | tile size vs the 107-way parallelism per level (a level needs >= 107 tasks to fill the chip) | engineering |
| 3 | Matmul tile efficiency: 30% of bf16 peak in situ -> cuBLAS class (see 1b) | replay: -15 ms (108 -> 93) at the free-scheduler end; ~-60 ms at today's schedule if the matmul share of busy time falls 3.5 -> 1.5 CTA-s | cp.async/ldmatrix mma on A100 has no wgmma/TMA; the best known A100 bf16 kernels reach ~70-75% of peak; 96 KiB smem, 1 block/SM | engineering, hard |
| 4 | Break up the long chain tasks (CE bwd 4.7 ms x 32, embedding bwd 4.4 ms x 25, colsum 0.9 ms, norm_bwd_params 2.4 ms) and fuse the memory-bound classes (3.2 CTA-s, 36% of busy time) | critical path 77 -> ~45 ms; replay floor 108 -> ~90 | HBM floor of the whole step 14.67 GB / ~1.35 TB/s = ~11 ms; these classes are ~2.7x above their own share of it | engineering |
| 5 | bf16 MMA attention (today 3xTF32 on fp32 shared tiles, bf16 storage only) | 2.09 CTA-s -> ~0.7: -8 to -12 ms at the free-scheduler end | flash-class bf16 on A100 ~150 TFLOP/s fwd, ~90 bwd (see 1b SDPA) | engineering |
| 6 | Dependency hop 28 us -> ~1-3 us | **1-8 ms** only (replay 108 vs 116 ms) | raw global-memory hop 0.66 us idle; ~1.5-3 us when 107 workers poll; the op body, not the hop, is the 5-11 us | engineering, but low value here |
| 7 | Power cap: 1,260 MHz under 400 W on the 162M step | up to 10% | the part's power limit | **fundamental** |

What is fundamental: the FLOP floor (~10 ms bf16), the HBM floor (~11 ms), the power cap, and that **a persistent-kernel VM cannot beat a CUDA graph
by more than what a graph wastes**. For this model that waste is small: graph 31 ms vs a ~13 ms cuBLAS-class floor leaves ~18 ms, of which edge
latency (~1,300 kernels x 1.3-2.5 us = 2-3 ms), wave-quantization tails and the elementwise kernels' bandwidth are all that a VM can reclaim.
My estimate for the best case with every engineering item done: **~35-60 ms** (1.1-2x above the graph), and a win over the graph of at most ~1.1x
at 162M. Everything else is engineering, in the order above; #1 is the one that moves the number by 2x.

## 4. The small-MLP wins (merged panel: MLP h72 d2, 1 run: 32.6 us vs PyTorch's best 72.6 us)

PENDING-MLP

## Provenance and limits

* Trace: `trace_162m_bf16s_1step.csv.gz` (`kvm_exec --quiet --trace-csv`, bundle generated by `bench/vs_pytorch/gen.py
  --model gpt_162m (precision bf16s) --steps 1 --mode tf32 --tiling a100`); `analyze_trace.py TRACE BUNDLE_DIR` needs the bundle's `tasks.bin` and
  `successors.bin` (slot = task index, checked: 58,634 unique slots).
* Not measured: the bf16s step on the merged executor; no GPU time was available for it (another build held the GPU lock for over 35 minutes), so
  the merged-tree statements rest on its committed panel and on counting its f32 bundle's tasks (53,939, 255 levels, 31,755 matmul tiles).
* The replay is a model. In-situ task times are not what the tiles cost alone.
* One run of `hopbench` and the fused-MLP bed was started without the GPU lock by mistake (a `--help` flag it ignores; a few seconds, no other
  process affected); every number reported was re-taken under the lock or is marked.
