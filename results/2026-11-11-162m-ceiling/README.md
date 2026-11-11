# Ceiling analysis: the 162M GPT step and the small-MLP wins (A100)

**Question.** Is 293 ms/step (bf16s, 162M GPT) the most the VM can be pushed to? **No.** It is about 15x above the compute floor at
realized tensor-core rates (18 ms) and 2.5x above what the *same* tasks would take with a free scheduler (108 ms). The gap is mostly the
scheduler: one task start per 1.6-2.2 us for 58.6k tasks is 94-128 ms by itself, and it is the term that survives when every tile is made
cuBLAS-class (replay: 45 ms with a free scheduler, 105 ms at 1.6 us per start). The ceiling is also not "beat the CUDA graph" by much: the
graph's 31 ms is 1.7x above the 18 ms cuBLAS+SDPA floor, and a VM dependency hop (1.4 us join, 0.66 us raw) is no cheaper than a graph edge
(1.05-1.24 us), so the VM can win only by hiding memory-bound work under GEMMs: best case ~22-28 ms (1.1-1.4x over the graph).

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
| the same at the realized rate | realized bf16 peak 251 TFLOP/s (8192³, 1410 MHz, 400 W); cuBLAS at the step's exact GEMMs: **14.27 ms** (213 TFLOP/s average) | **12.1 ms** at peak-realized, **14.3 ms** at the step's shapes |
| Attention FLOPs, causal | 3.307 - 3.035 = **0.272 TFLOP** (fwd 2 + bwd 5 matmuls of T²·hd/2 per head, 12 layers, B4 H12 T1024 hd64: 77 + 193 GFLOP) | bf16 MMA: 0.9 ms; the executor's 3xTF32 attention: 3 x 0.272 / 156 = **5.2 ms**; torch SDPA (flash) measured **3.7 ms** (12 x (0.074 fwd + 0.234 bwd)) |
| All FLOPs at TF32 (what the f32 bundle uses) | 3.307 / 156 | 21.2 ms (the repo's earlier floor) |
| HBM bytes | 14.67 GB / 1,555 GB/s spec; at the measured copy rate 1,381 GB/s (read-only 1,441) | **9.4 ms** spec, **10.6 ms** measured |
| Dependency chain | 326 logical levels (390 task levels) x hop | executor, earlier results (pick-up + release + op body) 28 us: 9.1 (10.9) ms; this tape's measured ready-to-start for an idle worker 4 us: 1.6 ms; raw hop (`hopbench`, one way, idle chip) 0.66 us, 0.69 us with 104 pollers, the executor's own join path (fence + indegree atomic + release) 1.40 us (1.49 with pollers): **0.5 ms** at 1.4 us; a CUDA-graph edge 1.05-1.24 us (measured, 1,024-4,096 nodes) = 0.4 ms |
| Largest of the three (valid because compute-bound GEMM tiles and memory-bound tiles can sit on different SMs at once) | | **~10 ms** (bf16 spec FLOPs); **~18 ms** with cuBLAS GEMMs (14.3) + SDPA (3.7) as the compute term |

So the bf16 step has a 10 ms spec floor and an 18 ms floor at what cuBLAS and flash attention actually reach on this part. torch's CUDA graph
(31 ms) is 1.7x above the 18 ms, eager (47 ms) 2.6x, the executor 267-293 ms is 15-16x above. The dependency chain is never the binding term at
this size (0.4-0.5 ms floor, 1.6-11 ms as built). Raw output: `raw/floors.json`, `raw/hop.jsonl`.

The VM's hop is not cheaper than the graph's: join path 1.4 us against 1.05-1.24 us per graph edge. A VM can therefore not win on dependency
latency alone, only on filling gaps (tails, co-scheduling memory-bound tiles under GEMMs). That is a fundamental property of global-memory
counters on this part.

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

**Why "nothing ready" exists.** Tasks become ready in bursts at level boundaries and are drained at the pop rate. A task that became ready
while workers were idle and the queue empty (only 585 of 58,634 did) started 4 us later (p10 3, p90 5 us for every idle-worker bucket): the
pick-up itself is cheap. The cost is the drain: the DAG has 390 levels, 150 tasks per level on average; 150 x 2.2 us = 330 us per level
(130 ms over the step) while the tasks it releases are in flight for only 100-300 us. Between a level's last start and its last end, and
before the next level is released, workers sit idle with nothing ready.

Admission look-ahead is **not** the cause, tested: the same bundle with the whole tape admitted at once (`--live-cap 65536 --max-tasks 65536
--ring-words 8388608`) is **slower**, 292-298 ms against 267-269 (3 reads each; `raw/execcfg.out`), and `--live-cap 32768` is equal (267-273).
Halving the workers (`--blocks 54`) gives 315 ms, so 2x the workers buys only 1.17x: the run is not worker-bound. (The trace shows at most 4,302 tasks
admitted-and-unfinished, mean 559, but that is where the tape's frontier is, not a limit.)

**Per op class (CTA-ms, share of the 8,782):**

| class | tasks | CTA-ms | mean us | note |
|---|---|---|---|---|
| matmul_mma_bf16 | 30,910 | 3,513 (40%) | 114 | 3.035 TFLOP / 3.513 CTA-s = 0.86 TFLOP/s per SM = **30% of the 2.89 TFLOP/s per-SM bf16 peak**; cuBLAS at the same shapes needs 14.27 ms x 108 SMs = 1.54 CTA-s, so the tiles are **2.3x slower than cuBLAS** (0.44x), in situ against isolated |
| attention (fwd 404, bwd dkv 958, dq 608, delta 117) | 14,208 | 2,087 (24%) | | 0.272 TFLOP in 3xTF32 = 0.82 TFLOP of TF32 work in 2.09 CTA-s = 0.39 TFLOP/s per SM = 27% of the 1.44 TFLOP/s TF32 per-SM peak; torch SDPA needs 3.69 ms x 108 = 0.40 CTA-s: **5.2x slower than flash** |
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
| free scheduler, matmul tiles at cuBLAS speed (x0.44) | 91 ms (128 ms at 1.6 us per start) |
| ... and attention at SDPA speed (x0.19) | 74 ms (123 ms) |
| ... and the memory-bound classes at half their time | **45 ms** (**105 ms** at 1.6 us per start) |
| measured | 240 ms (trace) / 267 ms (kernel) |

The model is crude (run times are the in-situ ones, which already include contention; no look-ahead window; priority = tape order) so read it as
"what the DAG allows", not a prediction. It says: the hop is worth 1-8 ms, the pop is worth 20-50 ms, and the part the replay cannot explain
(160 -> 240 ms) is not isolated: candidates are the pop slowing down under 100+ pollers and tasks in a burst starting late in tape order while the
long chain tasks wait. The one clear message of the last three rows: once the tiles are good, **tasks x pop cost is what is left** (105 ms of 105).

## 3. Ranked attacks (bf16s 162M, relative to 267 ms on this tree)

Gain is what the replay above and the trace allow; none is implemented here.

| # | attack | est. gain | what bounds it | kind |
|---|---|---|---|---|
| 1 | Remove the serial dispatch: pre-assigned per-worker queues (the merged tree's static schedule, compiled at load, declines bf16 GEMMs and casts because its cost model does not price them), and fewer tasks | 267 -> ~110-130 ms (2.0-2.4x) at today's tile speeds (replay: 131 ms at 1.6 us/start, 108 free); dispatch floor N x pop = 94-128 ms for 58.6k tasks, 24-40 ms for 15-25k | the DAG's width and critical path (77 ms); a level needs >= 107 tasks to fill the chip, so tasks cannot be made arbitrarily fat | engineering, the one that matters most |
| 2 | Fatter tasks: fuse the 7.2k round_bf16 into producers (weight shadows into AdamW), rope into attention loads, split only the long chain tasks (CE bwd 4.7 ms x 32, embedding bwd 4.4 ms x 25) | critical path 77 -> ~45 ms; with #1, 105 -> ~60-80 ms | HBM floor 10.6 ms for the whole step; the memory-bound classes (3.2 CTA-s, 36% of busy time) are ~2.7x above their share of it | engineering |
| 3 | Matmul tiles to cuBLAS class (0.44x now) | replay: 108 -> 91 ms free-scheduler, 131 -> 128 at 1.6 us/start: **small until #1 is done** | A100 mma.sync/cp.async, no wgmma/TMA; cuBLAS itself reaches 213 of 251 realized TFLOP/s here; 96-144 KiB smem, 1 block/SM | engineering, hard |
| 4 | bf16 MMA attention with flash-style fusion (today 3xTF32 on fp32 shared tiles, bf16 storage only) | 2.09 CTA-s -> ~0.4: 91 -> 74 ms free-scheduler (-17) | SDPA flash 3.7 ms for 12 layers (87 TFLOP/s fwd, 69 bwd) | engineering |
| 5 | Dependency hop 28 us -> 1.4 us | **1-8 ms** (replay 108 vs 116 ms at 28 us) | raw hop 0.66 us, join path 1.4 us (1.49 with 104 pollers): never below a graph edge (1.05-1.24 us) | engineering, low value here; floor is **fundamental** |
| 6 | Power cap (1,260 MHz under 400 W on the 162M step, merged panel) | up to ~10% | the part's limit | **fundamental** |

What is fundamental: the FLOP floor (10 ms spec, 18 ms with cuBLAS+SDPA), the HBM floor (10.6 ms), the power cap, the hop floor (no cheaper than a
graph edge), and that **a persistent-kernel VM cannot beat a CUDA graph by more than what the graph wastes**. Here the graph wastes ~13 ms of its
31 (31 - 18): graph-edge latency (~1,900 logical ops, 1.05-1.24 us each, about 2 ms), partial waves and the memory-bound kernels that run alone.
Best case with every engineering item done and tiles at cuBLAS speed: **~22-28 ms**, i.e. at most 1.1-1.4x faster than the graph; a realistic
near-term target is **60-110 ms** (#1 and #2 only), still 2-3.5x above the graph. Everything except the floors above is engineering; #1 moves the
number by 2x, #3-#5 together by about 1.5x only after #1.

## 4. The small-MLP wins (merged panel: MLP h72 d2, 1 run: 32.6 us vs PyTorch's best 72.6 us)

Model (`bench/exec_vs_torch/gen_mlp.py`): 96 tokens, vocab 100, embedding 48, hidden 72 x 2 layers, head 72 -> 100, SGD; 48 tasks per step per run
as unfused tiles. The 32.6 us row is the executor's fused-MLP path (a block-resident program per row slice, activations in shared memory, one
reduction hop per step). I re-ran the standalone bed of that program (`bench/fused_mlp/fused_mlp_bed.cu`, no executor around it; `raw/fmlp72*.txt`).

**Floors, per step (one run):**

| term | arithmetic | floor |
|---|---|---|
| FLOPs | 3 x 2 x 96 x (48x72 + 72x72 + 72x100) = 9.1 MFLOP; whole chip at TF32 133-156 TFLOP/s = 0.06 us; one SM (1.44 TFLOP/s) 6.3 us, 1.05 us per slice when cut into 6 | 0.06-1 us |
| Bytes | 20.8k parameters = 83 KB; each slice loads them (L2) and writes a partial-gradient copy; total L2/HBM traffic ~1.5 MB: 1.1 us at 1,381 GB/s, ~1 us per slice at one SM's L2 rate | ~1 us |
| Dependency chain | SGD chains step to step: load -> compute -> publish -> hop -> reduce+SGD -> hop; 2 hops at 0.66-1.4 us (join path) | 1.3-2.8 us |
| Sum of the serial chain | load 1 + compute 1-2 + publish 0.5 + hops 1.3-2.8 + reduce 1 | **~5-8 us** (my estimate) |

**Measured:** bed 30.0 us/step with 6 slices of 16 rows, 41.2 us with 3 slices of 32 (200 steps, 3 reads agree within 0.02 us); the executor row 32.6 us;
PyTorch's best (compiled whole-run graph) 72.6 us, eager 1,532 us, graph edge 1.05-1.24 us. Losses of the bed match the CPU reference to 1.4e-7
per step, final weights 2.0e-7 (L2).

**Where the bed's 30-41 us goes** (cycles per step, slice 0, from `raw/fmlp72.txt`; 57,278 cycles = 40.6 us): reduce + SGD + fence 27%, publish
partials 12%, forward/backward GEMM phases ~37% (small mma tiles between block syncs), softmax-CE 6%, weight/id load 6%, the two hop waits only
**3.4%** (1.5 + 1.9; 5.7% with 6 slices). The step is the serial instruction chain inside one CTA, not the hop and not the FLOPs.

So: the 2.2x win over the graph is real, and it exists because the graph's tiny kernels (count not inspected) cost 72.6 us in total, each paying a 1.05-1.24 us edge plus its own
body, while the fused program pays 2 hops and no kernel boundaries. It is **4-6x above its own chain floor** (5-8 us), and the gap is engineering
(reduce+SGD done by one slice's threads at 11-15k cycles, partial publish through atomics, one resident block per SM): worth ~1.5-2x more
(a 15-20 us step), i.e. a 3.5-5x win over the graph, bounded by the 5-8 us chain, which is **fundamental** (two dependent global round trips plus the
step-to-step SGD dependence, no pipelining possible across steps: `2026-11-09-a100-step-overlap`). Where it does not extend: the fused program needs
the weights in shared memory (146 KB at 3 slices, 108 KB at 6): h256 d4 (0.50x) and h1024 d2 (0.34x) fall back to the tile path, whose floor
(h1024: 0.69 GFLOP = 4.4 us at TF32 peak, ~5 MB of weights resident in the 40 MB L2) is again 5-10 us against 309 us measured: 30x above, bounded by
pop and hop per level exactly as in the 162M step. At 8 and 32 runs (32.3 and 46.3 us vs PyTorch 78.8 and 104.0) more runs ride along free until
slices exceed the 107 resident blocks (32 runs x 6 slices = 192 blocks = 2 waves); the FLOP floor at 32 runs is 1.9 us, so that row is wave-bound,
not compute-bound.

## Provenance and limits

* Trace: `trace_162m_bf16s_1step.csv.gz` (`kvm_exec --quiet --trace-csv`, bundle generated by `bench/vs_pytorch/gen.py
  --model gpt_162m (precision bf16s) --steps 1 --mode tf32 --tiling a100`); `analyze_trace.py TRACE BUNDLE_DIR` needs the bundle's `tasks.bin` and
  `successors.bin` (slot = task index, checked: 58,634 unique slots).
* Not measured: the bf16s step on the merged executor. No binary of the merged tree was on the box and a full-op-pack build ran 45+ minutes
  under the shared GPU lock (other agents' builds held it for that long, which also delayed every run here by ~25 minutes), so the merged-tree
  statements rest on its committed panel and on counting its f32 bundle's tasks (53,939, 255 levels, 31,755 matmul tiles).
* Commands: `python3 floors_bench.py`, `hopbench --quick` and `fused_mlp_bed --steps 200 --hidden 72,72 [--slice 16]` (all from this repository's
  `bench/`), `kvm_exec --quiet --trace-csv T bundle` and the three window / worker variants in `raw/execcfg.out`; one `flock` hold for all of them.
* The replay is a model. In-situ task times are not what the tiles cost alone.
* One run of `hopbench` and the fused-MLP bed was started without the GPU lock by mistake (a `--help` flag it ignores; a few seconds, no other
  process affected); every number reported was re-taken under the lock or is marked.
