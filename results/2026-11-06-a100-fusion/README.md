# gpt_162m: fewer dependency edges from the frontend (non-matmul, non-attention task graph)

Track 8 of the 2026-11-06 session. Question: how many tasks and edges does the frontend emit per op class for one 162M GPT
training step, and how much of it can the frontend remove without touching device code or numerics?

| | before (origin/main d9d9cee) | after (`perf/fusion-162m`) |
|---|---:|---:|
| tasks / step | 318,139 | 312,802 (-1.7%) |
| edges / step | 6,889,225 | 3,462,817 (**-49.7%**, 2.0x) |
| model critical path (inspect cost model) | 391 tasks / 59.97 ms | 257 tasks / 39.74 ms |
| executor, 1-step bundle, kernel time, best / median / worst of 7 | 1045.1 / 1047.3 / 1065.3 ms | **862.0 / 867.5 / 872.7 ms** (-17.2% median) |
| final weights (all 197 tensors, 162,322,944 values) and loss after the step | reference | **bitwise identical** |
| activation pool | 4.8 GB (x1.56 reuse) | 7.5 GB (no reuse inside a step) |

The 5x goal for tasks and edges is **not reachable from the frontend** and was not reached: 95% of the tasks are matmul tiles
(247,128) and attention blocks (55,296), which the matmul-tile and attention tracks own, and the task count they imply is fixed by
the tile size. Everything else this track owns is 15,715 tasks (4.9%) before, 10,378 after; even deleting all of them would be 1.05x.
Edges are the lever the frontend has, and 2.0x of them is gone. How the time splits is in "What the time says".

## Tasks and in-edges per op class (1 step, `a100` tiling, `python3 -m kvmfe.inspect --model gpt_162m --steps 1 --tiling a100`)

Edges are counted at the consumer (a task's predecessors). `noop` = join tasks. Raw: `classes-base.json`, `classes-after.json`.

| device op | tasks before | tasks after | in-edges before | in-edges after |
|---|---:|---:|---:|---:|
| matmul_tile | 247,128 | 247,128 | 2,531,097 | 1,032,723 |
| noop | 5,524 | 4,203 | 1,537,443 | 328,227 |
| attn_bwd_dkv | 18,432 | 18,432 | 772,608 | 663,552 |
| attn_bwd_dq | 18,432 | 18,432 | 405,504 | 313,344 |
| adamw | 2,850 | 842 | 394,569 | 378,968 |
| gelu_bwd | 384 | 384 | 289,504 | 73,728 |
| attn_fwd | 18,432 | 18,432 | 258,048 | 258,048 |
| rope_bwd | 768 | 768 | 168,576 | 36,864 |
| layernorm_bwd_dx | 800 | 800 | 154,288 | 105,184 |
| attn_bwd_delta | 384 | 384 | 120,928 | 27,648 |
| gradnorm_partial | 2,850 | 842 | 55,222 | 45,101 |
| softmax_ce_bwd | 32 | 32 | 50,400 | 50,400 |
| softmax_ce_fwd | 32 | 32 | 50,336 | 50,336 |
| norm_bwd_params | 75 | 75 | 40,128 | 40,128 |
| colsum | 324 | 324 | 20,580 | 20,580 |
| layernorm_fwd | 800 | 800 | 18,464 | 18,464 |
| rope_fwd | 768 | 768 | 18,432 | 18,432 |
| gradnorm_finish | 1 | 1 | 2,850 | 842 |
| embedding_bwd | 25 | 25 | 88 | 88 |
| softmax_ce_reduce | 1 | 1 | 64 | 64 |
| counter_inc | 1 | 1 | 64 | 64 |
| embedding_fwd | 32 | 32 | 32 | 32 |
| select_window | 64 | 64 | 0 | 0 |
| **total** | 318,139 | 312,802 | 6,889,225 | 3,462,817 |

## What changed (all in `frontend/kvmfe`; no device code, no ABI change)

| Change | Effect on the 162M step | Why |
|---|---|---|
| 1. Joins per group, not per combination (`dataflow.py`) | 6.89M -> 6.42M edges | `sources()` cached a join per *combination* of predecessor groups. The LM-head optimizer chunks each needed the same 50,304 logits tiles plus a different handful of others, so 12 joins of 51,157 edges each were built. A large group (>= `join_min`) is now joined on its own and shared by every later reader/writer. |
| 2. Join of joins, equal sets share a join | 6.42M -> 5.86M | A chunk that needs 786 per-column joins needed all 786 edges again for each of 768 chunks. A predecessor set that is still large and is needed twice is now joined as a whole. Equal sets share one join (`join_of`). |
| 3. `Tiling.join_min` = 8 for `a100`/`b200` (was the module constant 32) | 5.86M -> 4.75M edges, +4k joins | The 12 producer tiles of a d768 row strip are read by 12-36 consumer tiles each; below 32 they were never joined. |
| 4. Splice joins that end with one consumer (`_splice_thin_joins`) | 4.75M -> 4.61M edges and 322,082 -> 315,170 tasks | A join is created at the second use of a set, so a set used exactly twice left a join with one consumer: one task, one edge and one dependency hop for nothing. |
| 5. Optimizer chunks of 262,144 elements, rounded down to whole matmul-tile rows (was 65,536, whole rows) | AdamW and grad-norm tasks 2,850 -> 842 each | One weight-gradient tile now feeds exactly one chunk; chunks of the same tile row no longer each repeat that row's fan-in. Wide matrices (the LM head, 50,304 columns) keep short chunks. |
| 6. Reuse policy `step`, default for training (`plan.py`) | 4.61M -> 3.46M edges, critical path 428 -> 257 tasks | Slot reuse inside a step turns into WAR/WAW edges and serializes: the model's critical path was 60.3 ms with reuse and 39.7 ms without. `step` lets a slot be reused only by a *later step*, so one step costs its whole activation footprint (7.5 GB here, against 1.9 GB of weights and state on a 40 GB card), and a window of S steps costs no more than one. `shape` and `none` stay available (`--reuse`); serving keeps `shape`. |

Also: `examples/models/run.sh` honours `KVM_REMOTE` and no longer stops after the first case (its `ssh` read the case list from stdin);
`frontend/kvmfe/train.py` read `repeat_batch` off every config, which broke 17 zoo-model tests on origin/main (Llama/MoE/MLP/ViT configs do not have it).

## Measurements (A100, kernel time of one `kvm_exec` launch, gpt_162m, 1 optimizer step)

Interleaved in one lock hold (`bench/vs_pytorch/ab_bundles.sh`), 7 launches per bundle, one process each, same `kvm_exec` binary (built from
the unchanged executor sources). Raw: `raw/ab1-runs.txt`, `raw/ab2-runs.txt`, `raw/ab3-runs.txt`.

| bundle | tasks | edges | best | median | worst (ms) | run |
|---|---:|---:|---:|---:|---:|---|
| origin/main | 318,139 | 6,889,225 | 1045.1 | 1047.3 | 1065.3 | ab2 |
| changes 1, 2, 5 (join_min 32), `shape` reuse | 312,802 | 5,724,599 | 976.5 | 981.0 | 1055.8 | ab1 |
| 1, 2, 3, 5 (join_min 8, before change 4) | 322,082 | 4,620,543 | 978.3 | 982.3 | 992.7 | ab1 |
| 1-5 (`shape` reuse) | 315,170 | 4,613,631 | 949.6 | 953.5 | 958.1 | ab2 |
| 1-6 (**final**) | 312,802 | 3,462,817 | **862.0** | **867.5** | **872.7** | ab2 |
| 1-6 + `--fuse` (`matmul_tile_dgelu`, opt-in) | 313,738 | 3,643,369 | 878.8 | 881.4 | 887.4 | ab3 |

* The final bundle is 17.2% faster than main (median over median) and the spread is 1.2% (ab2), the spread of main 1.9%.
* **Edges, not tasks, move the time.** Between the second and third rows tasks go up 3% (+9,280 joins), edges go down 19% (-1.10M) and
  the time does not move. Solving the marginal differences of the four `shape` rows gives about 4 us per task and 36 ns per edge;
  from main to final that is 5.3k tasks (22 ms) and 3.43M edges (123 ms) of the 180 ms saved, the rest being the shorter dependency
  chains of change 6. The dependency tracker is the cost, as the track brief said; the frontend can only make it smaller.
* `--fuse` is **not** a win here: it removes the 384 `gelu_bwd` tasks (and 7% of the modelled critical path) but the fused tile reads
  the pre-activation with a full fan-in, +180k edges, and the measured step is 1.5% slower. It stays opt-in. (Bias, GELU and the
  residual add were already in the `matmul_tile` epilogue; norms cannot join a matmul epilogue, a tile holds no full row.)
* What is left of the 867 ms is not this track's: the brief's admission floor of ~604 ms plus tile work. The remaining 3.46M edges
  are 1.03M into matmul tiles (4.2 per tile, the K-fan-in), 1.28M into attention (36 per `attn_bwd_dkv` task, causal triangle),
  379k into AdamW, 328k into joins.

## Numerics

* `bench/vs_pytorch/dump_weights.py` runs `kvm_exec --dump-buffers` on a bundle and keeps the 197 parameter tensors and the loss; the
  `--compare` mode diffs two such dumps. origin/main bundle against the final bundle, and against the changes-1-5 bundle:
  **0 of 162,322,946 values differ** (max |diff| 0.0), loss included, and the two repeat runs of the same bundle agree as well. This
  holds because the changes alter only the edge set and the buffer slots, never what a task computes; the grad-norm partial sums are
  regrouped by change 5, which is exact here because the global norm is below the clip threshold (coefficient 1); a run with clipping
  active was not made, so that regrouping is untested at the bit level (it changes a float32 summation order, ~1e-7 relative).
* CPU: `python3 -I -m unittest discover -s tests` (145) and `-s tests/frontend` (68, +4 new: join sharing, `join_min` argument,
  `step` reuse policy, optimizer chunk alignment) pass.

## GPU suites run on the final frontend (A100, same box, each in its own lock hold)

| Suite | Result | Raw |
|---|---|---|
| `examples/models/run.sh` (24 model-zoo bundles: MLP, ViT, Llama, GQA, MoE, fused, remat, accum, bf16 and the `san_*` set, generated by this frontend, `kvm_exec` vs the compiler's CPU golden) | **24/24 PASS** (worst per-step loss rel err 9.7e-7 at 3xTF32; the 1xTF32 cases at their 2e-3 / 6e-3 bounds) | `raw/models/` |
| `run_exec_tests.sh --gpt` (gpt_tiny, scheduler off / global / static / dynamic / arbiter: buffers byte-identical across all, memcheck / racecheck / synccheck) | **GPT PASS**, md5 identical in all 5 modes, 0 sanitizer errors | `raw/exec-gpt-run.txt` |
| `demo/run_demo.sh large` (gpt_162m, 6 steps in windows of 1, a serve request every 2, 700 Doom frames, one launch) | **ALL CHECKS PASSED**: 1,916,342 tasks, versions 22,935 exact / 0 violations, Doom 700/700 frames identical to the CPU oracle, 170.8 frames/s; loss 10.9790 -> 10.9237. Train steps 2-5 take 1.45-1.58 s each (mean **1.50 s**, a serve request in every second one) against 1.81 s in `results/2026-11-06-demo-large` (-17%). Total kernel time 11.24 s. | `raw/demo-large/` |
| `run_gpu_ops.sh` (GPT op pack vs CPU doubles) | **not run**: this branch changes no `.cu`/`.cuh`/`.def` file, the op bodies are the ones already measured in `2026-11-04-a100-ops-perf` | - |

The brief's `bench/exec_losses.py` compares per-step losses with a PyTorch run; none was made for this tree. The check made instead is
stronger for this change: the step-1 loss and every updated weight are bit-identical to origin/main's, whose step-1 loss and 196 gradients
match torch-tf32 and the float64 reference (`results/2026-11-06-gpt162-vs-torch`, origin/main).

## Reproduce

```bash
python3 -m kvmfe.inspect --model gpt_162m --steps 1 --tiling a100            # from frontend/; --reuse shape for the old memory plan
python3 bench/vs_pytorch/gen.py w_162m --model gpt_162m --steps 1 --mode tf32 --tiling a100     # on the GPU box
KVM_HOST=... flock -w 7200 ~/.gpu.lock bash bench/vs_pytorch/ab_bundles.sh 7 OUT w_base/bundle w_new/bundle   # run from the dir holding ./kvm_exec
python3 bench/vs_pytorch/dump_weights.py ./kvm_exec w_new/bundle dump_new && python3 bench/vs_pytorch/dump_weights.py --compare dump_base dump_new
```

Environment: NVIDIA A100-SXM4-40GB (108 SMs, SM clock max 1410 MHz), driver 580.105.08, CUDA 12.8 (`nvcc` 12.8.r12.8), `kvm_exec` built with
`nvcc -O3 -arch=sm_80 -std=c++17 -Iinclude --fmad=false -Xcompiler -ffp-contract=off kvm_exec.cu`, bundles `--mode tf32 --tiling a100`.
The GPU was shared with other agents (lock-serialized, other builds running on the host's CPUs), commit of the frontend: see the branch tip.

## Where the next factor is (not done)

1. Attention edges (1.28M, 37%): a causal `attn_bwd_dkv` task reads every later query block, a triangle. The attention track's
   multi-block tasks will cut the task count; chained suffix joins (`j_i = {fwd_i, j_{i+1}}`) would turn the triangle's O(n^2) edges
   into O(n) at the cost of a dependency chain per head.
2. The LM-head optimizer: 154 short chunks of the 768 x 50,304 matrix read 786 gradient tiles each through joins (95k edges). One chunk
   per tile row would halve that and make 12 tasks of 3.2M elements (~1.5 ms on one SM): a tail-latency trade, not measured.
3. A counter-barrier opcode would replace a join's task (one hop, one admission) by a counter in the consumer: the executor's lane.
