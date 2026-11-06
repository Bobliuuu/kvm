# Where a persistent-kernel VM wins or loses (A100), first measurements

GPU: NVIDIA A100-SXM4-40GB, 108 SMs (107 workers + 1 admitter), driver 580.105.08, CUDA 12.8 (nvcc), torch 2.7.0. Shared box: ratios are from one process, back to back, best of 5.
Code commit for the runs: 2313da6 + 9830790 (branch `bench/vm-wins`; the runs used the tree at 2313da6, i.e. before the predicated-task commit, which does not touch these paths).
Build flags: `-O3 -arch=sm_80 -std=c++17 -Iinclude --fmad=false -Xcompiler -ffp-contract=off -diag-suppress 177` (`chain` adds `-DKVM_CHAIN`; hop_bench adds `-DKVM_EXEC_NO_MAIN`).

## Regime 1: R independent runs of the audit's `mlp_ce` (50 SGD steps each), us per step for ALL R runs
Command: `python3 bench/regimes/run_runs.py WORK --steps 50 --runs 1,2,4,8,16,32 --reps 5 --vm base=./kvm_exec_base --vm chain=./kvm_exec_chain --vm-args "--live-cap 131072 --ring-words 8388608 --max-tasks 262144"` (`runs_a100tiling.txt`).
Bundles from `bench/regimes/gen_mlp.py --tiling a100` (51 tasks/step; the audit used the `cpu` tiling, 138 tasks/step). R copies via `replicate.py` (no shared edges).
Correctness gate passed first, for both executors: single run vs CPU golden (rel L2 <= 1e-3), and 4 replicated copies bitwise equal to the single run in all 79 buffers.

| R | batched (vmap-like graph) | R graphs, 1 stream | R graphs, R streams | eager | VM base | VM chain |
|---|---|---|---|---|---|---|
| 1 | 124.7 | 109.2 | 109.9 | 604.6 | 399.5 | 376.7 |
| 2 | 106.2 | 205.8 | 115.1 | 1162 | 417.5 | 379.4 |
| 4 | 114.2 | 412.1 | 150.8 | 2636 | 494.2 | 426.4 |
| 8 | 131.4 | 821.9 | 285.3 | - | 754.0 | 617.4 |
| 16 | 156.2 | 1665 | 570.2 | - | 1453 | 1039 |
| 32 | 237.8 | 3435 | 1106 | - | 2896 | 2125 |

VM (chain) vs baseline: WINS vs PyTorch eager (1.6x at R=1, 6.2x at R=4); WINS vs sequential per-run graphs from R=8 (1.3x at 8, 1.6x at 16 and 32); LOSES to the R-stream graph (3.4x slower at R=1, 1.9x at R=32) and to the batched graph (3x at R=1, 8.9x at R=32). The batched graph needs identical runs; the VM does not, but heterogeneous runs were not measured (not done).
Versus the audit: at R=1 the VM lost to the whole-step graph 5.4x on 138 tasks/step; with a100 tiling it is 3.7x (base) / 3.4x (chain). `KVM_CHAIN` (inline successor hand-off) gives 6% at R=1 and 27% at R=16-32; it is opt-in and `test_exec` was NOT run on it (only the golden/bitwise gate above).

### Why (measured)
1. **Admission windows serialized the copies.** With default `ring_words` a bundle record may hold only ring/2 = 32768 words, about 1080 tasks, so a 40,800-task bundle was streamed in ~38 windows (trace: windows of 1083 tasks admitted every ~6.5 ms) and average concurrency was 1.74 tasks. Raising `--ring-words/--live-cap` so the whole set is one record cut R=16 from 5118 to 1453 us/step (3.5x). `runs_default_streamed.txt` has the unfixed numbers. Anyone running many bundles must size these.
2. **Task granularity.** 32 of 51 tasks per step are 1-CTA GEMM tiles (mean 11.6 us each). The marginal cost of one more run is 41-47 us/step = 51 tasks at 0.8-0.9 us, i.e. the executor's ~0.75M tasks/s dispatch ceiling, while a graph pays ~0.7 us per node for the same work but gets the tile fan-out of a kernel for free (hardware CTA launch). So the VM pays per tile what a graph pays per kernel. Fix to try: multi-tile tasks (one task loops over a tile range) and the frontend's fusion; not implemented.
3. Dependent hop: `hop_chain.txt` (KVM_CHAIN executor only; a base-executor micro was not obtained because that build was OOM-killed on the loaded box, so the hop saving is NOT measured in isolation): 9.1 us per dependent `add` task for a single chain, 0.83 us per task at R=16 and 0.44 us per task at R=108 chains. Against ordinary CUDA (same micro): per-op launches 2.6 us and graph nodes 1.2 us per step at R=1 (the VM is 7x slower per dependent step), but at R=108 independent chains the VM beats per-op launches 6.0x and one sequential graph 2.8x, and loses to a graph with 108 parallel branches (2.3x) and to a batched kernel (30x). Gate: every final buffer exactly L.

## Not done / blocked
Data-dependent control flow (predicated tasks `KVM_TF_SKIP_IF_*` are implemented in `include/kvm_exec.cuh` but untested and unbenchmarked; skipped tasks still cost a hop, so unrolled predication saves nothing: a real device-side loop-back is needed; CUDA 12.8 conditional WHILE nodes would be the baseline to beat), heterogeneous runs, train+serve co-residence, step overlap, torch.compile, H100/B200 (Modal launch was denied by the permission layer; nothing was launched or spent).
