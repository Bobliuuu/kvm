# Fewer, larger tasks: measured (A100-SXM4-40GB, shared box; one process, ratios within one lock hold)

Models: audit `newmodel` cases mlp_ce (200 steps) and xfmr_odd (50 steps), SGD, 1xTF32. Raw log: `logs/bench-o2-time.txt`, counts `logs/bench-o2-counts.txt`.
Script: `bench_job.sh`, `count.py` (task counts / bundles). Best of 5 launches of the whole run.

| variant | mlp tasks/step | mlp us/step | xfmr tasks/step | xfmr us/step |
|---|---|---|---|---|
| audit setup: `Tiling.preset('cpu')`, no fusion | 138 | 456 | 873 | 2364 |
| `a100` preset (row blocks 128, opt chunk 64k), fusion pass | 49 | **339** (1.35x) | 299 | **2054** (1.15x) |
| + GEMM tile merging (128x128 tile when GEMM has <= 32 64-tiles) | 26 | 447 (slower) | 240 | 2560 (slower) |
| PyTorch whole-step CUDA graph, same process | | 114 | | 981 |

VM / CUDA graph: mlp 4.0x -> 3.0x, xfmr 2.4x -> 2.1x (the audit's 5.4x / 3.9x came from a loaded box and the cpu tiling).

## Findings
1. The audit generated its bundles with the tiny `cpu` tiling preset (8-row blocks, 256-element optimizer chunks). The default `a100` preset plus the existing
   fusion pass (`fuse.py`, already in `compile_training(fuse=True)`) cuts tasks 2.8x (mlp 138 -> 49, xfmr 873 -> 299) and runs 1.35x / 1.15x faster. Free win, no code change;
   the audit's gap numbers are partly a tiling-choice artifact.
2. Fewer tasks is NOT the lever once the tiles of a GEMM are parallel: merging a GEMM's 64x64 tiles into one 128x128 tile cut tasks another 1.9x / 1.25x and made the run
   SLOWER (447 vs 339 us/step). Per step the VM is close to a serial chain (mlp: 49 tasks, 339 us); a bigger tile lengthens its link without removing a hop. Only fusion of
   tasks that sit ON the dependency chain (epilogues, consecutive dependent row-wise ops) can help; parallel siblings cannot.
3. Kept as opt-in (`Tiling.mm_merge`, default 0 in every preset): matmul_tile flag 32 (128x128 tile) through lowering, the CPU references (`runtime/ref_gpt.py`,
   `ref_extra.py`, `frontend/kvmfe/opset.py` now honour tile size from flags) and the epilogue wrappers (`matmul_tile_dgelu/silu_mul/silu_mul_bwd` accept flag 32 in
   `include/kvm_ops_extra.cuh`). GPU: `test_extra` 2568 passed with 480 new big-tile wrapper checks, all bitwise vs matmul_tile + the elementwise op; the single failure was the
   stale "wrappers reject flag 32" expectation, now flag 64.
4. Next (not done): chain fusions need new device ops: softmax_ce fwd+bwd in one task, rmsnorm into the GEMM tile when N <= 64 (xfmr d=48), SGD into the dW GEMM epilogue
   (alpha=-lr, accumulate into W; bitwise equal, but it must respect the weight-version barriers), and dropping the two no-op barrier tasks per step when nothing serves.

## Tests
`tests/frontend/test_fusion_tiles.py`: cpu-tiling/no-fusion baseline vs a100 tiling, fusion, and merged tiles (SGD and AdamW), program and random topological order: weights and losses BITWISE equal on the CPU golden VM, task count strictly lower.
Python suite 145 OK, frontend suite 82 OK (CPU, box). GPU-vs-golden and executor functional suite: see RUNS.md.
