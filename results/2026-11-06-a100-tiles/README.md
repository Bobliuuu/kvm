# gpt_162m tiles: task count and matmul tile (A100, 2026-11-06/07) - PARTIAL, see "Not done"

GPU: A100-SXM4-40GB (108 SMs), driver 580.105.08, CUDA 12.8, shared lab box. nvcc -O2/-O3 -arch=sm_80, executor flags as in results/2026-11-06-gpt162-vs-torch. Branch perf/tiles-162m.
Commands: tests/gpu_ops/bench_mm.cu (MM_GPT162=1, MM_RZ=1), tests/gpu_ops/bench_tile.cu, bench/vs_pytorch/tiles162.sh, bench/vs_pytorch/gen.py --tiling a100|a100_mm64|a100_trunc.

## Results
- Tasks per 162M step: 318,139 (a100_mm64) -> 88,226 (preset a100: 128x128 GEMM tiles + 4-block attention, from perf/frontend-bigtile).
- Executor step (1-step tape, best of 5, two interleaved rounds, one process family, MIN_BLOCKS=1 build): 1041-1044 ms (64x64 tiles) -> 366-367 ms (a100) -> 355-356 ms (a100_trunc). MIN_BLOCKS=2 is worse (440 / 414 ms). Torch: 58.8-69.6 ms. (These runs used an all-out-of-line tile build; final inline build not re-timed.)
- Trace of the 366 ms run: 38.7% worker utilisation, mean ready->start wait 1.3 ms, matmul 63.6% of task time (127 us mean): the step is now scheduling/admission bound (88k tasks x 1.9 us = 168 ms floor), not tile bound; truncation buys only 3% end to end.
- Matmul TFLOP/s (raw/mm_table.txt, same process as cuBLAS TF32): default rounding path unchanged, 4096^3 NN/TN/NT 68.6/60.1/70.8 vs cuBLAS 128-131 (0.54/0.47/0.54). New opt-in flag 256 (truncated operands, cuBLAS TF32 behaviour): 91.5/81.5/93.2 = 0.72/0.64/0.73 (target 0.70 met for NN/NT). 162M shapes: rounding 0.35-0.49x, truncation 0.46-0.64x (K=50304 NT 0.45 is under-filled).
- Numerics: truncation error vs cuBLAS fp32 at 4096^3: 6.4e-4 relative (rounding 2.6e-4), inside the existing 6e-3 TF32 tolerance. test_ops 2274 pass/0 fail incl. truncated cases and NaN propagation, memcheck 0 errors (raw/). NOT measured: 162M gradients vs float64 with flag 256 (default stays rounding).
## What did not work (profile-first, ablation of the 128x128 tile 1.856 ms: no mma 1.67, no fragment loads 1.55, no cp.async 1.50, no rounding 1.44)
- Operand rounding costs 22%: that is the lever (flag 256). Software-pipelined main loop: slower (62 vs 76). BK32 / 3-5 stages: no gain. A-only/B-only shared rounding: mixed, dropped. 2-op max() rounding: wrong for canonical NaN, caught by test_ops, reverted. A pass loop around the tile spilled 28 KB (-25%); out-of-line only for the 256x128 task.
## Not done
Final-build re-timing, MIN_BLOCKS=2 executor suites, 162M numerics with flag 256, demo large / Doom 35 Hz.
Proposed text for status tables: "matmul_tile flag 256 (truncated TF32 operands) 0.64-0.73x cuBLAS at 4096^3; rounding path 0.47-0.54x".
