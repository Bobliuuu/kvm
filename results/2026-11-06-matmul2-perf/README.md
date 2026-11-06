# Matmul and occupancy work, perf/matmul2 (2026-11-06)

Goal: matmul at >= 70% of cuBLAS TF32 (was 0.34-0.51x), fix the 255-register / one-block-per-SM whole-kernel build, keep every test green.

## Setup

- GPU: Modal `A100-40GB` (NVIDIA A100-SXM4-40GB, 108 SMs, 1410 MHz max SM clock), driver 610.57.04, CUDA 12.8.61, `nvcc -O2 -arch=sm_80 -std=c++17 -lineinfo`
  (some correctness runs landed on A100-80GB containers). The shared lab box was held by a long job, so nothing here ran on it.
- Estimated Modal spend: about 25 container runs of 10-60 min, mostly nvcc time, roughly 10-15 USD (A100-40GB list price, not read from a bill).
- Benchmark: `tests/gpu_ops/bench_mm.cu`, op library `matmul_tile` against cuBLAS `cublasSgemm` (TF32 tensor-op math) in ONE process, median of 5
  repetitions x 10 launches, 4096^3 unless stated, one CTA per output tile, tiles launched row-major. cuBLAS on Modal drifts 105-136 TFLOP/s
  between containers, so compare the ratio column.
- Commands: `modal run --detach tests/gpu_ops/modal_bench.py --what mm --nvflags "-DMM_MINB=2"` (after), same on the base commit 6d31aba plus
  `bench_mm.cu` (before; `MM_MINB=1`, the old single-block build). Tests: `--what test_ops --nvflags "-DBENCH_MINB=2"`, memcheck via `--wrap "compute-sanitizer --tool memcheck --error-exitcode 9"`.
- Code: branch `perf/matmul2`, `include/kvm_ops_gpt.cuh`, `include/kvm_exec.cuh`, `tests/gpu_ops/{bench_mm.cu,bench_ops.cu,test_ops.cu,modal_bench.py,dev_bench.sh}`.
- Raw outputs next to this file: `before_bench_mm.txt`, `after_bench_mm.txt` (two variants: `MM_MINB=2` = the new default, `MM_MINB=1`), `attn_minb1_vs_minb2.txt`,
  `test_ops_minb2.txt`, `test_ops_memcheck.txt`, `sweep_bk_nst_before_cvt_fix.txt` (an intermediate sweep, see "what failed").

## Results: TFLOP/s of matmul_tile, same process as cuBLAS TF32

| shape / tile | before | after (default build) | cuBLAS (same run) | ratio before -> after |
|---|---|---|---|---|
| NN 4096^3, 128x128 tile (flag 32) | 42.8 | 68.6 | 131 | 0.33 -> 0.52 |
| NT 4096^3, 128x128 | 44.4 | 71.1 | 131 | 0.34 -> 0.54 |
| TN 4096^3, 128x128 | 43.0 | 60.2 | 130 | 0.34 -> 0.46 |
| NN 4096^3, 256x128 task (flag 64) | 50.0 | 61.2 | 130 | 0.39 -> 0.47 |
| NN 4096^3, 64x64 (what `frontend/` emits today) | 23.5 | 30.5 | 106-136 (noisy) | 0.18 -> 0.29 (NT 36 -> 41) |
| NN 8192x768x768, 128 / 256x128 | 31.6 / 28.5 | 47.5 / 49.5 | 103 | 0.31 -> 0.46, 0.28 -> 0.48 |
| NN 4096x3072x768, 128 / 256x128 | 33.0 / 28.7 | 51.6 / 46.2 | 108 | 0.31 -> 0.48, 0.27 -> 0.43 |
| NN 2048^3, 128 | 32.6 | 40.5 | 101 | 0.33 -> 0.40 |
| NN 1024^3, 64 / 128 | 21.0 / 21.9 | 25.1 / 23.0 | 66 | 0.32 -> 0.38 / 0.33 -> 0.35 |
| NN 512^3, 64 | 12.2 | 11.5 | 23 | 0.52 -> 0.50 (128/256 tiles waste most of the 16 CTAs) |
| 3xTF32 NN 4096^3, 128 (cuBLAS fp32 19.0) | 24.8 | 28.5 | 19.0 | 1.31x -> 1.50x |
| **bf16 operands, NN 4096^3, 128x128 (new, flag 128)** | n/a | **111.8** | 128 (TF32) | **0.87** |
| bf16 NT / TN 4096^3, 128x128 | n/a | 109.6 / 90.5 | 136 / 128 | 0.81 / 0.71 |
| bf16 NN 4096^3, 256x128 | n/a | 98.6 | 132 | 0.75 |

**The 70% target is met only by the new bf16-operand path** (relative error vs fp32 2e-3 at K=4096, tolerance 3e-2, a different precision tier, opt-in by flag).
The TF32 path went from 0.33 to 0.52-0.54x at its best (about 1.6x faster) and did NOT reach 0.70.

## What changed

1. `__launch_bounds__(256, KVM_MIN_BLOCKS)` on the executor kernel. The macro defaults to 1 (unchanged behaviour) because the executor suites could NOT be run at 2
   (see below); `-DKVM_MIN_BLOCKS=2` is the intended setting (128 registers, the grid is occupancy sized so it doubles to 2 x SMs). The op harness kernel (`k_run`, all ops) was
   measured and tested at 2. This was the largest single lever: the same old matmul code went 42.8 -> 60.6 TFLOP/s (128 tile) just from two resident blocks.
   `bench_ops` `BENCH_MINB=2` on attention (`attn_minb1_vs_minb2.txt`, no attention code changed): backward B4 T2048 H16 Hkv4 hd64 6.81 -> 5.46 ms (+25%),
   backward B4 T1024 hd64 2.91 -> 2.34 ms (+24%), forward T1024 0.919 -> 0.970 ms (-5%), forward T2048 unchanged. hd=128 (not in the default union) is 12-25% slower.
2. 256x128 tile (flag 64) is now two 128x128 passes inside one task (4x2 warps of 64x64 need 128 accumulators, which cannot fit 128 registers); same task shape and ABI.
3. Operands are rounded to tf32 ONCE per CTA in shared memory (`tile_cvt`/`TileStream::cvt`: each thread rewrites the chunks it copied) instead of once per
   fragment per warp. Bitwise the same values as before (same rounding, NaN-safe), so results are unchanged; +22% at 2 blocks/SM (49.4 -> 60.6).
   (It is net-negative for the 64x64 tile at one block per SM: 23.5 -> 21.6; at two blocks it is positive, 30.5.)
4. `TileStream`: per-thread cp.async pointers computed once; fast path when whole tiles are in range, 16-byte aligned and K % BK == 0, else the old per-chunk path.
5. Pipeline: 4 stages of BK=16 (64 KiB; `kvm_gpt_smem_bytes` now sized for 256 tile rows, not 384). Two blocks per SM still fit.
6. bf16-operand matmul (flag 128, 128x128 / 256x128 and the 64 tile): fp32 tiles in shared memory, fragments loaded with the same ldmatrix / scalar loads as the TF32 path,
   packed with `cvt.rn.bf16x2`, `mma.m16n8k16`. The (k, k+4) pairs of each k8 half become one bf16x2, i.e. both operands use the same k permutation, which a sum over k ignores.
   Shared-memory traffic equals the TF32 path's while the tensor work per byte doubles. Accumulation is fp32.

## Why TF32 stops near 0.5x (analysis, not profiled: no ncu on the machines used)

At 2 blocks per SM the budget is 128 registers, so a warp tile is 64x32 (64 accumulators). Per m16n8k8 mma that tile reads 192 B of operands from shared memory,
against 128 B/clk/SM of shared bandwidth and one mma per clock per SM, before counting cp.async writes and the conversion pass: shared-memory bound at roughly
half of peak. A 64x64 warp tile (128 B/mma) needs 128 accumulators, hence 255 registers and one block per SM, which the executor wants to leave. bf16 halves the
shared traffic per FLOP, which is why it reaches 0.87. Measured, not modelled: 128 tile NN 49.4 (1 block, per-fragment convert) -> 60.6 (2 blocks, smem convert) -> 68.6 (+TileStream, 4 stages).

## Correctness

- `test_ops` (harness kernel built with `__launch_bounds__(256, 2)`): **1979 passed, 0 failed** (was 1396 on this branch's base; new cases: shapes that reach the cp.async fast path,
  bf16 flag for every tile, NaN propagation for every tile and precision). `compute-sanitizer --tool memcheck`: 1979 passed, 0 errors (`test_ops_memcheck.txt`).
- Golden tests: no tolerance changed. TF32 and 3xTF32 results are bitwise those of the old rounding.
- NOT run (jobs hit Modal's 60 min timeout or were stopped after 45 min of nvcc time): `test_extra` (2089), `test_exec` suites and the gpt_tiny bundle through `kvm_exec` at `KVM_MIN_BLOCKS=2`,
  racecheck/synccheck. `kvm_exec.cu` takes more than 40 min to compile on a Modal container at -O3. So the executor default stays at 1 until someone runs `tests/gpu/*` and `run_exec_tests.sh` with `-DKVM_MIN_BLOCKS=2`.

## What failed / did not help

- Deeper pipelines (BK16 x 4 or 6 stages, BK32 x 3) at ONE block per SM: no gain (41-43 TFLOP/s). The limit was occupancy, not latency. First hypothesis wrong.
- Rounding in shared memory alone, at one block per SM: no change (42.4 vs 42.8) and the 64 tile got slower; it only pays at two blocks per SM.
- `BK=16 x 4 stages` looked 25% SLOWER in an early sweep (46 vs 64): the smem sizing function still reserved 96 KiB, so only one block fit. Fixed by the sizing change.
- `MM_MINB=3` (80 registers): 17.8 TFLOP/s, 12 KB of spills. Not usable.
- Not done: attention changes (only the occupancy effect above), wiring the 128 tile into `frontend/` (it still emits 64x64 tasks, `MM_TILE = 64`, so real
  models see 0.29x not 0.52x), a 64x64-warp-tile TF32 kernel for the 255-register build, TF32 at >= 0.70.
- The lab GPU box was unavailable (its lock held by another long job), so Modal replaced it. Modal's cuBLAS numbers vary 105-136 TFLOP/s between containers.
