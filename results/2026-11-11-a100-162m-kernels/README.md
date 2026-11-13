# bf16 tensor-core causal attention for the 162M GPT step (A100)

Branch `perf/kernels-162m`. Op files only: `include/kvm_attn_bf16.cuh` (new), `include/kvm_ops_gpt.cuh` (dispatch, `attn_bwd_delta`),
`tests/gpu_ops/{bench_attn,test_bf16,bench_mma_bf16}.cu`.

**Verdict (micro-benchmark, whole step pending below).** At the gpt_162m attention shape (B=4, T=1024, H=12, hd=64, q/k/v/dO/outputs in bf16
storage, the `bf16s` step) one layer's attention costs **1,762 -> 419 us** (forward 330 -> 93, backward 1,432 -> 326, one task per block, 384 tasks
per op). Over 12 layers that is 21.1 -> 5.0 ms of GPU time per step, 16.1 ms saved. Every case in `test_bf16`, `test_ops` passes.

## What changed

* `attn_fwd`, `attn_bwd_dq`, `attn_bwd_dkv` take a new path when q, k, v (and dO for the backward) are raw bf16 storage, 16-byte aligned with
  ld a multiple of 8, hd = 64: `mma.sync.m16n8k16` bf16 with fp32 accumulate, operands from swizzled bf16 shared tiles through `ldmatrix`
  (`.trans` for V and K as the second operand), 64-key tiles in a 3-stage `cp.async` pipeline, P / dS packed to bf16 straight from the
  accumulator registers. Same task geometry as before (128 rows per task, one m16 tile per warp, no cross-warp reduction). The old
  3xTF32 path computed three products of values whose low halves are zero when the inputs are bf16; everything else (fp32 operands, hd != 64)
  still takes it. `-DKVM_GPT_ATTN_BF16=0` turns the new path off.
* Numerics: products of bf16 values are exact in fp32; the new path rounds P (forward, dV) and dS (dQ, dK) to bf16 before the tensor
  cores, as every bf16 flash attention does. Results are therefore not bitwise equal to the 3xTF32 path; `test_bf16` `t_attn_tc` holds them
  to the float64 reference (o 1.2e-2, lse 1e-5, dq/dk/dv 1e-2 relative to the maximum; measured o 1.6e-3, dq 2.0e-3, dk 2.1e-3, dv 2.6e-3,
  lse 1.2e-7) for ragged T (1, 70, 150, 257), GQA (Hkv < H) and every bf16/fp32 storage mix of the operands.
* `attn_bwd_delta`: `hd/8` lanes per (row, head) each load 8 elements as one 16-byte piece (the scalar loop kept a few KB in flight per SM and was
  latency bound); one summation order for fp32 and bf16 storage, so they stay bitwise equal (`test_bf16` delta checks).

## Setup

NVIDIA A100-SXM4-40GB, 108 SMs, 400 W, driver 580.105.08, CUDA 12.8.93 (`raw/env_gpu.txt`, `raw/env_cuda.txt`), clocks 1,410 MHz at rest; the box is
shared and clocks moved between runs (rounds 1 and 4 of the base ran slow: 422 / 559 us), so the table is the median of 5 interleaved rounds
(base, new, base, new, ...), each round `BENCH_REPS=9`. Commit: the branch head when this README was written (see git log).

```
nvcc -O2 -arch=sm_80 -std=c++17 -lineinfo -I<main include> -I. bench_attn.cu -o bench_attn_base      # origin/main 49f2a1a headers
nvcc -O2 -arch=sm_80 -std=c++17 -lineinfo -I../../include -I. bench_attn.cu -o bench_attn_new        # this branch
BENCH_BF16=1 BENCH_REPS=9 ./bench_attn_{base,new}
```

`bench_attn` checks every kernel against the float64 reference before timing (bf16 mode: tolerances 1e-2 / 2e-2 for one bf16 rounding of the
outputs); per-round output in `raw/attn_{base,new}_r{1..5}.txt`.

| op (4 x 1024 x 12 heads x 64, bf16 storage) | base us | new us | x | TFLOP/s base -> new (full-square count) |
|---|---|---|---|---|
| attn_fwd (384 tasks) | 330.1 | 92.6 | 3.56 | 19.5 -> 69.4 |
| attn_bwd_delta (32 tasks) | 162.2 | 42.0 | 3.86 | |
| attn_bwd_dq (384 tasks) | 438.7 | 107.9 | 4.07 | 18.4 -> 74.6 |
| attn_bwd_dkv (384 tasks) | 692.4 | 160.8 | 4.31 | 11.6 -> 50.1 |
| backward (delta + dq + dkv) | 1,431.6 | 325.8 | 4.39 | 11.2 -> 49.4 |

(base and new from the median round; the three backward kernels run back to back in the last row, so it is below the sum of the rows.)

## Tests

| suite | result | log |
|---|---|---|
| `test_bf16` (A100, `--fmad=false`): 953 checks including the new `attn tc` cases | 953 passed, 0 failed | `raw/test_bf16.log` |
| `test_bf16` under compute-sanitizer memcheck, racecheck, synccheck (`TEST_BF16_SMALL=1`) | 511 passed each, 0 errors, 0 hazards | (this README) |
| `test_ops` (A100, every case, hd=128 included) | 5,403 passed, 0 failed | `raw/test_ops.log` |

## Dropped (measured, no win)

* A mask-free fast path for key tiles entirely below the diagonal in the forward: inside the run-to-run spread (107-119 us against 110-119 us), removed.
* `attn_bwd_delta` with more (row, head) pairs in flight per warp on the scalar loop (U = 4/8/16/32): no change; the 16-byte-load form is the fix.
* GEMM tile variants for `matmul_mma_bf16` at 4096^3 and K = 16,384 (BK = 64 with 3 stages, 3 and 5 stages, spread issue): all within 0.65-0.70x
  of cuBLAS, the default (BK = 32, 4 stages) is as good. Removing the per-slab `__syncthreads` (wrong results, ablation only) read +15% at
  K = 16,384 (184 -> 217 TFLOP/s) and removing the `cp.async` issue or the `ldmatrix` each read +5-10% at 4096^3: the tile's gap to cuBLAS is spread
  over its main loop, not one stall; no change kept.
