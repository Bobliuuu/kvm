# Matmul tiles toward cuBLAS parity, A100 (2026-11-09)

Branch `perf/matmul-parity`. Op kernels and the frontend's tile choice only; executor and scheduler code are untouched.

**Result.** gpt_162m f32, 1 step through `kvm_exec`, 7 interleaved rounds in one container: **755.8 ms on main -> 251.5 ms (3.0x)**,
because the a100 preset now emits 256 x 128 TF32 tiles (52,615 tasks per step instead of 270,914) and those tiles run 1.6-2.2x faster
than before. The bf16 GEMM tile went from 0.46-0.48x to 0.69-0.75x cuBLAS bf16 at 4096^3 (0.77x at 8192^3). TF32 tiles inside the op
kernel reach 0.61-0.65x cuBLAS TF32 at 4096^3 (main: 0.37x). **The 0.8x target was not reached inside the op kernel**; the same tile
launched as its own kernel measured 0.70-0.80x (see "What limits it").

## Setup

| | |
|---|---|
| GPU | Modal NVIDIA A100-SXM4-40GB, driver 610.57.04, max SM clock 1410 MHz; nvcc 12.8.93 (`raw/env.txt`, `raw/steps162_env.txt`) |
| Benches | `tests/gpu_ops/bench_mm.cu` (TF32 `matmul_tile` vs `cublasSgemm` TF32 tensor-op math), `tests/gpu_ops/bench_mma_bf16.cu` (bf16 tile vs `cublasGemmEx` bf16), both in ONE process with cuBLAS, median of 5-7, every row checked against cuBLAS (all `ok`). Main and this branch built and run in the same container, one after the other. Build: `nvcc -O2 -arch=sm_80 -std=c++17 --fmad=false -Xcompiler -ffp-contract=off -lineinfo` |
| Step time | the method of `results/2026-11-06-a100-combined`: `bench/vs_pytorch/gen.py --model gpt_162m --steps 1 --mode tf32 --tiling a100` on both trees, `kvm_exec -O3` built from both trees in one container, 7 rounds interleaved (`raw/steps162_ab_runs.txt`) |
| Commits | base main `d12c206`; this branch's head is the PR's |

## TF32 `matmul_tile` (raw: `raw/tf32_main.txt`, `raw/tf32_new.txt`), TFLOP/s and ratio to cuBLAS TF32 in the same process

| shape | tile | main | this branch |
|---|---|---|---|
| NN 4096^3 | 256x128 | 50.0 (0.37) | 78.2 (0.61) |
| TN 4096^3 | 256x128 | 48.1 (0.37) | 80.5 (0.63) |
| NT 4096^3 | 256x128 | 49.6 (0.37) | 84.5 (0.65) |
| NN 4096^3 | 128x128 | 43.1 (0.33) | 69.3 (0.54) |
| NN 4096^3 | 64x64 | 30.2 (0.22) | 32.9 (0.25) |
| gpt_162m qkv fwd NN 4096x2304x768 | 256x128 | 29.9 (0.28) | 60.4 (0.57) |
| qkv dgrad NT 4096x768x2304 | 256x128 | 41.5 (0.35) | 74.0 (0.62) |
| fc1 fwd NN 4096x3072x768 | 256x128 | 29.5 (0.27) | 61.4 (0.56) |
| fc2 fwd NN 4096x768x3072 | 256x128 | 44.6 (0.36) | 71.0 (0.58) |
| fc1 dgrad NT 4096x768x3072 | 256x128 | 44.0 (0.36) | 76.2 (0.62) |
| fc1 wgrad TN 768x3072x4096 | 256x128 | 34.0 (0.29) | 56.4 (0.48) |
| logits fwd NN 4096x50304x768 | 256x128 | 32.2 (0.28) | 70.0 (0.58) |
| logits dgrad NT 4096x768x50304 | 256x128 | 53.4 (0.49) | 83.4 (0.77) |
| logits wgrad TN 768x50304x4096 | 256x128 | 50.6 (0.40) | 83.7 (0.61) |
| proj wgrad TN 768x768x4096 (18 tiles on 108 SMs) | 256x128 | 8.5 (0.10) | 14.2 (0.17) |

Numerics are unchanged: the TF32 result is bitwise the result of cuBLAS TF32 on 4096^3 (max |diff| 0), and on the test data of
`test_ops` this branch reproduces main's per-case errors exactly (same relerr on the same inputs).

## bf16 `matmul_mma_bf16` (raw: `raw/bf16_main.txt`, `raw/bf16_new.txt`), ratio to cuBLAS bf16

| GEMM | main | this branch |
|---|---|---|
| 4096^3 NN / TN / NT / TT | 127.6 / 123.6 / 123.0 / 123.3 (0.46-0.48) | 192.8 / 185.0 / 179.3 / 187.6 (0.69-0.75) |
| 8192^3 NN | 170.2 (0.65) | 203.6 (0.77) |
| qkv fwd 4096x768x768 NN | 64.9 (0.42) | 109.5 (0.70) |
| fc1 fwd 4096x3072x768 NN | 71.5 (0.36) | 122.3 (0.62) |
| fc2 fwd 4096x768x3072 NN | 102.2 (0.44) | 168.7 (0.73) |
| logits fwd 4096x50304x768 NN | 46.1 (0.21) | 116.8 (0.55) |
| logits dgrad 4096x768x50304 NT | 169.4 (0.62) | 166.2 (0.64) |
| qkv wgrad 768x768x4096 TN | 26.8 (0.18) | 32.0 (0.21) |

## gpt_162m step (raw: `raw/steps162_ab_runs.txt`, `raw/steps162_bundle_stats.txt`)

| run | tasks | edges | ms/step best / median / worst |
|---|---|---|---|
| main executor, main f32 bundle (64x64 TF32 tiles) | 270,914 | 2,251,649 | 739.6 / 755.8 / 762.4 |
| this branch, f32 bundle (256x128 TF32 tiles) | 52,615 | 562,511 | 249.4 / 251.5 / 257.1 |
| this branch's executor, main's f32 bundle | 270,914 | 2,251,649 | 742.1 / 745.1 / 756.2 |

The third row shows the executor build is not slower on identical work; the gain is the bundle (fewer, faster tiles). PyTorch on the
same model measured 59-70 ms (TF32) in `results/2026-11-06-gpt162-vs-torch`: the executor is now about 4x slower instead of 11-12x.
**bf16s could not be timed**: `kvm_exec` rejects the gpt_162m bf16s bundle with `bad buffer shape or dtype` (exit 1) on main `d12c206`
as well as on this branch (`raw/bf16s_run_main.log`, `raw/bf16s_run_new.log`); it ran on 2026-11-07 (`results/2026-11-06-a100-combined`,
293 ms), so this is a regression on main in the executor's bundle loader, not in the tiles. The bf16 tile itself is covered by
`test_bf16` and `bench_mma_bf16` above.

## What changed

1. **Staged GEMM epilogue** (`mm_epilogue_staged`, `include/kvm_ops_gpt.cuh`). Both tensor-core tiles finished with a register epilogue
   unrolled over every fragment row and testing each optional operand per element: about 180 KB of straight-line SASS run once per tile.
   ncu on the bf16 tile at 4096^3: `stall_no_inst` 36% of all warp samples and 42% of the time outside the k loop. The accumulators now go
   to shared memory one 64-row slab at a time and a rolled loop does bias / pre / gelu / residual / accumulate with 16- or 8-byte coalesced
   accesses, the same operations in the same order. This alone took the bf16 tile from 0.46-0.51x to 0.69-0.77x.
2. **`mm_tf32_tile`**, the fast path of the 128x128 (flag 32) and 256x128 (flag 64) TF32 tiles: cp.async stages of BK 32 x 3 (144 KiB,
   256x128) or BK 16 x 4 (96 KiB), fragment double buffering, one barrier per stage. MN-major operands no longer use scalar loads plus
   register shuffles: B loads with LDS.32 in the standard fragment layout, A with LDS.64 into the mma's register pairs through a permuted row
   ownership (rows 8g + 2(i ^ (g&1)) + {0,1}); both shared layouts swizzled to zero bank conflicts (ncu). Unaligned shapes and 3xTF32 keep
   the previous `mm_tile_big`. The function is `__noinline__`: inlined into the op switch, the loop spilled in the k loop.
3. **TF32 rounding in two instructions** (`f2tf32_op`): `|x| < inf ? bits + 0x1000 : bits` (CUTLASS's round_half_ulp_truncate) instead of
   a three-op integer compare; the operand the tensor core reads is identical, NaN and -NaN still propagate (`test_ops` checks both on every
   tile size). This is what cuBLAS does: our result equals cuBLAS TF32 bitwise.
4. **Shared memory**: single-arch sm_80 / sm_90 / sm_100 builds get `KVM_GPT_MMA_SMEM` = 144 KiB (from `__CUDA_ARCH_LIST__`), others keep
   96 KiB; the tile reads `%dynamic_smem_size` and picks BK 32 x 3 only when the launch has it. `kvm_gpt_smem_bytes` returns the largest
   requirement (it returned a flat 104 KiB for hd = 128 builds). Test harnesses set their attribute from it.
5. **Frontend**: `Tiling.mm_tile` with shape fallbacks (`mm_choice`, from `perf/frontend-bigtile`); the a100 preset uses 256 (256x128,
   stepping down to 128x128 / 64x64 for small or ragged GEMMs); `a100_mm64` keeps the old tiling. CPU references and the fused matmul
   wrappers accept the big-tile flags.

## What limits it (not fixed here)

- **Op context**: the identical `mm_tf32_tile` launched as its own kernel measured 0.70-0.73x cuBLAS (`bench_mm MM_DIRECT=1`, 4096^3), and a
  standalone kernel with compile-time epilogue arguments 0.76-0.80x; inside `matmul_tile` it is 0.61-0.65x. ptxas compiles the out-of-line
  callee per calling kernel; ncu shows more `wait` and `barrier` stall in the op-kernel build. The executor's kernel was not profiled here.
- **TF32 rounding**: truncating operands instead (cuBLAS does NOT do this; it rounds like us) measured 0.85-0.99x standalone; not adopted,
  it would change numerics.
- **cuBLAS's kernel** for this shape is CUTLASS `s1688gemm_128x256_32x3` (147 KiB, 230 registers, 4.2 instructions per mma, 93% tensor
  pipe); ours runs about 5 instructions per mma.
- **Skinny wgrad** (768x768x4096: 18 tiles on 108 SMs) stays at 0.17-0.21x: no split-K. Inside the executor other tasks fill the idle SMs.
- 128x128 tiles (warp tile 64x32) top out near 0.55-0.69x; 256x128 is the shape to use.

## Correctness (this branch, A100)

| suite | result |
|---|---|
| `test_ops` (adds aligned partial tiles, K = 1000, NaN and -NaN on every tile size) | 3119 passed, 0 failed (`raw/op_suites.txt`) |
| `test_extra` | 3670 passed, 0 failed |
| `test_bf16` | 685 passed, 0 failed |
| `run_suite.py --suite exec` (`tests/gpu/test_exec`) | 8609 checks, 0 failed, 0 suites failed, litmus PASS (`raw/exec_suite_summary.txt`) |
| CPU: `tests` / `tests/frontend` | 154 OK / 126 OK (15 skipped: torch not installed locally) |

Not run: sanitizers, the litmus mutant builds, `tests/generic`, `tests/gptdiff`, the model zoo, the demo, gpt_162m gradients vs float64
with the new tiling (the tiles' arithmetic is unchanged, so they are expected to be identical; not measured).
