# Matmul tiles toward cuBLAS parity, A100 (2026-11-09)

Branch `perf/matmul-parity-v2`. Op kernels and the frontend's tile choice only; executor and scheduler code are untouched.

**Result.** gpt_162m f32, 1 step through `kvm_exec`: **755.8 ms on main -> 229.0 ms (3.3x)**. The a100 preset now emits 256 x 128
TF32 tiles (52,615 tasks per step instead of 270,914), and those tiles run 1.6-2.5x faster than before (0.56-0.73x cuBLAS TF32 at 4096^3,
main 0.37x). The bf16 GEMM tile went from 0.46-0.48x to 0.69-0.75x cuBLAS bf16 at 4096^3 (0.77x at 8192^3).
**Not met:** the 0.8x target (best in the op kernel is 0.73x); and **gpt_162m bf16s is 11% slower than main** (257 vs 230 ms per step),
cause not isolated (see "Open: bf16s step").

## Setup

| | |
|---|---|
| GPU | Modal NVIDIA A100-SXM4-40GB, driver 610.57.04, max SM clock 1410 MHz; nvcc 12.8.93 (`raw/env.txt`, `raw/steps162_env.txt`) |
| Benches | `tests/gpu_ops/bench_mm.cu` (TF32 `matmul_tile` vs `cublasSgemm` TF32 tensor-op math), `tests/gpu_ops/bench_mma_bf16.cu` (bf16 tile vs `cublasGemmEx` bf16), both in ONE process with cuBLAS, median of 5-7, every row checked against cuBLAS (all `ok`). Build: `nvcc -O2 -arch=sm_80 -std=c++17 --fmad=false -Xcompiler -ffp-contract=off -lineinfo`. Ratios are to cuBLAS in the same process; main and this branch's TF32 default build ran in different containers of the same type (`raw/tf32_main.txt`, `raw/tf32_new.txt`); the 144 KiB variant ran in main's container (`raw/tf32_new_144k.txt`) |
| Step time | the method of `results/2026-11-06-a100-combined`: `bench/vs_pytorch/gen.py --model gpt_162m --steps 1 --mode tf32 --tiling a100` on both trees, `kvm_exec -O3` built from both trees in one container, rounds interleaved |
| Commits | base main `a99fdca` (step A/B 1 used `d12c206`; the difference is `kvm_exec.cu`'s bf16 storage fix, #10) |

## TF32 `matmul_tile`, TFLOP/s (ratio to cuBLAS TF32)

| shape | tile | main | this branch (default, 96 KiB) | 144 KiB build |
|---|---|---|---|---|
| NN 4096^3 | 256x128 | 50.0 (0.37) | 91.2 (0.69) | 78.2 (0.61) |
| TN 4096^3 | 256x128 | 48.1 (0.37) | 73.2 (0.56) | 80.5 (0.63) |
| NT 4096^3 | 256x128 | 49.6 (0.37) | 95.3 (0.73) | 84.5 (0.65) |
| NN 4096^3 | 128x128 | 43.1 (0.33) | 67.8 (0.52) | 69.3 (0.54) |
| NN 4096^3 | 64x64 | 30.2 (0.22) | 32.9 (0.25) | 32.9 (0.25) |
| gpt_162m qkv fwd NN 4096x2304x768 | 256x128 | 29.9 (0.28) | 68.1 (0.64) | 60.4 (0.57) |
| qkv dgrad NT 4096x768x2304 | 256x128 | 41.5 (0.35) | 80.7 (0.68) | 74.0 (0.62) |
| fc1 fwd NN 4096x3072x768 | 256x128 | 29.5 (0.27) | 69.1 (0.63) | 61.4 (0.56) |
| fc2 fwd NN 4096x768x3072 | 256x128 | 44.6 (0.36) | 80.8 (0.66) | 71.0 (0.58) |
| fc1 dgrad NT 4096x768x3072 | 256x128 | 44.0 (0.36) | 83.0 (0.68) | 76.2 (0.62) |
| fc1 wgrad TN 768x3072x4096 | 256x128 | 34.0 (0.29) | 51.0 (0.44) | 56.4 (0.48) |
| logits fwd NN 4096x50304x768 | 256x128 | 32.2 (0.28) | 78.1 (0.67) | 70.0 (0.58) |
| logits dgrad NT 4096x768x50304 | 256x128 | 53.4 (0.49) | 59.3 (0.54) | 83.4 (0.77) |
| logits wgrad TN 768x50304x4096 | 256x128 | 50.6 (0.40) | 76.7 (0.60) | 83.7 (0.61) |
| proj wgrad TN 768x768x4096 (18 tiles on 108 SMs) | 256x128 | 8.5 (0.10) | 12.8 (0.15) | 14.2 (0.17) |

Numerics are unchanged: the TF32 result is bitwise the result of cuBLAS TF32 on 4096^3 (max |diff| 0), and on `test_ops`'s data this
branch reproduces main's per-case errors exactly.

## bf16 `matmul_mma_bf16` (raw: `raw/bf16_main.txt`, `raw/bf16_new.txt`, one container), ratio to cuBLAS bf16

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

## gpt_162m step

| run | tasks | ms/step best / median / worst | raw |
|---|---|---|---|
| main, f32 (64x64 TF32 tiles) | 270,914 | 739.6 / 755.8 / 762.4 | `steps162_ab_runs.txt` (7 rounds) |
| this branch, f32, 144 KiB build | 52,615 | 249.4 / 251.5 / 257.1 | same container |
| this branch's executor, main's f32 bundle | 270,914 | 742.1 / 745.1 / 756.2 | same container |
| **this branch, f32 (default 96 KiB build)** | 52,615 | 229.0 / 229.3 / 230.1 | `steps162_smem_ab.txt` (5 rounds, second container) |
| main, bf16s | 58,634 | 225.9 / 230.3 / 232.6 | `steps162_smem_ab.txt` |
| **this branch, bf16s (default 96 KiB build)** | 58,634 | 255.6 / 257.2 / 259.0 | same container |
| this branch, bf16s, 144 KiB build | 58,634 | 266.9 / 268.9 / 272.2 | same container; `steps162_bf16s_144k_ab.txt` agrees (264.5-272.6 vs main 227.1-231.2) |

The executor build is not slower on identical f32 work (row 3); the f32 gain is the bundle. PyTorch on the same model measured 59-70 ms
TF32 (`results/2026-11-06-gpt162-vs-torch`): the f32 executor is now 3.3-3.9x slower instead of 11-13x.

**Open: bf16s step.** Same bundle (bf16s does not use `matmul_tile`'s new tiles), 11% slower. Ruled out: the launch size (96 KiB, as on
main). Not tested: whether it is the bf16 tile's staged epilogue inside the executor (in isolation the tile is 1.4-2.5x faster), or code
size / register allocation of the executor kernel after the out-of-line TF32 tiles were added. The next measurement is the bf16s step with
the bf16 tile's old epilogue restored, and a per-op trace of both builds.

## What changed

1. **Staged GEMM epilogue** (`mm_epilogue_staged`, `include/kvm_ops_gpt.cuh`). Both tensor-core tiles finished with a register epilogue
   unrolled over every fragment row and testing each optional operand per element: about 180 KB of straight-line SASS run once per tile.
   ncu on the bf16 tile at 4096^3: `stall_no_inst` 36% of all warp samples, 42% of the time outside the k loop. The accumulators now go to
   shared memory one 64-row slab at a time and a rolled loop does bias / pre / gelu / residual / accumulate with coalesced 16- or 8-byte
   accesses, the same operations in the same order.
2. **`mm_tf32_tile`**, the fast path of the 128x128 (flag 32) and 256x128 (flag 64) TF32 tiles: cp.async stages of BK 16 x 4 (96 KiB) or,
   in a `-DKVM_GPT_MMA_SMEM=147456` build, BK 32 x 3; fragment double buffering; one barrier per stage. MN-major operands no longer use
   scalar loads plus register shuffles: B loads with LDS.32 in the standard fragment layout, A with LDS.64 into the mma's register pairs
   through a permuted row ownership (rows 8g + 2(i ^ (g&1)) + {0,1}); both shared layouts swizzled to zero bank conflicts (ncu). Unaligned
   shapes and 3xTF32 keep `mm_tile_big`. `__noinline__`: inlined into the op switch, the loop spilled in the k loop.
3. **TF32 rounding in two instructions** (`f2tf32_op`): `|x| < inf ? bits + 0x1000 : bits` (CUTLASS's round_half_ulp_truncate) instead of
   a three-op integer compare; the tensor core reads the same operand, NaN and -NaN still propagate (`test_ops` checks both on every tile
   size). cuBLAS does the same: our result equals cuBLAS TF32 bitwise.
4. **Shared memory**: `kvm_gpt_smem_bytes` returns the largest requirement (it returned a flat 104 KiB for hd = 128 builds); test harnesses
   and `bench_ops` set their attribute from it. The 144 KiB build is opt-in: it is faster for the TN tile in isolation, but the executor's
   larger launch shrinks the L1 carveout of every other op and the 162M step got slower (f32 251 vs 229 ms, bf16s 269 vs 257 ms; medians).
5. **Frontend**: `Tiling.mm_tile` with shape fallbacks (`mm_choice`, from `perf/frontend-bigtile`); the a100 preset uses 256 (256x128,
   stepping down to 128x128 / 64x64 for small or ragged GEMMs); `a100_mm64` keeps the old tiling. CPU references and the fused matmul
   wrappers accept the big-tile flags.

## What limits the tiles (not fixed here)

- **Op context**: the identical tile launched as its own kernel measured 0.70-0.73x cuBLAS, and a standalone build with compile-time
  epilogue arguments 0.76-0.80x (BK 32 x 3); in the op kernel it is 0.56-0.73x. ptxas compiles the out-of-line callee per calling kernel.
- **TF32 rounding**: truncating operands instead measured 0.85-0.99x standalone; not adopted (cuBLAS rounds like us; truncation has
  2.4x the error).
- **cuBLAS's kernel** for 4096^3 is CUTLASS `s1688gemm_128x256_32x3` (147 KiB, 230 registers, 4.2 instructions per mma, 93% tensor pipe);
  ours runs about 5 instructions per mma.
- **Skinny wgrad** (768x768x4096: 18 tiles on 108 SMs) stays at 0.15-0.21x: no split-K. Inside the executor other tasks fill the idle SMs.
- 128x128 tiles (warp tile 64x32) top out near 0.52-0.60x; 256x128 is the shape to use.

## Correctness (this branch, A100)

| suite | result |
|---|---|
| `test_ops` (adds aligned partial tiles, K = 1000, NaN and -NaN on every tile size) | 3119 passed, 0 failed at 144 KiB (`raw/op_suites.txt`); matmul subset 889 passed, 0 failed at the 96 KiB default |
| `test_extra` | 3670 passed, 0 failed |
| `test_bf16` | 685 passed, 0 failed |
| `run_suite.py --suite exec` (`tests/gpu/test_exec`, 144 KiB build) | 8609 checks, 0 failed, 0 suites failed, litmus PASS (`raw/exec_suite_summary.txt`) |
| gpt_162m bundles through `kvm_exec` (96 and 144 KiB builds) | every run `OK`, all tasks completed |
| CPU: `tests` / `tests/frontend` | 154 OK / 126 OK (15 skipped: torch not installed locally) |

Not run: the exec suite on the 96 KiB default build (its executor ran the 162M bundles); sanitizers; litmus mutant builds;
`tests/generic`; `tests/gptdiff`; the model zoo; the demo; gpt_162m gradients vs float64 with the new tiling (the tiles' arithmetic is
unchanged, so they are expected to be identical; not measured).
