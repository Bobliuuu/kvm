# Hopper (sm_90a) GEMM tiles: TMA + mbarrier + wgmma on the H100 (2026-11-09)

**Verdict.** KVM now has a real sm_90a path (`include/kvm_hopper.cuh`): `matmul_tile` (1xTF32) and `matmul_mma_bf16` load their operands with TMA
(`cp.async.bulk.tensor`, tensor maps patched per task on the device), complete on mbarriers, and multiply with `wgmma.mma_async`. It is correct everywhere it
runs (`test_hopper` 1612/0 against float64 and against the sm_80 tile; the executor suite 8609/0 and the op suites on the sm_90a build), the sm_80 build is
unchanged (A100 exec suite 8609/0). Speed:

* **TF32 large tiles: 1.2-2.7x the sm_80 tile on the same H100** (128x128 NT 219 vs 84 TFLOP/s, 256x128 NT 232 vs 86 at 4096^3; TN/TT, which need the
  shared-memory transpose, 1.2-1.7x), **0.27-0.59x cuBLAS TF32**.
* **TF32 64x64 tile** (the one gpt_162m f32 is lowered to): wins only when both operands are K-major (NT 77 vs 60); NN ties, TN loses to the shared-memory
  transpose (38 vs 48 in an earlier build). The 64x64 tile therefore takes the Hopper path only for K-major A and B.
* **bf16 128x256: 1.16-1.19x the sm_80 `mma.sync` tile** (260-264 vs 220-224 TFLOP/s at 4096^3, 0.36x cuBLAS bf16 ~725); with the operands L2-resident 410
  vs 290. Operand traffic and per-slab overhead keep it far from cuBLAS (section 4). The 4096x50304x768 logits GEMM is 0.95x the sm_80 tile.
* **gpt_162m step: unchanged by the Hopper tiles** (f32 620.3 vs 621.3 ms, bf16s 204.4 vs 204.8 ms with `KVM_NO_HOPPER=1`, median of 5 interleaved runs).
  The step is task-throughput bound, not GEMM bound, and at the `a100` tiling the f32 GEMMs are 64x64 tiles. The H100 runs it 1.36x (f32) / 1.44x (bf16s)
  faster than the A100 measurement of the same model (845.5 / 293.3 ms, `results/2026-11-06-a100-combined`, an earlier commit and a different container, so a loose comparison). PyTorch on the same H100 is 16-20x (TF32) and
  8-15x (autocast bf16) faster than the executor.
* **Found and fixed on the way:** main's executor rejected bf16 storage buffers (the #8 landing merge dropped the bf16 hunks of `kvm_exec.cu`), so no
  `precision='bf16s'` bundle could run on main on any GPU; the loader hunks are restored here.

## Setup

| | |
|---|---|
| GPU | Modal NVIDIA H100 80GB HBM3 (SXM, 132 SMs, sm_90), driver 610.57.04. One early step-time run landed on an H200 (Modal upgraded `gpu="H100"`; the runner now pins `H100!`): its numbers are not used. |
| Toolchain | nvcc 12.8.93 (`nvidia/cuda:12.8.1-devel-ubuntu22.04`), PyTorch 2.7.0 |
| Builds | op tests `nvcc -O2 -arch=sm_90a -std=c++17 --fmad=false -Xcompiler -ffp-contract=off -lineinfo`; executor `-O3`, same flags; `tests/gpu/build.sh` with `KVM_ARCH=sm_90a` |
| Runner | `tests/modal/run_h100.py` (`build`/`cpu` on a 32-core CPU container, `test`/`bench`/`step`/`custom` on an H100) wrapping `tests/gpu_ops/run_hopper_box.sh`; the A100 check is `tests/modal/run_suite.py --suite exec` |
| Code | H100 builds and runs: this branch's Hopper commit on top of `d12c206` (main after #8) plus the bf16-loader hunks of #10. The branch was then rebased on main
`7ce1932` (#10, #15 multi-GPU); the Hopper device code is identical, the rebase only brings #15's `dist/` / comm-pack changes, which the H100 runs here did
not exercise. The A100 sm_80 check and the CPU suites ran on the rebased tree. |
| Raw output | `raw/` |

## 1. What was built

* **Tensor maps without host work per task.** `kvm_hopper_host_init()` encodes three template maps (fp32 128-byte swizzle, bf16 64-byte, bf16 128-byte) with
  `cuTensorMapEncodeTiled` (driver entry point, no `-lcuda`) once per process; `KvmExec` and the op-test harnesses call it. A task copies its template into a
  per-block slot in global memory and patches address, extents, row stride and box with `tensormap.replace`, then fences the tensormap proxy. Each slot keeps the
  key of the map it holds, so consecutive tiles of one GEMM operand reuse the map untouched.
* **TF32 tile** (`kvmhop::mm_f32<TM,TN>`, the 64x64 / 128x128 (flag 32) / 256x128 (flag 64) tiles of `matmul_tile`, non-precise only): BK 32 slabs (128-byte
  rows), up to 6 TMA stages, warpgroup split 2 x m64n32k8 (64x64), m64n128k8 per warpgroup (128x128), 2 x m64n128k8 per warpgroup (256x128). wgmma tf32 reads
  only K-major operands, so an MN-major operand lands as [32 k][32 mn] boxes and 256 threads transpose it into the K-major swizzled layout (double buffer).
* **bf16 tile** (`kvmhop::mm_bf16`, `matmul_mma_bf16`'s 128x256): one m64n256k16 per warpgroup and k16 step, the transpose bits handle all four layouts
  (no transpose pass); BK 64 (128-byte swizzle) when two 48 KiB stages fit, else BK 32 (64-byte swizzle) for the 96 KiB launch.
* **Pipeline.** `wait_group 1`: the wgmma group of slab i overlaps the TMA of later slabs (and the transpose of slab i+1); a stage is refilled as soon as the
  group that read it is complete in both warpgroups.
* **Integration.** On sm_90 `KvmExec` launches with 200 KiB of dynamic shared memory (`kvm_hopper_dyn_smem`; one block per SM either way: the kernel uses 255
  registers). The tiles read `%dynamic_smem_size`, pick the pipeline depth that fits and decline (the caller runs its mma.sync tile) when the launch is too small,
  an operand is not 16-byte aligned / strided, the host init did not run, or the 64x64 layout rule says so. The rule is deterministic (launch size minus 2 KiB) so
  `test_hopper` predicts every decline. `KVM_NO_HOPPER=1` disables the path for A/B runs in one binary. Every other target compiles exactly as before
  (`__CUDA_ARCH_FEAT_SM90_ALL` guard).
* **Numerics.** fp32 accumulation in a fixed order: bitwise reproducible run to run on one GPU. tf32 wgmma truncates its inputs to 19 bits (the sm_80 tile rounds),
  inside `matmul_tile`'s documented 1xTF32 bound (4e-3 of max|ref|) but not bitwise equal to the A100.

## 2. Correctness

| check | result | raw |
|---|---|---|
| `test_hopper` (H100): 4 layouts x 3 tf32 tile kinds x 7 shapes (ragged and aligned, padded rows half the time) x random epilogues, bf16 4 layouts x 5 shapes x fp32/bf16 out, at the 96 KiB and the 200 KiB launch; vs float64 of the tensor-core inputs, vs the sm_80 tile of the same binary, and "Hopper path taken" from device counters | 1612 passed, 0 failed at both launch sizes (final code); every expected decline predicted | `raw/` |
| `test_exec` full (all packs, sm_90a, H100) | 8609 checks, 0 failed, 0 suites failed (final build) | `raw/final_h100_test_exec.log` |
| `test_ops`, `test_bf16`, `test_extra` (sm_90a, H100, final build) | 2965 / 685 / 3669 passed, 0 failed | `raw/final_ops_sanitizers.log`, `raw/final_test_extra.log` |
| compute-sanitizer memcheck / racecheck / synccheck on `test_hopper` (final build) | 0 errors / 0 hazards / 0 errors | `raw/final_ops_sanitizers.log` |
| sm_80 unaffected: `tests/modal/run_suite.py --suite exec` on an A100 (rebased tree) | 8609 checks, 0 failed, 0 suites failed, litmus PASS | `raw/a100_sm80_exec_suite.txt` |
| CPU suites | `tests` 153 OK, `tests/frontend` 124 OK (15 skipped) | |

## 3. GEMM throughput (TFLOP/s, 4096^3 unless stated, one CTA per tile, median of 7; `test_hopper --bench`)

| case | Hopper tile | sm_80 tile (same GPU) | cuBLAS | Hopper / sm_80 | Hopper / cuBLAS |
|---|---|---|---|---|---|
| tf32 64x64 NT (K-major A, B) | 77.1 | 60.0 | 399.3 | 1.29 | 0.19 |
| tf32 64x64 NN / TN | declined (sm_80 tile: 44.9 / 47.5) | 44.9 / 47.9 | 393.4 / 260.9 | 1.00 | |
| tf32 128x128 NN / NT / TN / TT | 135.9 / 219.3 / 101.5 / 133.2 | 79.4 / 83.7 / 77.2 / 79.9 | 384.6 / 387.2 / 258.5 / 385.0 | 1.71 / 2.62 / 1.31 / 1.67 | 0.35 / 0.57 / 0.39 / 0.35 |
| tf32 256x128 NN / NT / TN / TT | 162.2 / 231.6 / 95.4 / 105.7 | 80.7 / 86.2 / 81.6 / 85.1 | 384.6 / 389.6 / 256.9 / 385.0 | 2.01 / 2.69 / 1.17 / 1.24 | 0.42 / 0.59 / 0.37 / 0.27 |
| tf32 qkv 4096x2304x768 / fc1 4096x3072x768 NN, 64x64 | declined | 38.1 / 38.3 | 209.9 / 259.1 | 1.00 | |
| bf16 128x256 NN / NT / TN / TT | 263.7 / 260.6 / 259.9 / 261.0 | 222.0 / 219.6 / 224.0 / 222.9 | 727.2 / 721.5 / 725.3 / 728.2 | 1.16-1.19 | 0.36 |
| bf16 logits 4096x50304x768 NT, fp32 out | 68.8 | 72.8 | 535.6 | 0.95 | 0.13 |

Raw: `raw/final_test_hopper_bench.log` (final code), the 64x64 rows from `raw/hop5_ops_bench_sanitizers.log` (same 64x64 code). TN / TT tf32 pay the shared-memory
transpose of the MN-major operand; NT (both operands K-major, the TMA box lands in exactly the layout wgmma reads) is the fast case.

cuBLAS: `cublasGemmEx` TF32 (`CUBLAS_COMPUTE_32F_FAST_TF32`) / bf16 with fp32 accumulate, same GPU, same container.

## 4. Why bf16 does not gain, and what was tried

Same bench with every CTA computing tile (0,0) (`HOP_SAME=1`: operands stay in L2; `raw/final_test_hopper_bench.log`, second table):

| case | normal | operands L2-resident | sm_80 tile, L2-resident | cuBLAS |
|---|---|---|---|---|
| bf16 128x256 (4 layouts) | 260-264 | 408-410 | 279-290 | 720-730 |
| tf32 256x128 NT | 231.6 | 308.7 | 94.1 | 390.9 |
| tf32 128x128 NT | 219.3 | 231.7 | 86.3 | 389.6 |

So at 4096^3 the bf16 tile loses about a third to operand traffic (one CTA per SM, no cluster multicast, tile order m-outer), and the rest of the gap to cuBLAS
is per-slab overhead of a 2-warpgroup tile: thread 0 issues the TMA inside warp 0, both warpgroups meet at a `__syncthreads` per 64-wide slab, and there is no
producer warpgroup.

* The first version stored the bf16 tile's epilogue one element at a time: 222 TFLOP/s (`raw/v2_test_bench.log`, `raw/v2_bench_same_tile.log`), no faster than the sm_80 tile. Column pairs as one 64-bit / bf16x2
  access give the 260 above (logits 56.6 -> 68.8). The first tf32 version called the tile through a `__noinline__` function: ptxas serialized every wgmma
  ("wgmma pipeline crossing function boundary"); force-inlining and `wait_group 1` took 128x128 NT from 175 to 219.

* A variant with per-stage *empty* mbarriers (each warpgroup arrives, thread 0 refills one iteration late, no block barrier in the main loop) was measured and
  rejected: it lost a stage of prefetch depth and was slower everywhere (bf16 205, tf32 128x128 NT 167, 64x64 NN 35; `raw/hop4_rejected_empty_barrier_variant.log`).
* Not done: TMA multicast across a 2-CTA cluster (what cuBLAS uses to halve operand traffic) does not fit the executor, which runs one independent task per block;
  a producer warpgroup with `setmaxnreg` needs a third warpgroup, and the executor's contract is 256 threads per op.

## 5. gpt_162m step on the H100

Command: `modal run --detach tests/modal/run_h100.py --stage step --tag hop7` (`tests/gpu_ops/run_hopper_box.sh step`): `bench/vs_pytorch/gen.py` bundles
(gpt_162m, 1 step, `--mode tf32 --tiling a100`), `kvm_exec --quiet` interleaved 5 times with and without `KVM_NO_HOPPER=1` in one container, then
`bench/vs_pytorch/torch_bench.py` (3-step bundle, median of 5) in the same container. Raw: `raw/step/`.

| gpt_162m, one AdamW step, H100 SXM | ms/step (median) | range |
|---|---|---|
| executor f32 (270,914 tasks), Hopper tiles | 620.3 | 616.3-624.3 |
| executor f32, `KVM_NO_HOPPER=1` (sm_80 tiles, 96 KiB launch) | 621.3 | 616.9-623.1 |
| executor bf16s (58,634 tasks), Hopper tiles | 204.4 | 203.1-205.7 |
| executor bf16s, `KVM_NO_HOPPER=1` | 204.8 | 203.1-225.7 |
| PyTorch TF32 eager / whole-step CUDA graph / `torch.compile` + graph | 39.3 / 37.9 / 31.8 | |
| PyTorch autocast-bf16 eager / graph / compile + graph | 24.8 / 22.7 / 13.9 | |
| A100-SXM4-40GB reference (executor f32 / bf16s, `results/2026-11-06-a100-combined`, earlier commit) | 845.5 / 293.3 | |

Every executor run completed all tasks (`OK`). The armed run logs `hopper: TMA + wgmma tiles armed, dynamic smem 204800 B` and a grid of 132 (1 admitter +
131 workers). The GEMM tiles are not what limits this step: on the A100 the same step was measured as about 3.3 us per task end to end with the workers 27%
busy (`results/2026-11-06-gpt162-vs-torch`), and making the bf16 tiles 1.2x faster does not move it. A first step run landed on an H200 (Modal upgraded
`gpu="H100"`), and the bf16s half failed at load there (the loader bug above); those numbers are not used.

## 6. Blackwell (tcgen05) and what is missing

Not built: no sm_100 box was used. Missing for a Blackwell path: `tcgen05.mma` with tensor-memory (TMEM) accumulators (`tcgen05.alloc` / `ld` / `dealloc`),
the 2-CTA `cta_group::2` MMA, and a build of the packs for `sm_100a`; the TMA / tensor-map / mbarrier machinery here carries over unchanged.

## 7. Caveats

* The tf32 path is 1xTF32 only; `precise` (3xTF32, flag 16) keeps the sm_80 tile on Hopper.
* The bench is one CTA per tile in tile order m-outer; the executor interleaves tiles of many ops, so the step numbers, not the bench, say what the executor gets.
* `test_ops` reports 2965 checks on this tree (A100 and H100 alike); `results/2026-11-06-a100-combined` recorded 3224 on the `perf/combined` branch.
* The step comparison uses the `a100` tiling preset; an H100 preset that lowers the f32 GEMMs to the 128x128 / 256x128 tiles (where the Hopper path is 2x+) was
  not built: the step is task-throughput bound either way.
* Spend: about $26 (Modal billing for this work's apps through 05:00 UTC: $20.43, plus the last hour and the A100 suite run, estimated); Vast: under $1
  (three H100 rentals; one host never started the instance and one went offline mid-build, so the work moved to Modal; all destroyed).
