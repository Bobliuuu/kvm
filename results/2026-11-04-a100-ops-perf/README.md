# Op-library performance on A100 (2026-11-04)

GPU: NVIDIA A100-SXM4-40GB, 108 SMs, 1410 MHz, driver 580.105.08, CUDA 12.8, sm_80, `nvcc -O2`. Shared box (GPU lock held, but
other jobs run unlocked around it: spreads are real). Code: branch `perf/ops`, commits f34bbd9..cd56ffe. Command:
`KVM_HOST=... TEST=1 REF=1 VARIANTS="BENCH_MINB=2" ./run_gpu_perf.sh <tag>` (7 repetitions per number, median [min..max]).
"Full dispatch" = the all-ops kernel (what a megakernel compiles: registers are the union's), as in the executor.

Logs: `baseline.txt` (before, plus torch/cuBLAS/FlashAttention reference lines `REF`), `final.txt` (after; default build, then
`BENCH_MINB=2` = all-ops kernel capped at 128 registers, then `KVM_GPT_MAX_HD=64`), `sanitizers_and_e2e.txt`, `gptdiff_run.txt`,
`exec_suite_tail.txt`, `gelu_vec_intermediate.txt`.

## Results (full dispatch, median)

| op / shape | before | after, default build | after, 128-reg cap (`BENCH_MINB=2`) | % of peak (cap) | reference |
|---|---|---|---|---|---|
| matmul TF32 NN 4096^3, 64x64 tile | 29.4 TFLOP/s | 27.0 (box busy; 40.5 in a quiet run) | 42.7 | 27% of 156 | cuBLAS TF32 105.6 (0.40x) |
| same, 128x128 tile (flag 32, new) | n/a | 42.8 | 43.5 (52.2 NT) | 28-33% | 0.41-0.49x |
| matmul TF32 TN / NT 4096^3 (64 tile) | 28.4 / 27.5 | 32.4 / 33.3 | 42.4 / 46.8 | 27-30% | 103.5 / 105.7 |
| matmul 3xTF32 NN 4096^3 (64 tile) | 17.1 | 20.4 | 24.6 | 47% of 52 | cuBLAS fp32 19.0 (1.3x) |
| matmul TF32 8192x768x768 | 23.4 | 25.7 (128 tile 31.4) | 35.3 (128 tile 40.5) | 23-26% | 79.8 |
| matmul TF32 4096x3072x768 | 24.4 | 26.7 (128 tile 33.2) | 37.1 (128 tile 43.5) | 24-28% | 100.1 |
| matmul TF32 1024^3 / 512^3 | 20.5 / 11.8 | 22.3 / 12.7 | 27.1 / 12.2 | 17% / 8% | 50.1 / 17.0 |
| attn fwd B4 T1024 H16 hd64 | 4.15 | 12.2 | 12.0 | | torch fp32 SDPA 16.2; bf16 flash 93.6 |
| attn bwd (same) | 3.24 | 9.5 | 11.8 | | fp32 15.8; bf16 69.4 |
| attn fwd B4 T2048 H16 Hkv4 hd64 | 4.58 | 13.9 | 14.8 | | fp32 19.7; bf16 133.0 |
| attn bwd (same) | 3.64 | 12.9 | 16.1 | | fp32 18.5; bf16 95.9 |
| attn fwd B2 T2048 H8 hd128 | 3.22 | 15.9 | 13.6 | | fp32 20.7; bf16 108.4 |
| attn bwd (same) | 2.59 | 14.3 | 11.6 | | fp32 17.0; bf16 90.0 |
| layernorm fwd 16384x1024 | 881 GB/s | 1127 | 1151 (74%) | 74% of 1555 | ATen 1237 |
| layernorm bwd dx | n/a | 911 | 1008 (65%) | | |
| norm_bwd_params, 16 tasks x 64 cols | 159 | 268 | 97 | | one block per task: too little parallelism |
| norm_bwd_params, 64 tasks x 16 cols | n/a | 801 | 399 | | |
| adamw 16M | 1032 | 1171 | 1292 (83%) | 83% | fused ATen 1230 |
| gelu fwd 64M | 306 | 1180 | 1337 (86%) | 86% | ATen 1365 |
| softmax_ce fwd 8192x50304 | 914 | 993 | 1361 (87%) | 87% | torch eager 324 |
| softmax_ce bwd | 1002 | 1301 | 1327 (85%) | 85% | torch eager 541 |

Attention TFLOP/s counts causal 4*T^2/2*hd per head (fwd; 2.5x for bwd) so it is comparable with the reference lines, but our
products run as 3xTF32 on the tensor cores: three mma per useful one. The ceiling at that cost is about 52 TFLOP/s. bf16/fp16 paths
were NOT added (no bf16 goldens or tolerances exist, and a bf16 buffer dtype needs executor/compiler/reference support).

## Occupancy (registers and blocks per SM)

Per-op registers (`cuobjdump -res-usage`, final.txt): matmul 200, attn_fwd 212, attn_bwd_dq 166, attn_bwd_dkv 241, all-ops kernel 255
(before: 171). All of them are above 128, so the persistent kernel runs ONE block per SM before and after (65536 / (256 x 171) = 1).
Dynamic shared memory: KVM_GPT_SMEM_BYTES 48 KiB -> 56 KiB (hd <= 64) or 80 -> 104 KiB (hd = 128); two blocks per SM still fit
(164 KiB). Capping the all-ops kernel with `__launch_bounds__(256, 2)` (128 registers, ~1 KiB of spill, `BENCH_MINB=2`) is
the single largest remaining lever and is the executor's call: it gave +50% on 64-tile matmul, +25% attn bwd T2048, +10-30% on the
memory-bound ops here, and -15% on hd=128 attention. `-DKVM_GPT_MAX_HD=64` (default) already keeps hd=128 out of the build.

## What did not work or did not help (honest list)

- Rounding to tf32 with `cvt.rna` was conversion bound (cvt is 1/4 of integer rate); two integer ops (bitwise equal rounding) helped;
  dropping the rounding (truncation) gave only +8% for 3x the error (relerr 7.6e-4 vs 2.7e-4): rejected.
- Making the tf32 rounding NaN-safe (review finding: +0x1000 turns the canonical NaN 0x7fffffff into -0) costs a few ALU ops; the
  quiet-box numbers before that fix were ~5-10% higher. Kept for correctness.
- The 64x64 task is bound by L2->SM traffic (16 KiB per 262 kFLOP slab): 40-47 TFLOP/s at best. The new 128x128 tile (flag 32)
  needs the lowering to step tiles by 128; it is not used by `frontend/` yet (agent H). It reached 45.7-57.7 TFLOP/s in a quiet run
  (not preserved: those logs were overwritten) but only 42.8-43.5 in the final busy run, 41-49% of cuBLAS either way.
- Gap to cuBLAS (105 TFLOP/s) remains 2x: no deeper (4+) pipeline within the 48 KiB cap, one block per SM, scalar B loads for
  [k][n] operands. Attention is 3xTF32, so it is below torch's fp32 SDPA (efficient-attention) by 15-30%, not above it.
- norm_bwd_params with few tasks cannot use the GPU: the task partition (frontend) must give >= 64 tasks (16 columns each).
- The first perf runs lost hours in the GPU-lock queue, one died of a dropped idle ssh session (`run_gpu_perf.sh` now detaches the job).
- No ncu/nsys on the box: bottlenecks were inferred from instruction counts and experiments, not profiled.
- Intermediate raw logs (runs between baseline and final) were deleted by mistake; the numbers in commit messages come from them.

## Correctness (all on the final code, cd56ffe)

- `test_ops` 1104 passed, 0 failed (was 444 at the start of this work; adds 128x128 tile, odd origins, NaN, dense/vector norm and CE paths,
  differing leading dimensions in attention). `test_e2e` 7 passed. compute-sanitizer memcheck / racecheck / synccheck / initcheck on `test_ops`:
  0 errors (`sanitizers_and_e2e.txt`).
- `tests/gptdiff/run.sh --sanitize` (own scratch dir): 8 bundles (1/2/5 steps cpu tiling, 2 steps a100 tiling; tf32 and precise)
  all exit 0 and PASS the golden check (precise: all buffers rel L2 <= 2e-4, worst 7.35e-5); memcheck/racecheck/synccheck 0 errors.
- `run_exec_tests.sh` functional suites: 8582 checks, 0 failed.
- An independent read-only review found 4 real bugs (misaligned cp.async for odd tile origins, dq/dk/dv ld overwrite, -inf CE NaN,
  NaN eaten by tf32 rounding), all fixed with tests before these runs.

# Round 2 (2026-11-04, same box; commits after d5fe58b on perf/ops)

## Merge and bf16
- Merged feat/integration (107a276). M's `include/kvm_ops_extra.cuh` still compiled against the old SIMT attention helpers (kept, unused by
  the new kernels) but its `mm_tile_bf16` used removed wmma/tile_async code: replaced by `kvmgpt::mm_tile<TA,TB,false,true>`, a real
  bf16 tensor-core tile (operands rounded RNE to bf16 with `cvt.rn.bf16x2.f32` when read from shared memory, `mma.m16n8k16.bf16`, fp32
  accumulate, fp32 containers). The extra pack's attn_full_* now use the separate dq/dk/dv leading dimensions (same bug as in the
  main attention) and are compiled out above KVM_GPT_MAX_HD. `test_extra` 2089 passed / 0 failed on the merged tree (r2b.txt, r2d.txt);
  `test_ops` 1396 passed (256x128 tile added to all transposes/edges/odd origins).
- bf16 speed was NOT measured separately (no bench entry yet); the 2x claim is untested. Attention in bf16 was not written.

## Matmul vs cuBLAS (target >= 70%): NOT reached
Best result: new `matmul_tile` flag 64 = 256x128 tile, warp tile 64x64 (smem-wavefront ratio 1.0 instead of 1.5 per mma), 72 KiB smem
(KVM_GPT_SMEM_BYTES becomes 72 KiB for hd<=64 builds). Median of 7, full dispatch, busy box (r2d.txt); cuBLAS TF32 (torch) from baseline.txt:

| shape | 64x64 | 128x128 | 256x128 | cuBLAS | best / cuBLAS |
|---|---|---|---|---|---|
| 4096^3 NN | 31.9 | 45.1 | 51.0 | 105.6 | 0.48 |
| 4096^3 TN / NT | | | 48.7 / 51.3 | 103.5 / 105.7 | 0.47 / 0.49 |
| 8192x768x768 | 25.4 | 32.4 (40.5 with 128-reg cap) | 29.7 | 79.8 | 0.41-0.51 |
| 4096x3072x768 | 26.3 | 34.1 (42.9 with cap) | 30.1 | 100.1 | 0.34-0.43 |
| 4096^3 3xTF32 | | 25.8 | 40.7 (r2b, invalid, see below) | fp32 19.0 | |

Tried and rejected (logs r2a.txt, r2d.txt): BK=32 stages (96 KiB): 33.1 vs 33.8 (no gain); 4 stages: 43.7 vs 33.8-45 (noise level);
launch-order grouping of 8 row tiles for L2: no change; 128-register cap: +30-50% on 64/128 tiles but the 256x128 tile collapses to
13 TFLOP/s under a cap (128 accumulator registers spill), so the cap and the huge tile are mutually exclusive; ldmatrix.trans for
[k][n] operands is impossible for 32-bit data (it transposes 16-bit elements). r2b.txt (first 256x128 run) is INVALID: the epilogue stored
only half of the column tiles (test and bench check failed, fixed before r2d); its apparent 73-78 TFLOP/s are not real.
End-to-end (`r2_regsweep.txt`, kvm_exec on the 206-task gpt_tiny a100 bundle, 7 runs each): 3.04-3.08 ms for base, maxrregcount 128, 168
and 200: this bundle is latency/dependency-bound, so op throughput and the register cap are invisible; a larger bundle is needed for a
real end-to-end number (not generated).

## Attention (r2c.txt): compile-time `KVM_GPT_ATTN_MMA` (default 3 = the 2e-5 contract)
| hd64 T2048 / hd128 | 3xTF32 (default) | 2xTF32 | 1xTF32 | torch fp32 SDPA |
|---|---|---|---|---|
| fwd hd64 | 13.6 | 16.8 (relerr 4.6e-4) | 19.5 (7e-4) | 19.7 |
| bwd hd64 | 12.7 | 16.0 (1.1e-3) | 18.4 (2e-3) | 18.5 |
| fwd hd128 | 15.3 | 19.1 (6e-4) | 22.4 (7e-4) | 20.7 |
| bwd hd128 | 14.3 | 17.7 (1.5e-3) | 20.2 (3e-3) | 17.0 |
1xTF32 reaches torch fp32 parity (and beats it at hd128) but at TF32 accuracy, so it is NOT the default; it needs a tolerance decision
(goldens) and an ABI flag, neither exists. The default 3xTF32 stays 15-30% below torch fp32 SDPA. Not done: single-pass online-softmax
backward (a fused dq/dkv needs atomics or a second pass; deterministic order was kept).

## Scratch directories I overwrote (rsync --delete), disclosed in the integration log
~/kvm-gptdiff-scratch (gptdiff run) and ~/kvm-exec-scratch (exec run), both ~08:28 UTC; all later runs used private dirs.

## Round-2 verification (final tree)
test_ops 1396 and test_extra 2089 passed; compute-sanitizer memcheck/racecheck/synccheck/initcheck on test_ops and memcheck/racecheck on
test_extra: 0 errors (`r2_sanitizers.txt`); tests/gptdiff/run.sh --sanitize: all 8 bundles PASS, sanitizers clean (`r2_gptdiff.txt`);
run_exec_tests.sh: 8582 checks, 0 failed (`r2_exec_tail.txt`).
