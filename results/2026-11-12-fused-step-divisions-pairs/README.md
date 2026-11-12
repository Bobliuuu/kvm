# The fused GPT step: attention's independent GEMMs at once, and grid_loop's divisions through a float reciprocal (A100)

Two changes to the fused GPT step (`include/kvm_fused_gpt.cuh`), each keeping every bit of the step:
- **Attention's independent GEMMs at once** (`perf(fused): run the attention items' independent GEMMs at once`). The attention items'
  backward has GEMMs that read nothing another writes: do with P, dV with dS, dq with dK, dWq beside q's dh rows, dWk and dWv beside k's
  and v's. `gemm_par` runs them on disjoint warps, sized by each GEMM's tiles times its k steps. Each output's k chain stays whole in one
  warp, in order, so every bit is the serial order's. `KVM_FGPT_PAIR` 0 runs them one after another, for a bed's control.
- **grid_loop's divisions through a float reciprocal** (`perf(fused): divide grid_loop's bounds through a float reciprocal`). Every
  staging pass, `zero()` and row loop of the step runs `grid_loop`, and each call divided by a run-time value twice on every thread.
  `div_small` divides through `__fdividef`'s reciprocal and fixes the float quotient with one integer step.
  - **Exact for 0 <= x < 2^21:** below 2^21 the float quotient's relative error stays under 2^-21, so it is off by at most one, and the
    integer step corrects that.
  - **grid_loop stays in range by construction:** it divides NT (at most 1024, by a `static_assert`) and the thread index (below 1024, by
    CUDA's block limit).
  - **No run-time range check:** nvcc does not fold a check of x against 2^21 away for the thread index. A checked form keeps a whole
    integer division beside the float path at every call site, and it ran slower than the division it replaced.
  - **`tests/static/test_div_small.cu`** checks `div_small` against `/` on the device: every x below 2^21 for d up to 512, grid_loop's own
    domain (x up to 1024, d up to 2^20), the range's top edge, and 2^24 random pairs. On the A100: `div_small: PASS, 2173941844 pairs
    checked` (`raw/device-test/`).

Every figure here was re-derived from these raw files by the team's verifier (Nov 13). In `raw/`, the host's working folders read
`/root/work/` and the GPU lock's own lines are removed; the panel's job log is renamed `panel/panel.log`. Everything else is as the runs
wrote it. The base is #62's head (`a48fcf0`); b0 and b1 in the bed logs are the base and this branch.

## The panel's 9 rows against the base

`raw/panel/` holds this branch against its base, each built with the panel's flags, on one A100-SXM4-40GB: 3 paired rounds a row,
`--warm-ms 500`, every weight buffer gated bitwise (`--gate-tol 0`). Paired is the base's time over this branch's, round by round
(the median of 3):

| row | base, ms | this branch, ms | paired |
|---|---|---|---|
| gpt-tiny | 6.190 | 5.831 | x1.0614 |
| gpt-small | 13.727 | 12.999 | x1.0562 |
| gpt-162m | 351.468 | 352.387 | x0.9963 |
| mlp72-r1 | 5.574 | 5.577 | x0.9995 |
| mlp72-r8 | 5.495 | 5.490 | x1.0009 |
| mlp72-r32 | 8.219 | 8.220 | x0.9999 |
| mlp-hetero-r8 | 6.276 | 6.282 | x0.9990 |
| mlp256-r1 | 4.535 | 4.548 | x0.9982 |
| mlp1024-r1 | 6.478 | 6.498 | x0.9954 |

All 9 are bitwise equal to the base, and the geometric mean is x1.0116. Only gpt-tiny and gpt-small run the fused GPT step. The other
seven run no code this branch changes: gpt-162m runs the per-op path, and the MLP rows their own steps.
- **Another load order** (`CUDA_MODULE_LOADING=EAGER` for both binaries, 3 paired rounds): gpt-tiny x1.0624, gpt-small x1.0560, within 0.1
  points of the default load's.
- **The MLP rows with the GPT pass off** (`--no-fuse-gpt`): x0.9971-x1.0002, geometric mean x0.9991.

Two builds also place their code differently in GPU memory, which alone has moved a panel row by up to 5.5% (same device code, another load order), so each row's gain here carries that uncertainty.

## The step's bed

`raw/bed/` holds `bench/fused_gpt/fgpt_bed.cu` built from each tree with the executor's flags (`-O3 -arch=sm_80 -std=c++17 --fmad=false
-Xcompiler -ffp-contract=off -diag-suppress 177 -DKVM_LEAN`): b0 is the base, b1 this branch. Over 20 steps, b1's losses, parameters,
moments and gradients are bitwise b0's on all four shapes. Then 5 rounds of 20 steps, the order alternating, in-kernel time a step:

| shape | base, µs a step | this branch | rounds |
|---|---|---|---|
| gpt-tiny at a gang of 16 | 124.2 | x1.0588 | x1.0579-x1.0615 |
| gpt-small at 32 | 270.8 | x1.0448 | x1.0444-x1.0451 |
| a d128 at 32 | 828.6 | x1.0292 | x1.0235-x1.0299 |
| the d128 hold-out draw at 64 | 925.7 | x1.0231 | x1.0217-x1.0242 |

## Kernels

- **Integer divisions** (`cuobjdump -sass`, the SASS of an integer division by a run-time value, `raw/census/`): the bed's step kernels
  run 129-158 a kernel instance on this branch against 278-349 on the base. The executor's four fused-step instances on this branch run
  131-158.
- **Resources** (`cuobjdump -res-usage`, `raw/panel/key-res-usage.txt`): 21 of the executor's 22 kernels keep the base's registers, stack
  and local memory. One fused-step instance drops from 168 registers to 166. Every instance keeps a 56-byte stack and no local memory.
