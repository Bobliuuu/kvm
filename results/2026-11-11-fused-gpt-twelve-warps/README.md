# The fused GPT step on 12 warps a member (A100)

`KVM_FGPT_NT` (include/kvm_fused_gpt.h) grows from 256 threads to 384: each gang member's block runs 12 warps, three to each SM
sub-partition where there were two. The step's tile loops are latency-bound: each k step waits on its fragments' shared-memory loads and
on the MMA chain. With a third warp on each sub-partition there is one more to switch to. The kernel's registers fall to fit 384 threads a
block without spilling.

The step is not bitwise to the 256-thread build: the column sums' chunks, the block reductions' partials and the squares' per-thread sums
follow the block size. It is judged by the panel's loss gates and by the step-1 first moments against float64 (below).

Every figure here was re-derived from these raw files by the team's verifier (Nov 11). In `raw/`, the host's working folders read
`/root/work/` and the GPU lock's own lines are removed; everything else is as the runs wrote it.

## The bed

`bench/fused_gpt/fgpt_bed.cu` on an A100-SXM4-80GB, built at 256, 384 and 512 threads from one source (`d6f6c75`, whose fused GPT
sources and bed equal #48's head's, with `KVM_FGPT_NT` made overridable for the bed). Each shape ran at its gang width, 20 steps with the phase clock, 5 rounds with the arms' order rotating, no slow
run. The 384- and 512-thread builds pass the workload's float64 checks first: gpt-tiny and gpt-small at 2 steps, the two d128s at one
step and resynced (`raw/bed/`).

| shape | members | 256 threads, µs a step | 384 | 512 | 384 over 256 (rounds) | 512 over 256 |
|---|---|---|---|---|---|---|
| gpt-tiny | 16 | 135.0 | 132.3 | 140.7 | x1.020 (x1.017-1.023) | x0.959 |
| gpt-small | 32 | 290.5 | 276.3 | 294.1 | x1.051 (x1.048-1.052) | x0.987 |
| d128 | 64 | 852.3 | 789.4 | 800.0 | x1.079 (x1.079-1.080) | x1.065 |
| d128 hold-out draw | 64 | 1,055.8 | 955.9 | 965.3 | x1.104 (x1.103-1.105) | x1.094 |

Registers (the plain and the clocked instance, `raw/bed/build-*.log`): 182 / 178 at 256 threads, 160 / 157 at 384, 128 / 128 at 512.
None spill except 8 bytes in the plain instance at 512. At 512 threads the 128-register cap costs the small shapes their gain.

## The executor

This branch (`67819f6`) against its base (`2a8692b`, the head of #48), each built with the panel's flags, both on one A100-SXM4-80GB: the
fused GPT step's three rows, 9 interleaved rounds a row, `--warm-ms 500`, no slow run in either arm (`raw/executor/ab_*.json`).

| row | members | base, ms | this branch, ms | paired | rounds |
|---|---|---|---|---|---|
| d128 hold-out draw (d 128, 4 layers, T 96, B 2, 30 steps) | 64 | 31.446 | 28.915 | **x1.088** | x1.086-1.089 |
| gpt-small (50 steps) | 32 | 14.323 | 13.812 | **x1.038** | x1.035-1.042 |
| gpt-tiny (50 steps) | 16 | 6.628 | 6.488 | **x1.021** | x1.019-1.024 |

The executor's `kvm_fgpt_kernel` goes from 178 registers to 168, with a 56-byte stack and no local memory in either (`cuobjdump
-res-usage`). Rows that run no fused GPT step run the same code: every other kernel's SASS is identical in the two builds, 18 of 19
functions, and `kvm_fgpt_kernel` goes from 41,120 instructions to 38,256 (`raw/executor/sass-identity.txt`, by `sass_ident.py` beside it).

## Numerics

- **Every step's loss**, this branch against the base: within 2e-3 on gpt-tiny (largest relative difference 2.99e-5) and gpt-small
  (4.76e-5), and within 1e-2 on the d128 hold-out draw (9.44e-4) (`raw/executor/run.log`).
- **Against PyTorch's eager step** (`run_gpt.py`, the panel's gate): the executor's losses on this branch differ by at most 1.05e-4
  (gpt-tiny) and 6.41e-4 (gpt-small) against 2e-3, and by 7.13e-3 on the hold-out draw against its 1e-2 (`raw/executor/gpt_*.json`).
- **The first moments after step 1 against float64** on the hold-out draw, this branch, the base and eager TF32 (`raw/executor/first-moments-ho128.json`):
  the geometric means of the relative L2 are 6.776899e-4, 6.776894e-4 and 7.037210e-4. 16 of the 68 first-moment buffers differ from
  the base's, by at most 3.4e-5 in their distance. This branch's geometric mean is within 1.05x the larger of the other two, and no buffer
  exceeds twice the larger reference, so the check passes; it passes for the base and for eager taken as the candidate too.
