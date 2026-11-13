# A gang crossing's arrival, released without the SC fence (A100)

A gang crossing (`include/kvm_gang.cuh`) is message passing. Each member stores its results, then thread 0 announces its arrival. The waiters'
acquire loads see every arrival, and a `__syncthreads` then makes every member's stores visible to its block. The arrival used to be released by a
`__threadfence()` and then the release operation: two fences in SASS, `MEMBAR.SC.GPU` and then `MEMBAR.ALL.GPU`. The release paired with the
pollers' acquires carries the protocol, so this branch makes the release operation alone the default (`KVM_GANG_FENCE` 1). The other two forms
stay selectable for the bench and the device test: the old one (0), and CUTLASS's `GenericBarrier` form (2), which is `fence.acq_rel.gpu`
followed by a relaxed operation.

Every figure here was re-derived from these raw files by the team's verifier (Nov 12). In `raw/`, the host's working folders read
`/root/work/` and the GPU lock's own lines are removed. Two more edits: the panel's job log is renamed `panel/panel.log`, and its first and last
lines lose an internal label (the candidate reads "this branch"). Everything else is as the runs wrote it.

The crossing bench and the device test (`raw/crossing/`, `raw/device-test/`) ran from the source before its fixups were squashed (`353618a`,
named in `raw/crossing/MANIFEST.txt`). They compile `kvm_gang.cuh` alone, which equals this branch's but for the default, and they set the
fence form with `-DKVM_GANG_FENCE`. The beds (`raw/bed/`) ran from this branch's own tree, 12 warps a member as it ships.

## The panel's 9 rows against the base

`raw/panel/` holds this branch against its base (#54 with #56's commit), each built with the panel's flags, on one A100-SXM4-40GB: 3 paired
rounds a row, `--warm-ms 500`, every weight buffer gated bitwise (`--gate-tol 0`).

| row | base, ms | this branch, ms | paired |
|---|---|---|---|
| gpt-tiny | 6.504 | 6.454 | x1.0077 |
| gpt-small | 13.770 | 13.722 | x1.0035 |
| gpt-162m | 351.536 | 351.989 | x1.0001 |
| mlp72-r1 | 5.585 | 5.572 | x1.0014 |
| mlp72-r8 | 5.503 | 5.512 | x0.9989 |
| mlp72-r32 | 8.255 | 8.253 | x0.9984 |
| mlp-hetero-r8 | 6.282 | 6.284 | x1.0006 |
| mlp256-r1 | 4.560 | 4.479 | x1.0208 |
| mlp1024-r1 | 6.472 | 6.380 | x1.0143 |

All 9 are bitwise equal to the base, and the geometric mean is x1.0051. The four rows whose steps cross a gang (gpt-tiny, gpt-small, mlp256-r1,
mlp1024-r1) gain. The other five run no gang crossing: gpt-162m, and mlp72-r1, r8, r32 and hetero-r8, which run the one-block MLP step.
Every kernel of the branch's executor keeps the base's registers, stack and local memory (`cuobjdump -res-usage`, 20 of 20:
`raw/panel/key-res-usage.txt`).

## A crossing

`raw/crossing/` holds `bench/gang/fence_bench.cu`: the three forms in one process, 15 paired repetitions, every arm faster in 15 of 15.

| gang | counted barrier, 0 → 1, ns | flag lines, 0 → 1, ns |
|---|---|---|
| 16 | 1,375 → 1,272 | 1,252 → 1,170 |
| 32 | 1,460 → 1,358 | 1,342 → 1,264 |
| 64 | 1,534 → 1,416 | 2,187 → 1,864 |
| 108 | 1,655 → 1,537 | 3,908 → 3,024 |

## The fused GPT step's bed

`raw/bed/`, forms 0 and 1 of `bench/fused_gpt/fgpt_bed.cu` built from this branch's tree, 5 paired rounds with the order alternating, 20 steps on
the phase clock; form 1 is faster in every round:
- gpt-tiny at a gang of 16: 133.6 → 131.6 µs a step (x1.0152);
- the d128 hold-out draw at 64: 949.8 → 944.1 µs (x1.0063).

These beds are built with plain `nvcc -O3 -std=c++17 -arch=sm_80`, not with the executor's flags (`--fmad=false -Xcompiler -ffp-contract=off
-DKVM_LEAN`), so they run other code generation of the same step: they show the mechanism, and the panel's 9 rows above, on the executor's
own builds, carry the gain.
Two builds also place their code differently in GPU memory, which alone has moved a panel row by up to 5.5% (same device code, another load order), so each row's gain here carries that uncertainty.

## The device test

`raw/device-test/` holds `tests/static/gang_device.sh`: `test_gang_device.cu`, built in each form and with its mutants.
- **The forms pass:** gangs of 1, 2, 3, 5, 8, 13 and 76 blocks side by side (108 in all), then one gang of 108. In each form, every member
  reads every slot of its round, from L2 and through L1, in the first launch and in a staggered second. An aborted third launch ends without a hang.
- **The no-wait mutant** is caught both ways.
- **Relaxed pollers** are caught through L1 in form 1, which rests on the pollers' acquire. Forms 0 and 2 release with a stronger fence that
  happens to hide that mutant, so the test only reports it there (`gang_device.txt`).
