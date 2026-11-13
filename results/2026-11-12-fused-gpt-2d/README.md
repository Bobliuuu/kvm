# The GPT step's 2-D form (A100)

Until this branch, the fused GPT step (#33) fused a GPT run only when some gang of at most 108 members fits the step's phases in a
member's shared memory. On an A100 the d384 draw (3 layers, d 384, 6 heads of 64, MLP 1536, vocabulary 256, 4 sequences of 96) fits
no gang, so it ran op by op. The 2-D form runs that step as one gang as well. The attention half runs on a 6 x 12 layout of members and
the MLP half on 6 x 16: 96 members, 3 query blocks a unit, 16 vocabulary shards and 38,080 floats of shared memory a member. The pass
picks the 2-D form only for runs the fused step has no gang for, so every run the fused step already takes stays as it was.

Every figure here was re-derived from these raw files by the team's verifier (Nov 12). In `raw/`, the host's working folders read
`/root/work/` and the GPU lock's own lines are removed. Two more edits: the panel's job log is renamed `panel/panel.log`, and the three
headers of `numerics/outputs.md5` are reworded to drop internal labels. Everything else is as the runs wrote it.

## d384 on one A100

`raw/d384/` holds one A100-SXM4-40GB (1,095 MHz at the start, 400 W cap), one hold, 20 steps of the draw. `run_gpt.py` timed PyTorch's
modes and main's executor in one run, and this branch's executor in a second, with 7 interleaved reps each. Main's executor is `4bbdaab`
built with the panel's flags, which runs this draw op by op. The 2-D form's bed (`bench/fused_gpt/f2d_bed.cu`) ran 3 times.

| arm | median µs a step |
|---|---|
| PyTorch, compiled and graph-captured (`compile_graph`, its fastest mode) | 1,325.2 |
| main's executor, op by op | 3,676.1 |
| this branch's executor, the 2-D form | **2,824.2** |
| the 2-D form's bed | 2,862.0 (2,860.6-2,866.5) |

- **This branch over main: x1.302.** The reps are tight: main 3,672.8-3,681.4 µs, this branch 2,821.4-2,825.5.
- **PyTorch's fastest mode still runs 2.13x faster** (this branch at x0.469 of it).
  - PyTorch's modes ran in main's run, so that ratio divides medians from two runs in one hold. It is not paired by rep.
  - Eager read 13,664 and 12,437 µs in the two runs, a 9% spread. The graph-captured modes are the stable ones.
- **The bench's loss gate:** 20 steps of this draw are chaotic. Every arm that changes eager's arithmetic drifts from eager's losses
  as the steps go on (`raw/d384-losses/losses_by_step.txt`); eager's own graph captures match it exactly.
  - The largest difference over the 20 steps: main's executor 8.5% and this branch 11.0%, both at step 19, and PyTorch's compiled
    modes 7.3-7.5% (the bench records only their largest).
  - Main's executor and this branch agree with eager to 2.1e-6 at step 0, and this branch to 2.1e-4 at step 1.
  - That earlier run of the same read failed the bench's default 2e-3 gate on every arm; its records are the `*.failed.json` beside
    the table. Timing was therefore read with the diagnostic gate at 0.15 (`--tol 0.15 --compile-tol 0.15`), and correctness is
    judged against float64 on the first step (below).

## The panel's 9 rows against the base

`raw/panel/` holds this branch against its base (#54 with #56's commit), each built with the panel's flags, on the same A100: 3 paired
rounds a row, `--warm-ms 500`, and every weight buffer gated bitwise (`--gate-tol 0`).

| row | tasks | base, ms | this branch, ms | paired |
|---|---|---|---|---|
| gpt-tiny | 6,749 | 6.511 | 6.519 | x0.9989 |
| gpt-small | 28,858 | 13.771 | 13.771 | x1.0000 |
| gpt-162m | 161,316 | 352.289 | 352.311 | x0.9990 |
| mlp72-r1 | 9,600 | 5.578 | 5.573 | x1.0007 |
| mlp72-r8 | 76,800 | 5.500 | 5.507 | x0.9989 |
| mlp72-r32 | 307,200 | 8.251 | 8.223 | x1.0032 |
| mlp-hetero-r8 | 68,400 | 6.285 | 6.277 | x1.0013 |
| mlp256-r1 | 8,394 | 4.538 | 4.551 | x1.0013 |
| mlp1024-r1 | 17,117 | 6.494 | 6.498 | x0.9992 |

No panel row takes the 2-D form. gpt-tiny and gpt-small keep the fused step's gang, and the other rows run as they did. All 9 are
bitwise equal to the base, and the geometric mean of the paired medians is x1.0003.

## Numerics

- **The device program** (`include/kvm_f2d.cuh`) is generated from the development beds the step was verified on.
  - On the A100, its d384 1xTF32 step writes the same five files as the bed's step, byte for byte: losses, parameters, both moments and
    gradients (`raw/numerics/outputs.md5`).
  - The bed's other layout of the gradient norm's partial sums adds the squares in other shares. With clipping engaged, its parameters
    and moments differ in rounding while its losses and gradients match.
- **Against float64** (the workload's VM), one step at d384 (`raw/numerics/`):
  - **The precise build (3xTF32)** passes: the loss is within 8.51e-8 relative, the gradients within 1.30e-5 of their tensor's scale,
    and both moments within 2.15e-5 (`judge-d384p.log`).
  - **The 1xTF32 build, the one the executor runs,** gives a loss of 5.600117 against 5.600113 (5.96e-7 relative). Its gradients sit
    1.30e-3 of their tensor's scale away. That is TF32's rounding, which the precise build's float64 bar refuses (`judge-d384.log`).
  - **The executor's own first loss** is 5.600117 against the VM's 5.600113 (`exec-d384.log`).
- The executor runs the 2-D form for 1xTF32 bundles only. A precise d384 bundle stays unfused, with its reason.

## Build

- **The fused step is unchanged:** its helpers now take the block's thread count, and with the default the fused step's sm_80 PTX is
  byte-identical (`raw/build/fgs_ptx_same.out`).
- **The 2-D form's kernel** uses 255 registers, a 96-byte stack and no local memory in both barrier instances (`raw/build/res-usage.txt`,
  from this branch's executor binary). It runs 256 threads a member: at 255 registers, 12 warps spill.
- **`tests/static/test_fuse_gpt.cpp`** now checks each run's form (`raw/build/test_fuse_gpt.out`):
  - d384 1xTF32 fuses as the 2-D form's gang of 96, and d192 as its gang of 32;
  - d128 stays the fused step's gang of 64, gpt-tiny its 16 and gpt-small its 32;
  - d384 precise stays unfused, with its reason.
