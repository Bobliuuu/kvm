# The fused GPT step on its bed and in the executor: correctness against the workload VM, and its time

GPU: NVIDIA GeForce RTX 5070 Laptop GPU (sm_120, 36 SMs; the SMs ran at 2.5-2.8 GHz, measured as cycles over globaltimer
nanoseconds), CUDA 12.8, `nvcc -O3 -std=c++17 -arch=sm_120`. **Not an A100**: these runs give correctness, the shares of a step, and a
hint of its time.

Tree: this branch. Workloads: `bench/vs_pytorch/gen.py` at this tree, `gpt_tiny` and `gpt_small`, 50 steps, `--tiling a100`, in the
default `--mode tf32` (matmul_tile's 1xTF32) and in `--mode precise` (3xTF32).

## What runs

`bench/fused_gpt/fgpt_bed.cu` launches the device program (`include/kvm_fused_gpt.cuh`) as one cooperative launch of G blocks, one
gang member each, for every step of the run. `bench/fused_gpt/fgpt_bed.py` writes the model and its token streams from a workload, runs
the bundle's own tasks for the same steps on the workload VM (float32 a op), and judges the bed's run against it:
- every step's loss;
- the last step's gradients and AdamW moments, each tensor's largest difference over its largest magnitude.

The executor runs the same program on a kernel of its own: [In the executor](#in-the-executor).

## Correctness

Precise mode (3xTF32), 2 steps:

| row | G | losses against the VM | worst gradient, of its tensor's scale | worst moment | G against the first G |
|---|---|---|---|---|---|
| gpt-tiny | 1, 2, 4 | bit-identical (4.146039, 3.993824) | 6.1e-6 | 6.3e-6 | 8.1e-6 |
| gpt-small | 16, 32 | within 1.1e-7 relative | 6.1e-5 (G 32), 5.5e-5 (G 16) | 8.0e-5 | 1.7e-5 |

Both rows are an earlier build's reads (`raw/v1f-w_tiny_p-k2.log`, `raw/v1f-w_small_p-k2.log`). The build that ran the d128 read gpt-small
again: gradients within 6.8e-5 of scale at G 32, G 32 against G 16 within 1.5e-5 (`raw/d128.log`).

The default 1xTF32 numerics, as the panel runs them:

| check | result |
|---|---|
| gpt-tiny, all 50 steps, G 4: every step's loss against the VM | worst 6.73e-4 relative (first 4.14605 against 4.14604, last 1.24525 against 1.24471) |
| one step, gradients and moments against the VM | gpt-tiny G 4: 9.5e-4 to 1.05e-3 of scale; gpt-small G 32: 1.16e-3 to 1.35e-3 |

After the first AdamW update the trajectories part from the VM's, in either numerics. The update is about lr x sign(g), and rounding
flips its sign where a gradient is near zero, float32's as well as TF32's. On these two rows precise mode stayed within the bars over 2
steps; a d128 GPT did not (within 3.8e-6 of scale after one step, 9.21e-4 after two: Reach, below). Started from the VM's own weights and
moments after step 1 (`fgpt_bed.py --resync 1`), its step 2's gradients agree within 5.0e-6 (second moments 5.4e-6;
`raw/resync-step2.log`): the split is the two
trajectories parting at AdamW's first update, not an error in the step. So the bed judges gradients, moments and the gang sizes against
each other after one step, a later step from the VM's own state (`--resync`), and a free-running longer run by its losses; the d128 run
that printed FAIL under the old rule reads "reported" under this one.

## Time

In-kernel time a step (member 0's globaltimer, 20 steps, the GPU warmed first), 1xTF32. Runs on this card vary by about 10%:

| row | G | us a step |
|---|---|---|
| gpt-tiny | 4 | 118.4, 118.7, 103.5 |
| gpt-small | 32 | 234.2, 252.3, 230.4 |
| gpt-small, precise | 32 | 257.7, 264.7 (1xTF32 beside it: 221.0, 240.0) |

**Where gpt-small's step goes** (`fgpt_bed ... 32 clock`, the largest member of each phase, both layers summed; `raw/t20f-small-G32-clock.log`):
- the (sequence, head) units' attention, about 86 us on the critical path (36.6 forward, 49.8 backward), while the other 24 members wait;
- the MLP's backward, 27.3 us; AdamW, 22.8 us; fc1 and fc2, 12.3 us;
- 22 gang barriers, about 2.8 us each on this card.

Precise mode, three tensor-core products for one on the linear layers, adds only 25-35 us. So the step is bound by instruction and latency
chains (dependent shared-memory loads, conversions, synchronizations, round trips to L2), not by tensor-core throughput.

## In the executor

`kvm_exec` takes the fused path by itself when every task of a bundle belongs to a GPT training run that fuses (its log says
`path: fused GPT kernel`). The pass (`include/kvm_fuse_gpt.h`) rewrites each run into the G members of one gang, and `KvmExec::run_fgpt`
launches each gang as one kernel of G blocks (`include/kvm_fgpt_kernel.cuh`). `--no-fuse` runs the same bundle unfused, on the path it
took before.

**Why a kernel of its own:** ptxas allocates registers per kernel. Compiled as a function inside the static kernel, the program got 153
registers, and its MMA loops reloaded up to 179 values a pass (`raw/census-static-kernel-with-the-step-inside-by-function.txt`, each loop
attributed to its function). Alone, before the sm_80 changes below, it got 239, with at most 69 (`raw/census-fused-gpt-kernel*.txt`); on its
bed, 243 and 75 (`raw/census-bed.txt`). In place, it had also moved the other fused functions' allocation (the split member 156 → 147
registers), read on the base before #31 gave every fused form a kernel entry of its own (`raw/census-static-kernel-*.txt`).

**On sm_80 (the A100's architecture) the same program spilled.** The executor's kernel held 252 registers there, with 518 local reloads and
a worst MMA loop (the step's own loop, which holds every MMA) reloading 431 values a pass, where on sm_120 it held 239, 129 and 69 (`raw/census-sm80-before-fused-gpt-kernel-*.txt`,
`raw/census-fused-gpt-kernel*.txt`). Its long-lived state rode in registers through every phase: the scratch layout, which a
runtime layer index put in each thread's local memory, then the sizes, slices and pointers into shared memory. sm_80's phases need more
registers of their own, so it spilled there. Three changes:
- the scratch layout in shared memory, one copy a block;
- `ld()` as a volatile load: `__ldcg`'s `asm volatile` was a memory barrier to the compiler, which read every value it kept in memory
  again after each load;
- the run's long-lived state in that shared copy, read in place, each looping phase taking the values it uses for its own body, and the
  GEMM's k-loop kept rolled as before (with registers free the front end had unrolled it four times).

The executor's kernel now holds 195 registers on sm_80 with 10 local reloads in all and a worst MMA loop of 7, and 138, 4 and 1 on sm_120
(`raw/census-sm80-fused-gpt-kernel-*.txt`, `raw/census-sm120-fused-gpt-kernel-*.txt`). Every output stays bitwise equal on five
shapes, and the gang's abort still ends every faulted run (`raw/sm80-fix-bed.log`); the executor runs four GPT workloads fused, with the
same losses (`raw/sm80-fix-exec.log`). On this card, against the form before the third change, the best runs read gpt-small x0.987,
gpt-tiny x1.017 and the d128 x0.993 (9 rotating rounds, 3 for the d128; `raw/sm80-fix-timing.log`), and the medians x0.812, x0.910 and
x1.004. The card has a slow mode: a run takes 1.1-1.5x the cycles a step of a fast one, at the same SM clock. This change ran in it in 7
of 9 gpt-small rounds against the form before's 1 of 9 (Fisher's exact test, two-sided p = 0.015), 6 against 4 of 9 on gpt-tiny (p =
0.64) and 2 against 2 of 3 on the d128; where both arms ran fast, it is within about 1.5%. Whether the slow mode is the
card's or the code's is open, so the A100 read, which holds one SM clock, judges by medians and slow-run counts. On sm_120 the census
read 129 local loads, 242 local stores and a worst loop of 69 before the three changes, and 4, 5 and 1 after.

**The gang's partial sums.** A member summed the gang's partials (each row's G member partials, an attention element's item partials, a
replicated parameter's member partials, the gradient norm's G partials) in rolled loops of dependent adds, so most of a sum's round trips
to L2 ran one after another. `sum_l2` issues up to 4 to 32 loads before its first add (by site) and adds them in the same order, so the
result is the rolled loop's, bit for bit: every output stays bitwise equal on five shapes (`raw/partial-sums.log`), and the executor runs
four GPT workloads fused with the same losses as before it (`raw/partial-sums-exec.log`). On this card, best runs, medians and paired
by round: gpt-small x1.049 (168.2 against 176.5 us), x1.131 and x1.060, slow in 3 rounds of 9 against 4 before it; a d128 x1.022,
x1.002 and x1.036 (4 rounds, 1 and 1 slow); gpt-tiny x0.996, x1.167 and x1.191, slow in 2 rounds against 5. Over the rounds where both
arms ran fast: x1.060 (3 rounds), x1.036 (2) and x0.979 (2) (`raw/partial-sums.log`). In cycles a step over the fast runs, which neither
the slow mode nor the SM clock moves, it takes 4.6% fewer on gpt-small, 1.6% fewer on the d128 and 1.6% more on gpt-tiny: gpt-tiny's
median comes from its control's slow rounds, not from the code. The executor's kernel holds 203 registers on sm_80 (195 before), its 10 local
reloads and worst MMA loop of 7 unchanged, and 145 on sm_120 with 4 and 1 (`raw/census-sm80-partial-sums-fused-gpt-kernel-*.txt`,
`raw/census-sm120-partial-sums-fused-gpt-kernel-*.txt`).

These runs use a test build of the executor: the GPT pack's op bodies are stubs (a fused bundle dispatches none of them), the comm and
extra packs are left out, and the split step's share list is cut to 2, so it compiles within the laptop's memory.

| check, 1xTF32, gang of 32 | result |
|---|---|
| gpt-tiny, all 50 steps' losses against the VM (`fgpt_bed.py --exec`; `raw/sm80-fix-exec.log`, `raw/partial-sums-exec.log`) | worst 6.90e-4 relative (first 4.146049 against 4.146039, last 1.245289 against 1.244709) |
| gpt-small, its first 3 steps' losses against the VM | worst 7.39e-5 relative |

**Time**, us a step over 50 steps. Three arms in rotating order, six rounds a row (`raw/exec-timing-three-arms.log`). This card has
a slow mode (a run takes about 1.2x the time a step; this log prints no clock), so the table pairs the rounds where both
executor arms ran fast:

| row | inside the static kernel | its own kernel | ratio |
|---|---|---|---|
| gpt-tiny | 97.4, 97.8, 97.5 | 87.5, 87.2, 87.0 | x1.12 |
| gpt-small | 211.9, 211.8, 212.1 | 188.1, 188.1, 188.0 | x1.13 |

Over all six rounds, the median ratio is x1.12 (gpt-tiny) and x1.13 (gpt-small); the own kernel ran slow in 2 rounds of 6 against 2
inside the static kernel (gpt-tiny) and 2 against 1 (gpt-small).

## The gang's barrier

A step crosses 8 L + 6 gang barriers: 14 on gpt-tiny, 26 on gpt-small, whose query blocks add 2 L. The program crosses them on the gang's flag
lines (`kvm_gang::barrier_lines`, from #29). Each member publishes its count of crossings in its own word of a 128-byte line, and warp 0
polls the whole line, one member a lane (a gang of 32 is one line). That is one round trip to L2 a crossing, with no atomic. The counter
barrier (`kvm_gang::barrier`) waits for an atomic's round trip, then a release, then a poll.

| read (RTX 5070) | counter barrier | flag line | ratio |
|---|---|---|---|
| one crossing alone, 32 members (`bench/gang/barrier_bench.cu`, `results/2026-11-10-static-gang-tasks/flag-lines/`; unverified) | 1,173 ns | 721 ns | x1.63 |
| the executor, gpt-tiny, us a step (medians of 8 rounds in alternating order, no run of either arm slow, `raw/exec-barrier-ab.log`) | 87.3 | 82.3 | x1.06 |
| the executor, gpt-small | 187.7 | 178.7 | x1.05 |

With crossings this cheap, gpt-tiny's attention also runs in 2 query blocks a unit: 4 items over its 2 heads, which its 30 idle members
had waited on. The executor's step went from 82.2 to 78.3 us by median (x1.049, paired by round x1.046, no run slow; 8 rounds, `raw/exec-qb-ab.log`);
gpt-small already ran 4 blocks.
Under the counter barrier, 2 blocks had made gpt-tiny 10% slower.

**Correctness:**
- The bench's ordering check is clean at every G: after each crossing, every block reads every other block's value.
- The bed's outputs are bitwise equal over five 50-step runs of each shape (`raw/bed-determinism.log`).
- On `kvm_gang::barrier_lines` in place of this branch's own copy of the flag line, every output stays bitwise equal
  (`raw/restack-bitwise.log`), and where both arms ran fast the executor's step is no slower: gpt-tiny 76.3 to 74.9 us over 3 rounds,
  gpt-small 182.3 to 174.0 over 1. By median it reads x0.798 and x0.926, with slow runs in 6 rounds of 10 against 4 (gpt-tiny) and 6
  against 5 (gpt-small), not a significant split (Fisher p 0.66 and 1.0; `raw/restack-exec-paired.log`, a loaded laptop).
- Every bar against the VM above holds, unchanged.
- A member's fault ends the gang (`fgpt_bed ... fault=M:F`, `raw/abort.log`): one member given too little shared memory fails a phase
  before the first crossing, or after about ten. Every other member then leaves at its next crossing: error word 5, no member
  finished, no hang. Each faulted run ends within about 0.1 s of a clean run of its shape, the wall time mostly the process's start: on
  the sm_80 head gpt-small 0.90 and 0.90 s against 0.86, gpt-tiny 1.15 against 1.03 (`raw/sm80-fix-bed.log`; `raw/abort.log` is an
  earlier build's).

## Reach: the hold-out's shapes

A run fuses only where a gang's phases fit shared memory. The MLP's forward and backward, the logits, the head's backward and the
embedding's gradient used to stage all N rows of a d-wide stream next to the member's weight slices, so past a small N x d no gang fit.
They now walk N in chunks of R rows, with the weights resident:
- `kvm_fgpt_fit` picks the gang G and the largest chunk that fits: all of N, else 128 rows, else 64.
- A weight gradient, a GEMM over the rows, is written by the first chunk and added to by each later one, in order. The embedding's
  gradient resumes each running sum, so its result is bitwise the whole N's.
- The attention backward's h rows and Wq columns now go below DQ, over its dead buffers, where they fit there.

A seeded hold-out panel draws GPTs with d from 48 to 512, T from 32 to 128, B from 1 to 4 and V up to 512. Over every shape it can draw, on
an A100 (163 KB a block; computed on the CPU by `raw/reach.cpp`, built against the tree before row chunks and against this branch:
`raw/reach.txt`):

| d | 48 | 96 | 128 | 192 | 256 | 384 | 512 | all |
|---|---|---|---|---|---|---|---|---|
| before | 83% | 67% | 53% | 12.5% | 12.5% | 0% | 0% | 35% (816 of 2,304) |
| row chunks | 100% | 92% | 69% | 21% | 17% | 0% | 0% | 47% (1,088 of 2,304) |

The rest need the weights themselves streamed: at d 512 a member's MLP slices alone exceed its shared memory.

**Checks on an RTX 5070** (`raw/row-chunks-checks.log`, `raw/row-chunks-batch4-one-step.log`):

| check | result |
|---|---|
| the panel's shapes (one chunk each), the bed against the previous head's | bitwise equal: gpt-tiny at G 32 and 8, gpt-small at G 32 |
| forced chunks, precise, against the VM | gpt-small at 64 rows (G 16, 32) and 32 rows: losses within 1.1e-7, gradients within 4.2e-5 of scale; gpt-tiny at 16 rows: bit-identical losses, gradients within 7.2e-6 |
| a batch-4 gpt-small (N 256: 128-row chunks by the fit), bed, precise | losses within 1.1e-7 of the VM's; after one step, gradients within 2.3e-6 of scale |
| the same, the executor, 1xTF32, 3 steps | fused, losses within 7.0e-5 of the VM's |
| the executor on the panel rows, us a step (8 paired rounds, against the previous head) | gpt-tiny 78.3 to 75.9 (x1.032), gpt-small 179.2 to 177.4 (x1.010); with the bias corrections' table, now filled once a run |
| chunks whose last one is shorter (N not a multiple of the chunk, as a hold-out d128 on an A100: 128 + 64 rows), precise, against the VM (`raw/row-chunks-shorter-last.log`) | gpt-small at 96, 80, 48 and 112 rows, the batch-4 at 96 and 112: losses within 1.1e-7 over 2 steps |
| a d128 GPT (4 layers, 4 heads of 32, T 64, B 2: N 128, V 256; 64-row chunks by the fit), the widest the draw makes that fits this card (`raw/d128.log`) | correct at steps 1 and 2. Bed, precise, after one step: gradients within 3.8e-6 of scale, moments within 6.7e-6; step 2 from the VM's own step-1 state: gradients within 5.0e-6, second moments 5.4e-6 (gpt-small 2.1e-6, the batch-4 model 1.9e-6; `raw/resync-step2.log`). Run free over two steps, the losses stay within 1.8e-7 but the gradients read 9.21e-4 of scale (`w.gpt.h0.mlp.fc1.w`): the bed printed FAIL against its 3e-4 bar, which gpt-tiny and gpt-small pass (6.1e-6, 6.8e-5), and this branch's change to the bed relabels that run "reported"; the resynced step shows the trajectories parting, not the step. The executor, 1xTF32, 3 steps: fused, losses within 2.6e-4 of the VM's |
| the same d128, time (`raw/d128-time.log`) | the executor 790-808 us a step over 30 steps in its fast runs (1,000-1,021 in its slow ones); the bed's phase clock 827 us at 2.40 GHz, of which the MLP's backward 21%, AdamW 14%, the attention's and the replicated parameters' sums 10% |

## Limits

- **A laptop GPU, not an A100.** The A100 runs instruction-bound code at a lower clock (1.41 GHz) and tensor math much faster, and its
  barrier costs about half. The A100's number is its own read.
- **gpt-small with G below 16** needs more than this card's 99 KB of shared memory a member; the bed refuses those gangs by name. The
  A100's 163 KB takes smaller gangs.

## Reproduce

```
python3 bench/vs_pytorch/gen.py W/w_tiny --model gpt_tiny --steps 50 --mode tf32 --tiling a100        # and --mode precise, gpt_small
nvcc -O3 -std=c++17 -arch=sm_120 -Iinclude bench/fused_gpt/fgpt_bed.cu -o fgpt_bed
python3 -I bench/fused_gpt/fgpt_bed.py W/w_tiny --bed ./fgpt_bed --steps 2 --gangs 1,2,4               # judges against the VM
./fgpt_bed W/fgpt-bed-w_tiny-k2 4 clock                                                                  # the phase table
nvcc -O3 -arch=sm_120 -std=c++17 -Iinclude --fmad=false -Xcompiler -ffp-contract=off kvm_exec.cu -o kvm_exec  # the executor
python3 -I bench/fused_gpt/fgpt_bed.py W/w_tiny --exec ./kvm_exec --steps 50 --tol 2e-3                  # its own path: the fused GPT kernel
./kvm_exec --no-fuse W/w_tiny/bundle                                                                     # the same bundle, unfused
g++ -std=c++17 -O1 -Iinclude results/2026-11-11-fused-gpt-step-bed/raw/reach.cpp -o reach && ./reach  # the reach table's row (-DBEFORE on the tree before row chunks: the first)
```

Files: `raw/`, the runs above (paths replaced by `<work>`).
