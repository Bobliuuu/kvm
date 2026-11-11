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

Precise mode (3xTF32, so the trajectories stay comparable), 2 steps:

| row | G | losses against the VM | worst gradient, of its tensor's scale | worst moment | G against the first G |
|---|---|---|---|---|---|
| gpt-tiny | 1, 2, 4 | bit-identical (4.146039, 3.993824) | 6.1e-6 | 6.3e-6 | 8.2e-6 |
| gpt-small | 16, 32 | within 1.1e-7 relative | 8.0e-5 (G 32), 7.3e-5 (G 16) | 8.0e-5 | 1.8e-5 |

The default 1xTF32 numerics, as the panel runs them:

| check | result |
|---|---|
| gpt-tiny, all 50 steps, G 4: every step's loss against the VM | worst 6.73e-4 relative (first 4.14605 against 4.14604, last 1.24525 against 1.24471) |
| one step, gradients and moments against the VM | gpt-tiny G 4: 9.5e-4 to 1.05e-3 of scale; gpt-small G 32: 1.16e-3 to 1.35e-3 |

After the first AdamW update the 1xTF32 trajectories part from the VM's float32 one. The update is about lr x sign(g), and TF32 noise
flips the sign where a gradient is near zero. So later steps' gradients are reported, not judged, element by element.

## Time

In-kernel time a step (member 0's globaltimer, 20 steps, the GPU warmed first), 1xTF32. Runs on this card vary by about 10%:

| row | G | us a step |
|---|---|---|
| gpt-tiny | 4 | 118.4, 118.7, 103.5 |
| gpt-small | 32 | 234.2, 252.3, 230.4 |
| gpt-small, precise | 32 | 257.7, 264.7 (1xTF32 beside it: 221.0, 240.0) |

**Where gpt-small's step goes** (`fgpt_bed ... 32 clock`, the largest member of each phase, both layers summed; `raw/t20f-small-G32-clock.log`):
- the (sequence, head) units' attention, about 85 us on the critical path (36 forward, 50 backward), while the other 24 members wait;
- the MLP's backward, 26 us; AdamW, 22.5 us; fc1 and fc2, 12 us;
- 22 gang barriers, about 2.8 us each on this card.

Precise mode, three tensor-core products for one on the linear layers, adds only 25-35 us. So the step is bound by instruction and latency
chains (dependent shared-memory loads, conversions, synchronizations, round trips to L2), not by tensor-core throughput.

## In the executor

`kvm_exec` takes the fused path by itself when every task of a bundle belongs to a GPT training run that fuses (its log says
`path: fused GPT kernel`). The pass (`include/kvm_fuse_gpt.h`) rewrites each run into the G members of one gang, and `KvmExec::run_fgpt`
launches each gang as one kernel of G blocks (`include/kvm_fgpt_kernel.cuh`). `--no-fuse` runs the same bundle unfused, on the path it
took before.

**Why a kernel of its own:** ptxas allocates registers per kernel. Compiled as a function inside the static kernel, the program got 153
registers, and up to 149 local reloads a pass in its staging loops. Alone, it gets 237, with at most 69, as on its bed (240 and 75). Alone,
it also leaves the static kernel as it was: every fused function there compiles exactly as in a build without the step. In place, the
step had moved theirs (the split member 156 → 147 registers). `raw/census-*.txt`, read from each build's SASS.

These runs use a test build of the executor: the GPT pack's op bodies are stubs (a fused bundle dispatches none of them), the comm and
extra packs are left out, and the split step's share list is cut to 2, so it compiles within the laptop's memory.

| check, 1xTF32, gang of 32 | result |
|---|---|
| gpt-tiny, all 50 steps' losses against the VM (`fgpt_bed.py --exec`) | worst 6.89e-4 relative (first 4.146049 against 4.146039, last 1.245304 against 1.244709) |
| gpt-small, its first 3 steps' losses against the VM | worst 7.39e-5 relative |

**Time**, us a step over 50 steps. Three arms in rotating order, six rounds a row (`raw/exec-timing-three-arms.log`). This card runs in
two clock states about 1.2x apart, so the table pairs the rounds where both executor arms ran in the fast state:

| row | inside the static kernel | its own kernel | ratio |
|---|---|---|---|
| gpt-tiny | 97.4, 97.8 | 87.5, 87.2 | x1.12 |
| gpt-small | 211.9, 212.1 | 188.1, 188.0 | x1.13 |

Over all six rounds, the median ratio is x1.12 (gpt-tiny) and x1.13 (gpt-small).

## The gang's barrier

A step crosses 8 L + 6 gang barriers: 14 on gpt-tiny, 26 on gpt-small, whose query blocks add 2 L. The program crosses them on a flag
line (`kvm_fgpt::cross_line`). Each member publishes its count of crossings in its own word of one 128-byte line, and warp 0 polls the
whole line, one member a lane. That is one round trip to L2 a crossing, with no atomic. The counter barrier (`kvm_gang::barrier`) waits
for an atomic's round trip, then a release, then a poll.

| read (RTX 5070) | counter barrier | flag line | ratio |
|---|---|---|---|
| one crossing alone, G 32 (`bench/fused_gpt/barrier_bench.cu`, `raw/barrier-bench.log`) | 1,194 ns | 710 ns | x1.68 |
| the executor, gpt-tiny, us a step (8 rounds in alternating order, `raw/exec-barrier-ab.log`) | 87.3 | 82.3 | x1.06 |
| the executor, gpt-small | 187.7 | 178.7 | x1.05 |

With crossings this cheap, gpt-tiny's attention also runs in 2 query blocks a unit: 4 items over its 2 heads, which its 30 idle members
had waited on. The executor's step went from 82.2 to 78.3 us (x1.046, 8 rounds, `raw/exec-qb-ab.log`); gpt-small already ran 4 blocks.
Under the counter barrier, 2 blocks had made gpt-tiny 10% slower.

**Correctness:**
- The bench's ordering check is clean at every G: after each crossing, every block reads every other block's value.
- The bed's outputs are bitwise equal over five 50-step runs of each shape (`raw/bed-determinism.log`).
- Every bar against the VM above holds, unchanged.
- A member's fault ends the gang (`fgpt_bed ... fault=M:F`, `raw/abort.log`): one member given too little shared memory fails a phase
  before the first crossing, or after about ten. Every other member then leaves at its next crossing: error word 5, no member
  finished, no hang. On gpt-small and gpt-tiny, each faulted run ends within 0.1 s of a clean run's wall time.

## Reach: the hold-out's shapes

A run fuses only where a gang's phases fit shared memory. The MLP's forward and backward, the logits, the head's backward and the
embedding's gradient used to stage all N rows of a d-wide stream next to the member's weight slices, so past a small N x d no gang fit.
They now walk N in chunks of R rows, with the weights resident:
- `kvm_fgpt_fit` picks the gang G and the largest chunk that fits: all of N, else 128 rows, else 64.
- A weight gradient, a GEMM over the rows, is written by the first chunk and added to by each later one, in order. The embedding's
  gradient resumes each running sum, so its result is bitwise the whole N's.
- The attention backward's h rows and Wq columns now go below DQ, over its dead buffers, where they fit there.

A seeded hold-out panel draws GPTs with d from 48 to 512, T from 32 to 128, B from 1 to 4 and V up to 512. Over every shape it can draw, on
an A100 (163 KB a block; `kvm_fgpt_fit`, computed on the CPU):

| d | 48 | 96 | 128 | 192 | 256 | 384 | 512 | all |
|---|---|---|---|---|---|---|---|---|
| before | 83% | 67% | 53% | 12.5% | 12.5% | 0% | 0% | 35% (816 of 2,304) |
| row chunks | 100% | 92% | 69% | 21% | 17% | 0% | 0% | 47% (1,088 of 2,304) |

The rest need the weights themselves streamed: at d 512 a member's MLP slices alone exceed its shared memory.

**Checks on an RTX 5070** (`raw/row-chunks-checks.log`, `raw/row-chunks-batch4-one-step.log`):

| check | result |
|---|---|
| the panel's shapes (one chunk each), the bed against the previous head's | bitwise equal: gpt-tiny at G 32 and 8, gpt-small at G 32 |
| forced chunks, precise, against the VM | gpt-small at 64 rows (G 16, 32) and 32 rows: losses within 1.1e-7, gradients within 4.3e-5 of scale; gpt-tiny at 16 rows: bit-identical losses, gradients within 7.3e-6 |
| a batch-4 gpt-small (N 256: 128-row chunks by the fit), bed, precise | losses within 1.1e-7 of the VM's; after one step, gradients within 2.3e-6 of scale |
| the same, the executor, 1xTF32, 3 steps | fused, losses within 7.0e-5 of the VM's |
| the executor on the panel rows, us a step (8 paired rounds, against the previous head) | gpt-tiny 78.3 to 75.9 (x1.032), gpt-small 179.2 to 177.4 (x1.010); with the bias corrections' table, now filled once a run |

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
```

Files: `raw/`, the runs above (paths replaced by `<work>`).
