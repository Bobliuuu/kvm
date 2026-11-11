# The fused GPT step on its bed: correctness against the workload VM, and its time

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

The executor's dispatch of the op, the pass in `fuse_bundle` and the executor's descriptors are not in this branch yet: the bed runs the
program alone.

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
```

Files: `raw/`, the runs above (paths replaced by `<work>`).
