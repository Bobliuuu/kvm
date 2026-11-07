# gpt_162m: executor vs PyTorch eager / CUDA graph / torch.compile, and the step-1 numerics check (A100, 2026-11-06)

**Verdict.** On the 162M GPT (12 layers, d768, 12 heads, seq 1024, batch 4, 4096 tokens/step) the executor is **numerically right and 15x to 42x
slower than PyTorch**. Its best measurement is 1.04 s per optimizer step (one-step tape) against 58.8 ms for `torch.compile` + whole-step CUDA graph
and 69.6 ms for eager. README goal 3 is not met at this size either, and the gap is larger than at the small sizes of `2026-11-06-vs-pytorch`
(3.2-12.6x there). Its step-1 loss and all 196 gradient tensors agree with a float64 reference exactly as well as PyTorch's own 1xTF32 does.

## Setup

| | |
|---|---|
| GPU | NVIDIA A100-SXM4-40GB, 108 SMs, sm_80, driver 580.105.08, CUDA 12.8 (nvcc 12.8.93), max SM clock 1410 MHz; shared box, GPU lock held for every timed run |
| PyTorch | 2.7.0+cu128, Triton 3.3.0; TF32 matmuls (`allow_tf32`), fused AdamW (`fused=True, capturable=True`), SDPA attention |
| Executor | `kvm_exec` built `nvcc -O3 -arch=sm_80 -std=c++17 -Iinclude --fmad=false -Xcompiler -ffp-contract=off -lineinfo -diag-suppress 177 kvm_exec.cu`; default grid (108 blocks); 1xTF32 GEMMs; frontend `compile_training`, tiling `a100`, `reuse='shape'`, no fusion; 317-318k tasks and 21.7 edges/task per step |
| Training | identical on both sides: initial weights and token stream from the executor's `workload.json` (bit-identical generator), AdamW lr 3e-3 (timing and the 3-step loss check) / 3e-4 + 2 warmup steps (the step-1 gradient check), betas (0.9, 0.95), eps 1e-8, wd 0.01 on matrices, global grad-norm clip 1.0 |
| Code | `bench/vs_pytorch/` at commit 129dca8 plus the scripts added in this change (`exec_bench.py`, `exec_losses.py`, `campaign162.sh`, `executor162.sh`, `sweep162.sh`, `grads162.sh`); reference `tests/gpt162/torch_ref.py` (from `test/gpt162-vs-torch`) |

Commands (box side, repository root, each in one `flock ~/.gpu.lock` hold; workloads from `gen.py`, which takes 14 min for 3 steps on the CPU):
`python3 bench/vs_pytorch/gen.py w_162m --model gpt_162m --steps 3 --mode tf32 --tiling a100` (and `w_162m_1` with `--steps 1`);
`bash bench/vs_pytorch/campaign162.sh` (PyTorch modes, executor, traced run); `bash bench/vs_pytorch/executor162.sh OUT torch_eager.json`;
`python3 bench/vs_pytorch/exec_bench.py w_162m_1 ./kvm_exec --reps 5`; `bash bench/vs_pytorch/sweep162.sh w_162m_1 OUT`; `bash bench/vs_pytorch/grads162.sh WORK OUT`.
Raw outputs: `raw/` (the 30 MB `trace.csv` files stay out of the repository; the kernel trace is `--trace-csv`, one row per task).

## 1. Numerics (the open gap of `2026-11-06-demo-large`): closed for step 1

Step 1 of the demo's own configuration (lr 3e-4, warmup 2), `reuse none` so every gradient survives, executor dump vs PyTorch on the same bytes
(`raw/grads/grads.log`). Reference = float64 PyTorch; "tf32" = PyTorch 1xTF32 (the executor's arithmetic class); "fp32" = PyTorch fp32-highest.

| quantity | executor | torch tf32 | torch fp32 |
|---|---|---|---|
| loss, difference to fp64 (10.979000) | +1.9e-6 | +1.2e-5 | -9.3e-7 |
| global grad norm (fp64: 4.074979) | 4.074694 | 4.074827 | 4.074980 |
| clip coefficient (fp64: 0.245400) | 0.245417 | | |
| gradient relative L2 vs fp64, worst of 196 tensors | **1.44e-3** (`h11.attn.k.w`) | 1.46e-3 | ~2e-6 |
| by group, executor / torch tf32: wte, norms, attn, mlp, final norm, head | 9.5e-4 / 9.4e-4, 8.8e-4 / 9.2e-4, 8.8e-4 / 8.9e-4, 9.2e-4 / 9.3e-4, 4.9e-4 / 5.0e-4, 7.7e-4 / 7.7e-4 | | |
| first moment `m` after step 1 vs fp64 (rel L2) | max 1.4e-3, median 8.9e-4 | | |
| `w1 - w0` vs fp64 (rel L2) | median 6.7e-3, max 6.8e-2; sign flips: median 0, max 0.26% of a tensor's elements | | |

The executor's error equals PyTorch's 1xTF32 error tensor by tensor (a pure TF32 rounding noise of ~1e-3), 700x above fp32. Nothing is systematically off: no
tensor is worse than torch-tf32, norm and clip coefficient agree to 1e-4. The update `w1 - w0` has the same sign-flip noise as at 16.9k parameters
(`2026-11-04-a100-gpt-golden`). The 3-step bundle of the timing runs (lr 3e-3) also tracks PyTorch eager on the loss:

| step | executor | torch eager | relative |
|---|---|---|---|
| 0 | 10.97900 | 10.97901 | 8.7e-7 |
| 1 | 10.93952 | 10.93950 | 1.8e-6 |
| 2 | 11.16294 | 11.16277 | 1.6e-5 |

Not covered: a long run. The chain stream cannot be learned in 40 steps at this vocabulary (`demo-large` README), and `tests/modal/run_gpt162.py` stages B (repeated-batch
curves vs torch over 40-50 steps) were not run here.

## 2. Speed, same model, same optimizer (best of 5; median and worst in `raw/`)

Per optimizer step. PyTorch: 3 steps timed with CUDA events, state reset before each rep, separate process per mode. Executor: `kernel_ms` of a `kvm_exec` launch
(separate processes, same lock hold, minutes apart; `kvm_exec` has no PyTorch hook), which includes persistent-kernel start and host admission of the whole tape.

| implementation | ms / step best (median / worst) | tokens/s | x floor (21.2 ms) | executor / this |
|---|---|---|---|---|
| PyTorch eager | 69.64 (69.72 / 69.97) | 58.8k | 3.3x | 15.0x |
| PyTorch whole-step CUDA graph | 68.21 (68.37 / 68.41) | 60.1k | 3.2x | 15.3x |
| `torch.compile` max-autotune, no graphs | 59.92 (59.98 / 60.09) | 68.4k | 2.8x | 17.4x |
| `torch.compile` reduce-overhead | 60.08 (60.21 / 60.66) | 68.2k | 2.8x | 17.3x |
| compile + whole-step CUDA graph | **58.79** (58.84 / 58.95) | 69.7k | 2.8x | 17.7x |
| **executor, 1-step tape** (`w_162m_1`, 318,139 tasks) | **1041.8** (1047.9 / 1050.4) | 3.9k | 49x | 1x |
| executor, 3-step tape, default config (952,011 tasks) | 2765.5 (2766.8 / 2779.4) | 1.5k | 130x | |
| executor, 3-step tape, demo config (live-cap 65536, 4M-word ring, 2M slots) | 2498.2 (2503.1 / 2510.1) | 1.6k | 118x | |
| (reference) `demo-large` run, steps 1-5 of 6, beside serving and Doom | 1810 | 2.3k | 85x | |

**Floor** per step = max(FLOPs at peak, bytes at bandwidth, dependency depth x hop): 3.307 TFLOP / 156 TFLOP/s TF32 = **21.2 ms**; 14.7 GB / 1555 GB/s = 9.4 ms;
680 logical ops deep x 7 us = 4.8 ms (`info.json`; the chain is a lower bound). The model is compute-bound, so the floor is the FLOP term. PyTorch is 2.8-3.3x above it
(48-56 TFLOP/s effective, cuBLAS TF32 plus memory-bound pieces); the executor is 49x above it at best.

## 3. Why the executor loses here (cost model, AGENTS.md)

1. **Per-task admission, not compute.** One step is 318,139 tasks. At the measured admission rate of 1.9 us/task (single FIFO) that is **604 ms**, 58% of the
   1,042 ms; the remaining 1.7x is ready-to-start wait. Mean ready-to-start wait in the traced run is 4.6 ms against a mean task of 64 us; the 107 workers run
   tasks 26.7% of the time (`trace_summary.py` on the last 262,144 tasks, which is all the 2^18 trace slots hold). A CUDA graph of the same work has on the order of a thousand kernel nodes per step
   (1,893 logical ops) and pays no per-task admission.
2. **Tile work, even with perfect admission, is 2.8x PyTorch's step.** `matmul_tile` is 76% of task time at 62.5 us per 64x64-output tile. The traced tasks add up to
   16.7 CTA-seconds for 262k tasks, so about 20 CTA-s per step: 20 / 106 workers = **~190 ms/step** if all workers were always busy, against 58.8-69.6 ms in PyTorch. That is the
   "matmul about 0.5x cuBLAS, attention 15-30% below SDPA" of `2026-11-04-a100-ops-perf` plus row-wise tasks that are slow per task (rope_bwd 282 us, colsum 445 us, layernorm_bwd_dx 148 us mean).
3. **No dependency-hop term is binding.** 680 levels x 7 us = 4.8 ms is 0.5% of the step; the executor is nowhere near hop-bound, so the VM's one advantage (removing
   launch and tail costs) is worth at most a few ms of a 60 ms step. This is the regime AGENTS.md calls "little to win in big GEMMs".
4. **Admission window does not help.** `sweep162.sh` on the 1-step tape: default (live-cap 16384) 1043.8 ms; live-cap 65536 / 262144: 1198 ms (+15%);
   64 blocks: 1258 ms. The default is the best of those tried; the cost is not backpressure.
5. **Unexplained, reported as measured:** the 3-step tape costs 8.3 s, not 3 x 1.04 s; steps 2 and 3 cost ~3.6 s each. The demo's one-bundle-per-step flow gets 1.8 s per step. Likely
   suspects are version-parked tasks and slot recycling (max_tasks 2^18 against 952k tasks), but the demo config (2M slots) only recovers 10%, so I did not isolate it.
   Use the 1-step number (best, includes launch and ramp) as the headline.

## What this does and does not show

- Shown: at 162M parameters on one A100 the executor trains the same model to TF32-noise accuracy and 15-18x slower than PyTorch's slowest mode.
  Together with `2026-11-06-vs-pytorch` (tiny: executor beats eager 2.3x, loses to a graph 3.5x; small: 1.6x / 4.4x; mid: 3.5x slower than eager) the picture is consistent:
  the executor's win region is launch-bound eager execution on tiny models, never a graph.
- Not shown: any size where it beats a CUDA graph; H100/B200; fused-op (`fuse=True`) bundles, bigger GEMM tiles (listed "not built"), multi-run batching.
- The executor and PyTorch times come from different processes minutes apart on a shared box; best-of-5 spreads are under 1% (PyTorch) and 1% (executor 1-step).
- The kernel trace (`--trace-csv`) is kept on the box and in local `logs/` (28 MB each) for the planned timeline figure; it covers only the last 2^18 task slots of a step.

## Proposed status-table text (not applied; AGENTS.md / README.md are not edited here)

- Move "an executor-vs-PyTorch/CUDA-graph comparison" out of "Not built". Implemented + measured: "Executor vs PyTorch on the same GPT and AdamW (`2026-11-06-vs-pytorch`, `2026-11-06-gpt162-vs-torch`): correct to TF32 noise (162M step-1 loss and 196 gradients match torch-tf32, `gpt162-vs-torch`), faster than eager only at 17k and 116k parameters (2.3x, 1.6x), slower than a whole-step CUDA graph at every size (3.2x at 17k to 15x at 162M)."
- Known failures: "At 162M parameters a step is 318k tasks and takes 1.04 s on the executor (admission-bound, 3.3 us/task, tile work ~190 ms) against 59-70 ms in PyTorch; a 3-step tape costs 2.4-2.7x more per step than a 1-step tape, cause not isolated."
