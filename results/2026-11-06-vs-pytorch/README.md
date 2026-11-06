# Executor vs PyTorch eager / CUDA graphs / torch.compile on the same GPT (A100, 2026-11-06)

**Verdict: the executor loses to a whole-step CUDA graph on every model tried (3.2x to 12.6x slower), loses to
`torch.compile` + CUDA graph by more, and on the 3.4M-parameter model it is slower than plain PyTorch eager (3.5x). It beats eager
only on the two smallest models (2.3x on `gpt_tiny`, 1.6x on `gpt_small`), where eager is launch-bound; there it also beats
`torch.compile` without graphs (1.1-1.7x) but not with `reduce-overhead` on `gpt_small`.** README goal 3 ("faster than the strongest
conventional execution") is therefore **not met** for transformer training; only the `kvm.cu` MLP result stands.

## Setup

- GPU: NVIDIA A100-SXM4-40GB, 108 SMs, sm_80, driver 580.105.08, CUDA 12.8 (nvcc 12.8.93), max SM clock 1410 MHz; shared box, one hold of
  the GPU lock for the whole campaign. PyTorch 2.7.0+cu128, Triton 3.3.0 (`round2/env.txt`).
- Executor: `kvm_exec` built with `nvcc -O3 -arch=sm_80 -std=c++17 -Iinclude --fmad=false -Xcompiler -ffp-contract=off -lineinfo -diag-suppress 177
  kvm_exec.cu`, base commit `6d31aba` (`fix/window-done-wrap`), run with the default grid (occupancy-sized, 108 SMs x resident blocks) and,
  as `kvm_b4`, `--blocks 4`. Bundle: frontend `compile_training`, `a100` tiling, `reuse='shape'`, no fusion, `gen.py`. Bench code: commit `454f0b4`.
- Models (`kvmfe.nn.GPT`: pre-LayerNorm, RoPE, causal attention, tanh-GELU MLP, untied bias-free head, mean CE):
  `gpt_tiny` (d32, 1 layer, 2 heads, T32, B1, V64, 16.9k params, 50 steps), `gpt_small` preset (d64, 2 layers, 4 heads, T64, B2, V128, 116k, 50 steps),
  `gpt_mid` (d256, 4 layers, 4 heads, T128, B8, V512, 3.42M params, 1024 tokens/step, 20 steps; chosen as the largest size whose
  155k-task bundle the executor ran in this campaign without tuning, not a limit probe).
- Same everything: initial weights and token stream are read from the executor's `workload.json` (counter-based generator, bit-identical, self-checked
  against `runtime/initgen.py`), AdamW lr 3e-3, betas (0.9, 0.95), eps 1e-8, weight decay 0.01 on matrices only, global grad-norm clip 1.0, 1 batch per
  step, N steps from the same start. Precision: **TF32** (`allow_tf32`; executor 1xTF32 GEMMs) for `tiny`, `small`, `mid`; **fp32 / 3xTF32**
  (`highest` matmul precision; executor `matmul_tile` flag 16) for `tiny_p`, `mid_p`.
- PyTorch variants (`bench/vs_pytorch/torch_bench.py`), all fused AdamW (`fused=True, capturable=True`) and SDPA attention:
  `eager`; `graph` (the whole step incl. device-side batch select, backward, clip, AdamW in one CUDA graph, replayed N times);
  `compile_mat` (`torch.compile(mode='max-autotune-no-cudagraphs')` on forward+loss, backward/clip/AdamW eager); `compile_ro` (same with
  `reduce-overhead`, i.e. CUDA-graph trees on forward/backward, plus one 1-element loss clone per step); `compile_graph` (compile_mat's compiled
  fwd/bwd captured with clip and AdamW into one whole-step CUDA graph: the strongest PyTorch configuration here).
- Timing: PyTorch = CUDA events around N steps, state (weights, Adam moments) reset before each of 7 timed reps after one untimed rep.
  Executor = the kernel's own cudaEvent time ("OK: n/n tasks, X ms"), 7 launches (separate processes, run right after the PyTorch modes in
  the same lock hold). One executor launch runs all N steps: persistent-kernel start and admission of the whole N-step tape are inside the time.
  **Ratios between PyTorch variants are from one process; ratios to the executor compare two processes minutes apart** (AGENTS.md asks for one
  process; `kvm_exec` has no PyTorch-side hook). Best of 7 is the headline, median and worst are in the tables. Round 1 (`round1/`) is a full
  repeat before a fix: its `compile_ro` loss list was invalid (CUDA-graph output aliasing, fixed in round 2), its timings are valid.
- Reproduce: `bench/vs_pytorch/gen.py` (workloads), `campaign.sh` (box side, one lock hold), `report.py` (tables), `kvm_bench.py`, `trace_summary.py`.

## Correctness first

Loss at step 0 (pure forward at identical weights) against PyTorch eager, relative: TF32 modes 1e-5 or better for every run (executor 3e-6 tiny,
1e-7 small, 1e-6 mid); precise modes 1.2e-7 (`tiny_p`), 1.5e-7 (`mid_p`). Worst relative difference over the whole run vs eager (the trajectories
diverge after sign flips in AdamW's first updates, the mechanism in `2026-11-04-a100-gpt-golden`): executor 1.1e-4 (tiny), 9.9e-4 (small),
5.6e-3 (mid, an unstable run: loss goes 6.28 5.47 5.10 5.43 at lr 3e-3), PyTorch variants of the same order, `compile_mat` on mid 4.1e-2; precise
runs 2.1e-7 (`tiny_p`, executor) and 4.0e-4 / 2.9e-5 (`mid_p`, executor / compile). All variants train the same curve (`tiny`: 4.146 ... 1.245 over 50
steps; tables below). No run failed. The `graph` and `eager` rows are identical (same kernels).

## Results (round 2; best / median / worst of 7, microseconds per optimizer step)

| model | tokens/step | eager | CUDA graph | compile (mat) | compile (red-ovh) | compile + graph | **executor** | executor vs eager | executor vs graph |
|---|---|---|---|---|---|---|---|---|---|
| gpt_tiny TF32 | 32 | 2733 / 2755 / 2941 | 334 / 429 / 429 | 1995 / 2005 / 2147 | 1653 / 1677 / 1736 | 202 / 202 / 202 | **1166 / 1255 / 1305** | 2.34x faster | 3.49x slower (2.92x on medians) |
| gpt_small TF32 | 128 | 4617 / 4640 / 5147 | 644 / 780 / 829 | 3023 / 3074 / 3168 | 2108 / 2116 / 2168 | 424 / 424 / 425 | **2818 / 2828 / 2832** | 1.64x faster | 4.37x slower |
| gpt_mid TF32 | 1024 | 8530 / 8569 / 9312 | 2384 / 2512 / 2803 | 6141 / 6583 / 6781 | 2759 / 2802 / 3172 | 1677 / 1678 / 2115 | **29922 / 29955 / 30084** | **3.51x slower** | 12.55x slower |
| gpt_tiny fp32/3xTF32 | 32 | 2631 / 2674 / 2926 | 385 / 435 / 435 | 2007 / 2014 / 2293 | 1499 / 1514 / 1684 | 227 / 227 / 227 | **1216 / 1301 / 1312** | 2.16x faster | 3.16x slower |
| gpt_mid fp32/3xTF32 | 1024 | 8954 / 9036 / 9312 | 3749 / 3758 / 3952 | 6825 / 7069 / 7388 | 3308 / 3336 / 4108 | 2928 / 2934 / 3716 | **30185 / 30309 / 30764** | **3.37x slower** | 8.05x slower |

Tokens/s at best: tiny 27.4k executor vs 95.9k graph vs 158.6k compile+graph (eager 11.7k); small 45.4k vs 198.7k vs 301.7k (eager 27.7k);
mid 34.2k vs 429.5k vs 610.5k (eager 120.0k). The executor vs the other variants, TF32 best/best: tiny 0.58x of compile_mat time (1.7x faster),
0.71x of compile_ro (1.4x faster), 5.8x slower than compile+graph; small 0.93x (1.07x faster) and 1.34x (slower) and 6.6x slower; mid 4.9x, 10.8x, 17.8x slower.
`--blocks 4` is slower or equal everywhere (1.1x tiny, 1.3x small, 3.3x mid), so the default grid is the executor's best configuration here.
Round 1 repeats: executor 1274 / 2822 / 29884 us (tiny / small / mid), graph 429 / 644 / 2381 us; every ratio moves by under 10% except the tiny
graph, whose best-of-7 flips between a 334 us and a 429 us mode (clock/box state; the median is the stable number there).
Full tables with step-0 and worst-step loss differences: `round2/tables.md` (round 1: `round1/tables.md`).

## Against the roofline floor (AGENTS.md "Measuring")

Floor per step = max(FLOPs at peak, bytes at HBM bandwidth, logical dependency depth x one hop). Peak 156 TFLOP/s TF32 (52 for 3xTF32), 1555 GB/s;
FLOPs/bytes from `gen.py` (fwd + dgrad + wgrad GEMMs and attention; weights/optimizer state/activations touched once); depth = the longest chain of
logical ops per step (28 tiny, 46 small, 82 mid); hop = 7.0 us for the executor (measured, `2026-11-04-a100-exec-recycling`), 1.3 us for a graph
edge (the AGENTS.md H100 figure, not re-measured). The chain term is a lower bound (tile-level chains are longer).

| model | compute | bytes | chain, graph edge / executor hop | floor graph-style / executor | graph (x floor) | compile+graph | executor (x its floor) |
|---|---|---|---|---|---|---|---|
| gpt_tiny TF32 | 0.0 us | 0.5 us | 36.5 / 196.3 us | 36.5 / 196.3 us | 9.1x | 5.5x | 5.9x |
| gpt_small TF32 | 0.6 us | 4.6 us | 59.9 / 322.3 us | 59.9 / 322.3 us | 10.8x | 7.1x | 8.7x |
| gpt_mid TF32 | 135.1 us | 182.4 us | 106.7 / 574.7 us | 182.4 / 574.7 us | 13.1x | 9.2x | 52.1x |
| gpt_mid 3xTF32 | 405.2 us | 182.4 us | 106.7 / 574.7 us | 405.2 / 574.7 us | 9.3x | 7.2x | 52.5x |

Every run is far from its floor; the best (compile + graph) is 5.5-9x above it. **What the gap is made of** (executor, one traced run per model,
`round2/*/trace_summary.txt`): on `gpt_tiny`, 103 tasks/step in a chain of 28 logical levels; workers (107) run tasks 0.8% of the time, mean
task 8.1 us, mean ready-to-start wait 7.9 us, step span 999 us = about 36 us per logical level (task 8 us + wait 8 us + hop 7 us + joins), i.e. 5x the
bare hop the floor assumes. On `gpt_mid`, 7753 tasks/step (112,000 `matmul_tile` of mean 22.5 us = 67% of run time), utilisation 7.5%, mean wait
103 us; the step span is 23.3 ms against 29.9 ms of kernel time. This is consistent with the single admission FIFO (1.9 us/task, README):
7753 x 1.9 us = 14.7 ms of every step, which no amount of faster tiles removes; I did not isolate it with a controlled experiment. Matmul at 0.34-0.51x
of cuBLAS (`2026-11-04-a100-ops-perf`) adds on top. PyTorch's graph variants pay no per-task admission or hop through global memory: a graph edge is cheap
and cuBLAS/Triton GEMMs are better, so the VM's two costs (README "Where a VM can win") are not repaid on these shapes.

## What this does and does not show

- Shown: with the same weights, batches, optimizer and precision, on one A100, the executor trains these GPTs correctly and slower than PyTorch's best
  configuration at 32, 128 and 1024 tokens per step.
- Where the executor wins: against eager (launch-bound, 2.2-2.3x at 32 tokens, 1.6x at 128) and against `torch.compile` without CUDA graphs
  at the smallest sizes. A user who does not use CUDA graphs would see a win on tiny models; that is not the baseline AGENTS.md demands.
- Not tested: fused-op bundles (`fuse=True`), larger models, an H100/B200 (the VM's design target), several runs per launch (vmap-like batching, the
  setting where `kvm.cu` wins), a `-fmad=true` executor build, other tilings. Executor timing is per launch of N steps (ramp included); N = 20-50.
- Executor and PyTorch numbers come from different processes on a shared box (the lock serialises GPU jobs, not CPU load or clocks); the repeat
  round agrees within 10% except where noted.
