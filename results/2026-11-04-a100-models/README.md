# Framework generality: five model families through the generic executor on an A100 (2026-11-04)

Claim under test: **nothing in the stack is hardwired to one model, and ops run fast without a hand-written kernel.**
Every model below is a few dozen lines of the frontend's tracing builder (`frontend/kvmfe/models.py`); the only device code added is
a pack of 26 *generic, shape-parametrized* registry ops (`include/kvm_ops_extra.def`, ids 96-121: activations, SwiGLU, row softmax,
gather / scatter-add, MoE routing, bidirectional attention, bf16 matmul, LR schedule, matmul epilogue wrappers). None of them knows
a model. Autodiff, tiling, dataflow edges, memory planning, fusion and the compiler's golden are the same code for every model.

- GPU: NVIDIA A100-SXM4-40GB, 108 SMs, sm_80, driver 580.105.08, CUDA 12.8.93 (shared box; `flock` on the GPU lock, one hold for the whole batch).
- Executor: `kvm_exec.cu` built with `nvcc -O3 -arch=sm_80 -std=c++17 -Iinclude --fmad=false -Xcompiler -ffp-contract=off -lineinfo -diag-suppress 177`,
  default grid (108 blocks, 1 per SM) for the checked runs and timings, `--blocks 4` under compute-sanitizer.
- Code: branch `feat/models`; bundles from `examples/models/prebuild.sh`, run by `examples/models/run_box.sh logs/prebuilt OUT` on the GPU machine
  (cases: `cases.txt`, raw log: `runs.txt`, one directory per case in `cases/`: `check.txt`, `check.json`, `meta.json`, `times.txt`).
- Reproduce: `examples/models/prebuild.sh logs/prebuilt && examples/models/run_box.sh logs/prebuilt logs/run` (needs `KVM_HOST`-style access only for the rsync; the
  scripts take no host names), or `KVM_HOST=user@host examples/models/run.sh` from a laptop.

## Models

| family | preset | what is exercised | parameters |
|---|---|---|---|
| Llama-style decoder | `llama_tiny` (2 layers, d64, 4 heads, **2 KV heads**) | RMSNorm, RoPE with separate q/k head counts, grouped-query attention, SwiGLU, untied head; training + KV-cache decode | 69,952 |
| same, tied | `llama_gqa_tied` | `linear(wt=True)` tied LM head, embedding gradient accumulated from both uses | 65,856 |
| mixture of experts | `moe_tiny` (4 experts, top-2, capacity 1.25) | softmax router, `moe_route`, gather dispatch, per-expert SwiGLU, scatter-add combine scaled by the router gate, load-balance loss as a second backward root | 74,176 |
| MLP classifier | `mlp_tiny` | plain MLP, relu via the generic `act` op, Gaussian-cluster data, cosine LR schedule with warmup (on the device) | 1,732 |
| conv-free ViT | `vit_tiny` (16x16 images, 4x4 patches, 1 layer) | patch embedding as a matmul, learned positions (`add_rowtile`), **bidirectional** attention, mean-pool as scatter-add | 13,956 |
| mixed precision | `llama_bf16`, `mlp_bf16` | bf16 operands on the bf16 tensor-core tile, fp32 accumulate, bf16-rounded activations, fp32 master weights in AdamW, static loss scale 1024 / 128 | as above |

## Correctness against independent oracles (CPU, float64)

`tests/frontend/oracle_models.py` is a numpy float64 forward pass of each model that shares only the initial values and the data stream with
the frontend (no tiles, buffers, device ops or autodiff). The frontend's program run on the CPU float64 VM must reproduce its loss, and its autodiff
gradient of every parameter tensor (the largest element plus a pseudo-random one) must match a central finite difference of the oracle:

| model | params | loss CPU f64 VM vs numpy oracle (abs) | worst autodiff vs finite-difference grad error (spot check of every tensor) |
|---|---|---|---|
| mlp_tiny | 1,732 | 4.4e-16 | 3.0e-10 |
| vit_tiny | 13,956 | 2.2e-16 | 6.0e-08 |
| llama_tiny | 69,952 | 8.9e-16 | 4.6e-07 |
| llama_gqa_tied | 65,856 | 0.0e+00 | 1.1e-06 |
| moe_tiny | 74,176 | 8.9e-16 | 7.3e-07 |
| moe_drop_k1 | 49,600 | 8.9e-16 | 2.6e-07 |

(`moe_drop_k1` has capacity 0.5, so tokens are dropped; `examples/models/oracle_report.py`.) The same tests check that any topological order gives bitwise
identical state (dataflow completeness), and KV-cache decode of the Llama family (GQA, tied embeddings, RoPE offset by the cache length) equals
full-forward greedy decode of the oracle.

## A100 runs through the generic executor

Each case: the compiler's float32 CPU golden of the same bundle is diffed against the GPU's final buffers (`check.py`). GEMMs are **3xTF32**
(matmul flag 16, `Tiling.precise`) for the exact comparison, and 1xTF32 (the default) for the `tf32` cases. Tolerance policy (see `examples/models/check.py` and
`results/2026-11-04-a100-gpt-golden`):

- `precise`: every per-step loss within 1e-5 relative and every f32 buffer (activations, gradients, moments, weights) within 1e-3 relative L2. The loss is the sharp
  check (measured <= 3.7e-6 on all 17 precise cases). The all-buffer bound is loose on purpose: AdamW divides by sqrt(v), so a gradient element below the fp32 noise
  moves its weight by +-lr in one run and not in the other; the measured worst is 4.9e-4 after 10 Llama steps and 5e-6 for the MLP.
- `tf32` (1xTF32 GEMMs): losses within 2e-3 (measured <= 4.3e-4) and every buffer produced before the first optimizer task of step 1 within 6e-3 of max|ref| (measured <= 1.1e-3). Later steps follow a different but equally valid
  trajectory (GPT root-cause note), so weights are only reported.
- `lowp` (bf16): per-step losses within 1e-3 (measured <= 1.6e-4), step-1 activations/gradients within 6e-3 of max|ref| (measured 1.1e-4), weights within 1e-2 rel L2 (measured 1.8e-3).
  A bf16 rounding that flips between the GPU's fp32 accumulation order and the golden's moves one element by 2^-9 relative, and the flips compound over steps.
- Signal-free buffers are skipped and listed in each `check.txt`: the gradient of an attention **key bias** is exactly zero (softmax is invariant to a per-query constant),
  so the golden holds 1e-11-scale noise there and AdamW turns the noise into +-lr steps of the bias, the second moment and the bias itself (ViT; 24 buffers).

| case | model | steps | tasks | mode | loss (steps spread over the run) | max error vs CPU golden | exec time ms (min of runs) |
|---|---|---|---|---|---|---|---|
| gpt_fused | gpt_tiny | 5 | 1,340 | precise+fuse | 4.146 3.994 3.768 3.866 3.640 | 3.7e-05 (rel L2, all buffers) | 6.93 (n=3) |
| gqa_tied | llama_gqa_tied | 10 | 8,016 | precise | 4.228 3.670 3.752 3.299 2.913 | 2.9e-05 (rel L2, all buffers) | 31.86 (n=3) |
| llama_accum | llama_tiny | 5 | 5,222 | precise | 4.180 3.872 3.627 3.510 3.286 | 1.1e-04 (rel L2, all buffers) | 20.74 (n=3) |
| llama_bf16 | llama_bf16 | 10 | 9,636 | lowp | 4.213 3.672 3.730 3.357 2.969 | 1.1e-04 (step-1 acts/grads, max|e|/max|ref|) | 39.46 (n=3) |
| llama_f32 | llama_tiny | 10 | 8,336 | precise | 4.213 3.543 3.704 3.169 2.646 | 4.9e-04 (rel L2, all buffers) | 32.05 (n=3) |
| llama_fused | llama_tiny | 10 | 8,158 | precise+fuse | 4.213 3.543 3.704 3.169 2.646 | 4.9e-04 (rel L2, all buffers) | 31.09 (n=3) |
| llama_remat | llama_remat | 5 | 6,498 | precise | 4.215 3.877 3.558 3.666 3.393 | 3.2e-04 (rel L2, all buffers) | 25.27 (n=3) |
| llama_tf32 | llama_tiny | 5 | 4,166 | tf32 | 4.213 3.894 3.543 3.657 3.213 | 1.1e-03 (step-1 acts/grads, max|e|/max|ref|) | 15.85 (n=3) |
| mlp_bf16 | mlp_bf16 | 30 | 2,790 | lowp | 2.493 0.697 0.092 0.027 0.012 | 1.3e-07 (step-1 acts/grads, max|e|/max|ref|) | 18.50 (n=3) |
| mlp_f32 | mlp_tiny | 40 | 2,760 | precise | 2.493 0.427 0.060 0.042 0.033 | 5.0e-06 (rel L2, all buffers) | 18.50 (n=3) |
| mlp_tf32 | mlp_tiny | 5 | 340 | tf32 | 2.493 1.867 1.539 1.409 1.144 | 8.8e-04 (step-1 acts/grads, max|e|/max|ref|) | 1.97 (n=3) |
| moe_f32 | moe_tiny | 10 | 9,990 | precise | 4.170 3.645 3.101 2.723 2.453 | 7.1e-05 (rel L2, all buffers) | 38.93 (n=3) |
| moe_fused | moe_tiny | 10 | 9,590 | precise+fuse | 4.170 3.645 3.101 2.723 2.453 | 7.1e-05 (rel L2, all buffers) | 34.69 (n=3) |
| moe_tf32 | moe_tiny | 5 | 4,995 | tf32 | 4.170 3.885 3.645 3.497 3.209 | 8.5e-04 (step-1 acts/grads, max|e|/max|ref|) | 19.75 (n=3) |
| san_bf16 | llama_bf16 | 2 | 1,922 | lowp | 4.213 3.893 | 1.1e-04 (step-1 acts/grads, max|e|/max|ref|) | 8.47 (n=3) |
| san_gqa_fused | llama_gqa_tied | 2 | 1,566 | precise+fuse | 4.228 4.002 | 2.1e-05 (rel L2, all buffers) | 5.96 (n=3) |
| san_llama | llama_tiny | 2 | 1,664 | precise | 4.213 3.894 | 1.5e-04 (rel L2, all buffers) | 6.38 (n=3) |
| san_mlp | mlp_tiny | 3 | 207 | precise | 2.493 1.867 1.709 | 1.1e-06 (rel L2, all buffers) | 1.08 (n=3) |
| san_moe | moe_tiny | 2 | 1,998 | precise | 4.170 3.885 | 4.1e-05 (rel L2, all buffers) | 7.74 (n=3) |
| san_moe_fused | moe_tiny | 2 | 1,918 | precise+fuse | 4.170 3.885 | 4.1e-05 (rel L2, all buffers) | 7.24 (n=3) |
| san_remat | llama_remat | 2 | 2,356 | precise | 4.180 3.872 | 9.8e-05 (rel L2, all buffers) | 8.89 (n=3) |
| san_vit | vit_tiny | 2 | 850 | precise | 1.382 1.416 | 8.8e-06 (rel L2, all buffers) | 3.50 (n=3) |
| vit_f32 | vit_tiny | 20 | 8,500 | precise | 1.382 1.251 1.125 0.902 0.779 | 5.9e-05 (rel L2, all buffers) | 38.19 (n=3) |
| vit_tf32 | vit_tiny | 5 | 2,125 | tf32 | 1.382 1.416 1.345 1.300 1.287 | 9.4e-04 (step-1 acts/grads, max|e|/max|ref|) | 9.35 (n=3) |

All 24 cases exit 0 (`runs.txt`); `loss` is the GPU loss at ~0/25/50/75/100% of the run (mean over microbatches for `llama_accum`). Execution time is the executor's own
(`OK: n/n tasks, X ms`, first admission to last completion), min of 4 launches of the same bundle; the box is shared, so absolute times move by a few percent.
Per task that is 3.8-4.5 us for the transformer models (dependency hop plus a small tile), 6.7 us for the MLP (a serial chain of tiny tasks).

**compute-sanitizer** (memcheck, racecheck, synccheck, `--blocks 4`) on the eight `san_*` bundles (2-3 steps each, all five families, fusion, remat + accumulation, bf16 + loss scale,
LR schedule): 24 runs, 0 errors, 0 hazards.

## Features added to the frontend, with measurements

| feature | evidence |
|---|---|
| **Op fusion** (`fuse.py`: `gelu_bwd(mm)` -> `matmul_tile_dgelu`, SwiGLU up-projection + gating -> `matmul_tile_silu_mul`, SwiGLU backward -> `matmul_tile_silu_mul_bwd`) | CPU result bitwise equal to unfused (test); GPU 3xTF32 results identical to the golden at the same 4.9e-4 / 7.1e-5 as the unfused bundles. Tasks: Llama -2.1% (8336 -> 8158), MoE -4.0% (9990 -> 9590), GPT -1.5% (cpu tiling), ViT -4.0%, MLP 0 (`fusion_*_tiling.txt`; the optimizer's chunks dominate task counts). A100 time, 10 steps: Llama 32.0 -> 31.1 ms (-3%; unfused repeats 32.0/34.9/34.0, fused 31.4/31.1/31.3), MoE 39.0 -> 34.7 ms (-11%; 39.0/38.9/39.7 against 35.0/34.8/34.7). Critical path: -2 tasks per fused pair of layers (modeled -3% to -4%). Bias+GELU and the residual add were already fused in the `matmul_tile` epilogue; norms cannot be fused into a matmul epilogue (a tile does not hold a row). |
| **Rematerialization** (`with b.remat()`, `LlamaConfig.remat`) | weights bitwise equal to the plain program (test, dense and MoE); activation pool at a100 tiling: L4 13.8 -> 8.0 MiB, L8 24.9 -> 11.6 MiB, L12 d256 139.9 -> 56.7 MiB (-41% to -59%) for +36-38% tasks (`remat_memory.txt`). A100: `llama_remat` (3 layers, 5 steps) matches the golden (buffers 3.2e-4, losses <= 3.7e-6 like every precise case). |
| **Gradient accumulation** (`TrainConfig.accum`) | accum=2 of batch 1 equals one step of batch 2 to 1e-11 (test); A100 `llama_accum` (5 steps, 2 microbatches) matches the golden to 1.1e-4. |
| **LR schedules as device scalars** (`lr_schedule` + `adamw_lr`) | `const` schedule bitwise equal to the baked-lr program (test); cosine/warmup buffer value checked against the formula; A100 `mlp_f32`/`san_mlp`. Program text of every step stays identical, so window templates still work. |
| **Loss scaling + bf16** | `loss_scale` transparent to 1e-8 on weights (test); `llama_bf16` loss 4.213 -> 2.969 in 10 steps tracks the golden to 1.6e-4; `mlp_bf16` 2.493 -> 0.012 with golden loss error <= 1.0e-6. |
| **Tied embeddings, GQA, multiple loss roots, float streams, bidirectional attention, kv-head caches** | covered by the models above. |

## Honest limitations (also in FRONTEND.md)

- **Static shapes and a static DAG.** No dynamic shapes and no data-dependent control flow in the frontend. Data-dependent behaviour is in buffers (`select_window`, `lens`,
  `moe_route`, `lr_schedule`); MoE experts always run `capacity` rows (zero-padded). Not implemented: `while`/`if` over tensor values, early exit, variable sequence length.
- MoE routing ops are one task over all tokens (serial O(N*E*k)), fine to a few thousand tokens; no MoE serving.
- **bf16 is emulated storage**: bf16 values live in fp32 buffers (ABI v2 dtype ids 0 f32 / 1 i32 unchanged), so no memory saving; no fp16, no dynamic loss scaling / overflow skipping.
- Fusion gains are small at this size (a few percent of tasks): the matmul epilogue cannot hold a row, so norms stay separate, and AdamW chunks dominate the task counts of tiny models.
- Single GPU box, shared: timings are indicative (min of 4 launches). The 1xTF32 cases check losses and step-1 buffers only, by design.
- The compiler's golden is a pure-Python float32 VM: 10-step Llama/MoE bundles (8-10k tasks) take minutes to generate on an idle core (the Mac was at load 200, so the
  bundles were built on the GPU machine's CPUs); `--golden-timeout` was added to `compile_workload.py` for that.
- Shared files touched with small hooks: `include/kvm_ops_list.inc` (+4 lines), `include/kvm_exec.cuh` (+3, marked `KVM_MODELS_HOOK`), `runtime/abi.py` (registry name),
  `runtime/ref_ops.py` (+2), `runtime/compile_workload.py` (`--golden-timeout`).
