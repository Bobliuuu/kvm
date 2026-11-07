# Real bf16 on the A100: bf16 storage + a bf16 tensor-core GEMM tile, gpt_162m (2026-11-06)

**Verdict.** `precision='bf16s'` now runs the 162M GPT with bf16 activations and a real `mma.sync m16n8k16` bf16 GEMM tile (`matmul_mma_bf16`, op 122).
The executor step goes **1103.6 -> 483.8 ms (2.28x)**, tasks per step **318,139 -> 103,803 (-67%)**, device buffers **6.29 -> 4.97 GiB (-21%)**.
The GEMM tile reaches **104-125 TFLOP/s at 4096^3, 0.47-0.51x of cuBLAS bf16** (13x the previous bf16 tile): it doubles the TF32 tile, it does **not**
double the tensor peak (312 TFLOP/s) and it does **not** make the executor competitive: PyTorch autocast-bf16 + CUDA graph runs the same step in 30.9 ms
(**15.7x** faster than the executor), eager in 47.1 ms (10.3x). The step is admission/dependency bound, not GEMM bound (section 3).

## Setup

| | |
|---|---|
| GPU | NVIDIA A100-SXM4-40GB, 108 SMs, sm_80, driver 580.105.08, max SM clock 1410 MHz; shared box, GPU lock held for every timed run |
| Toolchain | nvcc 12.8.93; PyTorch 2.7.0+cu128 (Triton 3.3.0) |
| Executor build | `nvcc -O3 -arch=sm_80 -std=c++17 -Iinclude --fmad=false -Xcompiler -ffp-contract=off -lineinfo -diag-suppress 177 kvm_exec.cu` |
| Test builds | `nvcc -O2 -arch=sm_80 -std=c++17 --fmad=false -Xcompiler -ffp-contract=off -lineinfo -Iinclude -I. X.cu [-lcublas]` |
| Code | branch `perf/bf16-a100`; device code at `8b07421`, scripts at `d6b916d` (`test_ops` was built one commit earlier, before the GEMM epilogue change: it does not exercise that op) |
| Model | gpt_162m: 12 layers, d768, 12 heads, seq 1024, batch 4 (4096 tokens/step), vocab 50304, untied head, RoPE; AdamW lr 3e-3 (timing, 3-step losses) / 3e-4 + 2 warmup (demo), clip 1.0 |
| Commands | `bash bench/bf16/run_box.sh perf\|profile\|correctness OUT DATA` on the box (one GPU-lock hold per mode, `KVM_EXEC=./kvm_exec_h`); `demo/run_demo.sh large --model <gpt_162m, bf16s>` |
| Raw output | `perf/`, `profile/`, `correctness/`, `demo-large-bf16s/`, `bundle_bytes.json` in this directory |

## What was built

* **`matmul_mma_bf16`** (`include/kvm_mma_bf16.cuh`, op id 122, own op id so it merges cleanly beside the TF32 tile work): 128 x 256 x 32 CTA tile, 8 warps of
  64 x 64, bf16 shared tiles (no widening), `ldmatrix` (`.trans` for m-/n-contiguous operands: all four transpose combinations take one code path), 4-stage `cp.async`
  (96 KiB), fp32 accumulate in fixed order, one RNE rounding per stored value. Epilogue = `matmul_tile`'s (alpha, bias, saved pre-activation, GELU, residual, accumulate), out /
  res / pre each fp32 or bf16; a bf16 output is staged through shared memory and leaves as 16-byte stores. Anything unaligned or fp32-operand falls back to `matmul_tile_bf16`.
* **Storage-aware ops** (norms fwd/bwd, rope, gelu_bwd, residual adds, embedding, colsum, cross entropy, `round_bf16`): read and write fp32 or bf16 storage, 4-wide
  vector paths, scalar fallback. Everything else still sees fp32 scratch (`kvmfe/storage.py` widens/narrows). Weight gradients, AdamW moments and masters stay fp32.
* **Frontend** `precision='bf16s'` for GPT, Llama, MLP, ViT: explicit bf16 copies of fp32 operands, casts folded into producers, bf16 weight shadows refreshed each step, bf16 saved
  pre-activation, `matmul_mma_bf16` chosen when both operands are bf16 storage, n-outer tile order for wide GEMMs (4096 x 50304 x 768: 46 -> 80 TFLOP/s).
* **ABI**: `KvmBuf.dtype` 2 / workload dtype `"bf16"` (merged from `feat/bf16-hopper`; bundles without bf16 are unchanged), `kvm_run` manifests accept it, `kvm_exec --dump-bin`.
  CPU golden: `ref_extra._BF16_STORAGE` rounds every bf16 output; `anydt` tells the compiler which args may be bf16.

## 1. GEMM, `bench_mma_bf16` (kernel vs cuBLAS bf16 `GemmEx`, fp32 accumulate, same process; median of 7 [min..max]; each row also checks the result)

| M x N x K | transposes | matmul_mma_bf16 | cuBLAS bf16 | ratio |
|---|---|---|---|---|
| 4096^3 | NN / TN / NT / TT | 103.6 / 124.5 / 123.8 / 124.8 | 204.9 / 266.3 / 263.1 / 266.2 | 0.51 / 0.47 / 0.47 / 0.47 |
| 8192^3 | NN | 170.3 | 269.5 | 0.63 |
| 4096^3, previous bf16 tile (`matmul_tile_bf16`, 64 x 64) | NN | 9.5 | 263.9 | 0.04 |
| qkv/proj fwd 4096x768x768 | NN | 67.4 | 161.6 | 0.42 |
| fc1 fwd 4096x3072x768 | NN | 71.5 | 200.2 | 0.36 |
| fc2 fwd 4096x768x3072 | NN | 102.1 | 228.5 | 0.45 |
| head fwd 4096x50304x768 (m-outer / n-outer tile order) | NN | 46.1 / **80.2** | 222.8 | 0.21 / 0.36 |
| head dgrad 4096x768x50304 | NT | 171.7 | 253.9 | 0.68 |
| head wgrad 768x50304x4096 (m-outer / n-outer) | TN | 120.4 / **131.6** | 257.8 | 0.47 / 0.48 |
| fc1 dgrad 4096x768x3072 / wgrad 768x3072x4096 | NT / TN | 97.2 / 86.9 | 226.0 / 198.3 | 0.43 / 0.44 |
| fc2 dgrad 4096x3072x768 / wgrad 3072x768x4096 | NT / TN | 70.6 / 87.1 | 174.1 / 203.0 | 0.41 / 0.43 |
| qkv wgrad 768x768x4096 (18 tiles on 108 SMs) | TN | 27.2 | 151.7 | 0.18 |

Full table: `perf/bench_mma_bf16.log`, n-outer order `perf/bench_mma_bf16_nouter.log`, fp32 output `perf/bench_mma_bf16_f32out.log`. A100 peaks: bf16 312, TF32 156 TFLOP/s;
the previous executor TF32 tile measured 51 TFLOP/s at 4096^3 (`2026-11-04-a100-ops-perf`), so the bf16 tile is 2.0-2.4x that.
Small-K and few-tile shapes pay prologue/epilogue and wave quantization on a 1-CTA-per-SM tile; inside the executor other tasks fill the idle SMs.

**What limits it (ablation, 4096^3 NN/NT, TFLOP/s, current code).** Compute only, no barrier, no copies: 191-197 (a bare `ldmatrix`+`mma` loop reaches 244, cuBLAS 265).
Add the per-k-step CTA barrier: 168-170. Add the `cp.async` global->shared traffic: 122-124. The copy path alone moves 3.6-4.6 TB/s in isolation (`tests/gpu_ops/micro_ldgsts.cu`),
so it is the interaction of barrier, copies and `ldmatrix` in one 8-warp CTA, not memory bandwidth. Tried and not faster: 3/6 stages, copies interleaved between `mma`
instructions, the plain (no zero-fill) `cp.async` form, padded leading dimensions. Not tried: BK=64 (needs 3 stages = 144 KiB with this barrier scheme). I did not find the
remaining 2x to cuBLAS; warp specialization with an `mbarrier` pipeline is the next candidate.

## 2. Numerics (bf16 has 8 mantissa bits: relative 2^-9 per rounding; comparisons are stated against what bf16 can be)

| check | result |
|---|---|
| `test_bf16` (`tests/gpu_ops/test_bf16.cu`): `matmul_mma_bf16`, 240 configs (4 transposes, ragged M/N/K, fast and fallback path, every out/res/pre storage mix) + 7 storage-aware ops x fp32/bf16 in/out, vs float64 | **685 passed, 0 failed**; GEMM worst relerr 3.7e-3 of max (tolerances 1.2e-2 for bf16 outputs, 1e-4 for fp32 outputs); bf16-output ops within 6e-3 (one bf16 ulp 3.9e-3) |
| compute-sanitizer memcheck / racecheck / synccheck on `test_bf16` (small mode, 429 checks) and on the `san_bf16`, `san_bf16s` bundles | 0 errors, 0 hazards |
| regression of the fp32 paths I edited: `test_ops` / `test_extra` | **1396 / 0** and **3669 / 0** |
| CPU suites | `tests` 152 OK; `tests/frontend` 71 OK (new: bf16s on Llama, MLP, ViT; loss tracks the bf16 emulation to 5e-3 and fp32 to 5e-2) |
| step-1 of gpt_162m, executor vs **PyTorch fp64** truth (`correctness/grads_*/exec_grads.log`) | see next table |
| 3 steps, executor bf16s vs PyTorch (same initial bytes and token stream) | below |

Step 1 (lr 3e-3, `reuse none`, all 196 gradient tensors, relative L2 against float64 PyTorch; "torch tf32" = PyTorch 1xTF32, "autocast" = `torch.autocast(bfloat16)`):

| | loss - fp64 | global grad norm (fp64 4.074979) | clip coef (truth 0.245400) | grad rel L2, groups (wte / norms / attn / mlp / head) | worst tensor |
|---|---|---|---|---|---|
| executor f32 (1xTF32 tiles) | +1.9e-6 | 4.074694 | 0.245417 | 9.5e-4 / 8.8e-4 / 8.8e-4 / 9.2e-4 / 7.7e-4 | 1.44e-3 |
| torch tf32 | +1.2e-5 | 4.074846 | | 9.4e-4 / 9.1e-4 / 8.8e-4 / 9.3e-4 / 7.7e-4 | 1.46e-3 |
| **executor bf16s** | -3.1e-5 | 4.074788 | 0.245411 | **1.2e-2 / 1.2e-2 / 9.9e-3 / 1.1e-2 / 9.8e-3** | 1.69e-2 |
| torch autocast bf16 | +3.2e-4 | 4.072725 | | 1.1e-2 / 2.3e-2 / 2.1e-2 / 1.8e-2 / 7.6e-3 | 3.36e-2 |

bf16s gradients are ~10x the TF32 error (8 vs 11 mantissa bits) and no worse than PyTorch's own autocast on any group but the head (autocast's norm / attention / mlp gradients are ~2x worse; I did not isolate why:
the two differ in where values are rounded, ours stores the residual stream in bf16 and keeps norms / softmax / loss in fp32).

3-step losses (`perf/losses_vs_torch_*.json`): executor bf16s 10.97897, 10.93990, 11.16655; torch tf32 10.97901, 10.93949, 11.16299 (relative 3.9e-6, 3.8e-5, 3.2e-4);
torch autocast-bf16 10.97932, 10.93954, 11.18112 (3.2e-5, 3.3e-5, 1.3e-3). The 1.3e-3 at step 3 is AdamW amplifying rounding noise at lr 3e-3, same order as autocast vs TF32.

Model zoo vs the CPU golden (`correctness/model_zoo_bf16s.log`, `examples/models/cases_bf16s.txt`, tolerances of the `lowp` mode: loss rel L2 1e-3, step-1 activations 6e-3 of max, weights rel L2 1e-2):

| case | losses | step-1 acts/grads | weights rel L2 | verdict |
|---|---|---|---|---|
| llama_bf16 (emulation) / llama_bf16s | 2.0e-4 / 4.3e-4 | 5e-6 / 4.8e-3 | 1.8e-3 / 8.5e-3 | PASS / PASS |
| mlp_bf16 / mlp_bf16s | 8.3e-7 / 8.3e-7 | 1.3e-7 / 1.3e-7 | 3.9e-8 / 3.9e-8 | PASS / PASS |
| vit_bf16 (emulation) / vit_bf16s | 6.0e-4 / 7.1e-4 | 5.3e-3 / 8.3e-6 | **0.98 / 0.90** | **FAIL / FAIL** |
| san_bf16 (emulation, 2 steps) / san_bf16s | 6e-8 / 2.6e-5 | 5e-6 / 4.8e-3 | 1.1e-6 / **1.7e-2** | PASS / **FAIL** |

The ViT failure is the same in the emulation baseline: the key-projection bias has a mathematically zero gradient (softmax is invariant to it), so bf16 noise alone
drives AdamW's sign-like update (`m.vit.h0.attn.k.b` differs by 1.8) and the weight metric of that tensor is meaningless; losses and activations pass. `san_bf16s` (2 steps)
exceeds the weight tolerance (1.7e-2 vs 1e-2) while its 10-step sibling passes (8.5e-3): real bf16 storage is further from the CPU golden than the emulation because stored
values expose the 1-ulp flips that the emulation hides inside the matmul; I did not tighten or relax the tolerance.

## 3. Step time, tasks, memory (gpt_162m, 1 step, `kvm_exec`, best of 5 [median / worst])

| implementation | ms / step | tasks | vs executor bf16s |
|---|---|---|---|
| executor f32 (1xTF32 tiles) | 1103.6 (1108.4 / 1110.7) | 318,139 | 2.28x slower |
| **executor bf16s** | **483.8** (485.1 / 493.8) | **103,803** | 1x |
| executor bf16s, 3-step tape | 560.2 (562.4 / 565.6) | 312,739 | |
| PyTorch tf32: eager / graph / compile+graph | 69.9 / 68.3 / 59.0 | | 7.0x / 7.1x / 8.2x faster |
| PyTorch autocast bf16: eager / graph / compile+graph | 47.1 / 44.7 / **30.9** | | 10.3x / 10.8x / **15.7x** faster |

Floor of the step: 3.307 TFLOP at the bf16 peak 312 = 10.6 ms (21.2 ms at TF32 peak), 14.7 GB at 1555 GB/s = 9.4 ms, 4.8 ms of dependency hops. PyTorch autocast + compile + graph is 2.9x above the
bf16 floor; the executor is 46x above it.

Where the executor's step goes (`profile/`, traced launch, sum of per-task run time = CTA-seconds; 107 workers):

| | f32 (last 262k of 318k tasks) | bf16s (all 103.8k tasks) |
|---|---|---|
| GEMM tasks, CTA-s (mean) | 12.56 (61.9 us, 202,845 tasks) -> about 15.3 for all 247k | **3.51** (113.6 us, 30,910 tasks) |
| in-tile GEMM rate (3.037 TFLOP / CTA-s x 107) | ~21 TFLOP/s | **~93 TFLOP/s** (4.4x) |
| attention (fp32 SIMT, unchanged), CTA-s | ~3 (the traced part misses the forward half) | 2.96 |
| `round_bf16` (weight shadows, narrow/widen), CTA-s | | 0.73 (was 2.16 before vectorizing) |
| total task run time | 16.95 (for the traced 262k tasks) | 9.65 |
| worker utilisation / mean ready->start wait | 25% / 4.6 ms | 24.5% / 0.84 ms |

So bf16 shrinks the part it can (GEMM: 74% of run time -> 36%), but a worker is busy only a quarter of the time: 103,803 tasks at about 4.7 us each (483.8 ms / 103.8k) is
the step, and attention (fp32, 55k tasks) is now the next biggest block. That is the admission/attention tracks' ground, not a bf16 one.

Device memory (`bundle_bytes.json`, bundle buffers, `reuse='shape'`): f32 **6.290 GiB** -> bf16s **4.973 GiB (-21%)** (2.011 bf16 + 2.962 fp32). It does **not** halve: fp32 masters,
moments and weight gradients are 2.6 GiB, attention q/k/v/o/lse stay fp32, and bf16 weight shadows add 0.3 GiB. The 4096 x 50304 logits and their gradient drop from 786 to 393 MiB each.

## 4. End-to-end demo, `demo/run_demo.sh large` with bf16s (`demo-large-bf16s/`)

One persistent launch: 6 training steps + 4 serving requests on versioned weights + real Doom (700 frames), one MANIFEST, 663,268 tasks, err_code 0.
All checks PASS: 22,935 version-carrying tasks exact, 0 violations; training losses 10.979 -> 10.924 (no CPU golden at this size); Doom **700/700 frames raw-identical** to the CPU oracle;
**cadence 170 frames/s against the 35 Hz needed, 0 of 698 tick gaps over 28.6 ms** (p50 5.8 ms, p99 10.3 ms). Training step (launch to done, steps 2-5, serve between): 708, 690, 690, 704 ms,
mean **698 ms against 1.81 s** in `2026-11-06-demo-large` (f32 1xTF32, same A100 box, an earlier commit, steps 1-5): 2.6x. Needed `kvm_run` to accept dtype `bf16` in the manifest (done).

## What did not work / limits

* The GEMM tile is at 0.47x of cuBLAS (section 1). The executor step stays 15.7x behind PyTorch's best bf16 mode; README goal 3 is not met at this size with bf16 either.
* Memory is -21%, not -50%; attention is fp32-only (no bf16 attention path), weight gradients and optimizer state are fp32 by design.
* Dynamic shared memory per block rose from 72 to 96 KiB (`KVM_GPT_MMA_SMEM`) so the 4-stage tile fits: two 72 KiB blocks per SM fit before, two 96 KiB blocks do not (the executor
  runs one block per SM at its register count, so the executor is unaffected; the op benchmarks' `BENCH_MINB=2` variant is).
* Zoo: 3 of 8 compared cases fail the weights criterion (`vit_bf16` and `vit_bf16s`: bias artifact; `san_bf16s`: 1.7e-2), reported above, tolerances unchanged.
* Not built: fp16, bf16 attention / SwiGLU / softmax / MoE ops, bf16 weight gradients, dynamic loss scaling, Hopper/Blackwell tiles (wgmma/TMA).
* Rebuilding the executor on the shared box takes 35-50 min under load and one set of builds was killed (cause unknown); the final binaries come from the tree named above.

## Proposed status text (not applied; README.md / AGENTS.md are not edited here)

* "Not built" -> remove "real bf16/fp16 storage"; add "fp16, bf16 attention, bf16 weight gradients". "Code only": drop "bf16 tile speed (unbenchmarked)".
* "Implemented + measured": "Real bf16 storage (ABI dtype 2) and a bf16 tensor-core GEMM tile `matmul_mma_bf16` (`results/2026-11-06-a100-bf16`): 104-125 TFLOP/s at 4096^3 (0.47-0.51x cuBLAS bf16),
  gpt_162m step 1103.6 -> 483.8 ms (2.28x), 318k -> 104k tasks, device buffers -21%; gradients within 1.7e-2 of float64 (PyTorch autocast-bf16: 3.4e-2); still 15.7x slower than PyTorch autocast-bf16 + CUDA graph."
* "Known failures": "bf16s matches the CPU golden's weights only to 1.7e-2 on a 2-step Llama (tolerance 1e-2); ViT weight check fails on the zero-gradient key bias in bf16 emulation and storage alike."
