# kvm.cu on the A100: timeline, exact weights, torch baselines, fused programs

Latest result: section "Fair precision comparison (run f16b)" directly below, which supersedes the headline of the f16a section (torch TF32 only) after it.

## Fair precision comparison (run f16b, Modal, commit 54d85d2)

One container, one process: the VM programs and every torch variant at three precision classes: tf32 (fp32 tensors, TF32 matmul), fp16 and bf16 (autocast: half matmul inputs, fp32 accumulate, fp32 master weights, gradients carried x1024, same SGD update). Variants per class: eager (manual backward), whole-run CUDA graph, compile + graph, hand-batched bmm graph, bmm compile + graph, vmap(grad) graph, compile(vmap(grad)) + graph. All numerically verified against the VM (loss rel err 1e-5 fp16, 2e-4 bf16, 2e-6 tf32 over 20 steps; weights 5e-5 / 2e-3 / 6e-5). GPU this time: A100-SXM4-80GB, 108 SMs (Modal assigned the 80 GB part; f16a was the 40 GB part), driver 610.57.04, torch 2.7.0+cu126, 2 rounds x 3 reps best. Raw: `f16b/`.

us per step per run, best (best torch variant of that class in brackets):

| runs | VM fused16 (fp16 operands) | VM fused (TF32) | torch fp16 | torch bf16 | torch TF32 |
|---|---|---|---|---|---|
| 1 | **15.6** | 24.9 | 31.5 (compile + graph) | 34.9 (compile + graph) | 37.3 (compile + graph) |
| 8 | **2.20** | 3.28 | 4.94 (compile vmap + graph) | 5.48 (compile bmm + graph) | 4.96 (compile bmm + graph) |
| 32 | **1.71** | 2.90 | 2.14 (compile vmap + graph) | 2.26 (compile bmm + graph) | 2.02 (compile vmap + graph) |

VM speedup over the best torch of each class:

| runs | LIKE-FOR-LIKE half: fused16 vs torch fp16 | fused16 vs torch bf16 | fused16 vs torch TF32 | LIKE-FOR-LIKE TF32: fused vs torch TF32 |
|---|---|---|---|---|
| 1 | **2.01x** | 2.23x | 2.39x | **1.50x** |
| 8 | **2.24x** | 2.49x | 2.25x | **1.51x** |
| 32 | **1.25x** | 1.32x | 1.18x | **0.70x (VM LOSES)** |

Reading it plainly:
- Like-for-like fp16 (fp16 operands, fp32 accumulate and master weights on both sides): the VM wins at every size, 2.0x at 1 run, 2.2x at 8, but only 1.25x at 32 runs.
- Like-for-like TF32 (VM `fused` vs torch TF32): the VM wins 1.5x at 1 and 8 runs and LOSES at 32 runs (0.70x: compiled vmap graph 2.02 vs 2.90 us). The R=32 win exists only with fp16 operands, so it comes from the precision switch (halved shared-memory bytes, two blocks per SM), not from the VM idea itself; torch's compiled graph also gains little from fp16 here (2.14 vs 2.02 us: elementwise and launch overheads dominate at this size, so tensor-core width does not help it).
- Advantage shrinks from ~2x to 1.2-1.3x between 8 and 32 runs against every torch class, because torch's batched bmm approaches its own floor (2 us per run-step) while the VM is still ~14x above the FLOP floor.
- Mixed-precision note: the torch fp16 path keeps the same master-weight semantics as fused16, so the half-precision row is a fair comparison; the VM's lack of dynamic loss scaling (fixed x1024) applies equally to the torch variants here.

Proposed README/AGENTS text (not edited): "kvm.cu MLP, A100, vs the best torch variant of the same precision (compile + whole-run CUDA graph, vmap/bmm batching): fp16 operands 2.01x / 2.24x / 1.25x at 1 / 8 / 32 runs; TF32 1.50x / 1.51x / 0.70x (the VM loses at 32 runs in TF32). results/2026-11-06-a100-kvm-cu/f16b."

Earlier: section "Update: fused16 (run f16a)" below (torch TF32 only, 40 GB part); the `s1` sections after it are the first run and are kept for the record.

## Update: fused16 (run f16a, Modal A100-40GB, commit 888ef2b)

One Modal container: kvm.cu (3 VM programs, each with its 3 launch-style baselines) and the torch variants timed in ONE process (`bench/mlp_vs_torch/modal_run.py`, 2 rounds x 3 reps, best). GPU A100-SXM4-40GB 108 SMs, driver 610.57.04, nvcc 12.8, torch 2.7.0+cu126 (TF32), flags `-O3 -arch=sm_80 -DKTILE=32 -DKKC=64 -DKSTG=4`. Raw: `f16a/` (logs, JSON, timelines, `timeline_R1.png` fused vs fused16, `timeline_R32.png`). Weights are bitwise identical across the four modes in every program and run (0 elements differ); loss vs CPU fp32 2e-5 over 20 steps; torch compile baselines now have correct loss curves.

| runs | VM tiles (original) | VM fused (TF32) | **VM fused16 (new)** | best torch (TF32) | fused16 vs best torch |
|---|---|---|---|---|---|
| 1 | 25.2 | 25.0 | **15.7** | 37.4 (compile(vmap(grad)) + graph) | 2.38x |
| 8 | 4.25 | 3.25 | **2.19** | 4.94 (compile bmm + graph) | 2.26x |
| 32 | 3.45 | 2.85 | **1.65** | 2.01 (compile bmm + graph) | 1.22x |

(us per step per run.) fused16 vs the best launch-style baseline of the SAME program: 1.33x at 1 run, 1.55x at 8, 1.40x at 32 (batched graph). So the VM's own contribution is now clear at all three sizes, and the R=32 loss to compiled bmm is reversed.

What changed and why (from the KSTAGE profile in `diag2`: ~35k cycles per step at R=1, 15x over the tensor-core bound, spread evenly over the 4 GEMM groups, i.e. shared-memory-traffic bound, not hop bound, not FLOP bound): `--prog fused16` runs the same two instructions with fp16 tensor-core inputs, fp32 accumulate and fp32 master weights. Halving operand bytes and doubling k per mma cut per-step time 1.6x; the block shrinks from 148 KB to 78 KB of smem so two blocks fit per SM (up to 27 runs resident). Gradients are carried x1024 (exact power of two) so they stay out of fp16's subnormal range; the lr/1024 is folded into the update. Weights stay fp32 in global memory; the fp16 shadow in smem is refreshed after each update.

Caveats, honest:
- The VM now uses fp16 operands while the torch variants use TF32. A torch fp16/bf16-autocast variant is NOT yet in the table; it could narrow the gap, so treat the 1.22x at R=32 as provisional until that baseline exists. Numerics: the VM's loss tracks the fp32 CPU reference to 2e-5 relative over the first 20 steps and the 300-step loss curves agree to the printed digits.
- Floors at R=32 per step-run: 37.7 MFLOP per run-step (fwd 2*B*(D*H+H*O) = 12.6 MFLOP, bwd twice that), i.e. 0.24 us at the 156 TFLOP/s TF32 peak (0.12 us at fp16 peak); bytes ~0.2 MB/run-step at 1.5 TB/s = 0.13 us. The VM is at 1.65 us: still ~7-14x above the tensor-core floor, limited by in-block shared-memory traffic and latency, not by hops (the hop is ~1 us once per step and overlaps other runs).
- fused16 only for B=128 D=128 H=256 O=64 (compile-time geometry), 8 slices.
- Not done: lane-level bank-conflict profile, mma.sync/ldmatrix direct path, larger-k per-run fusion, torch fp16 baseline, an RTX 5070 rerun.

Proposed README/AGENTS text (not edited): "kvm.cu MLP, A100: fused16 program 15.7 / 2.19 / 1.65 us per step-run at 1 / 8 / 32 runs, 2.38x / 2.26x / 1.22x over the best torch TF32 baseline (compiled graph); fp16 operands vs torch TF32, see results/2026-11-06-a100-kvm-cu."

## Earlier run s1 (lab box, commit b829c4f)

Run `s1/` (the only complete measurement so far; later work was blocked by the GPU lock queue, see "Not done").

- GPU: NVIDIA A100-SXM4-40GB, 108 SMs, driver 580.105.08, CUDA 12.8 (nvcc 12.8.93), torch 2.7.0 (TF32 matmuls), SM clock 1410 MHz.
- Build: `nvcc -O3 -arch=sm_80 -std=c++17 -DKTILE=32 -DKKC=64 -DKSTG=4 kvm.cu` (`-shared -Xcompiler -fPIC -DKVM_LIB` for the library the torch harness loads).
- Commit: b829c4f (branch perf/kvm-cu). Command: `KVM_HOST=<box> bench/mlp_vs_torch/run_session.sh <outname> 1 8 32` (`session.sh` on the box, one GPU-lock hold; `torch_mlp.py` times torch variants and both VM programs IN ONE PROCESS: kvm.cu is loaded as a library via ctypes).
- Model: B=128 D=128 H=256 O=64, 300 steps per run, R runs = LR sweep. Every number is best of 5 reps x 3 rounds; ratios come from one process.
- Raw: `s1/result_R*.json`, `s1/log_R*.txt`, `s1/selftest.txt`, timelines `s1/timeline_R*_{tiles,fused}.csv` (steps 100-102).

## What was closed

- (a) Timeline: `--timeline file --tl-steps a b` runs the VM once more with `prof` set; per instruction it records claim / deps-ready / done times (`%globaltimer`) and `%smid` + block. `bench/mlp_vs_torch/plot_timeline.py` draws `s1/timeline_R1.png`, `s1/timeline_R8.png` from the VM's own timestamps. Caveat: `%globaltimer` ticks in 1 us steps on this GPU. The profiled run is slower than the plain one, use it for shape not speed.
- (b) Weights: the check is now exact bit equality across all four modes (and counts differing elements). 0 differing elements in every run below, for both programs. Loss still checked vs CPU fp32 (rel err 2e-5).
- (f) Baselines for the same MLP, same bits (X, T, W1, W2, per-run lr dumped by kvm.cu): torch eager (autograd; manual backward), CUDA graph of the whole run, torch.compile (python loop; + whole-run graph), torch.func.vmap(grad) (eager; + graph; compiled + graph), hand-batched bmm (eager; whole graph; compiled + graph). Weights of every torch variant match the VM to 6e-5 relative (TF32 rounding).

## Results (us per step per run, best; VM = best of its two programs)

| runs | VM tiles (original) | VM fused (new) | best valid torch | VM vs best torch |
|---|---|---|---|---|
| 1 | 25.3 | 25.3 | 53.8 (whole-run CUDA graph) | 2.13x |
| 8 | 4.22 | 3.31 | 4.64 (compile(vmap(grad)) + graph) | 1.40x |
| 32 | 3.47 | 2.95 | 1.96 (compile bmm + graph) | 0.66x (VM LOSES) |

Hand baselines (same tape as ordinary kernels, from the same process): batched graph 28.5 / 6.40 / 3.67, sequential graph 28.5 / 28.8 / 28.9, per-op launches 35 / 35 / 36 (R=1/8/32, tiles). So the original "1.5x at 1 run" is really 1.13x over the batched graph and 2.1x over torch; at 32 runs a compiled torch bmm graph is 1.5x FASTER than the VM.

Honest caveats:
- The torch.compile variants are flagged MISMATCH in the logs: their loss curve is all zeros (an in-place write into a view argument is dropped under compile; fixed in a later commit, untested on the GPU). Their weights match the VM (5.8e-5) and their timing is valid, so the table uses them. The compile gain over the plain CUDA graph is real (fused elementwise).
- torch eager / eager-loop variants are 9-165x slower than the VM; they are in the JSON, not in the table.
- The 25.3 us single-step is unchanged by the fused program. Timeline (`s1/timeline_R1.png`, bottom): FWD 8.2 us, hop ~1 us, BWD 15.4 us per step. The expected ~10 us total did not appear: the in-block work (wmma from smem, TF32 conversion, 256 KB of partial-y reads) costs far more than the hop. Floors for one step: FLOPs 37.7 MFLOP = 0.24 us at 156 TFLOP/s, bytes ~0.2 MB = 0.1 us, dependency chain 1 hop (fused) vs 4 (tiles) at ~1-5 us; the VM is 25 us, i.e. in-block compute efficiency (about 1.5 TFLOP/s) is the gap, not hops.

## Fused program (`--prog fused`, new)

Megatron-style slicing: W1 by columns and W2 by the same rows into 8 slices; instruction FWD (h_c, partial y_c) then one hop then BWD (sum of partials in fixed order, loss, dy, dh, both weight updates, all local). 2 instructions per (run, slice) instead of ~120, static per-block queue (no claim atomic), loop opcode per step, slice weights and h resident in shared memory when one task per block (R <= 13 at 8 slices). Baselines run the same two instructions as one launch per phase: weights bitwise identical across the four modes. Gain: 1.28x over the tile VM at 8 runs, 1.18x at 32, none at 1.

## Not done / next

1. Profile where FWD 8 us and BWD 15 us go (stage-cycle profiler is in the code, `-DKSTAGE`, never run: the diagnostic session starved in the lock queue of ~20 waiters).
2. Likely fixes: shared-memory bank conflicts / TF32 conversion in the wmma path, the 256 KB partial-y read (use 4 slices with streamed X, or reduce-scatter), prefetch.
3. Rerun s1 with the fixed compile baselines and the latest kernel (t read straight from global). GPU work should move to the Modal A100 runner (tests/modal/run_suite.py on origin/tools/modal-suite).

Proposed text for README/AGENTS (not edited): "`kvm.cu` MLP: now has a timeline (results/2026-11-06-a100-kvm-cu), exact bitwise weight check, and torch baselines in one process. Versus torch's best valid variant: 2.13x at 1 run, 1.40x at 8, 0.66x (loses to compiled bmm + CUDA graph) at 32 runs."
