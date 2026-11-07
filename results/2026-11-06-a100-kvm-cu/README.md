# kvm.cu on the A100: timeline, exact weights, torch baselines, fused program (work in progress)

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
