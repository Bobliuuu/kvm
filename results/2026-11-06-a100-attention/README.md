# Causal attention at the 162M GPT shape: fused 128-row tensor-core tiles (A100, 2026-11-06)

**Verdict.** The three causal attention ops (`attn_fwd`, `attn_bwd_dq`, `attn_bwd_dkv`) were rewritten as flash-style tensor-core kernels where one
task covers **128 rows** instead of 32. At the 162M shape (batch 4, seq 1024, 12 heads, head size 64) the attention work per step is
**55,680 -> 14,208 tasks**, the forward op is **2.06x faster** and the backward **1.32x faster**, both now **ahead of PyTorch's fp32 SDPA**
(fwd 1.97x, bwd 1.14x, same box, same lock hold), and the executor step goes **1046 -> 905 ms (1.155x)**. Numerics are unchanged
(same 3xTF32 contract; the demo's six losses agree with the previous run to all printed digits).
Almost all of the end-to-end gain is task count, not kernel speed: the kernels save ~8 ms of GPU time per step, the 41,472 removed tasks save ~133 ms
of admission and ready-wait. That is also why the step is still 15x slower than PyTorch: attention is now 5% of the 276,667 tasks and ~17 ms of the 905 ms.

## Setup

| | |
|---|---|
| GPU | NVIDIA A100-SXM4-40GB, 108 SMs, sm_80, max SM clock 1410 MHz, driver 580.105.08, CUDA 12.8 (nvcc 12.8.93). Shared box: every timed run held the GPU lock; other agents' compiles ran beside it, so quote ratios from one process / one lock hold |
| Branch / commit | `perf/attention-162m`; device code `cce4880` (later commits change tests, comments and the result files only); base `origin/main` d657dde |
| Op benchmark | `tests/gpu_ops/bench_attn.cu`: single-op kernels (one block per task, 256 threads, heaviest tasks first), correctness against the fp64 reference before every timing. `nvcc -O2 -arch=sm_80 -std=c++17 -lineinfo -I../../include -I. [-DBENCH_HD=128] bench_attn.cu`; `./bench_attn [B T H Hkv]` (default 4 1024 12 12). "old" = the same file built against `origin/main`'s `kvm_ops_gpt.cuh` with `-DKVM_GPT_ATTN_ROWS=32` |
| Executor | `kvm_exec` `nvcc -O3 -arch=sm_80 -std=c++17 -Iinclude --fmad=false -Xcompiler -ffp-contract=off -lineinfo -diag-suppress 177 kvm_exec.cu`; bundle `gen.py w_162m_1 --model gpt_162m --steps 1 --mode tf32 --tiling a100` (from `bench/exec-vs-torch`), timed with its `exec_bench.py ... --reps 3` |
| Reference | PyTorch 2.7.0+cu128, `scaled_dot_product_attention(is_causal=True)`, EFFICIENT_ATTENTION backend for fp32 (`tests/gpu_ops/ref_torch.py`, `REF_ATTN_ONLY=1`) |

## 1. Per-op results (162M shape: B4 T1024 H12 hd64; `raw/bench_attn_old_vs_new_and_torch.txt`)

Old and new alternate in one lock hold, 5 rounds, median of 7 repetitions each; the number is the value of rounds 1-4 (identical to 0.5%); round 5 caught the
box in a slower state for `old` only (fwd 715 us, bwd 1995 us), `new` was unmoved. TFLOP/s = causal 2*B*H*T^2*hd (x2.5 for the backward), as `bench_ops.cu` and `ref_torch.py` count.

| op | tasks before -> after | us before | us after | speedup | TFLOP/s before -> after | torch fp32 SDPA |
|---|---|---|---|---|---|---|
| attn_fwd | 1536 -> 384 | 559 | **272** | 2.06x | 11.5 -> **23.7** | 536 us, 12.0 TFLOP/s |
| attn_bwd_dq | 1536 -> 384 | 454 | 364 | 1.25x | 17.7 -> 22.1 | |
| attn_bwd_dkv | 1536 -> 384 | 822 | 532 | 1.54x | 9.8 -> 15.1 | |
| attn_bwd_delta | 32 -> 32 | 188 | 188 | 1.00x | | |
| **backward (delta + dq + dkv)** | 3104 -> 800 | 1573 | **1190** | 1.32x | 10.2 -> **13.5** | 1351 us, 11.9 TFLOP/s |
| attention tasks per layer (fwd + bwd) | 4640 -> 1184 | | | | | |
| attention tasks per step (12 layers) | 55,680 -> 14,208 | | | | | |

Other shapes (3 rounds each, spread under 1%):

| shape | op | us before -> after | TFLOP/s before -> after |
|---|---|---|---|
| B4 T2048 H16 Hkv4 hd64 (GQA 4:1) | fwd | 2635 -> 1220 | 13.0 -> 28.2 |
| | bwd (delta+dq+dkv) | 6127 -> 4507 | 14.0 -> 19.1 |
| B2 T2048 H8 hd128 (not in the default `KVM_GPT_MAX_HD=64` build) | fwd | 1437 -> 785..1007 | 11.9 -> 21.9..17.1 |
| | **bwd** | **3103 -> 4520 (slower, 0.69x)** | 13.8 -> 9.5 |

PyTorch references measured in the same lock hold (`raw/...torch`): fp32 and TF32-allowed efficient attention give the same 0.535 ms forward and 1.35 ms backward;
bf16 FlashAttention is 0.092 / 0.305 ms (not a fair target for a 3xTF32 kernel; the bf16 track owns that).

## 2. Where the time goes (first-principles profile; no ncu/nsys exists on the box, so by experiment)

- **Floor.** The forward is 6.44 GFLOP of useful work; as 3xTF32 it costs 3 tensor products per useful one = 19.3 GFLOP-equivalents of mma, 124 us at the 156 TFLOP/s TF32 peak
  (184 us at the 105 TFLOP/s cuBLAS reaches on this box). The new kernel takes 272 us: 46% of peak, 68% of cuBLAS-class.
  The three kernels run at 71-80 TFLOP/s of raw TF32 mma (fwd 71, dq 80, dkv 73), the same as the best matmul kernels in this tree (43-51 for the 64x64 tile, 105 for cuBLAS).
- **Tensor bound, measured.** Dropping the `hi*lo` product (2xTF32, breaks the 2e-5 contract: error 5e-4, so this is a probe only) cuts fwd 272 -> 196 us, dq 364 -> 250, dkv 532 -> 392:
  time follows the mma count, not the exp/softmax/ALU work. The SASS of the hd=64 loop shows HMMA interleaved with LDS and FADD/LOP3 (the hi/lo splits), no spills in
  the executor-relevant kernels (fwd 211 regs, dq/dkv 255 with <= 100 B of stack in the single-op builds).
- **Why the old kernels were at 9-12 TFLOP/s.** 32-row tasks: 8 warps = 2 row halves x 4 key slices, each warp ran a single 24-deep dependent mma chain per score tile, every
  Q fragment was re-split from shared memory every tile, and four partial O/dQ/dK/dV tiles were merged through shared memory at the end of every task.
- **Causal imbalance inside a 128-row task is small.** Warp w stops at its own diagonal: 1056 warp-tiles vs 1024 ideal (3%), and the slowest warp of a block costs 8% of the block's
  issue slots on average (62% at the first block, 95% at the last).
- **The remaining 2x to the 3xTF32 floor** is the shared-memory fragment traffic and the hi/lo splitting that every warp repeats on the same K/V tile, plus one block per SM (255 registers).
  Splitting once into shared memory needs 2x the stage bytes (over the 72 KiB all-ops budget at 3 stages); I did not try a 2-stage variant.
- **The backward is 7 tensor products where PyTorch does 5.** `dq` and `dkv` are separate tasks (deterministic, one writer per output element), so S and dP are recomputed in both. A fused
  backward needs atomics or a partial-dQ scratch of 100 MB per layer (about the cost it would save); not done.
- **`attn_bwd_delta` is the worst op per byte** (188 us for 25 MB; 32 tasks, one per 128 rows, latency bound: each task is 1.5 us per row regardless of loads in flight). It costs
  ~2.3 ms per step, so it was left alone; a 16-row split takes it to 44 us for 256 tasks, which trades back more admission than it saves.

## 3. End to end

Executor, `w_162m_1` (one optimizer step), `raw/executor_step_time.txt`: 3 rounds x 3 launches, base and new alternating in one lock hold.

| | tasks | ms / step, best (median, worst) over 9 launches |
|---|---|---|
| `origin/main` | 318,139 | 1046.3 (1048.1, 1068.2) |
| this branch | 276,667 | **905.6** (907.7, 935.6) |

1.155x, -141 ms, -41,472 tasks (= 12 layers x 3456 attention tasks). 3.4 us per removed task: the 1.9 us of single-FIFO admission plus the ready-to-start wait it hides
(the traced baseline has 107 workers busy 26.7% of the time).

`demo/run_demo.sh large` (162M GPT, 6 steps + serving + 700 Doom frames, one launch, `demo-large/`): **ALL CHECKS PASSED**, 1,699,336 tasks (was 1,948,168), 12.07 s (was 13.66 s),
Doom 700/700 frames identical, 171.8 frames/s, version check 22,935/22,935 exact. The six step losses are
10.9790, 10.9369, 10.8757, 10.8827, 10.8925, 10.9237, the same as the previous run (`2026-11-06-demo-large`: 10.979, 10.937, 10.876, 10.883, 10.892, 10.924); steps 2-6 depend on the new
backward through AdamW, so a gradient error at the 1e-3 level would show.
Not rerun: the step-1 gradient comparison against fp64 (`grads162.sh`; needs ~3 GB of box disk, 2-4 GB were free) and the 3-step torch loss comparison (`executor162.sh`;
it needs a 14-minute 3-step bundle). The 1-step executor timing replaces it.

Rejected: emitting the query blocks heaviest-first in the lowering (so the FIFO admits the long tasks early): 909-911 ms vs 903-907 ms, no gain, reverted (`raw/executor_step_time_heaviest_first_rejected.txt`).

## 4. Correctness

- `test_ops`: **all passed, 0 failed** (was 1396 checks on `origin/main`; new cases: multi-block T=257/300 with partial last block, GQA 2:1, hd 16..128, T=1 at hd 64/128, and a 256-draw T=1 noise sweep that counts its checks). `test_e2e` passes.
  Raw: `test_ops.txt`.
- compute-sanitizer on the attention cases (`TEST_ONLY=attention`, 1955 checks): memcheck, racecheck, synccheck, initcheck 0 errors (`sanitizer_*`).
  An earlier racecheck run caught my own test: two overlapping 128-row tasks (origins 33 apart) wrote the same rows from differently grouped sums, so the last bits depend on who wrote last.
  Production lowering never overlaps; the overlapping cases were removed and the header now states the contract (tasks must tile rows in steps of `KVM_GPT_ATTN_ROWS`).
- Bitwise deterministic run to run (every case has the check). Accuracy vs the fp64 reference in the bench: fwd 2.5e-6, dq 1.5e-6, dk 1.7e-5 and dv 1.8e-5 relative to max (bounds 2e-5 / 5e-5);
  the old kernels measured 2.1e-6 / 1.2e-6 / 5.0e-6. The dk/dv increase comes from `ex2.approx` on `scale*log2(e)*s - lse*log2(e)` (relative 2^-22 plus the rounding of a
  product of magnitude up to ~10); it stays 3x inside the test tolerance.
- CPU: `python3 -I -m unittest discover -s tests` 146 OK, `-s tests/frontend` 60 OK (a new test pins the Python block sizes to the device constant).

### The open T=1 backward note (AGENTS.md) - resolved: a real, bounded noise floor; the bound was too tight

With T=1, `p = 1`, `o = v`, so `dS = dP - delta = dO.V - dO.O` is exactly 0 and so are `dq` and `dk`; the device returns rounding noise because `dP` comes out of a
3xTF32 product (error about 2^-22 relative per term, `hd` terms) and `delta` out of an fp32 dot (`attn_bwd_delta`). The noise is not relative to anything, so the test uses an absolute bound.
It was 3e-5 and measured 9e-6 / 1.5e-5 at hd=64 here: only 2x margin, which is how a different tree or data draw can fail it. A 256-draw sweep (64 seeds x hd 16/32/64/128) now gives a
maximum of **2.77e-5** for any |dq|, |dk| (already within 8% of the old bound), so the bound is **1e-4** (3.6x margin) with the derivation in the test.
Not reproduced as a failure on the A100 by this branch's runs or by `fix/readme-t1-attn-bwd`; I cannot see the failing run, so I do not claim it was this, only that the old margin allowed it.

## 5. What did not help / is not done

- Heaviest-first block order in the lowering: no gain (above). `attn_bwd_delta` rewritten with 8 pairs in flight and hoisted loads: no change (167 vs 188 us), reverted.
- hd=128 backward is slower than before (0.69x; 255 registers and 1.5 KB of spills: the resident K/V (or Q/dO) fragments of a 16-row tile are 128 registers at hd=128). It only
  affects `-DKVM_GPT_MAX_HD=128` builds; the default build compiles hd <= 64. Fixing it needs fragment streaming for hd=128.
- 3xTF32 is kept (the 2e-5 contract); a 2xTF32 / fp16-split path would be 1.4x / ~2x faster but changes numerics, and bf16 storage belongs to the bf16 track. The interface is unchanged
  (buffers fp32; the bf16 agent can add a dtype next to `Chk::mat`).
- Larger tasks (256 rows, two sequential halves) would save another ~13 ms of admission per step but leave 192 forward tasks on 108 SMs; not tried.
- No full backward fusion (see section 2) and no profiler counters (not installed on the box).

## Reproduce

```bash
cd tests/gpu_ops   # on the GPU box, headers from ../../include
nvcc -O2 -arch=sm_80 -std=c++17 -lineinfo -I../../include -I. bench_attn.cu -o bench_attn && flock ~/.gpu.lock ./bench_attn 4 1024 12 12
nvcc -O2 -arch=sm_80 -std=c++17 -lineinfo -I../../include -I. test_ops.cu -o test_ops && flock ~/.gpu.lock ./test_ops
TEST_ONLY=attention compute-sanitizer --tool racecheck --error-exitcode 9 ./test_ops
REF_ATTN_ONLY=1 python3 -I ref_torch.py
KVM_HOST=... demo/run_demo.sh large
```
