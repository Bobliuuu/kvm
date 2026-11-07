# Causal attention at the 162M GPT shape: fused 128-row tensor-core tiles (A100, 2026-11-06)

**Verdict.** The three causal attention ops (`attn_fwd`, `attn_bwd_dq`, `attn_bwd_dkv`) were rewritten as flash-style tensor-core kernels where one
task covers **128 rows** instead of 32. At the 162M shape (batch 4, seq 1024, 12 heads, head size 64) the attention work per step goes
**55,680 -> 14,208 tasks**, the forward op is **2.06x faster** and the backward **1.29x faster**, both **ahead of PyTorch's fp32 SDPA**
(fwd 1.99x, bwd 1.12x; same box, same lock hold), and the executor step goes **1046 -> 905 ms (1.155x)**. Numerics keep the 3xTF32 contract (the six
demo losses agree with the previous run to all printed digits).
Almost all of the end-to-end gain is task count, not kernel speed: the kernels save ~8 ms of GPU time per step, the 41,472 removed tasks save ~134 ms of
admission and ready-wait. That is also why the step is still 15x slower than PyTorch: attention is now 5% of the 276,667 tasks and ~17 ms of the 905 ms.

## Setup

| | |
|---|---|
| GPU | NVIDIA A100-SXM4-40GB, 108 SMs, sm_80, max SM clock 1410 MHz, driver 580.105.08, CUDA 12.8 (nvcc 12.8.93). Shared box: every timed run held the GPU lock, but other agents compile beside it, so quote ratios from one lock hold |
| Branch / commit | `perf/attention-162m` on base `origin/main` d657dde; the numbers below are from the device code of commit f8ab4a7 (later commits touch comments and result files only) |
| Op benchmark | `tests/gpu_ops/bench_attn.cu`: single-op kernels (one block per task, 256 threads, heaviest tasks first), result checked against the fp64 reference before every timing. `nvcc -O2 -arch=sm_80 -std=c++17 -lineinfo -I../../include -I. [-DBENCH_HD=128] bench_attn.cu`; `./bench_attn [B T H Hkv]` (default 4 1024 12 12). "old" = the same file built against `origin/main`'s `kvm_ops_gpt.cuh` with `-DKVM_GPT_ATTN_ROWS=32` |
| Executor | `kvm_exec` built `nvcc -O3 -arch=sm_80 -std=c++17 -Iinclude --fmad=false -Xcompiler -ffp-contract=off -lineinfo -diag-suppress 177 kvm_exec.cu`; bundle `gen.py w_162m_1 --model gpt_162m --steps 1 --mode tf32 --tiling a100` and `exec_bench.py ... --reps 3` from `bench/exec-vs-torch` |
| Reference | PyTorch 2.7.0+cu128 `scaled_dot_product_attention(is_causal=True)`, efficient-attention backend for fp32 (`tests/gpu_ops/ref_torch.py`, `REF_ATTN_ONLY=1`) |

## 1. Per-op results, 162M shape: B4 T1024 H12 hd64 (`raw/bench_attn_old_vs_new_and_torch.txt`)

Old and new alternate in one lock hold, 5 rounds, median of 7 repetitions each. Rounds 2-5 agree to 0.2%; in round 1 the box was in a slower state for `old` only (fwd 719 us, bwd 1998 us; `new` unmoved).
TFLOP/s = causal 2*B*H*T^2*hd (x2.5 for the backward), as `bench_ops.cu` and `ref_torch.py` count.

| op | tasks before -> after | us before | us after | speedup | TFLOP/s before -> after | torch fp32 SDPA |
|---|---|---|---|---|---|---|
| attn_fwd | 1536 -> 384 | 559 | **271** | 2.06x | 11.5 -> **23.8** | 539 us, 11.9 TFLOP/s |
| attn_bwd_dq | 1536 -> 384 | 454 | 370 | 1.23x | 17.7 -> 21.8 | |
| attn_bwd_dkv | 1536 -> 384 | 822 | 555 | 1.48x | 9.8 -> 14.5 | |
| attn_bwd_delta | 32 -> 32 | 188 | 188 | 1.00x | | |
| **backward (delta + dq + dkv)** | 3104 -> 800 | 1572 | **1222** | 1.29x | 10.2 -> **13.2** | 1365 us, 11.8 TFLOP/s |
| attention tasks per layer | 4640 -> 1184 | | | | | |
| attention tasks per step (12 layers) | 55,680 -> 14,208 | | | | | |

Other shapes (3 rounds each, spread under 1% except where noted):

| shape | op | us before -> after | TFLOP/s before -> after |
|---|---|---|---|
| B4 T2048 H16 Hkv4 hd64 (GQA 4:1) | fwd | 2635 -> 1217 | 13.0 -> 28.2 |
| | bwd (delta+dq+dkv) | 6130 -> 4617 | 14.0 -> 18.6 |
| B2 T2048 H8 hd128 (not built by default: `KVM_GPT_MAX_HD=64`) | fwd | 1437 (1118 in a faster box state) -> 829 | 12.0 (15.4) -> 20.7 |
| | **bwd** | **3100 -> 4071 (slower, 0.76x)** | 13.9 -> 10.5 |

PyTorch in the same lock hold: fp32 and TF32-allowed efficient attention give the same forward (0.54 ms) and backward (1.36 ms); bf16 FlashAttention is 0.093 / 0.306 ms
(not a fair target for a 3xTF32 kernel; the bf16 track owns that).

## 2. Where the time goes (no ncu/nsys exists on the box, so profiled by experiment and SASS)

- **Floor.** The forward is 6.44 GFLOP of useful work; as 3xTF32 it costs three tensor products per useful one = 19.3 GFLOP-equivalents, 124 us at the 156 TFLOP/s TF32 peak (184 us at the
  105 TFLOP/s cuBLAS reaches on this box). The new kernel takes 271 us: 46% of peak, 68% of cuBLAS-class. The three kernels run at 71-80 TFLOP/s of raw TF32 mma (fwd 71, dq ~78, dkv ~70);
  the 64x64 matmul tile in this tree reaches 43-51.
- **Tensor bound, measured.** Dropping the `hi*lo` product (2xTF32; error 5e-4, so a probe only, not kept) cuts fwd 272 -> 196 us, dq 364 -> 250, dkv 532 -> 392: time follows the mma count,
  not the exp/softmax/ALU work. The SASS of the hd=64 loop shows HMMA interleaved with LDS and the FADD/LOP3 hi/lo splits.
- **Why the old kernels sat at 8-12 TFLOP/s.** 32-row tasks: 8 warps = 2 row halves x 4 key slices, each warp ran one 24-deep dependent mma chain per score tile, re-split the Q fragments from
  shared memory every tile, and merged four partial O/dQ/dK/dV tiles through shared memory at the end of every task.
- **Causal imbalance inside a 128-row task is small.** Warp w stops at its own diagonal: 1056 warp-tiles vs 1024 ideal (3%); the slowest warp of a block holds the others idle for 8% of issue slots
  on average (62% efficient at the first block, 95% at the last).
- **The remaining 2x to the 3xTF32 floor** is shared-memory fragment traffic and the hi/lo splitting every warp repeats on the same K/V tile, plus one block per SM (255 registers). Splitting once
  into shared memory needs twice the stage bytes (over the 72 KiB all-ops budget at 3 stages). Not tried.
- **The backward is 7 tensor products where PyTorch does 5.** `dq` and `dkv` stay separate tasks (deterministic, one writer per output element), so S and dP are recomputed in both. A fused backward
  needs atomics or a partial-dQ scratch of ~100 MB per layer (about the traffic it would save). Not done.
- **`attn_bwd_delta` is the slowest op per byte** (188 us for 25 MB; 32 tasks of 128 rows are latency bound: ~1.5 us per row whatever the loads in flight). It costs ~2.3 ms per step, so it was
  left alone; a 16-row split takes it to 44 us for 256 tasks, which gives back more admission than it saves.

## 3. End to end

Executor, `w_162m_1` (one optimizer step), `raw/executor_step_time.txt`: 3 rounds x 3 launches, base and final alternating in one lock hold.

| | tasks | ms / step: best, median, worst of 9 launches |
|---|---|---|
| `origin/main` | 318,139 | 1045.8, 1048.4, 1054.6 |
| this branch | 276,667 | **905.3**, 910.8, 994.1 (3 launches ran 990-994 in a slow box state, the other 6 at 905-921) |

1.155x on best, 1.151x on median; -141 ms for -41,472 tasks (12 layers x 3456 attention tasks): 3.4 us per removed task = the 1.9 us of single-FIFO admission plus the ready-to-start wait it hides.
Order of the blocks in the lowering was tried both ways (heaviest first: 909-911 ms, ascending: 903-907 ms): no gain, kept ascending
(`raw/executor_step_time_heaviest_first_rejected.txt`, earlier build).

`demo/run_demo.sh large` (162M GPT, 6 steps + serving + 700 Doom frames, one launch; `demo-large/`): see the table; all checks pass, Doom 700/700 frames identical, version check exact.
ALL CHECKS PASSED, 172.0 Doom frames/s. The six step losses are 10.9790, 10.9369, 10.8757, 10.8827, 10.8925, 10.9237, the same as the previous run (`2026-11-06-demo-large`: 10.979, 10.937, 10.876, 10.883, 10.892, 10.924); steps 2-6 depend on the
new backward through AdamW, so a gradient error at the 1e-3 level would show.

| | tasks | run time |
|---|---|---|
| `2026-11-06-demo-large` (before) | 1,948,168 | 13.66 s |
| this branch (final code, `demo-large/`) | 1,699,336 | **12.05 s** (12.07 s with the first version of the kernels) |

Not rerun: the step-1 gradient comparison against fp64 (`grads162.sh`; needs ~3 GB of box disk, 2-4 GB were free) and the 3-step torch loss comparison (`executor162.sh`; needs a 14-minute 3-step bundle).
The six-loss agreement stands in for them.

## 4. Correctness

- `test_ops` **3224 passed, 0 failed** (`test_ops.txt`; 1396 checks on `origin/main`; the new ones are multi-block T=257/300/129 with partial last blocks, GQA 2:1, hd 16..128, T=1 at hd 64/128, and a
  256-draw T=1 noise sweep that counts its checks quietly). `test_e2e` 7 passed (`test_e2e.txt`).
- compute-sanitizer on the attention cases (`TEST_ONLY=attention`, 1941 checks): memcheck, racecheck, synccheck, initcheck all 0 errors (`sanitizer_*`).
  An earlier racecheck run caught my own test: two overlapping 128-row tasks (origins 33 apart) wrote the same rows from differently grouped sums, so the last bits depended on who wrote last.
  Production lowering never overlaps; the overlapping cases are gone and the header states the contract (tasks tile rows in steps of `KVM_GPT_ATTN_ROWS`).
- Bitwise deterministic run to run (every case checks it). Accuracy vs the fp64 reference (bench, relative to max; T=1024 / T=2048): fwd 1.3e-6 / 2.2e-6, dq 4.1e-6 / 2.8e-6, dk 9.0e-6 / 1.1e-5,
  dv 5.7e-6 / 9.2e-6. The old kernels: 1.3e-6 / 1.8e-6, 4.1e-6 / 2.6e-6, 7.2e-6 / 8.1e-6, 5.2e-6 / 7.8e-6. Tolerances are 2e-5 (fwd) and 3e-5 (bwd).
- **An accuracy regression found and fixed on the way.** The first version of the new backward had dk/dv errors that grew with T (2.1e-5 at 1024, 3.0e-5 at 2048, old 7e-6). The exp (`ex2.approx` vs
  `exp2f`) is not the cause (identical numbers); the cause is the tensor core truncating its fp32 accumulator on every mma: three adds per k-step into a ~1000-step accumulator. Keeping `hi*hi`
  in the long accumulator and summing the two small products in a zeroed temp that is added once per tile (round-to-nearest) cut it 2.4x (dk 2.1e-5 -> 9.0e-6) for +1.6% dq and +4% dkv. Two other probes:
  two interleaved accumulator chains (1.7x better, dkv +22% from spills) and doing `hi*hi` first (no accuracy change, dkv 4% faster but dq/fwd slightly less accurate; not kept).
- CPU: `python3 -I -m unittest discover -s tests` 146 OK, `-s tests/frontend` 60 OK (a new test pins the Python block sizes to the device constant).

### The open T=1 backward note (AGENTS.md): resolved. A real, bounded noise floor; the old bound was too tight

With T=1, `p = 1` and `o = v`, so `dS = dP - delta = dO.V - dO.O` is exactly 0 and so are `dq` and `dk`. The device returns rounding noise because `dP` comes out of a 3xTF32 product
(about 2^-22 relative error per term, `hd` terms) and `delta` out of an fp32 dot (`attn_bwd_delta`). The noise is not relative to anything, so the test uses an absolute bound. It was 3e-5 and the
cases measured 9e-6 / 1.5e-5 at hd=64 (2x margin), which is how a different tree or data draw fails it. A 256-draw sweep (64 seeds x hd 16/32/64/128, `attn T=1 backward noise sweep`) gives a
maximum of **2.77e-5** for any |dq|, |dk| (within 8% of the old bound), so the bound is now **1e-4** (3.6x margin), with the derivation in the test. The failure itself was not reproduced on the
A100 by this branch's runs or by `fix/readme-t1-attn-bwd`, and I cannot see the failing run: I show only that the old margin allowed it.

## 5. What did not help, and what is not done

- Heaviest-first block order in the lowering: no gain (above). `attn_bwd_delta` with 8 pairs in flight and hoisted loads: no change (167 vs 188 us), reverted.
- hd=128 backward is slower than before (0.76x): 255 registers and 1.5 KB of spills, because the resident K/V (or Q/dO) fragments of a 16-row tile are 128 registers at hd=128. It only affects
  `-DKVM_GPT_MAX_HD=128` builds; the default build compiles hd <= 64. Fixing it needs fragment streaming for hd=128.
- 3xTF32 is kept (the 2e-5 contract); a 2xTF32 path is 1.4x faster but breaks the contract, and bf16 storage belongs to the bf16 track. The interface is unchanged (fp32 buffers; a bf16 variant can add a
  dtype next to `Chk::mat`).
- Larger tasks (256 rows as two sequential halves) would save another ~13 ms of admission per step but leave 192 forward tasks on 108 SMs; not tried.
- No backward fusion, no profiler counters (not installed on the box).

## Reproduce

```bash
cd tests/gpu_ops   # on the GPU box
nvcc -O2 -arch=sm_80 -std=c++17 -lineinfo -I../../include -I. bench_attn.cu -o bench_attn && flock ~/.gpu.lock ./bench_attn 4 1024 12 12
nvcc -O2 -arch=sm_80 -std=c++17 -lineinfo -I../../include -I. test_ops.cu -o test_ops && flock ~/.gpu.lock ./test_ops
TEST_ONLY=attention compute-sanitizer --tool racecheck --error-exitcode 9 ./test_ops
REF_ATTN_ONLY=1 python3 -I ref_torch.py
KVM_HOST=... demo/run_demo.sh large
```
