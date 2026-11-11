# main + #13/#22 as one tree: the 9-workload panel against PyTorch (A100)

**Verdict.** On main `9a76574` with #13 and #22 merged in, the executor reads **1.090x** PyTorch's fastest conventional variant per row
(geomean of 9 rows, 7 repetitions, A100-SXM4-40GB). It is faster on the four small-MLP rows (2.2-3.8x) and slower on the five GEMM-bound
rows (0.34-0.73x). #13/#22 alone read 1.095x on the same machine with the same bench. The merge costs the executor about 1%: its own
times are x0.991 of #13/#22's (geomean), with 8 of 9 rows slower by up to 2.4%, where repeated reads of one binary held within 0.5%.
PyTorch's bar on the hetero row is bimodal (137.8 us here against 121.9 us in #13/#22's read, a difference in PyTorch's own runs); with the
fast bar the headline is 1.076x. Every row passed its correctness gates.

## Panel (A100-SXM4-40GB, 400 W, driver 595.91.07; PyTorch 2.15.0+cu132; nvcc 12.8)

Time per training step, median of 7 repetitions:

| workload | PyTorch's fastest (variant) | ours | ratio | #13/#22 alone |
|---|---|---|---|---|
| MLP h72 d2, 1 run | 72.6 us (`compile_graph_whole`) | 32.6 us | **2.234** | 2.292 |
| MLP h72 d2, 8 runs | 78.8 us (`batched_compile_whole`) | 32.3 us | **2.440** | 2.452 |
| MLP h72 d2, 32 runs | 104.0 us (`batched_compile_whole`) | 46.3 us | **2.246** | 2.288 |
| MLP h48-h104 d2, 8 runs | 137.8 us (`multi_compile_graph_whole`) | 35.8 us | **3.847** | 3.419 |
| MLP h256 d4 | 125.2 us (`compile_graph_whole`) | 250.2 us | **0.501** | 0.525 |
| MLP h1024 d2 | 103.8 us (`compile_graph_whole`) | 309.5 us | **0.335** | 0.343 |
| GPT tiny | 185.1 us (`compile_graph`) | 263.8 us | **0.701** | 0.716 |
| GPT small | 417.3 us (`compile_graph`) | 571.9 us | **0.730** | 0.738 |
| GPT 162M | 63.1 ms (`compile_graph`) | 118.0 ms | **0.537** | 0.543 |
| **geomean** | | | **1.090** | **1.095** |

A ratio is PyTorch's fastest conventional variant on that row over ours (above 1, ours is faster). Each row's PyTorch and executor runs
come from one process tree on one host. "#13/#22 alone" is the same panel read on #13/#22's tree, on the same machine with the same bench
and repetitions (Nov 10); it is the column this merge must keep.

Clocks: every stage ran at a 1,410 MHz median; GPT 162M dipped to 1,260 MHz under the power cap (throttle mask `0x4`), as in #13/#22's read
(`env.json`, `logs/clocks_*.csv`).

## What ran

* **Executor:** one `kvm_exec` for every row, sha256 `43cb2f0c…7bc7c` (`env.json`), built with
  `nvcc -O3 -arch=sm_80 -std=c++17 -Iinclude --fmad=false -Xcompiler -ffp-contract=off -diag-suppress 177 -DKVM_LEAN`.
  The tree is this branch's, before its last two changes: `softmax_ce_fused` still at id 134, and `runtime/abi.py`, `runtime/ref_ops.py`,
  `tests/test_registries.py` as on main. The id move changes one case label in the executor's dispatch; the `abi.py` change is Python only.
* **Bench:** `bench/` in this folder, the bench exactly as run (bench_version 3.3). It is the repository's `bench/exec_vs_torch` and
  `bench/vs_pytorch` with four additions:
  * whole-run CUDA-graph variants beside the whole-step ones, one launch for all N steps: `compile_graph_whole`, `batched_compile_whole`,
    `graph_whole`, `multi_compile_graph_whole`;
  * a 0.5 s sustained GPU warm-up before every timed window, and inside the executor's child (`--warm-ms 500`);
  * CUDA-graph capture in `thread_local` mode, with a retry;
  * the heterogeneous row (`run_hetero.py`, `hetero_merge.py`) and the zoo generators.

  Two comments differ from the files that ran (a host name and an internal run label removed).
* **Rows** (`bench/exec_vs_torch/run_mlp.py`, `run_hetero.py`, `run_gpt.py`):
  * MLP h72 d2, 200 steps, 1, 8 and 32 independent runs; MLP h48-h104 d2 (8 widths), 200 steps; MLP h256 d4 and h1024 d2, 50 steps;
  * GPT tiny and small (`gen.py --mode tf32 --tiling a100`), and GPT 162M, 3 steps.
* **Gates:** each row's losses and final weights are checked against PyTorch eager on the same bytes (MLP: 3e-3 relative per step and 1e-3
  relative L2 per weight buffer; GPT: 2e-3 per step), and the MLP rows against float64 as well. A row that fails its gate has no ratio.

## Tests (the same tree)

| suite | GPU | result | log |
|---|---|---|---|
| CPU `tests` / `tests/frontend` / `tests/dist` / `examples/models` | none | 157 / 133 / 74 / 7 OK, on this branch's tree | |
| `test_exec` suites | RTX 3090 (sm_86) | 8,609 checks, 0 failed | `tests/test_exec_sm86.log` |
| `test_ops` | RTX 3080 Ti (sm_86) | 4,999 passed, 0 failed, 67 skipped: the cases this build cannot compile (hd=128 attention, the T=1 sweep) | `tests/test_ops_sm86.log` |
| static path `run_check.sh`; `gptdiff` (CPU and a100 tilings, 1/2/5 steps, 1xTF32 and precise), two of its cases under memcheck, racecheck and synccheck | RTX 3080 Ti (sm_86) | PASS at 1, 4, 16 and 107 workers; 8/8 PASS; clean | `tests/static_run_check_sm86.log` |
| litmus: properties, mutants, probes and e2e scenarios | RTX 3090 (sm_86) | **exit 1, from a control this GPU cannot show:** every property reads 0 violations on the production headers, every mutant is detected, every probe and e2e scenario passes. The one failure is the relaxed-CAS `mp` control, which consumer Ampere does not exhibit (an undetected control counts as a failure, `tests/litmus/litmus.cu:136-140`; the A100 row below detects it). Three e2e controls also print "not detected" in this harness (`breakchunk` on dag and bigrecord, `breakcont` on versions); #24 makes them bite | `tests/litmus_sm86.log` |
| the same, three full passes | RTX 3080 Ti (sm_86) | 3 of 3 run to their end, none hanging, each with the same verdict lines (exit 1 by the same control) | `tests/litmus_full_sm86_pass*.log` |
| `recycle` (3,000,000 tasks in 100,000 bundles, 16,666 ring wraps) | RTX 3080 Ti (sm_86) | 0 failures in 10 runs | `tests/recycle_sm86_verdicts.txt` |
| `test_ops`, every case (hd=128 included) | A100-SXM4-40GB, production build | 5,403 passed, 0 failed | `tests/test_ops_a100.log` |
| `test_e2e` | A100-SXM4-40GB, production build | 7 passed, 0 failed | `tests/test_e2e_a100.log` |
| memcheck on the attention cases | A100-SXM4-40GB, production build | clean | `tests/green_ops_a100.txt` |
| static path `run_check.sh` on the production executor | A100-SXM4-40GB | PASS (fused step against float64 6.58e-6, bound 1.33e-4) | `tests/static_run_check_a100.log` |
| litmus `mp` on the production build; `weakpub` mutant | A100-SXM4-40GB | 0 violations in 2,000,000 rounds, every control detected; the mutant shows 63 violations (caught) | `tests/litmus_mp_a100.txt`, `tests/litmus_weakpub_a100.txt` |

The sm_86 builds add `-DKVM_GPT_BIG_BK=16 -DKVM_GPT_MMA_SMEM=90112`, since those GPUs opt into 99 KB of shared memory a block where the
production tiles launch with 144 KiB.

## The GPT rows on main, same host and session

#18's rows read on main `9a76574` as it is: the same bench, host, session and build flags as the panel above (binary sha256 `7f5e0258…`,
`main-9a76574/env.json`). PyTorch's bars of the two reads agree within 0.2% (each read's fastest-variant median), so the change is the
executor's alone.

| row | main: ours (ratio) | this branch: ours (ratio) | this branch over main |
|---|---|---|---|
| GPT tiny | 641.4 us (0.289) | 263.8 us (0.701) | **x2.431** |
| GPT small | 1,549.1 us (0.269) | 571.9 us (0.730) | **x2.709** |
| GPT 162M | 135.0 ms (0.467) | 118.0 ms (0.537) | **x1.145** |

Files: `main-9a76574/` (raw JSON, `env.json`, stage logs and clock traces).

## The roadmap's Phase B row: GPT 162M bf16s (PyTorch's side only)

Measured on the same A100 (`phaseb-162m-bf16s-torch/`, two rounds of 5 repetitions, autocast bf16, the 1-step 162M tape): PyTorch eager
**51.2-51.7 ms a step** (round medians), whole-step CUDA graph 42.3 ms. The roadmap's exit gate, within 1.3x of eager, is therefore
**67.1 ms a step or less** for the executor. The executor's side did not run: its build (every op pack) was still compiling when this
session's GPU time ended.

## Not run on the A100

* `test_exec` with sanitizers on the production build: its build did not finish in time. The RTX 3090 run above (8,609 checks, 0 failed)
  covers the same suites on the same tree.
* The executor's Phase B row (above).

## Files

* `raw/`: the JSON each row's bench wrote; the table is computed from it.
* `env.json`: what ran: the binary's hash and flags, versions, clocks, and the CPU throttling of each stage.
* `logs/`: each stage's bench log, clock trace and stage record.
* `tests/`: the suite logs in the table above (one tree path in a log header and the runs' internal queue labels renamed, nothing else).
* `main-9a76574/`: the GPT rows read on main as it is (the section above).
* `phaseb-162m-bf16s-torch/`: PyTorch's side of the Phase B row.
* `bench/`: the bench as run.
