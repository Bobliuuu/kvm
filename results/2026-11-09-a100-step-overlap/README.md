# Step overlap within one run (goal 5a), A100: bounded result

A100-SXM4-40GB (108 SMs, driver 580.105.08, nvcc 12.8), shared box, one `~/.gpu.lock` hold per job. Code: `kvm.cu --overlap 0|1|2` (this branch), scripts in `bench/step_overlap/`.

## What can legally overlap (first principles)
One SGD step of the 2-layer MLP: `H1 = relu(X W1)` -> `Y = H1 W2` (reduces over ALL hidden columns) -> `dH = (dY W2^T) * mask` -> `W1 -= lr X^T dH` -> next `H1`.
`H1(s+1)[:,c]` needs `W1(s+1)[:,c]`, which needs `dH(s)[:,c]`, which needs `dY(s)`, which needs every column of `H1(s)`. So the true read-after-write chain is
4 dependent GEMM phases per step (the fused program cuts it to 1 hop by slicing columns); no tile of step s+1 can start before step s's `Y` is complete. What is *not* a true dependence:
- `DW2` (W2 update) is off the chain: only `Y(s+1)` needs it, yet the tape made `DW2` wait for all of `DH1` (WAR on W2) and `H1(s+1)` wait for `DW2` (WAR on H1), `Y(s+1)` for `DW2` (WAR on dY).
- Waits were on whole phases, although a consumer reads one row or column.
Algebraic look-ahead (`Z(s+1) = X' W1 - lr X'X^T dH`) would shorten the chain but changes rounding; not built.

## Implemented (bitwise-equal)
- `--overlap 0`: original tape (aggregate waits).
- `--overlap 1`: per-row/per-column counters, same buffers (WAR stays as waits).
- `--overlap 2`: 1 + step-parity double buffers for H1, dY and W2 (W2 updated out of place, new `F_SGDSRC` flag, parity bits in `xflags`). `DW2` waits only for `Y`; `H1(s+1)` waits only for its `W1` column. Each tile's arithmetic is unchanged.
A/B isolation: same binary, same tiles, same 1000 steps; only wait edges and buffer ids differ. Correctness: final weights hash (FNV-1a) identical for levels 0/1/2 in all 6 runs (`4acfcd7dceb8d98a`) and 0 of 147456 weights differ bitwise between the VM and each of the three launch baselines at every level; loss vs CPU fp32 2.8e-5.

## Result 1: kvm.cu, 1 run, 1000 steps, best of 15 reps, 3 interleaved rounds (`ab/`)
| tiles | overlap 0 | overlap 1 | overlap 2 | 2 vs 0 | best launch baseline |
|---|---|---|---|---|---|
| default (64x64, kc32) ms/1000 steps | 42.34 | 41.58 | 39.82 | **1.063x** | 45.9 |
| `-DKTILE=32 -DKKC=64 -DKSTG=4` | 25.38 | 24.86 | 24.29 | **1.045x** | 28.2 |
(medians of the 3 rounds' best; round spread <=0.5%.) A small but consistent win, 1.05-1.06x; the launch baselines are unaffected (45.9 / 28.2).

## Result 2: the win is NOT step overlap (`tl/`, `timeline_overlap.png`, `tl_stats.json`)
From the VM's own per-instruction `%globaltimer` stamps (KTILE=32 config, steps 100-105, profiled run within 1% of unprofiled): at every level the last tile of step s ends before or at the moment the first tile of step s+1 is ready
(overlap <= 0 us at all boundaries; step span = step period = 24.6 us at levels 1 and 2, 25.6 us at level 0). Level 2 gains ~1 us/step by taking `DW2` off the chain (its 5.1 us tile ran serially before; now it runs beside `DH1`), not by pipelining steps.
Timestamps tick at 1 us (A100 globaltimer), so sub-us structure is not resolvable.

Dependency-chain floor, per step (median tile compute, us): H1 4.1 + Y 6.1 + DH1 3.1 + DW1 4.1 = 17.4 us of tile compute on the chain, +~7 us of dependency hops/launch-to-ready = ~24.6 us. The overlap-2 run is AT that floor of its own chain (it cannot go lower without shortening a tile or a hop); only the algebraic look-ahead or column-sliced fusion (`--prog fused`, 1 hop) change the chain itself. Per-column waits alone (level 1) gave only 0.7-2% because every phase has <=8 tiles on 108 SMs: tiles of a phase finish together, so there is nothing to skew.

## Result 3: executor, isolated (`exec/`)
`gpt_tiny`, 40 AdamW steps, `--precise`, same `kvm_run` binary (sha256 prefix 82aa75d625440331), same 2 admissions, 5 interleaved reps. A/B pair differs only in the task graph: `w40bar` adds a no-op barrier task per step (all step k-1 tasks -> barrier -> all step k tasks; `bench/step_overlap/mkbar.py`: +39 tasks, +8034 edges), so steps cannot overlap.
| variant | admissions | kernel_ms (5 reps) | median |
|---|---|---|---|
| w40 (steps overlap as the DAG allows) | 2 | 60.3 59.7 59.8 60.4 59.8 | 59.77 |
| w40bar (same admissions, step barrier) | 2 | 62.9 63.3 62.9 63.1 62.7 | 62.92 |
| w1 (one host admission per step; the closeout's "no overlap" arm) | 41 | 86.4 82.9 82.5 81.0 81.6 | 82.53 |
Isolated step overlap in the executor = 62.92 / 59.77 = **1.053x** (includes the barrier's own hop). The closeout's 1.41x (82.5 -> 59.8 here) is therefore ~87% host admission (82.5 -> 62.9 = 1.31x) and ~13% overlap. Final weight buffers: all 21 files byte-identical across w40, w40bar and w1.

## Verdict
5a: no cross-step pipelining exists in an SGD chain (floor above), so "overlap" cannot be more than removing false (WAR) and over-coarse waits. That is worth a measured, bitwise-equal 1.045-1.063x in `kvm.cu` (overlap 2) and 1.05x in the executor DAG; the earlier `pbt-overlap` negative (0.8-1.0x) is reproduced for per-column waits alone (level 1: 1.02x) and explained: it needs the double buffers to matter. Not tested: larger shapes with more tiles per phase, multi-run, the fused programs (already 1 hop). Executor logs list each task; `trace.csv` not committed (size).
