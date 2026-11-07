# Worker completion path (A100, 2026-11-06): cheaper task end, no end-to-end win yet

**Verdict.** The completion path is 17% cheaper per task (one worker: 7.0 -> 5.8 us/task, dependent chain 8.55 -> 7.56 us), all
executor tests, litmus (7 mutants detected), negative build and sanitizers pass, but the 162M GPT step does **not** move
(1042-1077 ms before and after): that step is bound by the ready-queue pop (one head CAS per ~1.5 us), not by how a task ends.
Per-edge release is already almost free. The pop fix belongs to `perf/admission-bottleneck`; this change should be measured on top of it.

| | |
|---|---|
| GPU | NVIDIA A100-SXM4-40GB, 108 SMs, sm_80, driver 580.105.08, CUDA 12.8 (nvcc 12.8.93), shared box, `flock ~/.gpu.lock` held for every timed run |
| Branch / commit | `perf/completion-path` from `origin/main` d657dde; code commit fa8ea56, bench budget on top (see `git log`) |
| Bench build | `tests/gpu/build_core.sh` (core ops only, `-O3 -arch=sm_80 --fmad=false -ffp-contract=off -lineinfo`); the full-ops `test_exec` link was OOM-killed repeatedly on the shared box (ptxas ~35 GB per build) |
| e2e build | `nvcc -O3 -arch=sm_80 -std=c++17 -Iinclude --fmad=false -Xcompiler -ffp-contract=off -lineinfo -diag-suppress 177 kvm_exec.cu` |
| Commands | `./test_exec_core bench` (3 alternating rounds per variant, `raw/bench_*.txt`); `kvm_exec --quiet w162/bundle` with `w162` from `bench/vs_pytorch/gen.py w162 --model gpt_162m --steps 1 --mode tf32 --tiling a100` (318,139 tasks, 6.89M edges = 21.7/task) |

## 1. Where a task's time goes (from the executor's own trace and micro cases)

Traced 1-step gpt_162m run (last 262,144 tasks, `raw/trace_timeline.py`): in the steady state the ready queue holds 1,400 tasks that drain at
167 starts per 250 us **with only ~20 of 107 workers busy**; tasks start 1.5 us apart however many workers are idle, and the mean ready->start
wait is 4.5 ms (median 0.82 ms). Same slope in the micro case "fan-out 1 -> 8000": 1.42 us/task at 108 blocks, 1.47 at 32. Cause: every idle
worker scans the bucket heads and CASes the same head, so exactly one wins per L2 round trip. This is the pop (`kvm_claim`), owned by
the admission branch.

Worker cycle with one worker (blocks=2, no contention), `independent noop`: 7.0 us = about 10 dependent L2 round trips (~0.65 us each):
pop 4 (head, cell, CAS, slot), descriptor 1, `exec_count` atomic 1 (before the op), fence on all 8 warps, window credit 1, free-slot ticket 1 +
cell check 1, done count 1. Per-edge cost is negligible: `layered, 20 edges/task` runs at the same or lower us/task than 1 edge/task
(6.36 vs 6.89 us), the 21 decrements of a task issue from one warp in parallel and cost one round trip in total.

## 2. What changed (include/kvm_exec.cuh, `kvm_run_task`)

| change | removes |
|---|---|
| `exec_count` atomic result consumed after the op (DOUBLE_EXEC is still reported, now after the op) | 1 round trip before every op |
| `__threadfence()` only in threads that release successors (was all 256) | 8-warp fence on every task, also edge-less ones |
| window credit (thread 0) and slot recycle (lane of another warp) issued together | 1 serial round trip |
| `done` count as posted `red.release` (nobody reads the old value) | 1 round trip; release still orders recycle and credit before it |
| first successor id loaded before the op, consumed after | 1 round trip on the dependent hop |

Ordering is unchanged in kind: the op's stores precede the successors' `atom.acq_rel` decrements through the block barrier plus the
releasing thread's own fence; `sched_done` still precedes the slot recycle; the version-publication and Dekker fences are untouched.

## 3. Numbers (one process per variant, 3 rounds alternating, spread in the raw files)

| case (1 worker, us/task) | main | + completion | spread |
|---|---|---|---|
| independent noop | 6.34-6.99 | **5.80-5.85** | +-0.03 |
| dependent chain (hop) | 8.54-9.36 | **7.56-7.73** | +-0.1 |
| fan-out 1 -> 8000 | 6.20-6.92 | **5.21-5.25** | +-0.02 |
| layered 1024x15, 20 edges/task | 7.05-7.87 | **6.29-6.40** | +-0.06 |

With 32 and 108 blocks the micro cases are pop-bound and unchanged (1.42-1.48 us/task; chain 6.3-6.4 us at 32 blocks).
**End to end** (gpt_162m, 1 step, `raw/e2e_gpt162_1step.txt`): main 1060.5 / 1062.7 / 1065.0 / 1076.8 ms, with change 1055.4 / 1064.8 / 1086.5 / 1091.4 (+1 traced
run each: 1056.9 / 1081.7). No difference; the traced wait profile is identical (median 0.82 ms, p99 25 ms).

## 4. Tests

core build (`build_core.sh`, all executor suites): 8586 checks, 0 failed (`raw/core_suite_negative_sanitizers.txt`); `-DKVM_BREAK_FENCE` negative build fails the
versions suite as required (1689 violations); compute-sanitizer memcheck / racecheck / synccheck on `smoke`: 0 errors; litmus
(`raw/litmus.txt`, built on the core headers): 15 properties/scenarios/probes pass, 7 mutants/controls detected, `LITMUS PASS`. New bench cases (fan-out,
fan-in, layered DAG) and a perf budget (`one-worker task cycle < 6.2 us`, A100 only; original 6.3-7.0, now 5.85) fail on `origin/main`.
**Not run:** the full-ops (`gpt`/`extra`/`comm`/`game` packs) `run_exec_tests.sh`, `--gpt`, `--fuzz`: the link needs ~35 GB and was OOM-killed three times by the
other jobs on the box. They do not touch this code path except through the same `kvm_run_task`.

## 5. Next (not done)

1. Rebase on `perf/admission-bottleneck` once its claim is green (at 92f34d1 its `bench` suite dies with `misaligned address` in my build) and re-measure: after the pop is
   herd-free the worker cycle (5.8 us against 30-60 us tiles = 10-20%) and the enqueue (tail ticket + cell check + release store + their `avail` add) become the next terms.
2. Overlap the next pop with the current task's bookkeeping (a second warp claims while thread 0 finishes): about 2 us/task, but a worker would hold two tasks briefly,
   which the header's invariants forbid today.

## Proposed status text (not applied)

Implemented + measured: "Completion path: one worker's task cycle 7.0 -> 5.8 us on A100 (`2026-11-06-a100-completion`); per-edge release costs ~0; no change to the
162M step (1.06 s) because the ready-queue pop, not completion, is the bound."
