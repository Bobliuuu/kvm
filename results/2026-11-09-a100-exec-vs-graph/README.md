# Goal 3c: executor vs whole-step CUDA graph, A100 (2026-11-09): faster than before, still loses 1.8x / 2.4x

> **Reconciliation (PR #9 on the landing branch).** The numbers in the first table were measured on `perf/exec-vs-graph` (PR #6, tip 7801809). The landing branch (PRs #1-#7) had dropped
> #6's device code because it rewrites the same hot paths as #1's completion hand-off. It is now ported back piece by piece onto #1's code, each piece measured on its own
> (table below); only the pieces that win were kept, and the landing branch's racecheck fixes (every warp fences its own stores before the op's barrier) stay in. The
> reconciled executor reproduces #6's numbers (761 / 1570 us per step against 760 / 1565 for the #6 build, same box, interleaved) and the verdict is unchanged.

## Reconciliation: each piece measured alone (executor us per step, best / median of 7, interleaved runs on one box)

`gpt_tiny` / `gpt_small`, A100-SXM4-40GB, `-DKVM_LEAN` builds unless noted, driver `bench/exec_vs_graph/ab.py` (round r runs every variant once, order rotated). Raw: `raw/reconcile/`.

| Step | tiny | small | Kept? |
|---|---|---|---|
| landing (PR #8 device code: #1 hand-off, head CAS) | 822.6 / 830.8 | 1785.9 / 1788.6 | baseline |
| 1. `-DKVM_LEAN` vs the full build (`raw/reconcile/p6.txt`, same code) | 818.9 / 827.3 vs 819.9 / 830.2 | 1782.3 / 1786.6 vs 1785.8 / 1787.2 | yes: no runtime effect, build 5-6 min instead of 25+ |
| 2a. continuation, first port (worker pops nothing, but still pays the previous task's bookkeeping before the barrier) | 827.4 / 836.1 (landing 818.6; same binary `--no-cont` 838.9) | 1788.9 / 1793.8 (landing 1784.0; `--no-cont` 1810.6) | no: +1.2% over `--no-cont`, but not better than landing |
| 2b. same, queue-empty test moved from task start to op end | 833.2 / 841.1 (continuation fires on ~47% of tasks) | 1787.9 / 1792.3 | no change |
| 2c. continuation + the previous task's `kvm_post_task` paid by an idle thread during the release phase | 775.4 / 783.5 (landing 819.9) | 1724.2 / 1729.8 (landing 1782.8) | **yes**: -5.4% / -3.3% |
| 3. ticket claim (reserve + fetch-add) on top of 2c, scheduler off | 748.1 / 767.9 (vs 776.7 / 785.9 without) | 1567.2 / 1569.8 (vs 1728.4 / 1731.4) | **yes**: -3.7% / -9.3%; without continuation 820.3 vs 831.0 and 1649.2 vs 1814.8 |
| reconciled (LEAN + 2c + 3), final tree | **760.6 / 768.0** | **1570.4 / 1571.4** | |
| same, `--no-cont` | 836.9 / 846.0 | 1646.9 / 1652.1 | |
| PR #6 tip 7801809 (reference build, same run) | 760.2 / 769.6 | 1565.2 / 1568.1 | |

Why 2a did not win and 2c does: in the landing code the pop's claim is cheap and its completion bookkeeping is hidden behind it (`kvm_post_task` runs on thread 32 during
the pop). A continuation skips the pop, so that bookkeeping (window credit, slot recycle, done count: three dependent L2 round trips) landed in front of the barrier
that starts the successor and cost what the pop used to; moving it to an otherwise idle thread of the release phase (`pay` in `kvm_run_task`) makes the skipped pop
a real saving. The ticket claim wins even though it adds round trips per pop, because with 100+ polling workers the head CAS retries (one winner per round trip); it is
used only with the scheduler off. With the scheduler on, the claim has to take exactly the scanned cell (the policy block counts ready/running/BW/deadline tasks from that
cell's `res` and deadline before the claim), so the CAS claim stays there. Continuation is likewise off with the scheduler on.

Not ported, and why: #6's admitter staging and host-side stream prep. #1 has its own staged admission, and the reconciled executor is within 0.1-0.3% of the #6 build on both
workloads, so nothing measurable is left for it on these sizes (admission is off the dependency chain here, see below). #6's one-fence-warp completion is not ported
either: it drops the per-warp fence that fixed the rare `versions` failure.

The graph baselines were re-measured the same day on the same box: whole-step graph 430.4 / 430.6 (tiny) and 645.1 / 829.7 (small), `torch.compile` + graph 201.7 and 425.2
(`raw/reconcile/torch_now_*.json`). Executor / graph: 1.77x (tiny) and 2.43x (small, best) or 1.89x (small, median; the graph flips between two clock states).

**Verdict: bounded negative.** The executor trains `gpt_tiny` at 761 us/step and `gpt_small` at 1570 us/step on the landing code (1166 / 2818 in `2026-11-06-vs-pytorch`, 822 / 1786
before this reconciliation), correct against PyTorch. A whole-step CUDA graph runs the same steps at 430 / 645 us (best of 7), `torch.compile` + graph at 202 / 425. The executor
**loses to the whole-step graph by 1.77x (tiny) and 2.43x (small)**; goal 3 does not hold for the executor on any workload measured. No worker count helps (below), so the loss is
dependency-chain latency, not contention.

| | |
|---|---|
| GPU | A100-SXM4-40GB, 108 SMs, driver 580.105.08, CUDA 12.8, shared box, one `flock ~/.gpu.lock` hold per run |
| Code (first table) | branch `perf/exec-vs-graph` (PR #6; PR #2 tip c9ab1bc + cherry-picks of `perf/admission-bottleneck` 882c173 0d5ab11 92f34d1 5014d26 af266bf a05b926 and `perf/completion-path` a44c1c0 5d825ae c4e3ab5, + continuation, see git log) |
| Executor build | `nvcc -O3 -arch=sm_80 -std=c++17 -Iinclude -DKVM_LEAN --fmad=false -Xcompiler -ffp-contract=off -lineinfo kvm_exec.cu` (`KVM_LEAN` drops the comm/extra packs, 6 min instead of 25+; the GPT pack the bundles use is in) |
| Workloads | `bench/vs_pytorch/gen.py`: gpt_tiny (1 layer, d32, 32 tokens/step) and gpt_small (2 layers, d64, 128 tokens/step), 50 steps, TF32, a100 tiling |
| Commands | `bench/exec_vs_graph/run.sh` (graph, compile_graph, executor with/without continuation, traced run), `verify.sh`-equivalent steps listed under Correctness; raw in `raw/` |

## Numbers of PR #6's own build (us per optimizer step, best / median of 7; reproduced above on the landing code)

| | whole-step graph | compile + graph | executor (this work) | executor, no continuation | executor 2026-11-06 | executor vs graph |
|---|---|---|---|---|---|---|
| gpt_tiny | 430 / 430 | 202 / 202 | **758 / 760** | 831 / 833 | 1166 / 1255 | 1.76x slower |
| gpt_small | 645 / 746 | 334 / 407 | **1567 / 1570** | 1622 / 1628 | 2818 / 2828 | 2.43x slower |

(The graph's best-of-7 flips between a fast and a slow clock state, 334 vs 429 us earlier in the day; medians are the stable figure: tiny 430, small 746,
executor/graph on medians 1.77x and 2.10x.) Of the improvement over 2026-11-06, the ported admission + completion work gives the bulk (1166 -> 831, 1.40x
tiny; 2818 -> 1622, 1.74x small) and continuation another 9% (tiny) and 3% (small).

## Where the time goes (traces in `raw/out3/*/trace_summary_*.txt`, tiny, 28 logical levels, 103 tasks/step)

Step span 712 us = about 25 us per dependent level; per task, medians: release (task end to successor ready, includes fence + indegree atomic + enqueue)
about 4 us, ready-to-start 6 us (4-8 us through the queue, 2-4 us for a continuation), run 11 us. Mean task 10.5 us against 3-5 us for the same op as a
graph kernel; utilisation 1.4% of worker time (107 workers). The three terms are about equal: pop ~6, release ~4, and an op that needs ~11 us because even
its trivial reductions (`gradnorm_partial` 5.5 us, `colsum` 6.5 us) pay a chain of dependent L2/HBM round trips (args, buffer table, data, store, ~0.65 us each)
inside the 256-thread block, plus descriptor/trace bookkeeping. A graph edge costs about 1.3 us and its kernels run 3-5 us, i.e. ~15 us per level against ~25.
The admission FIFO (1.9 us/task) is not on the path any more at these sizes: 103 tasks/step x 1.9 us = 0.2 ms is hidden behind the chain after the
ported batched admission (span minus admission ramp is flat from 8 to 107 workers).

**Worker count does not matter** (`--blocks` 8/16/32/64/default, tiny 0.79/0.77/0.70/0.69/0.76 ms/step; small 1.85/1.62/1.58/1.57/1.57): fewer than 100 workers
already cover the 3-8 ready tasks per level. The executor is latency-bound along the dependency chain, which is exactly where a graph edge (1.3 us) beats a
global-memory hop (about 5-7 us, 4 us on a continuation).

**Fusion did not help**: `gen.py --fuse` gives 5100 tasks vs 5150 (tiny) and 13798 vs 13898 (small); the fusion pass only fuses ops that these models barely
contain, so the level count stays 28 / 46.

## What was changed

1. Ported `perf/admission-bottleneck` (ticket-based ready-queue claim instead of head CAS, 16-byte descriptor staging, batched argument loads, host-side stream prep)
   and `perf/completion-path` (cheaper task completion) onto the closeout tip; the cherry-picks applied without conflicts.
2. **Continuation** (`include/kvm_exec.cuh`, `--no-cont` to disable): the worker that drops a successor's indegree to 0 runs one such successor itself without
   enqueue and pop, when (a) the scheduler is off, (b) the successor is in the same priority bucket, (c) its version wait is met, (d) the ready queue was empty
   when the task started. Trace flag `KVM_TRACE_CONT` (16). Rule (d) came from a failing test: without it a chain starved queued low-priority tasks
   (`sched` suite, aging check, 1 failure); with it the suite passes.
3. `-DKVM_LEAN` build flag (skip comm/extra packs) for fast iteration, `bench/exec_vs_graph/`.

Not done (and why it would not change the verdict): pre-resolving static dependencies and windowing admission remove the 4 us release and 1.9 us/task admission terms,
but leave the pop (2-6 us) and the 11 us op bodies; even at zero for all three the executor would still need 28 x (op ~5 us + hop ~1.3) only to tie the graph.
A persistent-kernel design that beats a graph on this chain would need per-level latency under 15 us, i.e. op bodies as fast as standalone kernels and a
hop at graph-edge cost, which global-memory dependency counters do not give. Where the VM can still win (many independent runs per launch, in-kernel data-dependent control)
is the `kvm.cu` and coopt-sh results, not a single dependent step.

## Correctness

Gates of the reconciled device code (commit f3c3e94; the device code is identical since b1bcbf4), A100; raw in `raw/reconcile/`:

- Executor loss vs PyTorch (whole-step graph losses of the same run, same weights/tokens/optimizer): step 0 relative difference 3.2e-6 (tiny), 9.7e-8 (small); worst over 50 steps
  1.3e-4 (tiny), 7.3e-4 (small), the same numbers as on PR #6's build (`losses_*.json`; the losses are from the default run, continuation on).
- `run_exec_tests.sh` (full build, every suite): 8609 checks, 0 failed (`exec_core_suite.txt`). The core build ran five times in a row: 8609 / 0 each time.
- `-DKVM_BREAK_FENCE` negative build (`run_exec_tests.sh --negative`): fails as required, 118 violations (`exec_negative.txt`).
- compute-sanitizer memcheck and racecheck on the core build's `smoke`: 0 errors, 0 hazards. The gpt_tiny bundle (544 tasks) gives byte-identical buffers in six scheduler modes
  and is clean under memcheck, racecheck and synccheck (`gpt_bundle.txt`).
- Scheduler ON (it takes the CAS claim, continuation is off): `run_sched_tests.sh` 499 checks, 0 failed; both negatives (`starve_neg`, `inherit_neg`) are detected (`sched_suites.txt`).
- CPU suites: unchanged by this PR (no host-side Python touched).
- Two pre-existing flakes surfaced and are in `gates_core_build_and_flakes.txt`: `front` (a test race on shared counters, 1/60 on landing, 31/60 once dispatch was faster; **fixed in the
  test**, 0/100 after) and `sched` aging (admission publish order: 13/300 on landing, 0/300 here, one failure in one full run here; **not fixed**, rerun passes).
- The litmus wrapper `tests/litmus/build.sh` died with `nv: command not found` (function defined after its first use); fixed. Its mutant `popplain` was rewritten to mutate the ticket claim's
  fetch-add; the full wrapper run (`run_exec_tests.sh --litmus`: production properties 0 violations, mutants must fail) was **not completed** (its builds exceeded the time available on the shared box). The litmus suite inside `test_exec` (queue and park/publish litmus on the production headers) passes in every run above.
