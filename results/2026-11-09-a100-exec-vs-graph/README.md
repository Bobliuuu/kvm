# Goal 3c: executor vs whole-step CUDA graph, A100 (2026-11-09): faster than before, still loses 1.8x / 2.4x

> **Superseded (2026-11-11).** Measured on the executor before #13/#22. Main's current read is `results/2026-11-10-a100-merged` (executor 2.2-3.8x faster than PyTorch's fastest variant on small MLPs, slower on GEMM-bound rows). Kept as the record of that executor; its verdict below is not main's.

> **Landing note.** The numbers below were measured on branch `perf/exec-vs-graph` (PR #6), whose device changes (ticket-based ready-queue claim, continuation fast path, one-fence-warp completion, admitter staging, `--no-cont`, `KVM_LEAN`) rewrite the same hot paths as the completion hand-off / staged admission of PR #1 and conflict with them in `include/kvm_exec.cuh`, `include/kvm_sched.cuh`, `tests/gpu/test_exec.cu` and `tests/litmus/build.sh`. The combined branch keeps PR #1's device code (GPU-verified there) and carries PR #6's results and `bench/exec_vs_graph/` scripts only; `nocont.sh` and the "executor (this work)" column need the PR #6 code (branch tip 7801809). The verdict (goal 3 negative for the executor) rests on the numbers in this file and on PR #7's independent measurement.


**Verdict: bounded negative.** After porting the unmerged admission/completion work and adding a continuation fast path, the executor trains
`gpt_tiny` at 758 us/step and `gpt_small` at 1567 us/step (was 1166 / 2818 in `2026-11-06-vs-pytorch`), correct against PyTorch. A whole-step CUDA graph
runs the same steps at 430 / 645 us (best of 7), `torch.compile` + graph at 202 / 334 us. The executor therefore **loses to the whole-step graph by
1.76x (tiny) and 2.43x (small)**; goal 3 does not hold for the executor on any workload measured. No worker count helps (below), so the loss is
dependency-chain latency, not contention.

| | |
|---|---|
| GPU | A100-SXM4-40GB, 108 SMs, driver 580.105.08, CUDA 12.8, shared box, one `flock ~/.gpu.lock` hold per run |
| Code | branch `perf/exec-vs-graph` (PR #2 tip c9ab1bc + cherry-picks of `perf/admission-bottleneck` 882c173 0d5ab11 92f34d1 5014d26 af266bf a05b926 and `perf/completion-path` a44c1c0 5d825ae c4e3ab5, + continuation, see git log) |
| Executor build | `nvcc -O3 -arch=sm_80 -std=c++17 -Iinclude -DKVM_LEAN --fmad=false -Xcompiler -ffp-contract=off -lineinfo kvm_exec.cu` (`KVM_LEAN` drops the comm/extra packs, 6 min instead of 25+; the GPT pack the bundles use is in) |
| Workloads | `bench/vs_pytorch/gen.py`: gpt_tiny (1 layer, d32, 32 tokens/step) and gpt_small (2 layers, d64, 128 tokens/step), 50 steps, TF32, a100 tiling |
| Commands | `bench/exec_vs_graph/run.sh` (graph, compile_graph, executor with/without continuation, traced run), `verify.sh`-equivalent steps listed under Correctness; raw in `raw/` |

## Numbers (us per optimizer step, best / median of 7)

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

- Executor loss vs PyTorch (whole-step graph losses from the same run, same weights/tokens/optimizer): step 0 relative difference 3.2e-6 (tiny), 9.7e-8 (small);
  worst over 50 steps 1.3e-4 (tiny), 7.3e-4 (small), identical with and without continuation (`raw/out3/verify/losses_*.json`); same order as the 2026-11-06 result.
- GPU suite (`tests/gpu/build_core.sh`, core ops, A100): 8613 checks, 0 failed (`raw/out3/verify/core_suite.txt`); `-DKVM_BREAK_FENCE` negative build fails as required
  (118 violations); compute-sanitizer memcheck and racecheck on `smoke`: rc 0.
- CPU suites: `tests` 153 OK, `tests/frontend` 105 OK.
- **Not run:** the full-ops `run_exec_tests.sh` (`--gpt`, `--fuzz`, litmus on the full build) and the sched/adaptive suites with the scheduler on (continuation is
  disabled there by construction). The core litmus suite inside `test_exec` passed.
