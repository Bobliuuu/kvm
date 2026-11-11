# 162M GPT step and small workloads: where the executor's idle time is, and the one scheduling-side win (A100)

**Verdict.** On current main the 162M step is not scheduling-bound: workers run task bodies 88% of the time (not 10-21%), and `gpt_tiny` / `gpt_small`
run on the static path, where 86-88% of the critical chain is op-body time. The one scheduling-shaped loss was a **tail of under-sized-parallelism
tasks**: 25 `embedding_bwd` tasks of 7.6 ms (the last 8 ms of the step at 23% of workers busy) and 32 `softmax_ce_fused` tasks of 4.3 ms (on 32 of 107
workers). Splitting them (`vocab=128` keys, `ce_rows=16` rows per task, `Tiling` fields; no device code changed) takes the 162M step from
**115.05 to 107.19 ms (x1.073, best of 5 interleaved)**, with the loss, `wte`, `head`, `h0.q.w` and AdamW `m.wte` **bitwise equal** to the unsplit run.
Nothing on the device side (admission, hop, queue, signalling) is worth a 50-minute rebuild at these sizes: the measured terms are below.

| | |
|---|---|
| GPU | A100-SXM4-40GB, 108 SMs, driver 580.105.08, CUDA 12.8 (nvcc), 1410 MHz max, shared box, every run under `flock ~/.gpu.lock` |
| Executor | tree = main `49f2a1a` (device code unchanged by this branch), `nvcc -O3 -arch=sm_80 -std=c++17 -Iinclude --fmad=false -Xcompiler -ffp-contract=off -diag-suppress 177 -DKVM_LEAN -lineinfo`, sha256 `ed41554d...`; the same binary for every row |
| Bundles | `bench/vs_pytorch/gen.py --mode tf32 --tiling a100 [--tiling-set vocab=128,ce_rows=16]`: 162M (1 step; 53,939 tasks before, 54,579 after), `gpt_tiny` / `gpt_small` (50 steps) |
| Commands | `bench/sched162/` (`trace_idle.py`, `idle_by_op.py`, `tail_phases.py`, `static_chain.py`, `static_wait.py`); A/B = `kvm_exec --quiet --warm-ms 500 BUNDLE`, bundles interleaved, best of N (raw in `raw/job_a.txt`, `raw/job_b.txt`) |
| Branch | `perf/sched-162m` |

## 1. Where the time goes (traces: `kvm_exec --trace-csv`, `raw/trace_analysis.txt`)

**162M, dynamic path (mean task 214 us, 228 logical levels, 107 workers):**

| share of worker-time | before | after |
|---|---|---|
| running a task body | 88.3% | 94.9% |
| hop (pop + bookkeeping, task was already ready) | 2.0% | 2.2% |
| starved (task not yet ready: dependency) | 9.3% | 2.5% |
| admission ramp / tail | 0.1% / 0.4% | 0.1% / 0.4% |

Admission is not a term: the 162M bundle is one record admitted before the first task starts; the ready queue is never empty (mean 262-296 ready+running
tasks for 107 workers). The 392,630-task / 5.2 s / 10-21% busy shape is not this bundle (current tiling is 53.9k tasks a step; the Nov 9 w_162m_3
bundle was 276k a step): at current tiling the lever is op efficiency (matmul 6.8 of 12.2 s of worker-time), which belongs to the kernel track.
The starved 9.3% was concentrated: idle worker-time 10.8%, of which 41.5% fell while `embedding_bwd` ran (25 tasks, 8 ms, nothing else ready: it needs
the last backward and gates gradnorm and AdamW) and the 30-40 ms dip (81% / 77% busy) sat on the 32 `softmax_ce_fused` tasks. After the split
`embedding_bwd` is 393 tasks of 0.52 ms (3.0% of idle) and `softmax_ce_fused` 256 of 0.62 ms.

**gpt_tiny / gpt_small, static path** (the default for these: mean estimated task <= 6 us; 24 / 64 of 108 workers; `raw/trace_analysis.txt`):
walking the critical chain back from the last task: tiny 1,431 tasks, exec 11.25 of 13.15 ms (86%), gaps (hop + idle, timer quantum 1.02 us) 1.9 ms (14%);
small 2,352 tasks, exec 30.9 of 35.2 ms (88%), gaps 4.3 ms (12%). Matmul tiles alone are 50% (tiny) / 56% (small) of the chain. The cross-worker hop is
at the timer's resolution (median 1.02 us). Per-task exec 6.4 us (tiny) against the planner's 4.5 us estimate, so the planner's 8.7 ms makespan reads
13.1 ms; but sweeping its two knobs (`--comm-us` 0.5-4, `--workers` 12-107, 5 runs each, `raw/sweep1.txt`) moves nothing beyond noise
(tiny 15.5-16.7, small 28.4-31.2 ms, default best or tied; the box's clock state moved the same bundle between 13.1 and 16.6 ms across sessions).
Upper bound for any gap-side change on the static path: 14% / 12%, and about half of that is the trace's own `globaltimer` stamps and clock quantum.

## 2. The change, measured (same binary, interleaved, best of N)

| bundle / tiling | tasks | best ms | vs base | outputs |
|---|---|---|---|---|
| 162M base (`vocab=2048`, one 128-row block per loss task) | 53,939 | 115.05 | | |
| `vocab=256` | 54,111 | 109.61 | x1.050 | |
| `vocab=256, ce_rows=32` | 54,255 | 107.89 | x1.066 | bitwise equal |
| **`vocab=128, ce_rows=16` (kept; now the `a100` preset)** | 54,579 | **107.19** (107.98 in the second session) | **x1.073** | **bitwise equal** (wte, h0.q.w, head.w, loss, m.wte) |
| `vocab=128, ce_rows=8` | 54,835 | 108.37 | slower than 16 | dropped |
| `+ opt_elems=131072` / `65536` (AdamW tail) | 56,385 / 59,583 | 107.44 / 107.50 vs 107.98 | -0.5%, inside the spread | dropped |
| gpt_tiny `ce_rows=16` / `8` | 6,799 / 6,899 | 13.069 / 13.038 vs 13.066 (best of 9) | no change | |
| gpt_small `ce_rows=16` / `8` | 29,208 / 29,608 | 27.79 / 27.69 vs 28.42 (best of 9) | -2.2% / -2.6% | |

Per-row (loss) and per-key (embedding gradient) arithmetic does not depend on the split: each key's row is summed in ascending token order by one warp,
each loss row is one warp's. `raw/job_a.txt` holds the `cmp` lines.

## Correctness

Only `frontend/kvmfe/lower.py` (new `Tiling.ce_rows`, `a100` preset values) and `bench/` changed; `include/`, `kvm_exec.cu` and `kvm_sched*` are untouched, so the
device suites (core 8,613 checks, `-DKVM_BREAK_FENCE` negative, memcheck/racecheck on `smoke`) read as on main and were not re-run. Run here: CPU suites `tests` 157 OK and
`tests/frontend` 133 OK; bitwise comparison above; every timed run ended `OK: n/n tasks`.

## Not done, and why

* Device-side scheduling (static pre-resolved ready lists, device-side tape generation, per-SM queues, release/acquire without fences, successor prefetch): the traces
  bound each at the terms above (hop 2.2%, admission 0, starved 2.5% at 162M; 12-14% of gaps, half of it timer noise, on tiny/small), and one device-code build on this box takes
  45-50 minutes (one ptxas on the persistent kernel; `--split-compile` does not help), so a candidate costs a build, the core suite, the negative build and the sanitizers.
* The 162M step's remaining 94.9% is op bodies: `matmul_tile` 6.8 s of 12.2 s worker-time (214 us a tile), attention backward 1.7 s, AdamW 0.6 s.
* Gaps of the static path are measured with the trace on; with it off the real gap is smaller, so the 14% is an upper bound.
