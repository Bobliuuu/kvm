# Scheduler tail latency: express (per-class) admission and a strict-priority class tier (A100, 2026-11-07)

- Code: branch `perf/sched-tail`, tip 5fd7682 (express admission + `KVM_CF_STRICT`, merged with `perf/admission-bottleneck` a05b926). Baseline: origin/main a47a5f5 (`raw/base-origin-main-sched-bench.log`).
- Hardware: Modal containers, NVIDIA A100 (108 SMs, sm_80), driver 610.57, CUDA 12.8, flags of `tests/gpu/build_sched.sh`, `build_adaptive.sh`, `build.sh`
  (`-O3 -arch=sm_80 -std=c++17 --fmad=false -Xcompiler -ffp-contract=off -lineinfo`). Launched with `tests/modal/run_suite.py --suite custom|exec|litmus`.
  The board differs by container (baseline and 35edaef: A100-40GB; 5fd7682 sched run: an 80 GB A100 with the same 108 SMs), so quote ratios inside one run only.
- Raw logs: `raw/`.

## What was built
1. **Express records** (`KvmBundle::express`, `KvmBundleBuilder::express()`, header word 7). The admitter copies a record in chunks of 128 tasks. Before the first chunk it claims
   all resources of the record (slots, window id, successor words) and it releases no task before the last chunk. Between chunks it admits the express records that DIRECTLY follow
   it in the ring (at most 4096 tasks each, window id must be the next one, never waits for room) and releases their roots at once. Version and litmus semantics are untouched:
   tasks of the big record become runnable only after it is fully copied, exactly as before, and an express record is a plain bundle with its own window.
2. **`KVM_CF_STRICT`** (class flag in `KvmClassParams.protect`, CLI `--strict`): in `kvm_pop` a strict class with ready work is served before every fair-share class
   (below inheritance, reservations and EDF-urgent).

## Results (ratios within one container, best-of-5 median)
### Admission head-of-line (one noop class-0 bundle right behind a K-task class-1 record; small-bundle end-to-end p50, 5fd7682 container)
| K | ordinary | express |
|---|---|---|
| 500 | 151 us | 63 us |
| 2,000 | 387 us | 75 us |
| 8,000 | 1,309 us | 106 us |
| 32,000 | 4,253 us | 234 us |
| 64,000 | 8,465 us | 397 us |

(35edaef container, before the merge: 64,000 -> 11,364 us ordinary vs 451 us express.)

### Dynamic scheduler vs the global queue with serve priority (`test_sched baselines`, serve p99, 106 workers)
| config | 35edaef container | 5fd7682 container |
|---|---|---|
| global, serve prio 20 | 1660 us | 1651 us |
| dynamic + controller (main) | 2900 us (0.57x) | 2881 us (0.57x) |
| dynamic, strict serve | 1742 us (0.95x) | 1656 us (1.00x) |
| dynamic, strict + train floor 8 | 2047 us (0.81x) | 1885 us (0.88x) |
| dynamic, strict + controller | 1778 us (0.93x) | 1597 us (1.03x) |

Train throughput of the strict rows is 0.98-1.00x of the priority queue. Problem (b) (dynamic 1.7-1.8x worse) is closed: strict serve matches the global priority queue.
Overload (serve flood, train reservation 26 CTAs): train 0.4-0.7 ktask/s on the priority queue against 19-23 on the dynamic scheduler (32-44x); serve p99 62-63 ms against 92-97 ms.
The reservation design wins on train isolation and loses about 1.5x on serve p99 under overload. Unchanged; no overload win for serve was found.

### Bandwidth arbiter (`test_sched arbiter`, 5fd7682 container)
ON with a 60 us target: chain p99 998 -> 999 us (1.00x), stream throughput kept 99.9%. ON with 25 us: 998 -> 964 us (1.04x) at 78.6% of the stream. In `baselines`,
dynamic+ctl+arbiter reaches p99 1.15x of the priority queue but keeps only 0.57x of train throughput. As in every earlier measurement it is a throttle without a latency win;
the evidence supports deleting it (not done here, see gaps).

### Correctness (5fd7682)
`test_exec` full: 8613 checks, 0 failed. Litmus: 15 properties, 7 mutants detected, 0 failures. `test_adaptive express` (500 chains of 12 dependent adds crossing chunk boundaries, 24 express
bundles, exact sums; first express task ready at 49 us, the big record's first root at 1182 us): PASS. `test_sched` cost/share/reserve/edf/inherit/rand: 499 checks, 0 failed;
starve_neg and inherit_neg detected; arbiter suite 12 checks pass.

## Not done / honest gaps
- Real Doom + big training record + serve (`adaptive_run --express`) is NOT measured: the adaptive_run build needed the Doom object and was not ready. Doom lateness before/after is unmeasured; only the synthetic numbers above exist.
- The adaptive controller: with strict serve, controller on/off is 1.03x against 1.00x on serve p99 (noise). No evidence it helps, but the real-mix comparison is missing, so it was not removed.
- The arbiter code is not deleted (about 50 references across `kvm_sched.*`, `kvm_adapt.h`, tests, CLI).
- In the sched container, `test_adaptive telemetry loop floor` printed 1 failed check (`floor` passed; the failing suite name was cut off by `tail`); it is being reproduced separately and is not yet explained.
