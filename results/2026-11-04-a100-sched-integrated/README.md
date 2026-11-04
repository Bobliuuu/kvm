# feat/sched merged onto feat/integration (slot recycling), review fixes, real Doom under the scheduler, A100 (2026-11-04)

- GPU: NVIDIA A100-SXM4-40GB, 108 SMs, sm_80, driver 580.105.08 (CUDA 13.0), nvcc 12.8.93. Shared box, every run under the GPU lock
  (`run_all.sh` takes it once and runs the stages with `KVM_NO_LOCK=1`). Commit shown in each file's first line (`+dirty` = only test/runner edits after the device code).
- Build flags: `tests/gpu/build.sh` / `build_sched.sh` (`-O3 -arch=sm_80 -std=c++17 --fmad=false -Xcompiler -ffp-contract=off -lineinfo`, GPT pack present, game pack only for Doom with `-rdc=true -DKVM_WITH_GAME`).
- Entry points: `run_exec_tests.sh [--negative|--negative-park|--sanitize|--litmus|--fuzz|--gpt|--bench]`, `run_sched_tests.sh [--negative|--sanitize|--bench]`, `doom/tests/run_sched.sh`, `run_all.sh <group>`.
- IMPORTANT for comparing with the earlier sched results (`../2026-11-03-a100-sched`): the merged kernel carries the GPT pack and 48 KiB of dynamic shared memory, so only 1 block fits an SM:
  the default grid is 108 blocks (was 324). "Whole GPU" runs below therefore have a third of the workers and about half the streaming bandwidth (478 against 1017 GB/s);
  absolute numbers are not comparable with the 2026-11-03 file, ratios are.

## Merge
`git merge feat/integration` into feat/sched: 16 conflict hunks in `include/kvm_exec.cuh` and `kvm_exec.cu`, resolved as follows.
- Class-aware ready rings are fed by recycled slots: `kvm_enqueue` takes the slot, the class is `workload_id`; per-slot side arrays (`info`, cost, preference) are re-initialised at admission, and
  `kvm_sched_done` runs BEFORE `kvm_free_slot` (after the slot is recycled its trace/info belong to the next task).
- Parking: `kvm_release_task` pushes onto the list and calls B's `kvm_scan_parked`; the SCHED hooks are `kvm_sched_park` before the push (waiter counted before it is reachable) and
  `kvm_sched_unpark` at the single CAS winner inside `kvm_scan_parked` (the node is owned by whoever detached it, so the count cannot be taken twice). Priority inheritance is unchanged.
- `registry_ok()`: B's sparse-id version taken. The cost table is keyed by name, so GPT/game ops use the default (MIXED, 0 bytes) and never conflict with the `.def` files.
- Hygiene fix found on the way: `tests/gpu/kvm_ops_test.def|.cuh` (needed by B's `build.sh`) were ignored by `.gitignore` (`kvm_*`) and never committed; they are force-added here.

## Review findings fixed (R's REVIEW.md, verified against the merged code; each has a test that failed before)
| ID | Fix | Evidence |
|----|-----|----------|
| C1 | `timeout` is a stall detector: pending work and no task finished or admitted for `timeout` (the admitter waiting for room also counts); idle never times out; checked at most once per ms | probe_timeout (idle 3.5 s, timeout 2 s): ok; probe_timeout_busy (40 x 100 ms tasks, timeout 2 s): 40/40 done |
| C2 | Parked tasks have their own budget (`live_cap`), not counted in the runnable window; ready rings are `4 * live_cap` deep so they cannot fill (bound in the header comment); more than `live_cap` parked is the explicit `PARK_STALL` | probe_deadlock (8 waiters then the publisher bundle, live_cap 8): 9/9 done; test_exec `cross-bundle wait`; test_exec stall test now ends in PARK_STALL |
| C3 | 64-bit range checks on host (`validate`) and device (`kvm_admit_record`) | fuzz targeted T01/T03/T04: rejected, no crash |
| C4 | JSON depth limit, `need()` for required keys, bounds check of `reads[].task` | T06-T12 rejected with a message; random corpus 0 crashes |
| C5 | `spin` is bounded (10 s, else BOUNDS); `task_limit` (default 30 s) reports a task that ran longer as `TASK_LIMIT` (after the fact: tasks are not interruptible) | probe_spin: spin(2e9) rejected in 0.00 s; 1.5 s task with a 0.5 s limit -> TASK_LIMIT |
| C6 | CLI numbers parsed strictly with ranges; `KvmExec` rejects negative `aging_us` and out-of-range `timeout_us` | T15-T17, T19, T20 -> usage errors; probe_aging: -5 and -2e6 rejected, 0/1e6 give priority order 11..0 |
| C7 | `submit()` returns `DEVICE_ERROR` once the device has failed | test_exec errors/admission |
| C8 | `validate()` rejects a version-wait cycle inside one bundle (publisher -> waiter edges added to Kahn) and a task waiting for a version it publishes itself | test_exec; cross-bundle waits stay legal (undecidable here: T14 wait_ver=INT_MAX is still found by the stall timeout after 3 s, reported SLOW_REJECT) |
| S1 | no `8 * max_tasks` product exists in the merged code; `--max-tasks` range-checked | T19 |
| S3 | the admitter's free-ring wait exits on a device error | by inspection + all error tests |

## Results (counts)
| Suite | Result |
|---|---|
| B's `run_exec_tests.sh` functional (ops, types, init, errors, dag incl. 267k tasks, sched, admission, versions, recycle incl. 2M-task DAG and 3M-task long admission, litmus) | **8582 checks, 0 failed** (exec-functional.txt; was 8578 + 4 new: PARK_STALL, cross-bundle wait) |
| `--negative` (KVM_BREAK_FENCE) | 7497 checks, 118 failed: the harness detects it, `NEGATIVE TEST PASS` |
| `--negative-park` (KVM_BREAK_PARK_RECHECK) | litmus suite fails (10 checks), `NEGATIVE TEST PASS` |
| `--sanitize` (memcheck, racecheck, synccheck, smoke) | 0 errors, 0 hazards |
| R's litmus harness (`--litmus`) on the merged headers | **15 properties/scenarios/probes passed, 7 mutants/controls detected, 0 failures** (litmus.txt) |
| fuzz, random corpus | 300 cases: 58 OK, 231 REJECT, 10 USAGE, 1 SLOW_REJECT, **0 crashes, 0 hangs**, 0 sanity failures |
| fuzz, targeted T01-T22 | **findings: []** |
| GPT bundle (544 tasks, frontend `gpt_tiny` 2 steps) | 544/544 in every mode (scheduler off x2, global, static, dynamic, dynamic+arbiter+affinity); the final buffers are byte-identical across all six (md5 `1735f893ac10`); compute-sanitizer 0 errors, 0 hazards |
| `run_sched_tests.sh` (cost, share, reserve, edf, inherit, rand) | **497 checks, 0 failed**; `--negative`: starvation and deadlock both detected; `--sanitize`: 0 errors/hazards |

Litmus detail. Required-detect mutants: exchbefore (52,005 and 161,136 violations), norecheck (207,484 and 81,455), weakpub/mp (77), popplain/queue (the broken claim livelocks the queue: killed by the 120 s timeout, counted as detected),
`KVM_BREAK_FENCE` e2e control (751 stale reads). Documented as NOT observable on the A100 (R's rows 1 and 4, S4; reported, not required to fail): nodekkerfence (0/500k, 0/125k), nofence/mp (0/1.6M), qrelax (0/1M).
Production headers: sb 2.5M rounds, dekker1 500k, dekker3 250k, mp 2M, queue 1.0M pops: 0 violations. e2e dag/versions/ring/bigrecord/closerace: 0 failures, 0 findings.

## Real Doom beside a saturating bursty compute load (doom-sched.txt; single runs, not best-of-N)
Doom = workload 0 (4200 tasks: `doom_tick` + 4 `doom_present` + `doom_status` per frame, `deadline_us = (f+1)/35 s`), compute = workload 1 (300 chains x 40 rounds of
matmul 256^3 -> spin of 5 / 40 / 200 ms, seeded, 24,000 tasks, 1012.7 CTA-seconds). ALL priorities are 0 in every scenario. 700 frames; **every frame was compared with the CPU oracle
(raw bytes, fingerprint, game state): 700/700 identical in all 12 scenarios**, and 300/300 matmul outputs equal A@B in all loaded scenarios.

| Scenario | fps | tick gap p50 / p99 / max (ms) | gaps over 28.57 ms (35 Hz) | compute finished after |
|---|---|---|---|---|
| alone, scheduler off (108 workers) | 184.4 | 5.41 / 9.62 / 10.2 | 0 | - |
| alone, 1 worker (every task on one SM) | 182.2 | 5.36 / 9.65 / 10.3 | 0 | - |
| (i) one global queue, no priorities (original executor) | 56.1 | 5.90 / **174.2** / 199.9 | **59** | 12.78 s |
| (i') same queue through the scheduler (`--sched global`) | 55.7 | 5.87 / 181.6 / 216.1 | 59 | 12.87 s |
| (ii) static split, Doom 10% of the SMs | 182.4 | 5.37 / 9.68 / 10.3 | 0 | 10.88 s |
| (ii) static split, Doom 25% of the SMs | 182.8 | 5.34 / 9.65 / 10.3 | 0 | 12.89 s |
| (iii) dynamic: reservation 1 CTA + `urgent_us` 20 ms for Doom, nothing else | 177.9 | 5.49 / 9.95 / 10.5 | 0 | **10.08 s** |
| (iii') dynamic with NO reservation and NO deadline boost (weights 1:1 only) | 177.7 | 5.51 / 9.88 / 10.4 | 0 | 10.07 s |

- The 35 Hz deadline holds under the dynamic scheduler without any hand-set priority: 0 of 698 tick gaps exceed 28.57 ms (p99 9.95 ms, a tick is 5.5 ms of serial work); the unscheduled
  queue misses 59 and has a 174 ms p99 (the earlier measurement on the plain priority queue, 37.9 fps / 205 ms p99 at equal priority, reproduces in kind).
  Ticks flagged against their ABSOLUTE deadline: 7 for the scheduled runs (the 170-330 ms boot tick), 386-390 for the global queue.
- What fair sharing does here, by itself: (iii') is as good as (iii). The likely reason (inferred from the numbers, not traced): Doom uses well under a CTA of the machine, so its class keeps the smallest virtual time and is served first at
  every task boundary; the reservation and the deadline tier add nothing in this scenario (they are the guarantee when Doom would NOT be the light class).
- Against the static split: Doom's latency is the same (it needs one CTA, a 10% split gives it 9 SMs), but the split strands the CTAs it does not use: the compute finishes in 10.88 s
  at 10% and 12.89 s at 25% against 10.08 s dynamic (7% and 28% longer), block utilization 87% / 74% against 94%. Dynamic costs Doom 2.5% of its fps (177.9 against 182.4) in dispatch overhead.
  The static 10% result is the favourable case for the baseline; it works because the load is one stream and Doom's footprint is known in advance, which is exactly what a static split has to guess.

### Task affinity on the real chain (alone, 700 frames)
| | tick gap p50 | fps |
|---|---|---|
| scheduler off, 108 workers (ticks land on 105 different SMs) | 5.41 ms | 184.4 |
| one worker (always the same SM) | 5.36 ms | 182.2 |
| dynamic, no affinity | 5.44 ms | 179.2 |
| dynamic, chain affinity (successor offered to the releasing block for 2 ms) | 5.73 ms | 171.6 |
| dynamic, op affinity (offered to the block that last ran the same op for 2 ms) | 5.52 ms | 176.8 |

**Affinity does not help, and the effect it was built for did not reproduce.** In this tree the tick takes 5.4 ms whether it migrates over 105 SMs or stays on one; the 12.4 ms spread-over-SMs
figure from the earlier Doom run was measured with 215 workers (2 blocks per SM, an older kernel) and is not seen with the 108-block grid. Whatever caused it (the extra resident block per SM, or the
fatter kernel's code footprint) is gone, so there is nothing for the hint to recover; the hint costs 0.1-0.3 ms per tick (the successor waits for its preferred block). Under load (iii) with op affinity: 5.49 ms,
unchanged. The synthetic L1-resident chain (bench.txt `affinity`) is also slower with affinity (hop p50 41.0 -> 45.1 us) although the successor lands on the same block 0% -> 100% of the time.
The hint is implemented, tested and off by default; it is not recommended on this evidence.

## Scheduler benchmarks on the merged tree (bench.txt; 108 blocks, best-of-5 where stated)
- Dispatch (32 workers, 200 us tasks): original 95.3% occupancy / idle gap p50 8.2 us; GLOBAL 95.0% / 9.2; DYNAMIC 1 class 94.0% / 11.3; 4 classes 93.5% / 12.3. The dynamic scheduler adds 3-4 us per task on top of the
  recycling executor's own 8.2 us (it was 3 us vs 7 us before the merge). Not reduced in this round: the two coalesced class-state loads, the acquire scan and about 8 accounting atomics per task are all on the critical path.
  For 20 us tasks occupancy is dispatch-bound in every mode (43.8% original, 40.3% dynamic).
- Baselines under the scripted bursty serve/train load (serve p99 us / train ktask/s / utilization): global flat 21,304 / 104.8 / 97.8%; global with serve priority 20 **1,236** / 104.1 / 97.7%; static 25% 194,903 / 101.1 / 91.8%;
  static 50% 2,860 / 70.0 / 69.0%; dynamic + controller 2,239 / 103.6 / 97.1%; dynamic + controller + arbiter 1,361 / 83.2 / 80.4%.
  Dynamic against: flat queue 9.5x lower p99 at 0.99x train; static 25% 87x lower p99, 1.02x train; static 50% 1.28x lower p99, 1.48x train throughput, 1.41x utilization.
  **Against a global queue that gives serve a higher priority, dynamic sharing is 1.8x WORSE on serve p99 (2,239 against 1,236 us) at equal throughput**, as before the merge.
- Overload (serve flood 9000 req/s): train 1.8 ktask/s on the priority queue, 14.6 on the dynamic scheduler with a 26-CTA train reservation (8.2x); serve p99 65 ms against 44 ms.
- Bandwidth arbiter: chain request p99 655 us (OFF) against 648 (ON, 60 us target, 99.7% of the stream kept) and 606 (ON, 25 us target, 76.7% kept). The static cap sweep (cap 96: p99 551 us for 94% of the stream; cap 32: 495 us for 35%)
  shows the same trade as before: a throttle, not a free lunch, and a small one at 108 workers because the stream no longer saturates HBM (478 GB/s).
- Controller (rebalance): reservation reaches 90% of its settled 53 CTAs 3.2 ms after the burst; stable in both bursts (0 changes in the second half); train keeps 98% of the workers when serve idles.

## Not done / unproven
- The Doom comparison is one run per scenario (no spread); frames are verified byte-identical, the timing numbers are not best-of-N.
- Dynamic-mode dispatch gap (3-4 us per task) was not reduced. Affinity is not beneficial here (above).
- C8 is only decidable inside one bundle; cross-bundle unsatisfiable waits are found by the stall timeout, not statically. `task_limit` reports a long task after it ended; nothing interrupts it.
- The litmus mutants nodekkerfence, nofence and qrelax stay undetected on the A100 (as R documented): those orderings rest on the PTX model.
- The earlier 2026-11-03 sched numbers (grid 324) were not re-run at that grid in the merged tree.
