# Dynamic co-scheduling in the GPU executor, A100 (2026-11-03)

- GPU: NVIDIA A100-SXM4-40GB, 108 SMs, sm_80, driver 580.105.08 (CUDA 13.0), nvcc/runtime 12.8.93, SM clock max 1410 MHz. Shared box: every run under `flock ~/.gpu.lock`.
- Branch `feat/sched` (based on `feat/executor` 083513e). Device code is commit a001893; later commits touch tests and the runner only. Each file's own header line
  names the commit it ran at (`+dirty` = tests/runner edits not yet committed; the device sources were identical to the commit).
- Build: `nvcc -O3 -arch=sm_80 -std=c++17 -Iinclude --fmad=false -Xcompiler -ffp-contract=off -lineinfo -DKVM_EXEC_NO_MAIN tests/gpu/test_sched.cu` (and `test_exec.cu`).
- Commands: `KVM_HOST=... ./run_sched_tests.sh` / `--negative` / `--sanitize` / `--exec` / `--bench [suite...]`.
- Files: functional.txt, negative.txt, sanitizer.txt, exec-suites.txt, bench.txt (calib, dispatch, arbiter, rebalance, baselines, affinity), overload.txt.

## What was built (all in `include/kvm_sched.h|cuh`, `include/kvm_ops_cost.inc`, hunks marked SCHED in `kvm_exec.cuh`)

The scheduler sees `workload_id` (the class), `priority`, `deadline_us` and the per-op cost table; nothing else.

| Part | Mechanism |
|---|---|
| Sharing | per-class queues; tier 0 inherited boost, tier 1 below reservation (`min_cta`), tier 2 deadline (EDF over bucket heads, `urgent_us`), tier 3 weighted virtual time (charged at claim, corrected at end, catch-up on activation = work conserving, no banked credit). Caps `max_cta`, `bw_cta`; aging inside a class unchanged |
| Runtime updates | host writes `KvmSchedParams` under a seqlock into the mapped control block; the policy block (block 1) mirrors it to device memory within a tick (about 2 us). No relaunch |
| Bandwidth | `kvm_ops_cost.inc` (by op NAME, default MIXED/0 bytes): resource profile + bytes from typed args (`matmul`: 4(mk+kn+mn)); BW tasks under 16 KiB are demoted to latency-bound. Concurrent-BW-CTA limit per class and global; the policy block measures achieved MB/s from finished tasks' bytes and device timestamps and runs AIMD on the global limit when a protected class's queue wait exceeds its target |
| Rebalancing | work conservation (idle share lent at once, reclaimed at the next task boundary) + `KvmController` (host thread over the mapped telemetry: smoothed demand, fast attack, slow release, deadband) |
| Inheritance | parked-waiter counts per (version, class), publisher classes per version, transitive closure each tick; a boosted class ignores caps and runs as if under reservation. Class granular |
| Affinity | a ready task is offered only to the block that released it for `affinity_us` |
| Telemetry | live: per-class running/ready/CTA-ns/tasks/bytes/misses/window waits/BW MB/s/limit/boost mask in the mapped block; after the run: trace CSV with bytes, resource, start-reason (resv/boost/urgent), and a per-workload report (CTA-seconds, wait p50/p99/max, misses, GB/s) |
| Modes | `GLOBAL` = the original single queue, `STATIC` = a worker serves only the class that owns its SM id (the static-split baseline), `DYNAMIC` |

Tasks are not preemptible: a reclaim happens at the next task boundary, so its worst case is the longest task of the other classes (measured below).
Chunking is host-side: a long op split into independent tasks over sub-views of the buffers shortens that bound; there is no device-side cooperative yield.

## Correctness (functional.txt, negative.txt, sanitizer.txt, exec-suites.txt)

| Claim | Evidence |
|---|---|
| Exactly once, edges ordered, values bitwise, in every mode | `rand`: 50 runs x {GLOBAL, STATIC, DYNAMIC}, random weights/reservations/caps/BW caps/deadline windows, 119,500 tasks, exactly-once, every edge ordered by device timestamps, node values bitwise vs the CPU reference |
| Weighted share | weights 3:1 -> 0.750/0.250; 1:2:4:1 -> 0.125/0.250/0.500/0.125 of CTA-time (window where all backlogged), occupancy 95% |
| Work conservation | class 0 alone keeps 96.5% of 32 workers busy; a class arriving with reservation 16 gets its 16th CTA 180 us later (task length 200 us) |
| No starvation (min share) | weight-1 class with reservation 6 vs a saturating weight-1000 class, 40 pulses: max reclaim wait 115 us (longest competing task 323 us, bound 473 us), 0 violations in 240 tasks |
| The harness detects starvation | `starve_neg` (guarantee flag cleared): 10 violations, worst wait 36.9 ms against a 472 us bound; `--expect-fail` returns 0 |
| Deadline tier | 6 low-priority deadline tasks overtake 20 high-priority ones only with `urgent_us`, in exact EDF order (one task per priority bucket; inside one bucket order is FIFO) |
| Priority inheritance | 3 shapes (publisher class throttled to 0 CTAs; transitive 2-hop with both throttled; waiter parked before the publisher bundle is admitted): the waiter finishes in under 1 ms, started under boost |
| The harness detects the deadlock | `inherit_neg` (inheritance off): the waiter does not finish in 400 ms behind the throttled publisher; the host then lifts the throttle (also exercising a run-time update) |
| Existing exec suites | `test_exec` with the new tree, scheduler off: 8505 checks, 0 failed (exec-suites.txt) |
| Compiler bundle | `kvm_exec --sched dynamic --classes 2 --weights 3,1 --min 1,0 --arbiter --golden` on `examples/live_training_inference.json`: 10/10 buffer values bitwise, versions and observed versions match (not in these logs) |
| Sanitizers | compute-sanitizer memcheck, racecheck, synccheck on the `smoke` suite (all three modes, a throttled-publisher inheritance case): 0 errors, 0 hazards |

## Dispatch cost: where dynamic sharing loses (bench.txt, `dispatch`; 32 workers, equal tasks, 3 runs)

| tasks | original | GLOBAL | DYNAMIC 1 class | DYNAMIC 4 classes |
|---|---|---|---|---|
| 200 us: occupancy / gap p50 | 97.8% / 3.1 us | 97.1% / 4.1 us | 95.9% / 7.2 us | 95.5% / 7.2 us |
| 20 us: occupancy | 43.4% | 42.1% | 39.4% | 42.0% |

Dispatch is serialized on the queue heads: the original executor already tops out near 0.7M tasks/s. DYNAMIC adds 2-4 us per task of idle gap, i.e. 2% for 200 us tasks and
a lower ceiling for tiny tasks. A first version put hot counters on shared lines and used returning atomics before the claim CAS: with 322 workers it lost 20% of
streaming throughput (66% utilization); posted adds and one cache line per field fixed that (97%). Tasks shorter than about 100 us per CTA are dispatch-bound in every mode.

## Bandwidth arbiter (bench.txt, `arbiter`; whole GPU, 5 runs, median [min,max])

Chain = 16 dependent 32 KB cold copies per request (5.1 us when alone), 4000 req/s. Stream = 6.3 MB `add` tasks, always 1200 outstanding (about 1.0 TB/s).

| config | chain request p50 / p99 (us) | chain hop run p50 | stream GB/s |
|---|---|---|---|
| global queue, flat priority | 90,509 / 91,997 | 19.5 us | 1002 |
| global queue, chain priority 20 | 710 / 853 | 19.5 us | 1017 |
| dynamic (reservation 8, EDF), arbiter OFF | 768 / 899 | 19.5 us | 1018 |
| dynamic, arbiter ON, target 60 us | 705 / 907 | 18.4 us | 895 (88%) |
| dynamic, arbiter ON, target 25 us | 464 / 752 | 12.3 us | 175 (17%) |

Static cap sweep on concurrent stream CTAs (same load, arbiter off): cap 320 -> p99 862 us, 965 GB/s; 256 -> 851, 848; 192 -> 771, 657; 128 -> 744, 452; 64 -> 715, 254; 32 -> 673, 141.
Reading: a chain hop alone takes 5 us and takes 19.5 us while the stream saturates HBM; capping stream CTAs recovers up to 12 us per hop but costs throughput almost linearly
below the knee (cap 192 buys 14% on p99 for 35% of the stream). With the lenient target (60 us, never exceeded) the arbiter is nearly idle; with a tight target (25 us) the
AIMD runs to its floor and keeps pressure, so it gives 1.2x on p99 for 17% of the throughput. **This is not a good trade at this load**: the dominant cost of a chain hop here is wake-up and
pop latency behind 1 ms non-preemptible tasks plus HBM latency under load, and priority (global or reservation + EDF) already captures most of the benefit. The arbiter works
(measures bytes/s, adapts, caps are hard) but the adaptive controller does not find a knee; it is a throttle, not an optimizer. Deadline "miss 100%" in these rows is the
250 us SLO being unattainable under any config (chain p50 >= 460 us); the arbiter ignores misses for that reason.

## Rebalancing (bench.txt, `rebalance`)

106 workers (108 blocks, one per SM). Serve = 48-task requests (matmul 128^3, 261 us); train = saturating mix of 6.3 MB adds and matmuls. Controller period 0.5 ms.

| phase | rate/s | serve p50 / p99 (us) | train busy | serve busy | reservation range (2nd half) | changes (2nd half) |
|---|---|---|---|---|---|---|
| 0 idle | 0 | - | 98.7% | 0 | 2..2 | 0 |
| 1 low | 200 | 746 / 1011 | 95.8% | 2.7% | 15..50 | 25 |
| 2 BURST | 2500 | 844 / 2074 | 63.8% | 33.3% | 52..52 | 0 |
| 3 low | 200 | 695 / 994 | 95.6% | 3.0% | 26..52 | 24 |
| 4 idle | 0 | - | 98.7% | 0 | 5..12 | 7 |
| 5 BURST | 2500 | 798 / 1912 | 65.0% | 32.2% | 51..53 | 3 |
| 6 idle | 0 | - | 98.5% | 0.2% | 8..30 | 22 |

Convergence: the reservation reached 90% of its settled value (52 CTAs) 0.16 ms after the first burst began; in both bursts it is stable (at most 3 changes, range 2) and the test asserts it.
When serve idles train gets 98.5-98.7% of the workers; during a burst the machine stays 96-97% busy (train 64% + serve 33%). The reservation hunts in the low-rate phases (24 changes in
200 ms): the requests are 48-task spikes every 5 ms and the controller follows the spikes. That is harmless (the reservation only matters under contention) but it is not a calm controller,
and no proof of stability beyond these traces is claimed.

## Baselines under the scripted load (bench.txt, `baselines`; best-of-5, same 108-block grid in every mode)

| config | serve p99 (us) | train ktask/s | utilization |
|---|---|---|---|
| global queue, flat | 18,584 [18396,19624] | 112.5 | 98.5% |
| global queue, serve priority 20 | **1,140** [1091,1178] | 111.9 | 98.5% |
| static split 25% SMs serve | 118,245 [115685,141352] (95% of requests miss 3 ms) | 104.8 | 91.0% |
| static split 50% SMs serve | 1,825 [1620,2974] | 72.7 | 67.0% |
| dynamic + controller | 1,983 [1875,2062] | 111.0 | 97.9% |
| dynamic + controller + arbiter | 1,224 [1195,1351] | 91.8 | 82.7% |

Ratios (dynamic + controller against): flat global queue 9.4x lower serve p99 at 0.99x train; static 25%: 60x lower p99 at 1.06x train; static 50%: 1.5x the train throughput and 1.46x
the utilization at 8% higher p99 (1983 against 1825). **Against a global queue that gives serve a higher priority, dynamic sharing is worse on the latency-sensitive tail (1983 against 1140 us, 1.7x)
at equal throughput.** Strict priority is hard to beat while serve does not overload the machine; dynamic costs dispatch time per pop and the reservation/EDF tiers do not make up for it.
Spread across the 5 runs is tight (min/max in brackets above).

## Overload: where the dynamic scheduler wins (overload.txt)

Serve flood of 9000 req/s x 48 tasks (demand about 115 of 106 CTAs) for 300 ms, train saturating. Global queue with serve priority 20: train gets 1.7 ktask/s [1.1, 2.0] (starved to 1.5%).
Dynamic with a 26-CTA train reservation: 19.3 ktask/s [17.9, 21.0] = 11x, serve p99 57 ms against 38 ms (both far beyond any SLO: this is overload, the point is who gets starved).

## Task affinity (bench.txt, `affinity`)

Chain of 300 dependent in-place adds over 128 KB with every worker busy: successor on the block that ran its predecessor 0% -> 100% with `affinity_us` 20; hop p50 35.8 -> 37.9 us (no gain, noise).
A 128 KB L1-resident chain barely benefits. The 12 ms -> 5 ms Doom effect is attributed to instruction/state caches; it was NOT measured here (see Not done).

## Not done / unproven

- Real Doom scenario (feat/doom) was not run: it needs the NVPTX engine build and WAD on the box and a bundle generator that assigns workload ids; the hooks it needs (reservation, `deadline_us` + `urgent_us`, affinity) exist and are tested on synthetic loads only. The 35 Hz deadline claim is untested on Doom.
- Affinity effect on a warm/cold-cache workload is not shown (synthetic chain: no gain).
- The arbiter does not beat priority at this load (see above); the adaptive limit has no knee detection.
- Controller stability is shown on the traces above, not proven; it hunts in low-rate phases.
- Inheritance is class-granular (over-approximates); EDF is exact only across bucket heads.
- No device-side yield for chunked tasks; reclaim latency is bounded by the longest task.
- Admission is one FIFO ring: a serve bundle waits behind earlier bundles in the ring (train bundles were kept small and paced).
- Dispatch ceiling (about 0.5M tasks/s dynamic) limits experiments to tasks of at least ~100 us per CTA at 100+ workers.
