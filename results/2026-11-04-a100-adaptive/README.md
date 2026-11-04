# Adaptive scheduling and multi-tenant stress, real Doom + GPT train + GPT serve on one persistent kernel (A100, 2026-11-04)

- GPU: NVIDIA A100-SXM4-40GB, 108 SMs (106 worker blocks: 1 admitter + 1 policy block), driver 580.105.08, nvcc 12.8, `-O3 -arch=sm_80 --fmad=false -ffp-contract=off`, shared box, every batch under ONE `flock ~/.gpu.lock` hold.
- Code: branch `feat/adaptive` (base `feat/integration` 1d3c15a + `feat/adaptive-doom`). Driver `adaptive/adaptive_run.cu`, controller `include/kvm_adapt.h`, scenario compiler `adaptive/make_scenario.py`, batches `adaptive/batch.sh` via `adaptive/run_adaptive.sh`, aggregation `adaptive/analyze.py`.
- Raw logs: `holdA-main-big/` (5 reps main, 3 reps big model, suites), `holdB-tuned-multi-bw/` (tuned gate, multi-tenant, bandwidth), `holdC-suites/`, `doom-instances/` (multi-instance Doom build/verification). Every run directory has `summary.json`, `status`, `run.log`, the Doom-oracle `compare_doom*.txt`, the per-frame, per-request and per-window CSVs; `rep1` also has `timeline.csv.gz` (2 ms telemetry), `controller.csv.gz`, `doom_tasks.csv.gz`.
- All numbers: median [min, max] over repetitions (5 reps main, 3 reps big/tuned/multi, 4 reps bandwidth) unless "pooled" (all repetitions' events together; p999 needs the samples).
- **Correctness**: all 162 scheduler runs here ended with every Doom frame of every instance raw-byte, fingerprint and game-state identical to the CPU oracle (`status` = `0 1`, `compare_doom*.txt`), serve tokens identical across all request slots, training loss finite.

## What was built
| | |
|---|---|
| `KvmClassTele` (device -> host) | lossless cumulative counters per class: `dl_n`, `near_miss` (finished in the last quarter of the deadline budget), `wait_over` (queue wait above `target_wait_us`), `wait_sum_ns`; verified equal to the per-task trace (`test_adaptive telemetry`) |
| `KvmAdaptive` (host controller) | every 1 ms: reservation = smoothed demand + boost (doubles on a near miss or queue-wait pressure, halves after 50 calm periods), weight and EDF window follow the boost, batch class holds a floor reservation while it has work, optional arbiter assist (bandwidth saturated and a protected class under pressure). No caps, no SM partition. 8 host unit tests (`tests/host/test_adapt.cpp`) |
| `KvmAdmissionGate`, `KvmGateTuner` | per-class in-flight task budgets on the host (the admitter is ONE FIFO, see below) and an AIMD loop on those budgets driven by the deadline class's end-to-end slack |
| `adaptive_run` | paced 35 Hz real Doom (1-8 instances, one 6-task bundle per frame, deadline = one period), N GPT training runs (one optimizer step per bundle, 9.9k tasks), open-loop Poisson GPT serving requests (1.1k tasks, prefill + 4 decode steps), optional AdamW sweep (bandwidth co-runner); phased load: serve burst, training pause, Doom scene change |
| cost table | AdamW, gradnorm, residual, GELU, zero, embedding, softmax-CE now have byte costs (they were invisible to the bandwidth arbiter before) |

Phased scenario (24 s, `PHASES` in `batch.sh`): serve 25 req/s, burst 400 req/s in 8-12 s, training paused 14-17 s, Doom scene change 18-22 s (a synthetic 12 ms `spin` after every tick; the real engine's own tick cost also varies by scene, 4-8 ms). Classes: 0 Doom, 1 serve, 2 train. Configurations: `global` (one queue), `static` (SM split 10/30/60, baseline only), `fixed_fair` (weights 1:1:1), `fixed_prio` (8:4:1, Doom reservation 1, EDF window = period), `fixed_resv_peak` (serve reservation 40 = sized for the burst), `v1_demand` (the earlier demand-tracking controller), `adaptive` (this work), and `*_gated` = same plus admission budgets (serve 20k, train 25k tasks in flight of a 65k live window).

## Headline (honest)
1. **The scheduler policy is not what limits tail latency here; the single admission FIFO is.** With this executor (admitter ingests one bundle record at a time, about 0.5 M tasks/s) the machine is 12-15% utilised in every configuration (21-25% with the 6-layer d512 model), and Doom's frame time, 13 ms median against 7.3 ms alone, picks up 20-45 ms stalls whenever a serve burst or a big training record is ahead of it in the ring. All SM-level policies (weights, reservations, EDF, controller) differ by little; gating the admission changes the tail by 2-4x.
2. **The closed-loop controller did NOT beat the best hand-set fixed configuration.** Pooled over 5 reps (below), `adaptive` vs `fixed_prio`: Doom frames late 470 vs 353, serve p99 70.7 vs 60.8 ms, train 40.6 vs 40.3 steps/s. It does what it is designed to (attack 0.6 ms after a burst, release 7 ms after, floor held, see suites) but a static priority setting is already near the best because the device lends idle CTAs (work conservation) and Doom needs only one CTA.
3. **Closing the loop on admission works, at a price.** `adaptive_tuned` (gate budgets shrink when Doom's end-to-end slack is low): frames late 43 vs 117-134 of 4030, burst-phase 0 late frames, Doom p99 28.7 vs 30.1-30.8 ms and serve p99 35.9 vs 44.9 ms, BUT it sheds 38% of the serve requests (797 of 2101 dropped) and cuts training 32%. This is load shedding, not a free win.
4. Static SM split is the worst on every latency metric (Doom p99 99 ms, 1213/4030 late) and strands CTAs; global queue gives serve a 150 ms median in a burst.

## Main phased experiment (5 reps; per-phase tables in `tables-main-phases.md`, flat in `tables-main-flat.md`)
Pooled over all 5 reps (4030 Doom frames, about 10k serve requests per config):

| config | Doom frame p50 / p99 / p999 / max ms | frames late (of 4030) | serve p50 / p99 / p999 ms | train steps/s | serve dropped (of 2101/rep) |
|---|---|---|---|---|---|
| global | 15.8 / 39.5 / 61.4 / 68.8 | 310 | 149 / 168 / 174 | 45.9 | 416 |
| global_gated | 13.9 / 30.3 / 32.3 / 37.5 | 110 | 56 / 74 / 76 | 40.3 | 589 |
| static 10/30/60 | 15.4 / 99.2 / 107 / 110 | 1213 | 107 / 127 / 131 | 43.2 | 101 |
| fixed_fair | 16.1 / 54.8 / 69.4 / 77.0 | 898 | 112 / 141 / 145 | 41.4 | 163 |
| fixed_prio | 14.8 / 37.8 / 51.5 / 58.2 | 353 | 29 / 61 / 74 | 40.3 | 0 |
| fixed_prio_gated | 13.0 / 30.7 / 37.0 / 45.1 | 133 | 25 / 44 / 48 | 34.7 | 67 |
| fixed_resv_peak | 15.2 / 38.0 / 49.2 / 56.0 | 322 | 30 / 63 / 69 | 40.3 | 0 |
| v1_demand | 15.7 / 40.2 / 50.8 / 60.8 | 395 | 30 / 64 / 69 | 40.3 | 0 |
| adaptive | 15.2 / 39.9 / 50.8 / 56.6 | 470 | 34 / 71 / 86 | 40.6 | 0 |
| adaptive_gated | 13.8 / 30.2 / 36.5 / 38.6 | 115 | 28 / 46 / 50 | 35.2 | 106 |

Burst phase only (8-12 s), median [min,max] of 5 reps: Doom frames late `fixed_fair` 136 [127,138], `static` 138 [137,138], `fixed_prio` 44 [35,54], `fixed_resv_peak` 43 [34,47], `v1` 58 [37,66], `adaptive` 57 [50,70], `adaptive_gated` 8 [2,14], `fixed_prio_gated` 11 [7,15], `global_gated` 5 [0,6]; serve p99 ms `fixed_fair` 141, `static` 128, `global` 169, `fixed_prio` 60, `adaptive` 68, `adaptive_gated` 46, `fixed_prio_gated` 44. Training during the burst: global 29.8, fair 25.8, static 26.2, adaptive 20.0, fixed_prio 18.2 steps/s (the priority configs pay for the latency with training throughput; adaptive gives training 10% more than fixed_prio and loses 10% on serve p99).
Calm phases are equal across configs (Doom p99 24-25 ms, serve p99 26 ms, except `global`: serve p99 64 ms, `static`: 70 Doom frames late in the 12-14 s phase). Training pause (14-17 s): everything idle, Doom p99 9-12 ms in every config (static: 17 late frames per rep in this idle phase, cause not traced - it is the only config with late frames while the GPU is idle).
**Doom alone** (`alone`, same harness, 5 reps): frame p50 7.3 / p99 16.7 ms, 0 late. Co-running adds 6-8 ms median (memory-system/clock interference on the tick itself: tick run time 8 ms vs 5.4 ms alone, see `doom_tasks.csv.gz`).
Utilisation (sum of per-class SM time / workers x time): 12-16% in every row, 21-25% (big model); "stranded CTA time" (idle workers while a task was ready) is 60-80% by this definition but it mostly measures the microseconds between a task becoming ready and a worker popping it at 0.5 M tasks/s, so it does not separate the policies; the static split's 60% against 77% is the only visible difference.

## Big model (6-layer d512: 47k tasks/step, 2.9k-task requests; 3 reps) - `tables-main-flat.md` rows `big_*`
Doom frame p99: `global` 185, `static` 194, `adaptive`/`fixed_prio`/`v1` 297-301 ms, gated 87-125 ms; frames late 1339-1928 of 2300-2400 in EVERY config. A 47k-task training record takes the admitter roughly 100-300 ms; everything submitted behind it waits, so a 35 Hz game cannot be served by any scheduler on this executor while such a record is in the ring (record-size advice below). Utilisation 13-25%.

## Admission is the bottleneck: measured in isolation (`holdC-suites/`)
The admitter copies one bundle record at a time, task by task, at **1.9 us per task (0.52 M tasks/s)**, whatever the task is (`noop`s here). A 1-task latency bundle submitted right behind a K-task record of another class (`test_adaptive admit_gran`, `suites_admit.txt`):

| K (tasks in the record ahead) | 500 | 2,000 | 8,000 | 32,000 | 64,000 |
|---|---|---|---|---|---|
| small bundle end-to-end p50 | 0.18 ms | 0.53 ms | 1.9 ms | 7.1 ms | 13.8 ms |

so the head-of-line delay is K x 1.9 us. The real records are 9.9k tasks (small training step, 19 ms), 47k tasks (big model, 90 ms) and 1.1k-2.9k tasks (serve request, 2-5 ms); a burst of 400 req/s x 1.1k tasks alone is 0.44 M tasks/s, i.e. 85% of the admitter. Shared live window (`hol` suite, flood of 256 x 200 us bundles, live window 4096 tasks): latency bundle end-to-end **112.7 ms ungated vs 0.26 ms with the flood limited to half the window** (`suites_hol.txt`). Advice: (a) bound each class's in-flight tasks (`KvmAdmissionGate`); (b) keep records of deadline-critical neighbours small: a record of K tasks costs every other class 1.9 us x K, so for a 28.6 ms game period with 50% slack keep every other class's records under about 7k tasks; (c) the real fix is in the executor (class-aware or round-robin admission, or admitting a big record in slices): `kvm_exec` is outside this work (owner O).
Note the device-side near-miss/miss counters start the clock at ADMISSION, so they are blind to this wait (a Doom frame stuck 20 ms in the ring and then run at full speed counts as on time); `KvmGateTuner` therefore takes the application's end-to-end slack.

## Multi-tenant scaling (N real Doom instances, 35 Hz, beside 3 training runs + 40 req/s serving; 3 reps; `tables-holdB-flat.md`)
| N | Doom frame p99 ms (global / fixed_prio / adaptive) | frames late (of 560 x N) global / prio / adaptive | serve p99 ms (global / prio / adaptive) | train steps/s |
|---|---|---|---|---|
| 1 | 25.7 / 25.8 / 25.9 | 0 / 1 / 0 | 73 / 28 / 28 | 54 / 53 / 53 |
| 2 | 26.2 / 25.8 / 25.8 | 0 / 0 / 2 | 73 / 28 / 28 | 54 / 53 / 53 |
| 4 | 26.2 / 25.6 / 26.2 | 0 / 3 / 4 | 72 / 28 / 28 | 54 / 53 / 53 |
| 8 | 25.9 / 26.3 / 26.5 | 0 / 4 / 9 | 72 / 28 / 29 | 54 / 53 / 53 |

(late counts are medians per run). Every instance of every run is oracle-exact. Unpaced (each instance runs as fast as it can, adaptive): 61 / 61 / 60 / 61 fps per instance for N = 1 / 2 / 4 / 8, frame p50 16.8 ms, i.e. per-instance rate is flat and the aggregate scales linearly to about 485 fps at N = 8 (the 163 fps per instance measured alone/unloaded in `doom-instances/` falls to 61 under training + serving). **No breaking point was found up to N = 8**: eight game ticks need 8 of 106 workers (7.5%). N is capped at 8 by the build (one engine object per instance, 1.3 MB each; `DOOM_INSTANCES`), so the real breaking point (tick duty > worker count, ~100+ instances at 35 Hz) is not reachable here; the executor-side limit that would hit first is the admission rate (6 tasks per frame per instance, trivial) and the per-instance state buffers (65 KB per frame buffer in this harness, one per frame).

## Bandwidth arbitration (AdamW sweep, 1.6 GB over p,g,m,v, weight 16, 4 reps; `tables-holdB-flat.md`)
| config | sweep GB/s | Doom p99 ms | serve p99 ms | train steps/s |
|---|---|---|---|---|
| no sweep | 0 | 24.8 | 27.4 | 51.1 |
| sweep, no arbiter | 658 | 24.6 | 28.4 | 17.1 |
| sweep + arbiter (protect Doom/serve, targets 500 us / 1 ms) | 646 | 24.3 | 28.1 | 16.8 |
| sweep + `bw_cta` cap 24 | 182 | 26.5 | 28.4 | 43.4 |
| adaptive controller + arbiter assist | 319 | 26.5 | 27.6 | 43.2 |
**The arbiter did not win, and I could not make it win with these victims.** The AdamW sweep takes 658 GB/s (42% of HBM) and the real latency tenants do not notice: Doom p99 24.6 vs 24.8 ms, serve p99 +1 ms. The device arbiter never throttles (646 vs 658 GB/s) because the protected classes never exceed their wait targets: there is nothing to protect. What the knobs do change is the training throughput: the sweep as a weight-16 class takes SM time (training 51 -> 17 steps/s); a `bw_cta` cap of 24 or the controller's assist restores it to 43 steps/s at 182-319 GB/s of sweep, i.e. it is a throughput-fairness tool, not a latency tool, for this mix. Doom and the tiny serve model are compute/latency-bound (weights fit the 40 MB L2). A bandwidth-bound latency tenant (a decode step on weights >> L2) was not built, so the latency benefit of the arbiter remains unproven on this tree (the older synthetic chain benchmark showed 655 vs 606 us p99, a 7% effect).

## Stress tests (synthetic spin/noop tasks; `tests/gpu/test_adaptive.cu`, `holdA`/`holdC` suites)
| suite | result |
|---|---|
| `telemetry` | device `dl_n`/`near_miss`/`missed`/`wait_over` equal the values recomputed from the per-task trace (300/296/296/396): PASS |
| `loop` (62 workers, serve burst beside saturating batch) | reservation reaches 10 CTAs 0.6 ms after the burst starts, 50 at peak, released to 0 within 7 ms of its end; batch keeps 34% of the SM time during the burst; 67 parameter writes: PASS |
| `floor` | weight-100 class floods, weight-1 batch class: batch SM-time share **0.213** with the controller's 20% floor: PASS |
| `floor_neg` (negative control) | floor disabled: batch share **0.001** -> harness reports STARVATION, `NEGATIVE TEST PASS` |
| `granularity` | latency task (200 us) beside saturating competitors of length L, queue wait p50/p99/max us. Dynamic, 62 workers: L=0.1 ms 7/8/8, 1 ms 11/78/79, 5 ms 73/390/426, 20 ms 268/1294/1612. 16 workers: L=20 ms 810/4886/5035. Model p99 = L(1 - 0.01^(1/W)) ~ 4.6 L/W: 358 and 1432 us at W=62, 5002 us at W=16: the measured p99 is within 0.85-1.1x of it. Global queue: the latency task waits behind the whole backlog (0.4-0.6 s), 150/150 past a 5 ms budget; dynamic: 0-2 of 150 at L=20 ms |
| `hol` | see admission section |

**Chunking advice (from these numbers).** A non-preemptible competitor task of length L delays a latency task by about 4.6 L/W at p99 and up to L in the worst case (all CTAs starting together): to hold a slack s at p99 keep L <= s W / 4.6 (62 workers, s = 2 ms: L <= 27 ms; 16 workers: 7 ms), to hold it in the worst case keep L <= s. Prefer many small tasks to one large one only down to a few hundred us: below that the admission cost (1.9 us per task, section above) and the 0.5 M tasks/s ceiling dominate instead. Doom's own tick is one 4-8 ms task (not splittable): with a 28.6 ms period and 106 workers it is never the problem, but 4-8 ms is also the floor of its frame time.

**Priority inversion / starvation.** Version-wait priority inheritance, reservation reclaim bound (max 115 us measured earlier), and the starvation negative controls are unchanged and re-ran green on this tree (`run_sched_tests.sh` 499 checks, 0 failed; `--negative`: starve_neg and inherit_neg both detected; sanitizers 0 errors). New here: the controller's batch floor (anti-starvation, above) and the admission gate. The phased driver keeps training, serving and Doom independent (no weight-version coupling), so inversion through a version wait is not part of the real-workload mix.

**Verification on this tree** (`run_all.sh verify`, one lock hold; logs not committed, summary): exec functional 8582 checks with **2 failed, both in `recycle` long-running admission** (see failure list); `--negative` (KVM_BREAK_FENCE) 119 failed = detected; `--negative-park` detected; sanitizers memcheck/racecheck/synccheck 0 errors; litmus harness 15 properties passed, 7 mutants detected, 0 failures; fuzz random 0 crashes/hangs, targeted findings []; GPT bundle 544/544 in all six scheduler modes with identical buffer md5 `1735f893ac10` and clean sanitizers; sched suites 499 checks 0 failed (+2 new cost checks), negatives detected, sanitizers clean. Doom frame-exactness: every run of this README (162 scheduler runs, up to 8 instances) is oracle-exact.

## Failure list and honest gaps
1. **`recycle` long-running admission (3M tasks through 4096 slots) failed 2 of 3 times on this tree** (verify hold and first re-run: producers gave up after their 60 s window wait with half the bundles submitted, done == admitted, counters exact for what ran; passed in 6.3 s on the third try). Scheduler is OFF in that test, my device changes do not execute there. Baseline comparison on the unmodified base commit: RECYCLE_BASELINE. Not root-caused; looks like an intermittent window-completion/stall in the executor (owner O) and should be rerun on a quiet box.
2. The adaptive controller gave no improvement over a hand-set fixed priority configuration in the real mix (headline 2); its value is robustness (the floor and the attack/release are verified), not better numbers here. A phase where fixed configs fail and the controller does not (e.g. the training floor under a sustained flood) exists only in the synthetic `floor` suite.
3. With the 6-layer d512 model no configuration meets 35 Hz (1339-1928 late frames of 2300-2400): the admitter, not the policy, is the limit. I did not implement class-aware admission (kvm_exec is not mine).
4. The tuned admission gate (`adaptive_tuned`) buys deadlines by dropping 38% of serve requests and 32% of training throughput; the right operating point depends on how much shedding is acceptable and was not explored (one setting: serve 20k / train 25k tasks max, min 10%).
5. "Doom scene change" is a synthetic +12 ms serial task per frame (frame = about 20 ms + admission + interference > the 28.6 ms budget in the tail): no scheduler can recover work that is serial and longer than the slack; it needs a faster or split tick. The real engine's own scene variation (4-8 ms ticks) is present in every run.
6. No breaking point for N Doom instances was reached: N is capped at 8 by the build and 8 ticks use 7.5% of the workers. Per-instance device memory was not measured.
7. The bandwidth arbiter showed no latency benefit: Doom and the small serve model are not bandwidth-sensitive (see section), and no bandwidth-bound latency tenant (decode on weights much larger than L2) was built.
8. "Stranded CTA" time as defined (idle workers while a task was ready, sampled every 2 ms) is dominated by dispatch latency and is not a discriminating metric here; SM utilisation is 12-16% (21-25% big model) because the executor is admission/dispatch-bound, so the claim that dynamic sharing avoids static-split waste is visible only as `static` having 60% against 77% on that metric, and in its 3x worse Doom/serve tails.
9. Runs on a shared box: other tenants held the GPU lock between my batches, but the GPU itself showed other processes (3.6 GB, 100% utilisation while queued) at times; absolute numbers carry that noise (spreads are in the tables).
10. `summary.json` `fps` was wrong in holds A and B (format-string bug, fixed afterwards); the tables recompute it from `doom_frames.csv`. The `submit_call_us` field is absent from those runs.
11. Priority inheritance was not exercised together with the controller (the controller sets reservations/weights/EDF only, never caps, so the boost path is untouched, but that is by construction, not by test).
12. The Doom oracle covers 1000 frames (about 28 s); all runs are shorter.
13. Multi-instance Doom: `tests/` shows 3 failures (missing comm ops, unrelated attn/gelu ops) that look pre-existing and the `tests/frontend` suite did not finish in the fork's session; not re-run here.

## Reproduce
```bash
python3 adaptive/make_scenario.py --out scn     --train-runs 3 --serve-pool 48 --train-total 600          # d256 x 4 layers, vocab 2048 (default model)
python3 adaptive/make_scenario.py --out scn_big --model '{"name":"big","vocab":4096,"d_model":512,"layers":6,"heads":8,"seq":256,"batch":8}' --train-runs 2 --serve-pool 32 --train-total 300
KVM_HOST=user@host REPS=5 DUR=24 OUT=runs-a adaptive/run_adaptive.sh all_a scn      # suites + main + main_big   (scn_big synced beside it)
KVM_HOST=user@host REPS=5 DUR=24 OUT=runs-b adaptive/run_adaptive.sh all_b          # tuned + multi (8-instance binary) + bandwidth
KVM_HOST=user@host OUT=runs-c adaptive/run_adaptive.sh suites                        # test_adaptive: telemetry loop floor floor_neg granularity hol admit_gran
python3 adaptive/analyze.py RUNS --workers 106 --phases "base:1-8,burst:8-12,calm:12-14,pause:14-17,post:17-18,scene:18-22,tail:22-24"   # or --flat
```
Multi-instance Doom objects: `DOOM_INSTANCES=8 doom/build.sh` (see `doom-instances/README.md`). Host unit tests: `c++ -std=c++17 -Iinclude tests/host/test_adapt.cpp && ./a.out` (8 pass).
The rows of the tables come from commit cf69b4b/8a4262a of `feat/adaptive` (holds A-C); the scheduler-device change is only `kvm_sched.cuh`/`.h` telemetry (+4 posted adds per task, scheduler-on only).
