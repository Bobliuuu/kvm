# Doom as a parallel task graph (audit gap 6) - A100-SXM4-40GB, shared Lambda box, 2026-11-06

Branch `perf/game-parallel` (base `feat/integration` 1ca85d1). Code: `doom/glue/gpu_par.c`, `doom/patch_engine.py` (r_plane / d_main patches), `include/kvm_ops_game.{cuh,def}`
(ops `doom_sim` 134, `doom_draw` 135, `doom_post` 136, `doom_planes` 137), `doom/make_workload.py` (`--split`, `--planes`, `--tenant train`), `doom/modal/harness.py` (job runner,
despite the directory name it ran on the shared Lambda box: the Modal launch was refused by the permission system, no Modal app exists), `adaptive/adaptive_run.cu` (`--doom-split`).
Every GPU run took `~/.gpu.lock`; the box is shared, so absolute numbers vary a few percent between holds; ratios below come from the same process/hold unless stated.
Raw per-run JSON lines (metrics, executor-trace phase breakdown, oracle comparison) sit next to this file; exec logs are kept for representative runs.

## What was built
The serial Doom tick (`doom_tick`, one CTA, ~5.4 ms) became a dependent task graph per frame, frame-exact against the serial/CPU result:

`doom_sim(f)` (thread 0: game logic, BSP walk, wall/sprite setup; the 3D view is only LOGGED) -> `doom_planes(f, 0..P-1)` (every flat plane's span discovery and per-span
arithmetic, one thread per plane, spans appended to the log) -> `doom_draw(f, 0..T-1)` (T tasks, each draws a band of screen columns from the log, epoch by epoch) -> `doom_post(f)`
(HU messages, border, menu, via re-entering `D_Display` after the view) -> `doom_present/doom_status`. Anything that has to read the finished screen (fuzz/spectre columns,
translated columns) or an overfull log drains the log serially on thread 0 (classic order by construction). `doom_post` fails loudly if a stage the logged view needs never ran;
`doom_present/status` refuse a frame whose post has not run.

Why those pieces (stage 0, `classic-phase-profile.jsonl`, clock64 on the game thread, 699 ticks, per tick): simulation 275 us, status bar 292 us, **R_SetupFrame + BSP walk + wall setup 2873 us**,
**R_DrawPlanes 1547 us**, sprites 333 us, HU 36 us; the 256-thread pixel draw the engine already had is only 147 us of the 5.4 ms (753 column + 266 span commands, 2 flushes).
So the tick is 97% serial work on the game thread; pixel drawing was never the bottleneck. The plane stage is the one big piece that is independent per plane (planes never overlap each other or the walls).

## Result 1: frame-exactness (gate) - passed
`stage*/verify.jsonl`: raw bytes, fingerprint and game state of every frame equal to the CPU oracle for serial and split T = 1, 2, 4, 5, 8, 16, 32, 64 (300 frames), T = 8 and serial at 700 frames,
alone and beside the GEMM+AdamW tenant under the dynamic and the global scheduler. Every other run that was compared (speed, study, instances, adaptive_run: 100+ runs) was identical too.
`sanitize/`: compute-sanitizer **memcheck 0 errors, racecheck 0 hazards** for serial and split (12 frames each). synccheck: split 0 errors; the **serial** path reports 32 "barrier error, divergent threads"
at `gpu_bar` (the in-CTA named-barrier draw that the split path no longer needs; behaviour unchanged, it was not run under synccheck before).

## Result 2: speed (`speed/speed.jsonl`, 700 frames, 3 repetitions, scheduler off, one doom instance alone)
| | fps (median [min,max]) | tick gap p50 / p99 ms |
|---|---|---|
| serial `doom_tick` | 179.9 [179.8, 179.9] | 5.46 / 9.83 |
| split, 1 draw tile | 218.5 [218.5, 220.0] | 4.49 / 8.12 |
| split, 8 draw tiles | 221.4 [221.3, 221.5] | 4.42 / 8.08 |
| split, 16 draw tiles | 221.9 [221.9, 222.0] | 4.42 / 8.06 |
| split, 64 draw tiles | 217.6 [217.4, 217.8] | 4.52 / 8.15 |

**+23% fps, tick 5.46 -> 4.42 ms (-19%)**, bought entirely by moving the plane work off the game thread. Executor trace (700 frames, T=8): game-thread task 5.36 -> 3.83 ms; the
plane stage plus its dispatch hop ~0.33 ms; column tiles 0.06 ms; post 0.03-0.13 ms. Draw tiles beyond ~8 do not help (a tile is ~50 us of work, a task hop ~11 us).
Intermediate stages (kept as ablations): stage 1 = draw tiles only (`stage1-draw-tiles/`): **no speed-up** (182.1 vs 182.0 fps at 700 frames; draw wall time 142 -> 68 us of a 5.4 ms tick); stage 2a = planes drawn pixel-by-pixel by
one thread each (`stage2a-planes-one-thread-each/`): **slower than serial** (155-161 fps) because the biggest plane's pixel loop ran on one thread (plane stage 1.9 ms) - fixed in stage 2 by having the planes stage only compute
spans and the column tiles draw the pixels.

What is NOT parallelised: the BSP walk + wall setup (2.9 ms, 65% of the new tick) is a front-to-back occlusion walk over a shared clip list; the simulation (P_Ticker, RNG-ordered) is serial by nature. One game still occupies roughly
one CTA for ~3.8 ms of every 4.4 ms. The task graph exposes tile/plane parallelism of ~50-300 us per frame, not whole-GPU parallelism for a single game.

## Result 3: more concurrent games (item 2, `instances/`)
The instance limit (8, a `static_assert` + hand-written macros) is now 64 (`doom/gen_inst.py`, `include/kvm_doom_inst.inc`; `gen_inst.py N` regenerates for more). Instance objects build in parallel (`JOBS`).
Measured with 120 frames per instance, every instance of every run raw-byte/fingerprint/state identical to the oracle, scheduler `dynamic`:
| instances | serial: per-instance fps (min-max) / aggregate | split T=4: per-instance fps / aggregate |
|---|---|---|
| 1 | see Result 2 | see Result 2 |
| 2 | 167-169 / 336 | 199-201 / 401 |
| 4 | 165-167 / 663 | 197-199 / 793 |
| 8 | 162-165 / 1305 | 193-196 / 1553 |
| 16 | 156-158 / 2510 | 182-186 / 2932 |
| 16 beside the GEMM+AdamW tenant (dyn) | 143-146 / 2315 | 164-169 / 2664 |
16 independent games at 35 Hz need 560 fps aggregate; the machine delivers 2500-2900 (aggregate fps, unpaced capacity, not paced behaviour). **64 instances: not verified.** See "Not done".

## Result 4: real tenants instead of spin tasks (item 3, `study/`, `study/tables.md`)
The load is `make_workload.py --tenant train`: per round two forward GEMMs (64x64 `matmul_tile` tasks with a 16384-deep reduction, TF32) and an AdamW sweep over 8M parameters per chain (7 streams, 2M-element tasks),
joined per round; no `spin` task. Medium = 6 chains x 60 rounds (186k tasks, 87 CTA-s, 85 GB moved per run, block utilisation 35-43%); heavy = 16 chains x 25 rounds. It does not saturate the machine: the tenant is dependency/admission
limited (tasks wait ~3 ms in the ready queue, 43 GB/s), so "real pressure" here means GEMM+HBM traffic on ~40% of the CTAs, not 100%.
Medium tenant, 3 repetitions x 300 frames (median fps; p99 = tick-gap p99; "gaps>28.6" = tick gaps over one 35 Hz period, summed over reps):
| scheduler | serial Doom: fps / p99 ms / gaps>28.6 | split Doom (T=8): fps / p99 ms / gaps>28.6 |
|---|---|---|
| alone | 170.4 / 10.2 / 0 | 206.3 / 8.2 / 0 |
| one global queue, no priorities | 136.8 / 18.4 / 0 | 149.5 / 28.0 / **3** |
| `--sched global` | 136.2 / 18.6 / 0 | 148.8 / 28.2 / **7** |
| static split 10% | 153.1 / 10.5 / 0 | 178.1 / 9.5 / 0 |
| static split 25% | 153.1 / 10.3 / 0 | 177.7 / 11.2 / 0 |
| dynamic, reservation 1 + urgent 20 ms | 150.2 / 11.1 / 0 | 174.1 / 10.0 / 0 |
| dynamic, no reservation, no boost | 150.0 / 11.2 / 0 | 174.3 / 10.0 / 0 |
Findings: (a) the split Doom keeps its +16% fps advantage under the static and dynamic schedulers with a p99 of 9.5-11 ms and no late tick; (b) **under a single FIFO queue the split graph is worse in the tail**
(p99 28 ms vs 18 ms, 3-7 tick gaps over the 35 Hz period vs 0): a frame is now ~14 dependent tasks and every hop can queue behind the tenant's tiles - the more a game is split, the more it needs the scheduler;
(c) the scheduler modes separate exactly as in the old spin study (global worst, static and dynamic equal for Doom). Heavy tenant (1 rep): same ordering, global p99 38-80 ms with 9-19 late gaps for split, serial 35-40 ms with 18.
The old "load" was spin tasks (no memory traffic); here the tenant moves 85 GB and Doom under dynamic scheduling runs 12% slower than alone (150 vs 170 fps serial, 174 vs 206 split); I did not isolate the cause (shared L2/HBM versus clock/power), spin tasks showed 3% (177.9 vs 184.4 fps in `results/2026-11-04-a100-sched-integrated`).

## Result 5: honest comparison with the earlier numbers (item 4, `adaptive-gamepar/`)
`adaptive_run` (real GPT training + open-loop GPT serving + paced 35 Hz Doom, serve burst 8-12 s, train pause 14-17 s, scene change 18-22 s; 24 s; 3 reps; every Doom frame of every run oracle-identical):
see `adaptive-gamepar/tables.md`. **The earlier "470 vs 353 late frames" (adaptive vs fixed_prio, 5 reps pooled, p99 frame time 38-40 ms) does not reproduce on this tree**: on this base (executor with slot recycling and the merged sched),
the *serial* baseline already has 0 late frames of 2520 for both adaptive and fixed_prio (frame-time p99 17.9 ms) and 1 late frame under `global`. There is no headroom left to show a late-frame improvement, and I did not try to
re-create the old base. What the split does change here: Doom frame time p50 8.6 -> 7.4 ms (adaptive) and 8.5 -> 7.2 ms (fixed_prio), p99 17.9 -> 16.9 ms, with serve p99 and train throughput unchanged
(adaptive 50.9 vs 51.3 ms serve p99, fixed_prio 29.5 vs 29.6 ms; train 46.8-47.0 steps/s) - so the extra tasks did not cost the other tenants anything measurable. The earlier finding that adaptive is worse than fixed_prio on serve p99
(here 51 vs 30 ms) does reproduce. The 35 Hz target holds in every row (fps 34.95).

## Not done / caveats
- Modal: authorised, but the launch was denied by the permission system, so everything ran on the shared box under the lock. No Modal app was created.
- BSP walk + wall setup (65% of the remaining tick) and the simulation are still serial; a single game does not use the whole GPU. Pipelining `doom_sim(f+1)` with the draw of frame f was not done (the draw stages are only ~0.4 ms of 4.4; it would need a zone-purge guard).
- 64 instances: the header/generator/build support 64, a 64-instance binary was built, but **it faulted at run time** (illegal instruction in `P_SetupLevel`, even for 2 instances), see `instances/`; the cause was not found in the time available
  (suspects: 1.7 GB of static device memory from 64 x 27 MB engine images, or something in the 64-way dispatch). The verified limit is 16 instances (binary built with 16); 32 and 64 are untested.
- The study uses unpaced frames (the executor does not pace; `deadline_us` is metadata), so "late" there means a tick gap above 28.57 ms or a missed absolute deadline; the paced late-frame count comes from the `adaptive_run` section.
- The tenant does not saturate the GPU (see Result 4); the "dynamic with reservation" and "without" rows are equal, as in the earlier study.
- Single process per hold; 3 repetitions for speed/study/adaptive, 1 for the heavy tenant and instances. Numbers are from a shared box.
