# Real Doom on the generic executor, A100 (2026-11-03)

Hardware and toolchain: NVIDIA A100-SXM4-40GB, 108 SMs, driver 580.105.08, CUDA 12.8 (nvcc 12.8.93), clang 14.0.0 (NVPTX target, Ubuntu build),
doomgeneric commit `e257a09527062bfb986b6e5e0780839b6cbd5b71`, shareware `doom1.wad` (md5 `f0cefca49926d00903cf57551d901abe`), skill 3, E1M1.
Box shared with other agents: every GPU run under `flock ~/.gpu.lock`. Code: branch `feat/doom` (this directory's commit; see `git log`).

Build: `doom/build.sh` = clang `--target=nvptx64-nvidia-cuda -march=sm_80 -O2 -ffreestanding -fno-jump-tables` per engine file, `llvm-link`, our IR passes
(`fix_varargs.py`, `widen_i1_globals.py`), `llc -mcpu=sm_80 -mattr=+ptx75`, `nvcc -gencode arch=compute_80,code=sm_80 -dc -maxrregcount=112` -> `doom_dev.o`.
Test programs: `nvcc -O3 -arch=sm_80 -std=c++17 -rdc=true -I. -Iinclude --fmad=false -Xcompiler -ffp-contract=off ... doom_dev.o`.

## What is checked

The oracle is the same engine, glue and scripted player built for the CPU (`doom/cpu_build.sh`, `doom/cpu_run.py`), on a virtual clock of exactly one 35 Hz tic per
frame. Frame f = the screen after `doom_tick(frame=f)`. `doom/compare.py` compares, per frame, the raw bytes (64000 palette indices + 256 palette words,
sha256), the 64-bit fingerprint the device computes in `doom_status`, and the 14-word game state. Every comparison below printed `PASS`.

| Test | Command | Result |
|---|---|---|
| Standalone, one 256-thread CTA, same device functions as the ops | `KVM_HOST=... doom/tests/run_gpu.sh 2000` | 2000/2000 frames raw-identical, fingerprint-identical, state-identical. Final: 6/6 kills, 0 deaths, 5 items, level finished at tic 1500 (42.9 s) |
| compute-sanitizer memcheck, 12 frames (boot + 11 tics) | `SANITIZE=memcheck doom/tests/run_gpu.sh 12` | 0 errors, frames still identical |
| compute-sanitizer racecheck, 8 frames | `SANITIZE=racecheck doom/tests/run_gpu.sh 8` | 0 hazards, frames still identical |
| Through the executor, 700 frames, alone | `doom/tests/run_exec.sh 700` | 700/700 identical (raw, fingerprint, state) |
| Through the executor, 700 frames, beside a load, Doom at equal priority | same | 700/700 identical; 300/300 matmul outputs equal A@B |
| Through the executor, 700 frames, beside the load, Doom at higher priority | same | 700/700 identical; 300/300 matmul outputs equal A@B |
| Whole game through the executor beside the load, Doom higher priority | `ONLY=load_doom_first doom/tests/run_exec.sh 1600` | 1600/1600 identical, 300/300 matmul correct; 6/6 kills, 0 deaths, level finished tic 1500 |

The executor runs are one compiled bundle (`runtime/compile_workload.py`, ABI v2): per frame `doom_tick`, 4 x `doom_present` (50 rows each), `doom_status`, with
tick f+1 after all of them; with the load, a second graph of 300 chains of (matmul 256^3 -> spin 200 ms) x 40. The 300 chains exceed the 215 worker blocks, so
the ready queue is never empty and every block is busy (block utilization 99%): Doom and the load share the SMs through one ready queue, nothing is partitioned.
The 35 Hz pacing is only `deadline_us` on the tasks; the executor runs flat out.

## Cadence (700 frames; boot tick excluded; `metrics_*.json`)

| Scenario | fps | tick gap p50 / p99 | tick queue wait p99 | SMs used by Doom | block utilization |
|---|---|---|---|---|---|
| alone (215 workers, ticks migrate over SMs) | 79 | 12.4 / 23.4 ms | 6 us | 108 | 0.5% |
| alone, 1 worker block (every task on one SM) | 183 | 5.3 / 9.6 ms | 4 us | 1 | 99% |
| beside the load, EQUAL priority | 37.9 | 12.9 / 204.9 ms | 136.7 ms | 108 | 61% |
| beside the load, Doom HIGHER priority | 179 | 5.5 / 9.8 ms | 4 us | 30 | 99% |
| same, whole game (1600 frames) | 196 | 4.6 / 9.6 ms | 4 us | 59 | 99% |

Reading it honestly:
- The serial game tic is the limit, not the scheduler: one tic runs on one CTA (thread 0 plays, 255 threads draw), 3.8 ms median standalone, 4.6-5.5 ms in the
  executor when it stays warm. That is 180-200 fps ceiling, 5x the 35 Hz needed; Doom cannot go faster than one tic per ~4-5 ms however many SMs are free.
- When idle workers pick the tick up, it lands on a different SM almost every frame (100+ SMs) and takes 12.4 ms (2.3x slower than on a warm SM), most
  likely because the engine's code (1.3 MB object) and hot state are cold in that SM's L1/instruction caches (inferred from the one-worker control, not profiled). With priority under load the freed block is usually the one Doom just left, so it stays
  warm (30-59 SMs, 5 ms ticks). An affinity hint (prefer the block that ran the previous task of the same chain) would make the idle case as fast. Not done here.
- Equal priority under saturation: Doom's task waits behind ~185 queued spins (p99 137 ms queue wait, 205 ms frames): 37.9 fps overall, below 35 Hz in the bad frames.
  Priority fixes it without preemption because a block frees every ~1 ms (200 ms spins over 215 blocks); with non-preemptive 200 ms tasks and fewer blocks it would not.
- Deadline metadata: 7 of 1600 ticks carry a missed flag in the trace, all in the first frames (the 170-330 ms boot tick makes frames 0-6 late); none after.

## Things found on the way

- compute-sanitizer found a real code-generation bug: LLVM GlobalOpt shrinks 0/1 int globals to `i1` and llc NVPTX loads them with `ld.u16` from a 1-byte symbol
  (out-of-bounds read). `doom/glue/widen_i1_globals.py` rewrites those 15 globals to i8; memcheck is clean.
- The engine keeps `argv` after start-up; a stack-allocated argv crashed the CPU oracle at the level exit (tic 1500). Fixed (static storage) in both builds.
- `include/kvm_exec.cuh` `registry_ok()` still demands dense op ids (`id == position`), so any id >= 32 (gpt, game) fails its static_assert although
  `kvm_ops_list.inc` and its own message promise sparse ids. Not my file; `doom/tests/run_exec.sh` removes that one condition from the scratch copy on the box
  before building (logged as WORKAROUND). Fix = drop `kKvmOps[i].id != i ||`; `kvm_op()` already searches by id.
- The stock `kvm_exec` CLI cannot run Doom (no hook to upload the WAD and set the stack limit): `doom/exec_main.cu` (`doom_exec`) reuses its `KvmExec` and
  bundle reader and does both through `kvm_doom_host_init()`.

## Watch it

`doom_executor_700frames.mp4` (700 frames, 320x200, 35 fps) is the executor run beside the load. Any run's `frames.bin` converts with
`python3 doom/view.py frames.bin --video out.mp4` (needs imageio + imageio-ffmpeg) or `--png DIR --every 5` (stdlib only).

Raw logs: `logs/<UTC>-doomtest`, `logs/<UTC>-doomexec` (git-ignored); per-task traces (`trace.csv`) stay on the box in `~/kvm-doom-scratch/logs/doomexec/`.
