# doom/ - real Doom as scheduler tasks

The actual Doom engine ([doomgeneric](https://github.com/ozkl/doomgeneric), shareware E1M1, real BSP renderer, monsters, doors) compiled
for the GPU and run as ordinary operators of the generic executor (`include/kvm_ops_game.def`, `include/kvm_ops_game.cuh`). The scripted
player plays the level without cheats. The engine source (GPL) and the WAD are fetched, never committed.

```
setup.sh        fetch the pinned engine commit + shareware doom1.wad (md5 checked) into third_party/  (git/curl only)
build.sh        engine + libc + glue + player -> build/doom_dev.o   (clang NVPTX, ARCH=sm_80 default; link with nvcc -rdc=true)
cpu_build.sh    the same engine/glue/player for the CPU -> build/libdoom_cpu.so (the correctness oracle)
cpu_run.py      run the oracle N tics -> frames.bin + summary.json;  compare.py diffs any two runs frame by frame
patch_engine.py our patches to the engine (player hook, PTX workarounds, parallel-draw hooks); each asserts it applied
glue/           doom_glue.c (virtual 35 Hz clock, input word, state; shared GPU/CPU), gpu_glue.c, gpu_par.c (255 workers draw the
                queued columns/spans), libc*, fix_varargs.py + widen_i1_globals.py (IR passes for NVPTX), frame_hash.h
player/         the scripted player (hooks G_BuildTiccmd) + the E1M1 route and its planner
make_workload.py  workload JSON for runtime/compile_workload.py: N frames of tick/present/status, optional matmul load
exec_main.cu    doom_exec: runs a compiled bundle on KvmExec, writes frames/hashes/trace/cadence metrics
view.py         frames.bin -> PNG sequence (stdlib only) or mp4/gif (imageio)
tests/          doomtest.cu (standalone, no scheduler), run_gpu.sh, run_exec.sh (KVM_HOST=user@host)
```

## Use

```bash
doom/setup.sh && doom/build.sh && doom/cpu_build.sh              # needs clang+llvm with NVPTX, nvcc, gcc, python3
python3 doom/cpu_run.py --frames 700 --out /tmp/cpu              # the oracle
KVM_HOST=user@host doom/tests/run_gpu.sh 2000                    # standalone GPU vs CPU, every frame (SANITIZE=1: memcheck)
KVM_HOST=user@host doom/tests/run_exec.sh 700                    # through the executor, alone and beside a matmul load
python3 doom/view.py /tmp/cpu/frames.bin --video doom.mp4        # or --png DIR --every 5
```

Everything logs at `LOG_LEVEL=debug` by default (`info`, `warn`). Frame f is the screen after `doom_tick(frame=f)`.

## Executor requirements

`include/kvm_ops_game.cuh` (header comment) is the contract: blockDim >= 256 (thread 0 plays, 1..255 draw, named barrier 2), nvcc
`-rdc=true` with `build/doom_dev.o`, a per-thread stack limit (16 KiB), one tick in flight (tick f+1 after tick f and after the present and status
tasks that read frame f), the WAD uploaded by `kvm_doom_host_init()`. The 35 Hz pacing is `deadline_us` metadata on the tasks.
Game state is device-global: one Doom per process.
