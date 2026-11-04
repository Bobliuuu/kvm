# GPU executor: slot recycling, streaming, optional/i32 buffers, seeded/file init, park litmus (A100, 2026-11-04)

- GPU: NVIDIA A100-SXM4-40GB, 108 SMs, sm_80, driver 580.105.08 (CUDA 13.0), nvcc/runtime 12.8. The box is shared; every run was under `flock ~/.gpu.lock`.
- Code under test: commit 00e229d (tests, device runtime); the logs say `00e229d+dirty` because `kvm_exec.cu`'s CLI `main` (`--golden-tol`) was edited while the batch ran. Device code and test sources are identical to HEAD.
  The include directory held the real GPT pack (`kvm_ops_gpt.def/.cuh`, ids 32-63) during every run, so the dispatcher in these runs contained the 30 GPT ops. The game pack (Doom) is NOT built: it needs `-rdc` and `doom_dev.o`; opt in with `-DKVM_WITH_GAME`.
- Build: `tests/gpu/build.sh` = `nvcc -O3 -arch=sm_80 -std=c++17 -Iinclude -Itests/gpu --fmad=false -Xcompiler -ffp-contract=off -lineinfo -DKVM_EXEC_NO_MAIN -DKVM_EXTRA_OPS_DEF=... tests/gpu/test_exec.cu`
  (`--fmad=false`: per-op IEEE arithmetic so fp is compared bitwise with the CPU reference). Kernel with the GPT pack: 186 registers, 0 spills, 32 B stack, 4352 B static smem + 48 KiB dynamic smem (`KVM_GPT_SMEM_BYTES`), so 1 resident block per SM (grid 108).
- Commands: `KVM_HOST=... ./run_exec_tests.sh [--negative|--negative-park|--bench|--sanitize]`.

## What passed (files in this folder)

| Claim | Evidence (functional.txt unless noted) |
|---|---|
| Whole suite | 8578 checks, 0 failed: checker, ops, types, init, errors, dag, sched, admission, versions, recycle, litmus |
| 2,097,152-task DAG through 65,536 slots | `stream`: 6,289,920 edges, 128 windows (127 waited on predecessor windows), 3.95 s wall = 1.89 us/task, final layer bitwise vs CPU. Pool = 1/32 of the tasks |
| 200,000-task dependent chain through 1,024 slots | counter == 200000 after 782 windows (each task ran once, in order) |
| Long-running admission | 3,000,000 tasks in 100,000 bundles from 2 producer threads: 4096 slots, 1024-entry window table (wrapped ~100x), 8192-word successor ring, 16,666 admission-ring wraps, 6.2 s, all 64 counters exact |
| Recycling is not blocked by a parked task | 600,002 tasks in 20,002 bundles with one task parked on an unpublished version, 1024 slots; the parked task ran once when published (sees version 1) |
| Park/publish litmus | 12 runs x 156,000 tasks (3/5/9/33 blocks): every iteration races a waiter's park against the publish of its version (the waiter and the publisher are released by the same publish): no lost wakeup, early run (`WAIT_VIOLATION`) or double run (`DOUBLE_EXEC`) |
| The litmus can fail | negative-park-recheck.txt: with `-DKVM_BREAK_PARK_RECHECK` (waiter never re-checks the version after parking) 5 of 12 runs lose wakeups (TIMEOUT with 20-155,000 tasks unfinished). Run `--negative-park` |
| Version negative test | negative-break-fence.txt: `-DKVM_BREAK_FENCE` -> 1688 reader violations detected |
| `buf?`, i32 buffers, dtype errors | `types`: opt_add with b present/absent, gather through an i32 index buffer, iota/optional i32, 5 device error codes (BAD_BUF for dtype and for -1 on a required buf, BAD_BUF_INDEX, BOUNDS), builder rejects an absent required buf. The GPT pack (matmul_tile has `bias:buf?`) also ran this way |
| `seeded` init == runtime/initgen.py | `init`: 30 cases (6 seeds incl. 2^63+5 and 2^64-1, uniform/normal, 5 shapes), 9798 values: device kernel AND host reference bit-identical to vectors regenerated from initgen.py on the box (`make_initgen_vectors.py --check`); 3,000,001 normal values device == host |
| `file` init | 20 MB streamed through one 8 MiB staging buffer; size mismatch rejected |
| Real GPT ops through the executor | gpt-bundle.txt: frontend `gpt_tiny` 2-step AdamW training (544 tasks, 3,992 edges, seeded + values inits, i32 token buffers, buf? args) runs 544/544 with the GPT pack; compute-sanitizer memcheck, racecheck and synccheck: 0 errors, 0 hazards (4 blocks) |
| Dispatch cost | bench.txt |

## NOT proven / honest notes
- **GPT golden mismatch.** The same bundle against the compiler's CPU golden (`kvm_exec --golden`): losses agree (`loss.0` rel L2 2.5e-6, `loss.1` 1.4e-5), but updated weights/moments/later activations differ up to 8% relative L2 (`clip_coef` 1.7e-3). The result is identical for 2, 3, 16 and 64 blocks (same 5255 bitwise-equal values, same worst error 0.0809), so it is deterministic and schedule-independent; it points at the GPU GPT ops versus the CPU reference (gradient norm / clipping / Adam), not at the executor. ROOT-CAUSED afterwards (results/2026-11-04-a100-gpt-golden/README.md): 1xTF32 gradient noise flips the sign of near-zero gradients and AdamW's first update is a step function of the sign; with 3xTF32 GEMMs the same bundle matches the golden to 2.4e-5. Not an op or executor bug. D's `test_e2e` was not re-run through the executor.
- The bundle ran on the box as a single bundle: the frontend's MANIFEST (multiple admissions, shared buffers, `base_versions`) is not consumed by the CLI.
- Game pack / Doom not built or run here.
- `file` init through the CLI (bundle-relative path) is untested; the library path is tested.
- Python tests in this merged branch (`python3 -I -m unittest discover -s tests`): 145 run, 34 fail after merging feat/gptops + feat/doom into feat/compiler (compile_workload on this branch loads only the core registry while include/ now holds the GPT/game defs). feat/compiler alone: 145 OK. Not an executor issue; the lead's merge must pick one registry loader.

## Dispatch cost (bench.txt, 1 resident block/SM because of 186 registers)
Single worker 7.1 us/task (was 4.7 us before, core ops only, 5 blocks/SM); 108 blocks 1.38 us/task = 0.73M tasks/s; dependent hop 7.0 us (was 5.6 us). The extra work per task is the window credit, the slot recycle push and a fatter kernel; not optimised.

## Design and limits (details in include/kvm_exec.cuh)
- Slots recycle through a free-slot ring (a task's slot returns when it completes), arguments live inline in the slot (20 words/slot, so operators may take up to 20 arguments; was 15), successor lists live in a ring reclaimed per bundle record ("window") in order. A task parked on a version pins its slot, its window-table entry and (only if its window has edges) its window's successor words; with `win_cap` windows in flight and `succ_cap` words the pool can run ahead of such a task that far, then admission stalls and the device reports TIMEOUT.
- `submit_stream()` cuts one bundle (task order must be topological) into windows; an edge that crosses windows is enforced by the HOST (it submits a window only after the windows holding its predecessors completed, read from per-window done marks mirrored to pinned memory). Cost: a window with a cross-window dependency starts after its producers drain.
- A bundle record needs `nt <= min(live_cap, max_tasks)` and must fit in the admission ring.
