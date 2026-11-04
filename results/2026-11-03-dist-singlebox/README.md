# Distributed execution on ONE A100 (2026-11-04 05:44 UTC)

Everything that can be exercised without a second GPU, plus CPU stages. The stages that need 2+ GPUs or several hosts were
**skipped by the script, not run**; the table says which. Nothing here shows multi-GPU, NVLink, multi-node or WAN behaviour.

- GPU: NVIDIA A100-SXM4-40GB, 108 SMs, sm_80, driver 580.105.08, nvcc 12.8.93. The box is shared; the script held its GPU lock for
  the whole run, but other users' unlocked jobs may have run alongside, so latencies are indicative. Host: x86-64 Linux,
  `cudaDevAttrHostNativeAtomicSupported = 0`.
- Commit: 77b7775 (branch feat/multigpu, clean tree, the sources were rsynced from it). The directory name keeps the local date of
  the first recorded run.
- Command (from a laptop; `KVM_HOST` is an environment variable, never committed): `KVM_HOST=<user@gpu-box> dist/run_multigpu_tests.sh --remote`.
  On the box itself it is `dist/run_multigpu_tests.sh`. Stage timeout 600 s.
- Build flags (GPU stages): `nvcc -O3 -arch=sm_80 -std=c++17 -Iinclude --fmad=false -Xcompiler -ffp-contract=off -lineinfo -DKVM_EXEC_NO_MAIN`
  (`--fmad=false` so device floats compare bitwise with the CPU reference); negative builds add `-DKVM_DIST_BREAK_ORDER` /
  `-DKVM_DIST_NO_BP_POLL`.

## Result: RESULT: PASS (14 passed, 0 failed, 4 skipped); no stage needed a retry

| Stage | Result | What it covered |
|---|---|---|
| cpu-partitioner-sim | PASS (32 tests) | all strategies bitwise vs the unpartitioned CPU VM under random interleavings; fuzz; the simulator can fail |
| cpu-transport-proxy | PASS (90 checks) | CRC, exactly-once in-order delivery under drop/dup/delay/corruption, windows (an emulated 10 ms RTT: window 1 / 8 / 64 took 616 / 78 / 8 ms for 60 messages), dead peers, proxy + send-queue protocol |
| cpu-launcher-dry-run, cpu-multiprocess-tcp | PASS | launcher output; 3 OS processes over TCP with 10% drop, 10% dup, 20% delay |
| gpu-build, gpu-executor-regression | PASS | test_dist, both negative builds, test_exec; the executor's `smoke` suite with the hooks compiled in |
| gpu-ops | PASS (148 checks) | 240 random strided tiles (scalar and 16-byte paths) bitwise; reduce2 / reduce_put_signal bitwise; 1001 bf16 values bitwise; 10 misuse cases -> decoded device errors |
| gpu-pingpong-versions | PASS (20 checks) | 200 rounds between two executors with peer pointers; 100 rounds each over the host proxy (loopback, TCP) and 60 over TCP with 10% drop / 10% dup / 20% delay; every one of 819,200 values checked per run |
| gpu-errors-no-hang | PASS (7 + 2 checks) | unsatisfiable remote wait -> TIMEOUT after 0.8 s; full send queue without a proxy -> DIST_SENDQ_TIMEOUT after 0.3 s; silent proxy -> DIST_PROXY_DEAD after 0.3 s; killed peer -> DIST_PEER_DOWN within milliseconds; a full live window waiting for remote signals completes (pump during backpressure) |
| gpu-negative-break-order | PASS | each negative build must fail in the expected way: signal published before the data -> stale data detected (`--expect-fail=stale`, 1 detection, 0 timeouts); pump removed from admission backpressure -> receiver device TIMEOUT (`--expect-fail=timeout`) |
| gpu-sanitizer | PASS | compute-sanitizer memcheck 0 errors, racecheck 0 hazards, synccheck 0 errors on the ops suite (not on the two-executor suites: the sanitizer serialises launches) |
| gpu-plans-in-process | PASS | 9 partitioner plans (TP direct/ring/tree, ZeRO mixed p2p+proxy, ZeRO bf16 wire, DP floating point within 1e-5, pipeline x4 microbatches) with all ranks as threads on the one GPU: every expected value bitwise equal |
| gpu-plans-tcp-faults | PASS | 4 plans over TCP between rank threads with 5% drop, 5% dup, 10% delay: bitwise equal |
| gpu-multiprocess-tcp | PASS | 3 plans, one OS process per rank (4 processes sharing the GPU), TCP between them: bitwise equal |
| multigpu-p2p-bandwidth | SKIP: needs 2 GPUs (found 1) | not measured |
| multigpu-p2p-pingpong | SKIP: needs 2 GPUs | peer-to-peer across physical GPUs untested |
| multigpu-plans-one-rank-per-gpu | SKIP: needs 2 GPUs | not run |
| multinode-tcp-plans | SKIP: needs KVM_NODES with 2+ nodes | not run |

## What the idle-GPU runs found (fixed, and one open item)

The first recorded run (commit 6167fad, 14/14 on a busy GPU) hid two real bugs that only showed once the lock gave the GPU to the
tests alone:

1. `KvmExec`'s constructor memsets and buffer-table upload run in the legacy default stream while the kernel runs in a
   non-blocking stream, so nothing ordered them (random BAD_BUF / TIMEOUT). Fixed with a `cudaStreamSynchronize(0)`.
2. `start()` called `cudaMalloc` while a peer executor's persistent kernel was already waiting for it; `cudaMalloc` may wait for
   all kernels on the device (deadlock until the peer's timeout). Fixed by splitting `start()` into `prepare()` (allocations) and the
   launch; the harness prepares every executor before launching any. Without both fixes 2 of the first 2 plans failed; with them
   0 of 60 in a loop.

**Open item.** About 4% of runs of the in-process, single-GPU, several-executors stages still fail (about 10 of 230 runs across the loops and full runs after the two fixes), with
`CUDA_DEVICE_MAX_CONNECTIONS` and `CUDA_MODULE_LOADING` having no effect. Signature (from the watchdog and host timeline the
harness now prints): every rank thread waits in `finish()`, one executor's kernel started only when its peers' timeouts fired (its
event-measured kernel time is 0.14 ms), and the peers' pumps polled ~8 million times for signals that had not been sent. Not seen
with one process per rank; cannot occur with one device per rank. The script retries those stages (up to 3 attempts, every
failed attempt printed); this run needed none. The root cause was not established.

## Numbers worth keeping (A100, shared box, indicative)

- Cross-executor version hop, same GPU, peer pointers (publisher task end -> waiter task start, `%globaltimer`): rank0->1 median
  57 us (p90 203 us), rank1->0 median 78 us (p90 298 us). The executor's local dependent hop is 5.6 us
  (`results/2026-11-03-a100-exec`); the remote hop includes a 16 KiB tile copy, a system-scope fence, the admitter's inbox poll and
  the local wakeup. Not profiled or tuned.
- Round trip (two hops, two 16 KiB tiles, includes launch/submit/finish; resolution 1 ms over 100-200 rounds): ~0.5 ms peer
  pointers; ~1 ms through the host proxy with loopback transport and with TCP on loopback; ~20 ms with 10% drop / 10% dup / 20%
  delay (retransmit timeouts dominate).
- Not measured: NVLink/PCIe bandwidth, anything over a real network, anything on more than one GPU.
