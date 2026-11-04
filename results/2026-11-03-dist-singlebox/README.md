# Distributed execution on ONE A100 (2026-11-03 local, 2026-11-04 00:58 UTC)

Everything that can be exercised without a second GPU, plus CPU stages. The stages that need 2+ GPUs or several hosts were
**skipped by the script, not run**; the table says which. Nothing here shows multi-GPU, NVLink, multi-node or WAN behaviour.

- GPU: NVIDIA A100-SXM4-40GB, 108 SMs, sm_80, driver 580.105.08, nvcc 12.8.93 (the box was shared: other users' jobs ran on the
  GPU and CPU during the run, so latencies are indicative). Host: x86-64 Linux, `cudaDevAttrHostNativeAtomicSupported = 0`.
- Commit: 6167fad (branch feat/multigpu, clean tree, the sources were rsynced from it). Later commits changed only the test harness
  (stricter negative tests, a diagnosis line, the lock taken once per run) and docs.
- Command (from a laptop; `KVM_HOST` is an environment variable, never committed): `KVM_HOST=<user@gpu-box> dist/run_multigpu_tests.sh --remote`.
  On the box itself it is `dist/run_multigpu_tests.sh`. Stage timeout 600 s, GPU stages hold `~/.gpu.lock`.
- Build flags (GPU stages): `nvcc -O3 -arch=sm_80 -std=c++17 -Iinclude --fmad=false -Xcompiler -ffp-contract=off -lineinfo -DKVM_EXEC_NO_MAIN`
  (`--fmad=false` so device floats compare bitwise with the CPU reference); negative builds add `-DKVM_DIST_BREAK_ORDER` /
  `-DKVM_DIST_NO_BP_POLL`.

## Result: RESULT: PASS (14 passed, 0 failed, 4 skipped)

| Stage | Result | What it covered |
|---|---|---|
| cpu-partitioner-sim | PASS (32 tests) | all strategies bitwise vs the unpartitioned CPU VM under random interleavings; fuzz; simulator can fail |
| cpu-transport-proxy | PASS (90 checks) | CRC, exactly-once in-order delivery under drop/dup/delay/corruption, windows (window 1 / 8 / 64 over an emulated 10 ms RTT: 596 / 75 / 5 ms for 60 messages), dead peers, proxy + send-queue protocol |
| cpu-launcher-dry-run, cpu-multiprocess-tcp | PASS | launcher output; 3 OS processes over TCP with 10% drop, 10% dup, 20% delay |
| gpu-build | PASS | test_dist, the two negative builds, test_exec |
| gpu-executor-regression | PASS | executor `smoke` suite with the hooks compiled in |
| gpu-ops | PASS (148 checks) | 240 random strided tiles (scalar and 16-byte paths) bitwise; reduce2 / reduce_put_signal bitwise; 1001 bf16 values bitwise; 10 misuse cases -> decoded device errors |
| gpu-pingpong-versions | PASS (20 checks) | 200 rounds between two executors with peer pointers; 100 rounds each over the host proxy (loopback, TCP) and 60 over TCP with 10% drop / 10% dup / 20% delay; every one of 819,200 values checked |
| gpu-errors-no-hang | PASS (7 + 2 checks) | unsatisfiable remote wait -> TIMEOUT after 0.8 s; full send queue without a proxy -> DIST_SENDQ_TIMEOUT after 0.3 s; silent proxy -> DIST_PROXY_DEAD after 0.3 s; killed peer -> DIST_PEER_DOWN within milliseconds; full live window waiting for remote signals completes (pump during backpressure) |
| gpu-negative-break-order | PASS, with a caveat | the stage then counted ANY failure as detection. In this run the break-order build (signal published before the data) ended with a device TIMEOUT on both ranks, i.e. the two kernels did not make progress together on the shared GPU, not with stale data; the backpressure build (pump removed) timed out as intended. Stale-data detection itself: 8 of 8 repeats of the same binary, run by hand right after, reported 483,328 to 516,096 of 819,200 values wrong, and the previous full run (commit 958f253) reported 504,320. The stage was tightened afterwards (`--expect-fail=stale` / `=timeout`: it must fail in the expected way); that version has not been re-run to completion because the GPU lock stayed busy |
| gpu-sanitizer | PASS | compute-sanitizer memcheck 0 errors, racecheck 0 hazards, synccheck 0 errors on the ops suite (not on the two-executor suites: the sanitizer serialises launches) |
| gpu-plans-in-process | PASS | 9 partitioner plans (TP direct/ring/tree, ZeRO mixed p2p+proxy, ZeRO bf16 wire, DP floating point within 1e-5, pipeline x4 microbatches) run with all ranks as threads on the one GPU: every expected value bitwise equal |
| gpu-plans-tcp-faults | PASS | 4 plans over TCP between rank threads with 5% drop, 5% dup, 10% delay: bitwise equal |
| gpu-multiprocess-tcp | PASS | 3 plans with one OS process per rank (4 processes sharing the GPU), TCP between them: bitwise equal |
| multigpu-p2p-bandwidth | SKIP: needs 2 GPUs (found 1) | not measured |
| multigpu-p2p-pingpong | SKIP: needs 2 GPUs | peer-to-peer across physical GPUs untested |
| multigpu-plans-one-rank-per-gpu | SKIP: needs 2 GPUs | not run |
| multinode-tcp-plans | SKIP: needs KVM_NODES with 2+ nodes | not run |

## Numbers worth keeping (A100, shared box, indicative)

- Cross-executor version hop, same GPU, peer pointers (publisher task end -> waiter task start, `%globaltimer`): rank0->1 median
  58 us (p90 208 us), rank1->0 median 71 us (p90 332 us). The executor's local dependent hop is 5.6 us (`results/2026-11-03-a100-exec`);
  the remote hop includes a 16 KiB tile copy, a system-scope fence, the admitter's inbox poll and the local wakeup.
- Round trip (two hops, two 16 KiB tiles, includes launch/submit/finish): 250 us peer pointers; 500 us through the host proxy
  with loopback transport; 750 us through the proxy over TCP loopback; 18.7 ms with 10% drop / 10% dup / 20% delay (retransmit
  timeouts dominate).
- Not measured: NVLink/PCIe bandwidth, anything over a real network, anything on more than one GPU.
