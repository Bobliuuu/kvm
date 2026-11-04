# Scale-out without NCCL: multi-GPU and multi-node execution

The criticism this answers: *"It's single-GPU only. Flapping could train a trillion-parameter model with a megakernel without NCCL
across nodes and datacenters."* The honest answer has three parts. The design below is built. The single-box parts of it are
tested for real. The multi-GPU, multi-node and trillion-parameter parts are not run, and the last is only arithmetic.
Nothing in this repository trains a trillion-parameter model, or any model on more than one GPU, yet.

## What is built, what is tested, what is only modeled

| Piece | State | Evidence |
|---|---|---|
| Remote version variables (inbox + pump), ABI v2 unchanged | built, tested on one A100 (two executors in one process, same GPU) | `test_dist pingpong`, `errors` |
| Comm ops 64-71 (`put`, `put_signal`, `signal`, `reduce2`, `reduce_put_signal`, `send_proxy`, `pack_bf16`, `unpack_bf16`) | built, tested numerically (bitwise) on A100 | `test_dist ops`: 240 random tiles, reductions, bf16 |
| Data-before-signal ordering across executors | built; the checker is shown to fail when the order is broken | `-DKVM_DIST_BREAK_ORDER` negative run |
| Peer-to-peer over NVLink/PCIe (`cudaDeviceEnablePeerAccess`, peer pointers) | built, **untested on >1 physical GPU** (same-GPU "peer" = another pointer) | `multigpu-*` stages SKIP below 2 GPUs |
| Host proxy + reliable channel (ack/retransmit/ordering/windows/liveness) | built, tested on CPU and with the real device queue on A100 | `test_transport` (Mac and box), `gpu-plans-tcp-faults` |
| TCP transport (framing, CRC-32, reconnect-free, WAN buffers) with drop/dup/delay/corrupt injection | built, tested on loopback and between processes on one host | `test_transport`, `cpu-multiprocess-tcp`, `gpu-multiprocess-tcp` |
| RDMA / GPUDirect transport | **not built**; documented stub with a TODO list | `dist/transport_rdma.hpp` |
| Partitioner (DP, ZeRO/FSDP, Megatron TP, all-to-all, pipeline + microbatches, HEFT) | built; verified on the CPU simulator against the unpartitioned VM; plans also run on the A100 | `tests/dist/test_partition.py` (29 tests), `gpu-plans-*` |
| Multi-node, multi-datacenter runs | **not run**; launcher exists and dry-runs | `dist/launch.py`, `multinode-tcp-plans` (needs `KVM_NODES`) |
| NVLink / PCIe P2P bandwidth numbers | **not measured** (one GPU only) | `multigpu-p2p-bandwidth` |
| Overlap of communication with compute at tile granularity | partly: pipelined microbatches and chunked collectives are built; row-tile splitting of compute ops needs offset-capable tile ops | modeled only, see below |
| Expert-parallel routing with data-dependent token placement | **not built**; the static-capacity all-to-all is | `ROWS <-> COLS` resharding |
| Trillion-parameter training | **not built, not run, not claimed**; section "Scaling math" is arithmetic | `dist/scaling.py` |

## 1. Distributed model

N devices each run their own executor instance (one persistent kernel) over their shard of the task graph. A *version variable*
is a monotone counter owned by one device. The executor's existing protocol (a task waits for `version >= n`; a publisher
stores the next version after its writes; parked waiters are woken by the publisher) is already a distributed-memory protocol
if "publish" can happen across a link. So no new scheduler concept was added:

* **ABI v2 is untouched.** A consumer is an ordinary task with `wait_var/wait_ver`. A *remote* variable is just a variable id in
  the consumer's own table that is fed from outside. A remote buffer is a `KvmBuf` entry whose pointer is valid on this device
  (a peer pointer, or on one GPU another allocation). Producer-side addressing uses typed op arguments: a `flag` buffer (the
  target's inbox) plus the target's variable id. Variable ids are therefore namespaced by owner device: `(inbox of device d, var)`.
  The one extension to shared structures is a trailing `const KvmDistDev* dist` in `KvmDevState`, null on a single device.
* **Inbox.** `u32[num_vars]` in the owner's device memory. A producer on device A finishes its writes, then does
  `fence.sc.sys; st.release.sys inbox[var] = ver` into the owner's inbox through a peer pointer. Each inbox slot is a
  single-writer chain with strictly increasing versions (the partitioner guarantees it; a test checks it), so a plain release
  store suffices and no PCIe atomics are needed.
* **Pump.** The owner's admitter block (block 0, idle between bundles) polls `KvmDistDev::in_vars`; when `inbox[v] > version[v]`
  it does the local publish (`fence`, `st.release version[v]`) and wakes the parked tasks exactly like `kvm_publish`. After the
  last bundle the admitter keeps pumping until every admitted task is done. Workers never poll, and a remote waiter occupies no
  block (it is parked like any waiter).
* **Why one-shot variables.** Every cross-device landing buffer is written once (fresh, single assignment) and announced by its own
  variable at version 1. No remote write can race a local reader, and no credit/flow-control protocol is needed between
  devices at the DAG level. The cost is memory for landing buffers; recycling them with credit variables (consumer signals "free")
  is the obvious extension and is **not built**.

### Memory-ordering argument (system scope)

Producer task T on device A: all threads store tile data into the consumer's memory; `__syncthreads()`; thread 0:
`fence.sc.sys`, then `st.release.sys inbox[var]`. `bar.sync` plus a system-scope fence is cumulative over every thread's stores,
the same argument the executor already makes at gpu scope. Owner poller: `ld.acquire.sys inbox[var]` synchronizes with that
release, then `fence.sc.gpu; st.release.gpu version[var]`. A waiter's `ld.acquire.gpu version[var]` followed by `bar.sync`
observes it, and causality order is transitive across the two scopes, so a waiter that sees `ver` sees all of T's data. The fence
is what drains posted writes over NVLink/PCIe; it is not an optimisation target. Host notes: x86-64 (TSO) needs the compiler
and acquire/release ordering the C++ atomics already give; aarch64 (Grace) needs the hardware acquire/release `std::atomic`
emits, which is why the proxy never uses `volatile` for synchronisation. `-DKVM_DIST_BREAK_ORDER` publishes before the data
(plus a 200 us delay) and the harness must report stale reads: it does.

## 2. Comm ops (`include/kvm_ops_comm.def`, ids 64-95)

Ordinary registered ops, one X-line each, parsed by `runtime/abi.py` (merged in by `dist/commops.py`) and implemented in
`include/kvm_ops_comm.cuh`. Tiles are row-major `rows x cols` with leading dimensions, 16-byte vectorised when aligned.

| op | meaning |
|---|---|
| `put` | tile copy; with a peer pointer it is a device-to-device copy |
| `put_signal` | tile copy, then `kvm_remote_publish` of (var, ver) into the target inbox |
| `signal` | bare remote publish (cross-device version variables) |
| `reduce2`, `reduce_put_signal` | fixed-order add; fused add-and-forward for ring reduce-scatter |
| `send_proxy` | tile into the host send queue (mapped pinned memory), slot published with `st.release.sys`; the proxy moves it |
| `pack_bf16`, `unpack_bf16` | wire compression (round to nearest even), optional accumulate |

Collectives are not ops. The partitioner decomposes them into these tasks in the DAG: ring reduce-scatter (the chunk-`c` sum is
`x[c+1] + x[c+2] + ... + x[c]` in that order on every run, so results are deterministic and the same on all devices), ring and
direct all-gather, ring / binary-tree / direct all-reduce (results are copied, never recomputed per device, so replicas stay
bitwise identical), and all-to-all as `P(P-1)` tile puts. `chunks=k` cuts a collective into k independent rings whose landings
are tracked separately. Misuse (tile out of range, bad signal slot, wrong queue, oversized payload, bad destination) ends the
kernel with a decoded device error (`KVM_ERR_DIST_*`, codes 64-70), never undefined behaviour.

## 3. Transport

* **Intra-node:** CUDA peer access. The device that owns the data is never involved; the producer's block stores through the peer
  pointer. Pairs that cannot access each other fall back to the proxy path.
* **Inter-node:** one host *proxy thread* per device drains the device-written send queue (multi-producer/single-consumer ring in
  mapped pinned memory, ticket order, sequence numbers published with release stores; the ticket counter lives in device memory
  because atomics on pinned host memory need PCIe AtomicOps, which x86 hosts do not report: `HostNativeAtomicSupported=0` on the A100 box) and moves messages over a pluggable
  `Transport` (`dist/transport.hpp`): `send(dst, bytes)`, `poll(event)`, events `RECV` and `PEER_DOWN`, allowed to be lossy.
  Implemented: loopback (tests), **TCP** (portable, any host, no libraries; framing + CRC-32, bad frames dropped and counted,
  `SO_SNDBUF/RCVBUF` for long fat links), and a **fault injector** (drop, duplicate, delay/reorder, plus corruption inside TCP).
  `RdmaTransport` is a documented stub: what ibverbs and GPUDirect-RDMA need, in order, is listed in `dist/transport_rdma.hpp`.
* **Reliability above any transport** (`dist/reliable.hpp`): per-peer sequence numbers, exactly-once in-order delivery
  (deterministic per source-destination pair), cumulative acks, retransmit with backoff, and a **send window** (messages and
  bytes). The window is the pipeline depth: for a WAN link set `window_bytes >= bandwidth x RTT` and use large chunks
  (the send-queue slot size). A full window makes the proxy stop draining, the queue fills, and the producing task waits on the
  device with a timeout (`DIST_SENDQ_TIMEOUT`): nothing buffers without bound.
* **Receive:** `cudaMemcpyAsync` of the tile, then a 4-byte copy of the version into the inbox, in one stream (data completes
  before the signal starts); the pump reads the inbox with `ld.acquire.sys`.
* **Failure model.** Retransmit limit, silence past `dead_ms`, a lost connection or a refused send mark a peer dead; the proxy
  writes `KVM_ERR_DIST_PEER_DOWN` into the control block and the pump fails the device. A stalled proxy stops heartbeating
  and the pump raises `DIST_PROXY_DEAD`. A wait that nobody satisfies ends with the executor's own `TIMEOUT`. Shutdown
  is a FIN exchange over the same ordered stream, so no rank leaves while a peer may still deliver to it, and it ends at the
  deadline when a peer is dead. There is no checkpoint/restart: a failure ends the run with a code.

## 4. Partitioner (`runtime/partition.py`)

Input: an ABI-v2 workload, a cluster (devices, nodes, link kind/bandwidth/latency) and a strategy. Output: one ordinary
sub-workload per device (compilable by `compile_workload.py`; the bytes are what the GPU receives), plus the table of off-device
buffers and the inbox variables.

* **`shard` mode** (layout algebra). Every buffer has a layout: replicated, `ROWS`/`COLS` blocks, or `PART` (per-device addends).
  Ops have rules (elementwise, `matmul`: `(ROWS,R)->ROWS`, `(R,COLS)->COLS`, `(COLS,ROWS)->PART`, `(R,R)->R`, `mse_grad`), and the
  planner inserts the cheapest resharding (volume-based choice) where operands do not match. Data parallel = `x` rows; ZeRO/FSDP =
  parameters rows (gathered per use, gradients reduce-scattered); Megatron tensor parallel = `W1` cols, `W2` rows; sequence/expert
  style all-to-all = `ROWS <-> COLS`.
* **`place` mode.** Explicit task-to-device map, balanced pipeline stages (linear-partition DP on the cost model), or HEFT.
  Buffers follow a coherence rule (read elsewhere = one cached transfer; write invalidates). `microbatches` clones the
  per-microbatch part of the graph; weights are shared, partial gradients are combined in microbatch order. Cross-device
  `at_least` reads of a published variable get a mirrored inbox variable and a `signal` chain; cross-device `exact` reads are refused.
* **Dependencies** come from replaying program order with RAW/WAR/WAW tracking per buffer row range, a superset of the author's
  `after` edges. The compiler's checks (acyclic, bounds, versions) run on every shard.
* **Cost model** (`op_cost_us`, `Cluster`): roofline-ish compute plus the executor's measured dispatch cost (1.4 us per task,
  from `results/2026-11-03-a100-exec`), links as latency + bytes/bandwidth. `estimate()` is a discrete-event simulation of the
  executor's dispatch rule. It is a model; its numbers have not been compared with a multi-device run.
* **Verification** (`tests/dist/sim.py`, `verify.py`): all shards run in a multi-device CPU simulator under random
  interleavings, including "data landed, version not yet visible" and per-pair message order on proxy links. Integer-valued
  workloads match the unpartitioned CPU VM **bitwise**, for every strategy listed above; the floating-point training loop within 1e-5.
  The simulator is shown to fail when `put_signal` publishes first or a signal is missing.

## 5. Using it

```bash
# partition: workload + cluster + strategy -> shard0.json ... and (with --plan) per-rank bundles for the GPU harness
python3 runtime/partition.py workload.json cluster.json strategy.json out/ --plan
# run every rank as a thread of one process (ranks spread over the GPUs that exist), or one OS process per rank / per host
build/test_dist plan out/plan [--net tcp --fault drop=0.05,dup=0.05,delay=0.1]
python3 dist/launch.py --nodes "$KVM_NODES" -- build/test_dist plan out/plan --rank {rank} --peers {peers}
```

`cluster.json`: `{"devices": [{"node": 0}, ...], "links": {"intra": {"kind": "p2p", "bandwidth_gbs": 300, "latency_us": 2},
"inter": {"kind": "proxy", "bandwidth_gbs": 12.5, "latency_us": 30}}, "tflops": 19.5}`. `strategy.json`: `{"mode": "shard",
"layouts": {"W1": "cols", "W2": "rows", "x": "rows"}, "collective": "ring", "chunks": 2, "wire": "f32"}` or `{"mode": "place",
"auto": "pipeline", "stages": 4, "microbatches": 8, "split": ["x", "h0"], "accumulate": {"grad": "mean"}}`. Python API:
`partition.partition(spec, Cluster.uniform(8, nodes=2), strategy)` returns a `Plan` with `.shards`, `reconstruct()` and
`estimate(plan)` (the modeled makespan).

## 6. Tests

`dist/run_multigpu_tests.sh` is the single command. It prints one line per stage (PASS / FAIL / SKIP), bounds every stage with a
timeout, exits 1 on any FAIL, and `--dry-run` lists everything without running it. Stages that need more than one physical GPU
print `SKIP: needs 2 GPUs` below that and run automatically otherwise; the multi-node stage runs when `KVM_NODES` is set.

| Stage | Needs | Proves |
|---|---|---|
| `cpu-partitioner-sim` | python3 | every strategy equals the unpartitioned run (bitwise on integers), under random interleavings; 60 random layout/topology/collective combinations and random placements; the simulator can fail |
| `cpu-transport-proxy` | g++ | CRC, exactly-once in-order delivery under drop/dup/delay/corruption, windows, latency hiding by window depth, dead peers, the proxy and send-queue protocol with threads as producers, 3 ranks (also clean under TSAN) |
| `cpu-launcher-dry-run`, `cpu-multiprocess-tcp` | g++ | `launch.py` output, and the same reliable channel over TCP between 3 OS processes with faults |
| `gpu-build`, `gpu-executor-regression` | nvcc + GPU | builds (incl. the two negative builds); the executor's own smoke suite still passes with the hooks in |
| `gpu-ops` | 1 GPU | comm ops bitwise vs CPU (random strided tiles on both copy paths, reductions, bf16), 10 misuse cases each ending in the right device error |
| `gpu-pingpong-versions` | 1 GPU | 200 round trips between two executors (p2p pointers; proxy; proxy over TCP; proxy over TCP with faults), every value checked; cross-executor hop latency |
| `gpu-errors-no-hang` | 1 GPU | unsatisfiable remote wait, full queue without proxy, silent proxy, dead peer, and a full live window waiting for remote signals: each ends with a decoded code or completes |
| `gpu-negative-break-order` | 1 GPU | the harness detects a signal that overtakes its data, and a pump removed from admission backpressure |
| `gpu-sanitizer` | 1 GPU | compute-sanitizer memcheck, racecheck, synccheck on the ops suite (the two-executor suites cannot run under it: it serialises launches, and two persistent kernels waiting on each other would deadlock by construction) |
| `gpu-plans-in-process`, `gpu-plans-tcp-faults`, `gpu-multiprocess-tcp` | 1 GPU | partitioner output on the GPU against golden: all ranks as threads, over TCP with faults, and one OS process per rank |
| `multigpu-p2p-bandwidth`, `multigpu-p2p-pingpong`, `multigpu-plans-one-rank-per-gpu` | **2+ GPUs** | real peer-to-peer stores, NVLink/PCIe bandwidth, ranks on distinct GPUs |
| `multinode-tcp-plans` | **`KVM_NODES`** | ranks on several hosts over TCP (sources rsynced to `~/kvm-dist-run`, built on every node) |

### Measured so far (one A100 shared with other jobs; `results/2026-11-03-dist-singlebox/`)

* Cross-executor version hop on one GPU with peer pointers (publisher task end to waiter task start): median 58-71 us, p90 208-332 us.
  The executor's local dependent hop is 5.6 us, so a remote hop costs roughly ten times a local one here; it includes the tile
  copy, a system-scope fence, the admitter's poll and the wakeup. Not yet profiled or tuned.
* Through the host proxy a round trip of two hops and two 16 KiB tiles takes about 0.5 ms (loopback transport) and 0.75 ms (TCP on
  loopback); with 10% drop, 10% duplicates and 20% delay it is about 19 ms (retransmit timeouts).
* Nothing measured over a real network or between physical GPUs.

## 7. Scaling math (arithmetic, not measurement)

`python3 dist/scaling.py` regenerates this; every assumption is a flag. Assumptions: 1.0e12 parameters, 128 layers, hidden 25600,
sequence 4096, global batch 4096 sequences (16.8M tokens), 4096 GPUs = TP 8 x PP 16 x DP 32, 64 microbatches, H100-class peak
989 bf16 TFLOP/s at 40% MFU, 50 GB/s of NIC per GPU, 450 GB/s NVLink per GPU. None of these is a measurement of anything here.

* **State.** bf16 weights 2 + fp32 master 4 + Adam m,v 8 + bf16 gradients 2 = **16 B/param = 16 TB**. ZeRO-3 over DP on top of
  TP x PP stores every byte once: **3.9 GB/GPU**.
* **Per-GPU memory.** state 3.9 GB + gathered weights (two layers prefetched: 3.9 GB) + stored layer inputs under full recompute with
  sequence parallelism (3.4 GB) + recompute workspace (0.45 GB) = **11.6 GB**; **23.3 GB** if the whole stage stays gathered for
  the step instead. Fits an 80 GB part with room; it is not the constraint.
* **Per-step communication per GPU.** ZeRO-3: gather forward 15.1 GB + gather backward 15.1 GB + gradient reduce-scatter 15.1 GB
  if weights are gathered once per step; re-gathering for every microbatch would be about 1950 GB. Tensor parallel about 940 GB
  (NVLink). Pipeline point-to-point 3.4 GB.
* **Overlap budget.** Step compute is about 83 s at the assumed MFU. ZeRO-3 traffic needs **0.55 GB/s** per GPU to hide (the NIC
  provides 50), so it is 1.1% of compute if gathered once per step, 47% if re-gathered per microbatch: that is the real design
  choice. Tensor-parallel traffic is 2.5% of compute on NVLink. The pipeline bubble is 19% at 64 microbatches (48% at 16).
* **Across datacenters.** Put the datacenter boundary on the *pipeline* axis, not the data-parallel axis: a stage boundary carries
  859 GB per step = about 83 Gb/s aggregate, and a 15 ms hop is 1.2% of one microbatch's time. Putting it on the DP axis would need
  every GPU's 45 GB per step across the WAN.
* **What this does not cover:** the 4096-rank collective latency tail, failures at that scale (MTBF of thousands of GPUs is
  hours; there is no checkpoint/restart here), the numerics of a 1T-parameter run, the dispatch cost of the megakernel at
  these tile counts, and the fact that a 1.4 us/task executor has so far only been shown to win on small, dependent work
  (see `AGENTS.md`). Whether a persistent-kernel runtime beats NCCL-based stacks at this scale is unanswered.

## 8. Known limits

* Landing buffers are single-assignment; long unrolled runs grow memory (credits not built).
* The proxy path stages through pinned host memory (no GPUDirect RDMA); one proxy thread per device.
* `matmul` needs exact shapes; ops cover whole buffers when sharded; `matmul` row tiling needs offset-capable tile ops.
* Cross-device `exact` snapshot reads are refused (use `at_least` plus the hazard ordering the planner derives).
* bf16 wire format: all-gather of `ROWS` weights and place-mode transfers; collective reductions are fp32.
* `KvmExec` takes at least `num_vars` slots in the inbox; plans with thousands of one-shot variables need `num_vars` raised.
* x86-64 and aarch64 little-endian hosts only (wire structs are not byte-swapped).
