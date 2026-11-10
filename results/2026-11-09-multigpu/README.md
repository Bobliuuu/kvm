# KVM on more than one GPU, and across the internet (2026-11-09/10 UTC)

Every multi-GPU and multi-node stage of `dist/run_multigpu_tests.sh` ran for the first time on hardware that has them. Before this run
nothing in the repository had used a second physical GPU, NVLink or a network (`results/2026-11-03-dist-singlebox`: 4 stages SKIP).

**Hardware (Vast.ai, rented for this run, destroyed afterwards; hosts and ports are not recorded):**

| Name here | What | Used for |
|---|---|---|
| box D | 2x NVIDIA GeForce RTX 3090 (sm_86, 82 SMs, 24 GB), **NVLink bridge NV4** (4 links x 14.06 GB/s per direction), driver 615.78.08, CUDA 12.8, 96 cores, US (Ohio) | every GPU stage, multi-GPU stages, model plans, scaling |
| node B | 1x RTX 3090, 32 cores, Romania | second node for the multi-node runs; build host |
| (A) | 2x A100-SXM4-40GB, NV12 | lost: the host went offline after ~40 min, before any GPU stage finished; no result is from it |

Build: `nvcc -O3 -arch=sm_86 -std=c++17 --fmad=false -Xcompiler -ffp-contract=off -lineinfo -DKVM_EXEC_NO_MAIN -DKVM_MMA_NST=3 -DKVM_GPT_MMA_SMEM=73728`.
sm_86 needs the 3-stage bf16 MMA variant: its opt-in shared memory is 99 KiB per block (sm_80: 163 KiB), and the default 96 KiB of
dynamic plus the kernel's static shared memory does not fit (`cudaFuncSetAttribute ... invalid argument`); the suite now adds those two
flags for sm_86/87/89/120. Commit: branch `feat/multigpu-real` (this PR). Raw logs are kept outside the repo; the `.txt` files here are
extracts with addresses and ports removed.

## Result: every stage that hardware allows PASSES

`dist/run_multigpu_tests.sh` on box D (`KVM_PREBUILT=1`, binaries built by the same flags): **23 passed, 0 failed** (after the RDMA
guard fix), plus the multi-node stage launched from a separate host with `KVM_NODES` (box D x2 ranks + node B x2 ranks), `KVM_ADVERTISE`
and `KVM_LISTEN_BASE` taken from the environment: **PASS**. 25 stages, 24 PASS, 1 SKIP that the hardware cannot satisfy:

| Stage | Result | What it measured here |
|---|---|---|
| cpu-* (9 stages, incl. the new `cpu-rdma-ibverbs-build`) | PASS | as before, plus: the real libibverbs backend (`IbvVerbs`, never compiled before) compiles and links against rdma-core and its shim-backed tests pass |
| gpu-build, gpu-executor-regression, gpu-ops, gpu-pingpong-versions, gpu-errors-no-hang, gpu-negative-break-order, gpu-sanitizer, gpu-plans-in-process, gpu-plans-tcp-faults, gpu-multiprocess-tcp | PASS | the single-GPU suite on an RTX 3090 (first non-A100 run of it) |
| **multigpu-p2p-bandwidth** | PASS | peer stores over NVLink: **48.3 GB/s** GPU0->1, **39.2 GB/s** GPU1->0 (256 MiB, 82 blocks; NV4 nominal 56 GB/s); same-device HBM read+write 110 GB/s |
| **multigpu-p2p-pingpong** | PASS | 200 round trips between executors on different GPUs, every value checked; round trip on rank 0's clock (two version hops + a 16 KiB tile copy): **median 111.6 us** (p90 266) with peer pointers, 382 us through the host proxy, 591 us proxy over TCP, 630 us TCP with 10% drop / 10% dup / 20% delay (p90 56.7 ms: retransmit timeouts) |
| **multigpu-plans-one-rank-per-gpu** | PASS | tensor-parallel, ZeRO, pipeline, ring and mixed p2p/proxy plans with ranks on distinct GPUs: every value bitwise equal to the unpartitioned CPU run |
| **multigpu-model-plans** (new) | PASS | frontend models partitioned over the 2 GPUs, table below |
| **multinode-tcp-plans** | PASS | 5 proxy plans (ZeRO f32 and bf16 wire, ring TP, DP float, 4-stage pipeline) x 4 ranks on two hosts in two countries over TCP: every value bitwise equal |
| rdma-device-present (new) | SKIP | **impossible on this hardware**: the hosts have Mellanox NICs (mlx5 on box A, mlx4 on box D, visible in sysfs) but no `/dev/infiniband/uverbs*` node is passed into the containers, and they lack `CAP_NET_ADMIN`, so Soft-RoCE (`rdma link add ... type rxe`) is refused (`Operation not permitted`). The first full run reported this stage FAIL because its guard looked at sysfs; fixed, rerun = SKIP. The RDMA transport therefore ran only against the in-process shim. |

Extracts: `suite_2gpu.txt` (every 2-GPU stage), `multinode.txt` (every multi-node run).

## Frontend models on two GPUs (`dist/run_model_plans.sh`, new)

New partitioner support (`runtime/partition.py`): `place` mode now works on GPT/Llama/ViT training graphs, and a new `dp` mode does
data parallelism for any training workload (see DISTRIBUTED.md section 4). Each run below has a CPU golden; the "1-GPU reference" is the
same reference program run unpartitioned on one GPU through the same harness, so the two numbers show whether the partition adds error.
GEMMs 3xTF32 (`--precise`), 2 AdamW steps. Error = max over elements of |gpu - golden| / max(1, |golden|) (the harness's own measure).

| Plan | Ranks | Compared values | Worst error vs CPU golden | 1-GPU reference, same golden | Replicas |
|---|---|---|---|---|---|
| gpt_tiny, dp x2 (golden: accum=2 on one device) | 2 GPUs | 101,184 | 2.86e-6 | 2.86e-6 | bit-identical |
| gpt_tiny, dp x4 | 4 on 2 GPUs (41 blocks each) | 202,368 | 3.66e-4 | 3.66e-4 | bit-identical |
| llama_tiny, dp x2 | 2 GPUs | 419,712 | 9.83e-5 | 9.83e-5 | bit-identical |
| gpt_small, dp x2 | 2 GPUs | 698,880 | 1.24e-4 | 1.24e-4 | bit-identical |
| gpt_tiny, 2-stage pipeline (golden: the unpartitioned program) | 2 GPUs | 171,902 | 8.23e-6 | 8.23e-6 | - |
| llama_tiny, 2-stage pipeline | 2 GPUs | 574,608 | 1.74e-3 | 1.74e-3 | - |
| gpt_small, 2-stage pipeline | 2 GPUs | 1,837,490 | 1.63e-4 | 1.63e-4 | - |

In every row the partitioned run is exactly as far from the golden as one GPU running the reference program (for the pipelines even
the count of bitwise-equal values is the same: 24,616 / 30,972 / 87,473), so the partition adds nothing measurable; the error is GPU-vs-
CPU float noise amplified by AdamW (largest for llama_tiny, whose 1-GPU run has the same 1.74e-3). In the CPU simulator all pipelines are
**bitwise** equal to the unpartitioned program (`tests/dist/test_model_plans.py`). This table is the suite stage `multigpu-model-plans`
(`model_plans.txt`).

Across the internet (box D in the US + node B in Europe, every gradient message through the host proxy and TCP):

| Plan | Ranks | Worst error vs CPU golden | Kernel time |
|---|---|---|---|
| gpt_tiny dp x2 | 1 per node | 2.86e-6 (= the 1-GPU reference) | 0.87-0.97 s (2 steps, 40 sequential all-reduces, WAN round trips) |
| gpt_tiny dp x4 | 2 per node | 3.66e-4 (= the 1-GPU reference) | 0.84-0.97 s |

## Data-parallel speedup on two NVLinked GPUs

Same global batch both ways: DP x2 (each GPU runs batch b) against the same model compiled with gradient accumulation over 2
microbatches of b on ONE GPU, through the same harness (`--blocks 82`, kernel time from the harness), state dumped and compared
(`tests/dist/compare_dumps.py`).

| Model | Tasks per rank (reference) | Gradient all-reduces | DP x2 kernel | 1 GPU, accum=2 | Speedup | State vs reference |
|---|---|---|---|---|---|---|
| Llama-style d512, 6 layers, 8 heads, seq 256, batch 4/rank, vocab 1024, 3 AdamW steps, 3xTF32 | 34,797 (67,605) | 171 (134 MiB) | 71 ms | 121 ms | **1.70x** | replicas bit-identical; worst element 2.0e-5 of max over 171 buffers |
| **gpt_162m** (d768, 12 layers, seq 1024, batch 4/rank, vocab 50,304), 1 AdamW step, 1xTF32, a100 tiling | 274,450 (544,590) | 196 (1.24 GB) | 685 / 679 ms | 1,208 ms | **1.76x** | **bitwise identical** to the reference in all 588 state buffers (weights, m, v); replicas bit-identical |

At 162M the two ranks process 8,192 tokens per step in 0.685 s (about 12,000 tokens/s) against 6,780 tokens/s for one GPU. The 2-rank
speedup is below 2 because each rank still pays the executor's per-task cost for its whole graph (274k tasks) plus 1.24 GB of NVLink
all-reduce traffic, and the reference executes the dispatch of both microbatches back to back. The bitwise match at 162M follows from the
construction: scaling by 1/2 is exact, and a 2-rank reduction adds the two halves in the same order the accumulation does. With 3 steps
and 3xTF32 (row above) the tile-level accumulation order differs and the state agrees to 2e-5 instead. This is one step of one model on
one 2-GPU box: not a scaling curve, and no comparison with NCCL/DDP was made.

Running 162M needed harness capacity, not executor changes: the whole rank graph is one bundle record, so `KVM_DIST_LIVE_CAP` (2^19 / 2^20),
`KVM_DIST_SUCC_CAP` (2^25 successor words) and `KVM_DIST_RING_WORDS` (2^26) are raised from the defaults (the first attempts stopped
with `TOO_BIG` and `POOL`). Device buffers per rank: 8.05 GB (dp), 7.40 GB (reference).

## Multi-node TCP across two countries

The reliable channel (`dist/reliable.hpp`) over real TCP between box D and node B (about 140 ms to connect), `test_transport proc`:
500 messages each way exactly once and in order, clean and with 10% drop / 10% dup / 20% delay injected (retransmits 1,569-1,954 because
the test's RTO is 20 ms, far below the WAN RTT; correctness does not depend on it).

Containers sit behind port maps: a rank binds port `KVM_DIST_LISTEN_PORT` inside its container while peers dial the advertised
`host:port` (`dist/launch.py --advertise` / `KVM_ADVERTISE`, `--listen-base` / `KVM_LISTEN_BASE`). Ranks on the same node dial each other
on loopback: a container cannot reach its own public mapped port (no hairpin NAT), which is what failed the first 4-rank attempt
(rank 0 never connected to rank 1, the peer timeout fired: `DIST_PEER_DOWN`).

## Found and fixed on the way

* `place` mode partitioned GPT graphs into stale copies: it only knew the written arguments of six toy ops. `OP_WRITES` (measured,
  re-derived by `tests/dist/test_op_writes.py`) fixes it; control-only `after` edges (barriers that publish the weights version) were
  dropped and are now kept; dtypes were lost (i32 step counters became f32 buffers).
* i32 buffers could not cross devices: the device's `put`/`put_signal`/`send_proxy` take f32 views (`BAD_BUF`). New comm ops
  `put_i32`, `put_signal_i32`, `send_proxy_i32` (ids 72-74) move them bit-exactly; the planner picks them by dtype.
* Ranks sharing a GPU must split its SMs: two persistent executors that each ask for all 82 blocks cannot be co-resident and
  deadlock (diagnosed by the harness as a lost wakeup). `run_model_plans.sh` gives each rank SMs / ranks-per-GPU blocks.
* The never-run `multinode-tcp-plans` stage passed with zero checks: it passed every word of `make_plans --print-proxy-only` output as
  a plan name, and `test_dist plan` exited 0 on a missing plan directory. Both fixed; the rerun compares every value.
* The suite's build stage built four full-op-set binaries serially under the 600 s stage timeout (each takes 15-25 min); it now
  builds them in parallel, at most MemAvailable / 48 GB at a time (two builds were OOM-killed at 4 x ~45 GB), with its own timeout.

## Not shown (read before quoting)

* **No RDMA.** No RDMA device was reachable (see the SKIP above). `IbvVerbs` is now compiled and linked against real rdma-core, but it
  has never touched a NIC; GPUDirect RDMA remains design notes.
* **No 4-8 GPU box.** The only NVLink offers inside the budget were 2-GPU boxes (V100 boxes are sm_70, below the sm_80 floor of the op
  set; 4-8x A100 SXM4 cost 6.4 $/h, over the 3 $/h cap). Two PCIe 4-GPU boxes were rented for a 4-rank scaling run and never became
  usable (one stuck for 10+ min in its image's package step, one never left `loading`); both were destroyed without a result. The 4-rank
  runs here put 2 ranks on each of 2 GPUs or span two hosts, so there is no scaling curve beyond 2 GPUs.
* The MoE family is excluded from the dp-equals-accumulation claim: the frontend builds the balance loss with a fixed coefficient, not
  scaled by 1/accum, so accum=P sums the aux gradient where dp averages it (a frontend inconsistency, reported, not changed here).
* Kernel times come from the harness's host timer (1 ms resolution); the multi-node times are dominated by WAN round trips, not compute.
