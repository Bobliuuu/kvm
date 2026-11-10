# Scale evidence for F10a/F10b (2026-11-06), branch feat/scale-evidence

Legend: **MEASURED** = ran on hardware; **CHECKED** = exact equality against the real partitioner's output on CPU; **MODELLED** = arithmetic
or cost model, never a measurement.

## What exists now

| Item | Status | Where |
|---|---|---|
| Planner: params / bytes per device / FLOPs / comm volume of a ZeRO, DP and Megatron-TP sharded AdamW training step, emitted as KVM task graphs by `runtime/partition.py` | CHECKED: predictor (no tasks built) equals the emitted tasks exactly (comm bytes, FLOPs per device, buffer bytes per device) for 8 configs; partitioned CPU simulation equals the unpartitioned VM bitwise (1 step) / 1.3e-5 (3 steps, AdamW makes values non-integer) | `dist/scaleplan.py`, `tests/dist/test_scaleplan.py` (5 tests), `step1_check_cpu.txt` |
| New partitioner rule: sharded AdamW (ZeRO: p/m/v ROWS, gradient PART -> reduce-scatter, no all-reduce) and symbolic init (shapes too big for Python lists) | CHECKED + MEASURED (below) | `runtime/partition.py`, `tests/dist/test_dist.cu` (`gen` init, i32 buffers, `--dump`, `--timeout`, memory report) |
| Sharded stacks, ranks as in-kernel partitions of one A100 (2 and 4 ranks, ZeRO and tensor parallel, ring and direct collectives) vs the CPU golden | MEASURED: bitwise equal (1 step; every value of 8 plans), 1.3e-5 after 3 AdamW steps; **device buffer bytes equal the planner's prediction exactly** | `a100_golden/` |
| Trillion-parameter planner output (4096 GPUs, TP8 x PP16 x DP32, ZeRO-3): 9.7 GB/GPU state+activations, comm volumes, 1F1B lower bound | MODELLED | `plan_1T_4096gpu_modelled.json`, `taskgraph_1T_representative_gpu.json` |
| Step-time projection for 512-8192 GPUs with NVLink / NIC and KVM's hop costs | MODELLED (inputs: hop 5.6 / 57 / 203 us were measured on ONE GPU; links and GEMM efficiency are assumptions) | `projection_modelled.md`, `dist/scale_projection.py` |

## What the registry's ops cannot do (limits of the stack)

No attention and no nonlinearity in the sharded stacks: the partitioner has no rules for them, and the registry has no transpose op,
so each parameter is stored twice (W and its transpose WT, both trained) to get true gradients from plain matmuls. The 1T figures are
closed-form arithmetic over a GPT-shaped model, not a run of anything that size. Planner finding: the partitioner emits fresh
single-assignment buffers for every gathered or landed copy and never frees them, so ZeRO-3's per-layer gather window is NOT realised;
every rank holds the whole gathered model (see the memory numbers). A buffer-liveness pass is the missing feature.

## MEASURED on one A100-40GB: sharded AdamW, ranks as partitions (ZeRO, 3 steps, m = 64 tokens, lr 1e-3)

Each run: the same seeded data generated on the device, an unsharded run (1 rank, replicated) and a sharded run (ZeRO: W, WT, m, v ROWS
over the ranks, batch ROWS, gradients reduce-scattered) executed by the in-process multi-executor harness (`tests/dist/test_dist.cu plan`),
final parameters and per-step losses compared. "Logical params" counts W once (each is stored twice). Collectives are KVM tasks, no NCCL.

| logical params | shape | ranks | device bytes, all ranks | unsharded bytes | comm bytes planned (3 steps) | kernel ms sharded / unsharded | max abs diff of final W | worst loss rel diff | planner bytes and comm == emitted |
|---|---|---|---|---|---|---|---|---|---|
| 0.26 M | d256 x 4 | 4 | 0.087 GB | 0.010 GB | 0.040 GB | 14 / 21 | 4.5e-5 | 9e-8 | yes |
| 16.8 M | d1024 x 16 | 4 | 3.81 GB | 0.555 GB | 1.65 GB | 538 / 1582 | 7.6e-6 | 9e-8 | yes |
| 67.1 M | d2048 x 16 | 4 | 13.5 GB | 2.18 GB | 5.71 GB | 2060 / 6604 | 3.6e-4 | 2e-6 | yes |
| 134 M | d4096 x 8 | 4 | 25.2 GB | 4.33 GB | 10.5 GB | 4487 / 14322 | 6.6e-4 | 3e-7 | yes |
| 201 M | d4096 x 12 | 2 | 23.4 GB | - | 5.15 GB | 11627 | - | - | yes |
| 268 M | d4096 x 16 | 2 | 31.2 GB | - | 6.87 GB | 15568 | - | - | yes (largest tried: 31.8 GB device used) |

Reading it honestly:
* Equality: step-1 values are bitwise equal in the small CPU-golden plans; at scale the sharded and unsharded losses agree to <= 2e-6 and final
  parameters to <= 7e-4 absolute. Not bitwise because the gradient is a sum whose order differs (reduce-scatter vs one sequential k loop) and
  AdamW then amplifies it. With lr 1e-3 the third step often diverges (loss goes up, a deep linear net without normalisation); both runs diverge
  identically. The 0.26 M run descends 0.340 -> 0.327 -> 0.316 in both. This is "same numbers", not "trains well".
* Memory: the planner's per-rank buffer bytes equal what the harness allocated, to the byte, in every run (the planned and the actual number are the
  same integer). The finding is bad news for ZeRO on this planner: per-rank bytes are only 11% below the unsharded run at 4 ranks (e.g.
  952 MB vs 555 MB per rank-vs-unsharded at d1024: 3.81 GB / 4 = 0.95 GB per rank, unsharded 0.55 GB) because gathered weights, landing buffers and
  reduce-scatter temporaries are fresh buffers that are never freed. Sharding DOES cut the persistent state per rank (state / P) but the
  emitted program also holds the whole gathered model on every rank. A liveness pass in the partitioner is required before ZeRO-3 saves memory.
* Time: the sharded run is 3x faster than unsharded ON ONE GPU. That is not scaling: the registry's matmul is one 16x16 tile loop per task
  (one block), so four ranks' tasks simply occupy more SMs. It is not a compute benchmark of anything.
* Largest that fits: 268 M logical parameters (2.1 G fp32 words of parameters+optimizer+gradients, both orientations) on 2 ranks, 31.2 GB. 4 ranks
  need 1.7-2x the bytes (4 gathered copies), so the 4-rank ceiling is about 134 M-170 M. Not the trillion-parameter shape: that is the MODELLED plan.
* Harness limits found: a rank's whole graph is one bundle, capped by `live_cap` (now 4096): 5487 tasks per rank (32 layers, 4 ranks) is rejected
  (TOO_BIG) and the in-process harness then hangs at 100% CPU until the device timeout instead of failing. Documented, not fixed.

## What is still not evidence

No second physical GPU, no NVLink, no network. The A100 numbers above are one device; the 57-78 us cross-executor hop is the same-GPU number.
The projection is a model. Real 2/4/8-GPU runs: `HARDWARE_PLAN.md`; the Modal launch was blocked by the permission classifier in this session
(no cloud spend occurred), so those rows remain "not run".

## Reproduce

CPU: `python3 -I -m unittest discover -s tests/dist -p test_scaleplan.py` and `python3 -I dist/scaleplan.py check`.
GPU (box with nvcc, under its lock): build `tests/dist/test_dist.cu` as in `dist/run_multigpu_tests.sh`, then
`KVM_TEST_DIST=./test_dist python3 dist/scale_gpu.py golden OUT` and `... pair OUT --d 2048 --layers 16 --ranks 4 --lr 1e-3`.
