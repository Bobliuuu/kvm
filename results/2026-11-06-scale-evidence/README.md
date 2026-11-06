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
