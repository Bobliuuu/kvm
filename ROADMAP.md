# KVM roadmap: what is needed to finish

Status as of 2026-11-07 (evening). Numbers come from committed results; branch names are real. "Not run" means no result exists, not that it passes.

## Where things stand

| Area | State | Evidence |
|---|---|---|
| Correctness of the executor | green on every branch tested: 8613 checks, 0 failed, litmus pass | `results/2026-11-07-chunk-r1/`, `results/2026-11-07-a100-admv2/` |
| 162M GPT step vs PyTorch | **not met.** main: bf16s ~290 ms, f32 ~840 ms. `perf/chunked-release`: bf16s ~135 ms, f32 ~258 ms. torch eager 47 ms, CUDA graph 31 ms, TF32 59-70 ms | same |
| 162M learning | works (f32, 300 steps, matches torch to 4.2e-4) | `results/2026-11-06-a100-learning-162m` |
| Matmul tiles | 0.34-0.68x cuBLAS; skinny wgrad shapes 0.17-0.21x | `results/2026-11-07-gemm-tiles/` |
| Where the VM wins | launch-bound small models and in-kernel population control (see below) | `results/2026-11-07-a100-coopt-sh/` |
| Multi-GPU | not run | README |

## Branches to land (in this order)

1. `perf/admission-v2` : merges `perf/admission-bottleneck` + `perf/completion-path`, auto-sizes the admission window. bf16s 291.8 -> 150.6 ms.
2. `perf/chunked-release` (built on 1) : chunked descriptor release, edges first, add-indegree race handling. bf16s 135 ms, f32 258 ms. Gates passed: exec suite, float64 grads.
3. `perf/gemm-tiles` : BK=64 / spread-copy GEMM variants. Only gain is head NT (0.54 -> 0.67x). Not integrated; gates not run. Merge only the `-DKVM_MMA_SPREAD=1` BK=32 variant after the gates.
4. `results/coopt-sh` : in-kernel retire/PBT benchmarks, self-contained under `bench/coopt/`; safe to merge any time.
5. `integration/green` : integration branch for the all-green work (see P0). Not finished.

## P0: make everything green and reproducible

- [ ] **Standalone litmus-mutant suite hangs / prints nothing on admission-v2 and later.** Built, then no output in 2 h. Root-cause (deadlock in the new admission path vs stdout buffering) and fix. Nothing is "all green" until the mutants run.
- [ ] **Not yet re-run on the new executor:** sanitizers, negative controls, model zoo, 300-step learning curve vs torch, Doom + serve demo (700/700 frames identical, 35 Hz).
- [ ] **`kvm_run` demo path does not use the auto-sized/chunked admission** (only the `kvm_exec` main does). Wire it in and re-run the demo.
- [ ] **Zoo: 2 of 28 cases fail the weight check.** `vit_bf16s` (key-projection bias has a mathematically zero gradient, AdamW amplifies noise: needs a fix to the check, not the kernel) and `san_bf16s` (2 steps, weight error 1.7e-2 vs 1e-2 gate: confirm it is the golden/tolerance, not a kernel bug).
- [ ] **`test_ops` T=1 attention-backward tolerance.** Absolute bound `kT1Abs = 1e-4` is in; confirm it passes on every shape and close the README item.
- [ ] **`recycle` long-running admission flake** (3M tasks through 4096 slots failed 2 of 3 runs; passes in ~6 s otherwise; base commit shows it too). Not root-caused. Re-test on the chunked-release admitter, which changed the indegree/release protocol.
- [ ] **Merged executor ~6% slower than pre-merge.** Found to be additive (+1.8% attn, +4.4% bf16); decide to accept or tune each part.
- [ ] **Pending harvest:** Modal volumes `kvm-suite` (`/adm-g1-sched2`, `/c2-litmus`, `/c2b-exec`) and `kvm-gpt162` (`/learn256b`). The in-flight runs were stopped on 2026-11-07, so expect to re-launch.
- [ ] One script, `scripts/run_all_green.sh` (Modal entry), printing a single PASS/FAIL table. Then make every README badge and claim match it.

## P1: performance (biggest first)

Profile of the chunked build (bf16s): utilisation ~71%, admitter idle 1.4 ms, queue wait p50 55 us but **p99 3.2 ms**, matmul tiles 43% of task time, per-hop cost ~7 us.

- [ ] **Dependency-chain and tail effects** (next executor rebuild, batch these together because each nvcc build is ~40 min):
  - persistent-tile chaining: a worker continues into a ready successor without a queue round trip
  - critical-path priority (longest-remaining-path rank computed in `compile_workload`)
  - per-SM-group local ready queues with stealing to cut p99 queue wait
  - cheaper completion path (fewer fences/atomics); target per-hop 7 us -> <3 us
- [ ] **Matmul tiles to >=0.9x cuBLAS.** BK=64 and spread copies did not close it. Untried: split-K / stream-K for the skinny wgrad shapes (check grid fill and wave quantization on 108 SMs first, the 0.17-0.21x shapes look like occupancy, not pipeline), warp-specialized producer/consumer with mbarriers, per-shape tile selection at compile time (128x256 / 256x128 / 64x256), fused epilogues. Benchmark with many-rep medians; cuBLAS itself varies 124-161 TFLOP/s run to run in a shared container.
- [ ] **Fuse elementwise work into GEMM epilogues.** `--fuse` in `bench/vs_pytorch/gen.py` changes nothing today (task count stays 58,634); it needs op-level kernel changes. Larger elementwise/norm/optimizer tiles were measured and are slower, so reduce task count by fusion, not by coarsening.
- [ ] **Attention fwd/bwd** is 15-30% below torch SDPA: flash-style online softmax with bf16 mma.
- [ ] Measurement hygiene: compare variants interleaved inside one container; GPUs differ between Modal containers (A100-40GB SXM vs 80GB PCIe changed baselines by ~10%). Re-measure torch in the same container.

## P2: the use cases where the VM can win (co-optimization)

Done: `bench/coopt/` trains a population in ONE kernel with in-kernel retire rules and PBT (copy top weights over the worst, perturb LR, held-out eval inside the VM). 1.4-2.0x vs torch + CUDA graphs on small models, 0.59-0.9x on 128/256 where cuBLAS wins.

- [ ] Ping-pong buffers so the optimizer can overlap the next forward (single-buffered H1/dY made the overlap slower).
- [ ] Adam/momentum state in PBT copies (current prototype is plain SGD, 2-layer MLP, 4 batches).
- [ ] A real model for the same trick (e.g. 10-50M GPT) once the GEMM tiles are faster; today the win disappears once GEMMs dominate.
- [ ] Other "weird" fusions from the pitch: whole-training-run in a single launch, in-kernel data pipeline/RNG, train + serve co-location (the Doom + GPT demo already shows serve-while-train).

## P3: scope and honesty

- [ ] Multi-GPU: not run; at least write the scoping note in `DISTRIBUTED.md` (what a persistent VM per GPU + NCCL-free peer queues would need).
- [ ] Keep the README's "Honest limits" in sync after every merge: the VM still does not beat a CUDA graph at any size measured.
- [ ] 162M demo loss over the full 50k vocabulary only drifts; learning is shown on the 256-token stream. Say so wherever the demo is quoted.

## Suggested order (about a week of agent work)

1. P0 litmus hang + re-run all gates on `perf/chunked-release`, then merge branches 1-2 to main.
2. One batched executor rebuild with chaining + critical-path priority + local queues (P1), gates, measure.
3. Split-K / per-shape tiles / epilogue fusion (P1), gates, measure end-to-end.
4. Regenerate every README number from `run_all_green.sh`; merge `results/coopt-sh`.
5. Then P2 (ping-pong overlap, Adam PBT) and P3.

## Practical notes for whoever picks this up

- Each `kvm_exec` nvcc build takes ~40 min: cache binaries on a Modal volume, `-arch sm_80` only, batch changes per build.
- Modal has a 10-GPU workspace cap; queues were hours long when many agents ran at once. Run few jobs at a time and use `modal run --detach` or a deployed app (a backgrounded local client that dies can kill the run).
- Stop ephemeral Modal apps when done; check `modal app list` first.
- Repo history was rewritten on 2026-11-07 to remove a leaked address. Local clones need `git fetch && git reset origin/<branch>` (mixed reset keeps the working tree); do not push old history (the pre-push hook will block it).
- No IPs, hosts or keys in committed files; ssh only through `~/.ssh/config` aliases.
