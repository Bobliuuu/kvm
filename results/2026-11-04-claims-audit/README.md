# KVM claims audit vs Flapping Airplanes (2026-11-04/05)

Auditor X (independent). Branch `review/claims-audit` = `feat/integration` 0d945bc + this directory. GPU: A100-SXM4-40GB (108 SMs, sm_80), a shared box:
other users' jobs ran around mine, so absolute numbers are indicative; ratios are what to read. FA's published numbers (none, see below) are for H100/B200; nothing here is
a statement about those GPUs. All A100 groups ran in ONE detached job under one `~/.gpu.lock` hold in a private scratch dir
(`run_audit_job.sh`), raw logs in `logs/merged-0d945bc/`; pre-merge runs (1d3c15a) in `logs/prior/`. No host names, addresses or user paths are in this directory (scrubbed).

## 1. What FA publicly claims (primary text is thin; "inferred" = I found no primary text)

| # | Claim | Source |
|---|---|---|
| F1 | A "kittens" virtual machine "takes over the whole GPU" and allows "new kinds of co-optimization"; "fusing entire training runs into a single kernel and even weirder stuff" | tweet 4/5, https://x.com/flappyairplanes/status/2052107357585649810 (x.com returned HTTP 403 to me; text as quoted in this repo's AGENTS.md, not independently re-fetched) |
| F2 | "the world's most complete megakernel framework" | TechCrunch 2026-10-21 https://techcrunch.com/2026/10/21/year-old-ai-lab-flapping-airplanes-raises-1b-at-a-6b-valuation/ (fetched; founders quoted) |
| F3 | "It can perform entire training runs, serve inference, play video games, and even do all three simultaneously in a single kernel" | same TechCrunch article |
| F4 | "can overlap memory-bound inference with compute-bound training" so GPU hardware is rarely idle | same article |
| F5 | Does not use Torch, NCCL or CUDA's normal runtime; one kernel on one B200 trains, serves inference and plays Doom | co-founder interview, SECONDHAND: a search-result snippet of https://x.com/MTSlive/status/2113067421376585965 (403 on fetch); treat as unverified |
| F6 | Framework not hardwired to one model | inferred (implied by "framework"; no primary text) |
| F7 | Ops fast without hand-written kernels | inferred; no primary text |
| F8 | Dynamic SM sharing, memory-bandwidth arbitration, work rebalancing across co-scheduled workloads | inferred from F3/F4; no primary text uses these words |
| F9 | Weight sync between training and live inference | inferred |
| F10 | Scaling across GPUs/nodes/datacenters without NCCL; trillion-parameter training | inferred (F5 mentions NCCL only); nothing found on trillion parameters |
| F11 | Any number (latency, utilization, tokens/s, fps, speedup) | NONE found. Their only artifact is a timeline image (73 rows, ~1.75 ms, 7 colours, no legend; per AGENTS.md) |

Context only (Hazy Research, same lead author; fetched): Llama-1B single-sequence decode < 680 us on B200 and 78% of H100 bandwidth, > 1.5x SGLang
(https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles); Tokasaurus megakernel +22% tokens/s vs SGLang (Llama-70B TP, H100), global work queue +14.2% on its
ablation, NCCL replaced by in-kernel NVSwitch stores, launch 2.1 us (1.3 us in a graph) (https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main).
FA's own corpus gives nothing to benchmark against, so below I ran the closest *shape* of experiment and report ratios.

## 2. Claim matrix against the current tree (0d945bc)

Status: **IM** = implemented + measured; **I** = implemented, not measured (by me or anyone); **C** = code only / untested on its target; **M** = missing.
"Re-run" = I re-ran it on the A100 in this audit.

| Claim | Status | Evidence (file:line; results) | Re-run by me |
|---|---|---|---|
| F1 whole training run in one persistent kernel | IM | `kvm_exec.cu:803` `kvm_exec_kernel`, workers `include/kvm_exec.cuh:593` `kvm_worker`; training bundles up to 43,650 tasks / 50 steps ran in one launch (`logs/merged-0d945bc/time_kvm.txt`) | yes |
| Correct vs independent reference | IM | 4 NEW models vs CPU golden (section 4); `exec_functional.txt` 8582 checks 0 failed; model-zoo subset 7/7 PASS (`models-subset-check.txt`); litmus 15 passed | yes |
| F6 not hardwired to one model | IM, with limits | executor/scheduler/ABI/compiler have no model or game names (`logs/hardcoding-grep.txt`, `hardcoding-grep-merged.txt`); new model shapes + an MLP pass on the A100 (section 4). Limits: op registries are compile-time CUDA, game ops need a separate `-rdc` build (`include/kvm_exec.cuh:389-404`), the training builder is LM-only (`frontend/kvmfe/train.py:86,108`), `ModelConfig` rejects non-attention shapes even for an MLP (`nn.py:30-34`) | yes |
| F7 ops fast with no hand-written kernels | **M / contradicted** | Every op is a hand-written CUDA body (83 registered ops in `include/kvm_ops*.def`); there is no kernel generator. Quality: TF32 matmul 4096^3 = 27.6 TFLOP/s default build, 41.2 with the 128-register cap, vs cuBLAS 105.6 (0.26x / 0.39x) (`ops_bench_*.txt`, `time_torch.txt`) | yes |
| F3 train + serve + game together in one kernel | IM (by D, not by me) | Real Doom + GPT train + GPT serve in one executor: `results/2026-11-04-a100-adaptive/README.md`, harness `adaptive/adaptive_run.cu`; Doom + synthetic load: `results/2026-11-04-a100-sched-integrated/`. Not on the integ demo runner (`demo/` is agent H's, uncommitted at audit time) | only `test_adaptive` suites + gpt modes |
| F8a dynamic SM sharing | IM, narrow | per-class queues, weights, reservations, EDF: `include/kvm_sched.cuh:196-340`; `sched_functional.txt` 499 checks 0 failed; sharing is at task boundaries on whole CTAs (tasks are not preemptible) | yes |
| static SM split only a baseline | confirmed | `kvm_static_split` callers: `kvm_exec.cu:705` (`--shares`, needs `--sched static`), tests and `doom/tests/run_sched.sh:50-51` only; CLI default is scheduler off (`include/kvm_sched.h:106`); frontend/manifest code has no reference (`hardcoding-grep.txt`). H's runner (not audited) defaults to `dynamic` | grep |
| F8b memory-bandwidth arbitration | C-ish: implemented, blind to the main ops, no measured benefit | arbiter `include/kvm_sched.cuh:234-290,437`; input is a hand-written byte-cost table `include/kvm_ops_cost.inc` with 20 entries; `matmul_tile`, all attention, all norms, rope, MoE ops, Doom ops have NO entry (default MIXED, 0 bytes, never throttled) (`hardcoding-grep-merged.txt`). D's own result: "The arbiter did not win" (adaptive README). My re-run of the synthetic bench: arbiter ON (60 us target) p99 660 -> 654 us (1.01x) at 99.8% stream retained; the 25 us target costs 29% of stream throughput for 1.08x | static + prior bench |
| F8c work rebalancing | IM (synthetic) | host controller `include/kvm_sched.h:135`, `kvm_adapt.h`; `test_adaptive` telemetry/loop/floor PASS, floor_neg detected (`adaptive*.txt`); real-mix result: controller did NOT beat hand-set fixed priority (D: 470 vs 353 late frames) | suites yes, real mix no |
| F9 weight sync train -> live inference | IM | version variables, `exact`/`at_least`, FRAMEWORK.md; `exec_functional.txt` versions suite + `KVM_BREAK_FENCE` negative; demo bundles serve from training versions (`frontend/kvmfe/train.py:1-15`) | functional yes |
| F10a multi-GPU / multi-node without NCCL | C (single GPU only) | `runtime/partition.py`, `include/kvm_ops_comm.*`, `dist/`; `results/2026-11-03-dist-singlebox/README.md`: 14 stages pass on ONE A100, 4 stages needing 2 GPUs/nodes skipped; "Nothing here shows multi-GPU, NVLink, multi-node or WAN behaviour" | CPU partitioner/sim tests only (python suite 145 OK); GPU dist stages NOT re-run |
| F10b trillion-parameter training | **M** | largest model anywhere is the 162M GPT compile shape (`nn.py:64`), no GPU run of it on this tree; no sharded-optimizer run | - |
| F11 any quoted number | n/a | FA gives none | - |
| Timeline from VM timestamps (AGENTS.md goal 4) | I/IM | per-task trace exists (`--trace-csv`, `kvm_exec.cu`); not audited | no |

## 3. Honest gap list, ranked by importance

1. **The strongest conventional baseline beats the VM on these workloads (A100).** Whole-step CUDA graph: 132.5 vs 721 us/step (VM 5.4x slower) on a 2-hidden-layer MLP; 981 vs 3,810 us/step (3.9x slower) on a 3-layer transformer (table 5). The VM only beats PyTorch *eager* (1.04x and 1.9x) and `torch.compile` default (1.3x, 0.87x). The VM's dispatch floor is the cause: the admitter ingests 1.96 us per task (`adaptive.txt` admit_gran, 0.51 M tasks/s) so 38-45% of the run is admission alone, and each task here is tiny (4.4-5.2 us per task end to end). This does not refute FA (their regime and hardware are unknown) but the repo's central goal (beat the best conventional execution) is **not met** at these sizes on an A100.
2. **"No hand-written kernels" is false for this implementation, and the hand-written kernels are 2.5-4x off cuBLAS** (F7). Attention is 3xTF32/fp32 (12-16 TFLOP/s vs bf16 flash 70-130 in ops-perf README). Efficiency claims rest on ratios to torch fp32 SDPA, which is the weakest comparator.
3. **Multi-GPU, NCCL-free scaling and trillion-parameter claims have zero multi-device evidence** (F10). Everything is one-GPU emulation (ranks as threads/processes on one A100). The remote hop costs 57-78 us median vs 5.6 us local; nothing was measured over NVLink or a network.
4. **The headline co-scheduling demo does not show the scheduler helping where it matters.** D's own results: admission, not policy, limits tail latency (12-15% SM utilisation in every config); the controller loses to a hand-set priority; the tuned gate buys deadlines by dropping 38% of serve requests; with a 47k-task training record no config meets 35 Hz. My re-run of the scripted bench: dynamic+controller serve p99 2,157 us vs a plain global queue **with serve priority 1,214 us** (1.8x worse at equal throughput); it beats only the strawman flat queue (9.5x) and static splits. The Doom "load" in the sched-integrated study is `spin` tasks (busy waits), not real bandwidth/compute pressure.
5. **Bandwidth arbiter is blind to the ops that matter** (F8b): no cost entry for `matmul_tile`, attention, norms, rope, MoE, Doom; and it showed no measurable benefit on real tenants.
6. **Game side is one serial CTA.** Doom's tick is a single-block 5.4 ms task (`doom/exec_main.cu`, one tick in flight); "playing a game" uses ~1 SM, the other 105 are idle in the alone run. Fine as a latency tenant, weak as evidence of whole-GPU co-optimisation. Multi-instance Doom is capped at 8.
7. **Generic frontend, but not generic training.** `build_training` hard-wires token stream + softmax-CE + select_window (`train.py:86-108`); an MLP/regression/RL-style loop needs a new builder (I wrote one, 30 lines, `newmodel/gen.py`). `ModelConfig` validates attention geometry (head dim 16/32/64/128, seq % 32) even for an MLP. No dynamic shapes or data-dependent control flow (models README). bf16 is emulated in fp32 storage.
8. **Test-suite hygiene on the integration tip.** (a) `tests/frontend/test_models.py::test_vit_two_layers_patch2` fails deterministically (autodiff -1.5e-19 vs finite difference 1.1e-11 for an attention *key bias* whose true gradient is exactly zero; the relative-error check divides by ~0; absolute error 1e-11, so numerically benign but the committed test is red and the models README's "worst 1.1e-6" excludes it). (b) litmus mutant `weakpub/mp` was not built in two A100 attempts (binary `litmus_weakpub` missing, exit 127; first attempt shows `Killed`, probably an OOM kill of the parallel nvcc on the loaded box): litmus reports 15 passed, 6 of 7 mutants detected, 1 failure (`litmus.txt`, `litmus_rerun.txt`); the same harness passed 7/7 on 1d3c15a (`logs/prior/exec-litmus-1d3c15a.txt`). Unresolved here; could be environment, could be the merged `kvm_exec.cuh` no longer matching the mutant's sed pattern (that would give "sed matched nothing", which I did not see).
9. Verified-weak claims: flaky `recycle` long-admission run reported by D (2 of 3 failures on their tree) passed for me (`exec_functional.txt` recycle PASS 14.4 s, one run only); dist negative stage originally counted any failure as detection (since tightened).
10. Timing caveat: the shared box added noise (zoo subset times 10-25% above the models README's: llama_f32 33.7 vs 32.1 ms, moe_f32 43.9 vs 38.9, llama_bf16 47.3 vs 39.5).

## 4. New-model test (does the frontend generalise? what breaks)

`newmodel/gen.py` (public frontend API only, nothing in `frontend/` edited) builds four models the repo never shipped, compiles bundle + CPU golden, runs on the A100 with
`kvm_exec --blocks 4 --dump-buffers`, compares every f32 buffer (`newmodel/compare.py`; SGD, so no Adam sign-flip noise; 1xTF32 GEMMs).

| case | what it is | tasks (3 steps) | max weight rel L2 | max loss rel err | verdict |
|---|---|---|---|---|---|
| `xfmr_odd` | 3 layers, 3 heads (hd 16), d48, seq 64, vocab 100, rmsnorm, no bias, ffn x2 | 2,619 | 1.06e-4 | 8.2e-6 | PASS |
| `mlp_ce` | embedding -> 2x(linear+GELU) -> linear -> CE; no attention/norm/rope | 414 | 2.0e-4 | 6.5e-6 | PASS |
| `resmlp_rms` | 3x(linear+GELU+rmsnorm) -> CE | 753 | 1.4e-3 | 4.2e-4 | PASS (all-buffer worst 7.5e-3, activations under 1xTF32; loss falls 7.36 -> 1.39 like the CPU) |
| `bad_dims` | MLP with hidden width 7 (non-tile-multiple) | 300 | 2.1e-4 | 2.4e-6 | PASS |

Same results on the pre-merge and merged tree (`logs/newmodel-*.cmp.txt`, `logs/merged-0d945bc/newmodel-*.cmp.txt`). Exactly what broke (first attempt, all front-end, none on the device):
- `ModelConfig(d_model=40, heads=2)` for an MLP: "head dim 20 must be 16, 32, 64 or 128 (attention ops)" - the config class is attention-shaped; I worked around with a dummy valid config. `seq=48`: "must be a multiple of 32".
- `compile_training` is GPT-only; I called `build_training` + `lower_program` + `to_workload` directly with my own `Module` (works).
- `gen.py` initially dropped `file`-init streams (my script bug, not the repo's).
Refusals are clean `FrontendError`s (`logs/newmodel-bad_head.txt`), no silent miscompiles.

## 5. Closest comparable benchmark (A100; ratios, not absolutes)

FA gave no numbers, so the comparison is the repo's own premise: a whole N-step training run in one `kvm_exec` launch vs PyTorch 2.13 on the same GPU, same shapes
(`newmodel/torch_bench.py`, TF32 on, SGD, best of 5; VM time = executor's own "kernel done" ms, includes streaming admission, excludes process start and bundle load; merged tree).

| run | VM total (min of 5) | VM us/step | PyTorch eager | whole-step CUDA graph | `torch.compile` | VM vs graph | VM vs eager |
|---|---|---|---|---|---|---|---|
| `mlp_ce`, 96 tokens, 200 steps, 27,600 tasks | 144.0 ms | 720 | 749 | **132.5** | 928 | 5.4x slower | 1.04x faster |
| `xfmr_odd`, 128 tokens, 50 steps, 43,650 tasks | 190.1 ms | 3,803 | 7,417 | **981** | 3,313 | 3.9x slower | 1.95x faster |

(Pre-merge repeat: 143.0 ms / 189.9 ms VM, 744 / 8,097 eager, 132.2 / 1,029 graph: `bench-newmodel-vs-torch-1d3c15a.txt`.) Admission floor: 27,600 tasks x 1.96 us = 54 ms (38%) and 43,650 x 1.96 = 86 ms (45%). Why the VM loses: ~100-150 tiny tiles per step with a ~5 us dependency hop each (AGENTS.md: ~5 us/hop), and a graph replays the same 300 launches at ~1.3 us. Where it should win (many independent runs, data-dependent control) was not tested here. Not comparable to Hazy's numbers (B200/H100, bandwidth-bound decode).

Library reference points (same box): cuBLAS TF32 4096^3 105.6 TFLOP/s; ATen layernorm 1,120 GB/s. KVM ops (re-run of ops-perf headline numbers, `ops_bench_*.txt`): matmul TF32 NN 27.6 (claimed 27.0), 41.2 with `BENCH_MINB=2` (claimed 42.7); layernorm fwd 1,141 GB/s (claimed 1,127); adamw 1,181 (1,171); gelu 1,181 (1,180). `test_ops` 1396 passed, 0 failed. Claims reproduce; the claims are modest and honest, and say the same gap to cuBLAS I measured.

## 6. Other agents' claims, re-checked

- **Adaptive (D)**: `test_adaptive` 135 checks pass; `admit_gran` reproduces the 1.96-2.06 us/task admission cost (claimed 1.9); `floor_neg` detected. I did not re-run the Doom+GPT 24 s phased batches (hours of lock). Their README is unusually frank (controller no better than fixed priority; admitter is the bottleneck); I found no overclaim, but the "dynamic beats static" story is mostly shown against a no-priority global queue and a static split.
- **Models**: 7 of 24 cases re-run (mlp/vit/llama/gqa_tied/moe f32, moe tf32, llama bf16): all PASS with errors at or below the README's. Red unit test noted above (gap 8a).
- **Ops-perf (P)**: headline numbers reproduce within noise (above). The 128x128 tile is not used by the frontend yet (their note).
- **Dist**: CPU partitioner/simulator unit tests green in the python suite; the GPU dist script was not re-run (it takes the lock itself; I did not want to hold the shared lock a third time). Single-GPU only by their own statement.
- **Sched/exec suites on the merged tree**: exec 8582/0, sched 499/0, negatives detected, gpt modes byte-identical across 6 scheduler modes (md5 `40d3cfa93158`) + sanitizers clean, python 145 + frontend 59 (1 red).

## 7. Process notes (what I did wrong)
- My first launch of `examples/models/run_box.sh` inside an already-held lock deadlocked on its own `flock` and held the shared A100 lock for about 4 hours before timing out; the rerun used `run_all_full.sh` directly. I also rsynced an early run into two shared scratch dirs (`kvm-exec-scratch`, `kvm-sched-scratch`) before switching to private ones; the tree was the integration tree. Two orphaned flock waiters from that remain on the box (a remote `kill` was refused by the permission layer).
- x.com pages could not be fetched; F1 is quoted from AGENTS.md, F5 is from a search snippet.

## Reproduce
```
python3 -I -m unittest discover -s tests; python3 -I -m unittest discover -s tests/frontend
python3 -I results/2026-11-04-claims-audit/newmodel/gen.py OUT xfmr_odd      # or mlp_ce resmlp_rms bad_dims bad_head
KVM_HOST=user@host PYBIN=<python with torch on the box> results/2026-11-04-claims-audit/run_audit_job.sh WORK PREBUILT_MODELS NEWMODEL_BUNDLES   # detached, one lock hold
```
Files: `newmodel/` (generator, comparer, torch bench, libref), `logs/` (raw), `run_audit_job.sh`.

## Update 2026-11-06 (branch feat/scale-evidence): F10a / F10b, measured vs modelled

* F10a stays **C** (single GPU) but is now far better supported: the partitioner gained a ZeRO sharded-AdamW rule; sharded stacks (ZeRO, tensor parallel, 2-4 ranks as
  in-kernel partitions of one A100) are bitwise equal to the CPU golden in 8 plans and agree with an unsharded run to <= 2e-6 loss / 7e-4 parameters up to 134 M logical
  parameters (4 ranks) and ran at 268 M (2 ranks, 31 GB). The planner's bytes/FLOPs/comm volume equal the emitted tasks exactly. Still **no** second GPU, NVLink or network measurement.
* F10b stays **M** for "trillion-parameter training": the 1T plan (9.7 GB/GPU at 4096 GPUs, 30% MFU lower bound) is a closed-form MODEL; the largest thing executed is 268 M
  logical (536 M stored) parameters of a deep linear network without attention. Found: the planner never frees gathered buffers, so per-rank memory barely shrinks.
* Details and the hardware that would turn this into evidence: results/2026-11-06-scale-evidence/README.md and HARDWARE_PLAN.md.
