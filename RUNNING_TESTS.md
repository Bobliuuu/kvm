# RUNNING_TESTS.md: every test entry point, what it proves, what it costs

Hosts and keys come from the environment (`KVM_HOST=user@host`, optional `KVM_KEY`, `KVM_ARCH`, default `sm_80`); nothing is committed.
"Lock" is the GPU box's `~/.gpu.lock` (`flock`). Times are upper bounds taken from the scripts' own `timeout`, or the measured figure when a
result README or a local run recorded one; the GPU box is shared, so queueing for the lock can add hours.

## No GPU needed (run these before every commit)

| Command (repo root) | Checks | Runtime |
|---|---|---|
| `python3 -I -m unittest discover -s tests` | compiler, ABI v2 writer, CPU VM, registries, lowering, GPT reference, queue integration: 145 tests | about 3 s (measured) |
| `python3 -I -m unittest discover -s tests/frontend` | frontend: autodiff vs finite differences of float64 oracles (6 model families), dataflow fuzz, golden models, manifest, serve, game CLI, generic training (any module, MSE/CE, four never-shipped models): 78 tests (not found by the command above: no `__init__.py`) | about 30 s (measured) |
| `python3 -I -m unittest discover -s tests/dist -p 'test_*.py'` | partitioner and simulator (CPU) | seconds (32 tests, 4 s measured) |
| `c++ -std=c++17 -Iinclude tests/host/test_adapt.cpp -o /tmp/t && /tmp/t` | adaptive controller, 8 host unit tests | seconds |
| `dist/run_multigpu_tests.sh --cpu-only` | CPU stages of the distributed suite (simulator, transport, launcher) | minutes |
| `for f in $(git ls-files '*.sh'); do bash -n "$f"; done` | shell syntax of every script | seconds |

## GPU box, remote-driven (from a laptop; each script rsyncs, builds, runs)

"Scratch" is the remote directory. Rule for the scripts marked *private*: the default is `~/kvm-scratch-<name>-$USER-$$` (nobody else's run can
overwrite it), rsync never uses `--delete`, a successful run removes it, `KVM_KEEP_SCRATCH=1` keeps it, `KVM_SCRATCH=<dir>` names one (shared
on purpose; yours to manage). `KVM_NO_LOCK=1` means "an outer hold exists" (see `run_all.sh`): the script runs its GPU part without taking the
lock itself. Any non-empty value counts.

| Script | Checks | Runtime bound | Lock | Scratch |
|---|---|---|---|---|
| `run_exec_tests.sh [--negative\|--negative-park\|--sanitize\|--bench\|--litmus\|--fuzz\|--gpt]` | executor suites (8582 checks), mutants that must fail, sanitizers, litmus, CLI fuzz, GPT bundle | 2400 s per suite (fuzz 3000 s) | own `flock` per call; `KVM_NO_LOCK=1` skips it | fixed `~/kvm-exec-scratch` or `KVM_REMOTE`, **rsync --delete** (use `KVM_REMOTE` if others use the box) |
| `run_sched_tests.sh [--negative\|--sanitize\|--bench\|--exec]` | scheduler suites (about 500 checks), starvation/inheritance negatives, benchmarks | per script `timeout` | own `flock`; `KVM_NO_LOCK=1` skips it | fixed `~/kvm-sched-scratch` or `KVM_REMOTE`, **rsync --delete** |
| `run_all.sh <correctness\|verify\|fuzz\|bench\|final>` | the two scripts above in one lock hold; `verify` = everything that must stay green after a scheduler change | hours (lock wait up to 4 h) | takes the lock ONCE, runs stages with `KVM_NO_LOCK=1` | those of the stages |
| `run_gpu_ops.sh <test_ops\|test_e2e\|bench_ops\|bench_attn\|test_extra> [args]` | GPT op pack vs CPU double references (`WRAP="compute-sanitizer --tool memcheck --error-exitcode 9"` for sanitizers; `WRAP="env TEST_ONLY=attention"` runs just the attention cases, add `KVM_T1_SEEDS=2000` to the WRAP env for the deep T=1 noise sweep; `bench_attn [B T H Hkv]` is the attention-only single-op benchmark, builds in seconds, `-DBENCH_HD=128` for hd=128) | 600 s | `flock`; `KVM_NO_LOCK=1` skips it | *private* (`gpu-ops`) |
| `tests/gpu_ops/run_extra.sh [--sanitize]` | extra (model-zoo) op pack, `test_extra` (2089 checks), optional memcheck/racecheck/synccheck | 1500 s | one hold for all runs; `KVM_NO_LOCK=1` skips it | *private* (`extra`) |
| `tests/gpu_ops/run_bf16.sh [--sanitize] [--no-bench]` | bf16 tensor-core GEMM `matmul_mma_bf16` (4 transposes, ragged shapes, every out/res/pre storage mix) and the bf16-storage paths of norms / rope / gelu_bwd / residual / embedding / cross entropy vs float64 (`test_bf16`), then `bench_mma_bf16` (TFLOP/s vs cuBLAS bf16, correctness per row) | 3000 s | one hold for all runs; `KVM_NO_LOCK=1` skips it | *private* (`bf16`) |
| `bench/bf16/run_box.sh correctness\|perf OUT DATA` (on the box) | the GPU measurements of `results/2026-11-06-a100-bf16`: op tests + sanitizers + step-1 gradients vs PyTorch fp64 / TF32 / autocast-bf16, or GEMM TFLOP/s, executor step time, PyTorch baselines, 3-step losses | per stage `timeout`, one lock hold per stage | `flock` per stage | none (runs in place) |
| `tests/gptdiff/run.sh [--sanitize]` | gpt_tiny AdamW bundles (1/2/5 steps; 1xTF32 and 3xTF32) through `kvm_exec` vs the CPU golden, `golden_check.py` | 1200 s on the box | one hold; `KVM_NO_LOCK=1` skips it | *private* (`gptdiff`) |
| `tests/generic/run.sh` | never-shipped models (regression MLP, residual RMSNorm MLP, 1-D CNN, transformer encoder) x {cpu, a100} tiling, 3xTF32, vs the CPU golden (`golden_check.py --mode precise`, every f32 buffer); two MANIFEST demos through `demo/kvm_run` + `demo/check_demo.py`; builds `kvm_exec` and `kvm_run` outside the lock | 1500 s on the box (hold) | one hold for all runs; `KVM_NO_LOCK=1` skips it | *private* (`generic`) |
| `run_gpu_perf.sh <tag>` (`TEST=1 REF=1 VARIANTS=... ONLY=mm\|attn\|mem GROUP=g`) | op throughput, median of 7 [min..max], optional torch/cuBLAS reference lines; output in `results/2026-11-04-a100-ops-perf/<tag>.txt` | detached job, polled; lock wait up to 8 h | `flock` inside the detached job; `KVM_NO_LOCK=1` skips it | *private* (`perf`); an interrupted run leaves its dir and the job alone |
| `examples/models/run.sh [--sanitize] [--cases F] [--prebuilt D]` | model-zoo bundles through `kvm_exec` vs golden (`check.py`), one hold | 3000 s | `flock` unconditional (no `KVM_NO_LOCK`) | fixed `~/kvm-models-scratch`, **rsync --delete** |
| `doom/tests/run_gpu.sh [frames]` | standalone Doom CTA vs CPU oracle (`SANITIZE=memcheck\|racecheck\|synccheck`) | 600 s | `flock` unconditional | fixed `~/kvm-doom-scratch`, **rsync --delete** |
| `doom/tests/run_exec.sh`, `run_sched.sh`, `run_instances.sh` | Doom through the executor / schedulers / N instances, every frame diffed with the oracle | 600-3000 s | `run_sched.sh` honors `KVM_NO_LOCK`; the others `flock` unconditionally | `run_sched.sh`: `KVM_REMOTE` (default `~/kvm-sched-scratch`); `run_instances.sh`: `KVM_REMOTE` (default `kvm-adaptive-doom-scratch`); `run_exec.sh`: fixed `~/kvm-doom-scratch`; all **rsync --delete** |
| `adaptive/run_adaptive.sh <smoke\|main\|multi\|bw\|suites>` | adaptive controller experiments, one lock hold per batch | hours | takes the lock once | `KVM_REMOTE` (default `~/kvm-adaptive-scratch`), **rsync --delete** |
| `dist/run_multigpu_tests.sh [--remote\|--cpu-only\|--dry-run]` | distributed suite: PASS/FAIL/SKIP per stage; stages needing 2+ GPUs SKIP on one GPU | stage timeout 600 s | takes `$KVM_GPU_LOCK` once (`KVM_LOCK_HELD` marks a hold) | fixed, **rsync --delete** |
| `tests/litmus/run_remote.sh <stage>` | memory-ordering litmus, mutants, e2e probes, fuzz | 1500 s per stage | `flock` unconditional | fixed `~/kvm-review-scratch`, **rsync --delete** |

## Run on the GPU box itself (no ssh)

| Script | Checks | Runtime bound | Lock |
|---|---|---|---|
| `examples/models/run_box.sh PREBUILT OUT [--sanitize] [--cases F] [--check-only]` | the `examples/models/run.sh` check with bundles from `examples/models/prebuild.sh` (CPU only); only scratch is `OUT` | 3400 s | `flock` unless `KVM_NO_LOCK=1` or `--check-only` |
| `tests/gpu/run_gpt.sh`, `tests/litmus/run_litmus.sh`, `tests/litmus/run_fuzz.sh` | called by `run_exec_tests.sh --gpt/--litmus/--fuzz` | 900 / per stage / 3000 s | `flock` unless `KVM_NO_LOCK` is non-empty |
| `tests/gpu_ops/run_hopper_box.sh build\|test\|bench\|step OUT` (sm_90 box) | the Hopper TMA + wgmma tiles (`include/kvm_hopper.cuh`): `build` = every sm_90a binary in parallel; `test` = `test_hopper` (every layout x tile kind x epilogue x launch size vs float64 and vs the sm_80 tile, path-taken counters), `test_ops`, `test_bf16`, `test_extra`, `test_exec`, sanitizers on `test_hopper`; `bench` = TFLOP/s Hopper vs sm_80 tile vs cuBLAS; `step` = gpt_162m step time with/without the Hopper tiles (`KVM_NO_HOPPER=1`) and PyTorch eager / graph / compile | build ~20 min on 32 cores; test ~30 min | none (no lock) |
| `modal run --detach tests/modal/run_h100.py --stage build\|test\|bench\|step\|custom --tag NAME` | the same script on Modal: `build` on a CPU-only container, the rest on an H100 | as above | none |

## Not in the table on purpose

The multi-GPU and multi-node stages of `dist/run_multigpu_tests.sh` have never run (no such hardware): their SKIP lines are the honest result.
A GPU that opts into less than 104 KiB of shared memory a block (sm_86, sm_89, sm_120) builds `test_ops` and `test_extra` in a reduced key,
`-DKVM_GPT_BIG_NST=3 -DKVM_GPT_MMA_SMEM=90112 -DKVM_GPT_MAX_HD=64` (`test_extra` also `-DKVM_MMA_NST=3`): that build does not compile hd=128
attention, so those cases print SKIP and the summary counts them; the A100 runs them.
`results/*/README.md` name the command, commit, GPU and flags of each recorded run; a number without them is not a result.
Scripts still using `rsync --delete` into a fixed directory can destroy a concurrent run on the same box (it happened once, disclosed in
`results/2026-11-04-a100-ops-perf/README.md`): give such a script its own `KVM_REMOTE`.
