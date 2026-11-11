# AGENTS.md

`kvm` is a "kittens"-style GPU virtual machine: one persistent kernel takes over the whole GPU and interprets an instruction tape,
so a whole training run is one kernel launch. Version 2 of
[open-flapping-airplanes](https://github.com/Bobliuuu/open-flapping-airplanes); not affiliated with Flapping Airplanes.

## Goal: reproduce Flapping Airplanes' "kittens" VM

[Their post](https://x.com/flappyairplanes/status/2052107357585649810) (4/5, 2026-05-06): "One thing we’ve built is a “kittens”
virtual machine that takes over the whole GPU and allows new kinds of co-optimization. We can go past the traditional sequential
kernel model – for example, fusing entire training runs into a single kernel and even weirder stuff." Why (3/5): algorithms
">1,000x more data efficient" that "look like they’re going to be weird and different enough to require new systems under the hood."

They have published no code, paper or numbers (as of 2026-11-03), only
[this timeline](https://pbs.twimg.com/media/HHqMSLdagAAbYVh?format=jpg&name=orig): 73 rows (SMs or workers) over about 1.75 ms,
7 instruction colors, no legend, GPU and model unstated. So we reproduce the capability, and each claim needs a result file here:
1. one launch runs a whole training run (forward, loss, backward, optimizer, every step) on every SM;
2. it is correct: final weights bitwise equal to the same tape run as ordinary kernels, and the loss within a stated tolerance of
   an independent reference (CPU fp32 or PyTorch);
3. it is faster than the strongest conventional execution of the same work on the same GPU: one launch per op, the same in a CUDA
   graph, batched across runs (vmap-like), and a framework (PyTorch eager, a whole-step CUDA graph, `torch.compile`);
4. it draws a timeline like theirs from the VM's own per-instruction timestamps;
5. it shows co-optimizations the sequential kernel model cannot do: a step overlapping the next, many runs filling one GPU,
   different workloads sharing it.

**Where a VM can win.** It removes what a kernel boundary costs: the launch (about 2.1 µs, 1.3 µs in a CUDA graph, on an H100),
the last wave's tail, the next kernel's cold start, all-or-nothing dependencies. It adds costs of its own: a dependency hop through
global memory (about 5 µs in `kvm.cu`), and every op compiled under the union's register and shared-memory budget. So it wins on
small, dependent work (small models and batches, many steps or runs, data-dependent control) and has little to win in big GEMMs.
The evidence points both ways: `kvm.cu` beats its best baseline 1.50x on a one-run MLP step (25.3 against 37.9 µs), while
open-flapping-airplanes' own task code, launched as per-op kernels, trained 13-25% faster than its megakernel on a 56 µs step, and
Megatron + Transformer Engine with CUDA graphs beat the megakernel on a 162M GPT (43.1 s against 48.1 s, B200). Explain every
result by these costs.

## Where we are (tip of main; keep this current)

Two layers. `kvm.cu` is the original sm_80 prototype (80-byte instruction tape, 2-layer MLP, TF32 `wmma`, loop opcode, `--runs R`). Around the
generic executor `kvm_exec.cu` + `include/kvm_exec.cuh` the tree on main has: ABI v2 (`include/kvm_task_abi.h`), operator registry packs
(core 0-31, GPT 32-63, comm 64-95, extra/model-zoo 96-121, game/Doom), the compiler and CPU golden (`runtime/`), the scheduler
(`include/kvm_sched.*`) and adaptive controller (`include/kvm_adapt.h`), the frontend (`frontend/kvmfe`: GPT, Llama/GQA, MoE, MLP, ViT, real bf16 storage,
fusion, remat, generic training of any traced module), real Doom (`doom/`), multi-GPU code (`dist/`, `runtime/partition.py`). Map: README.md, FRAMEWORK.md, FRONTEND.md,
DISTRIBUTED.md. Entry points and cost of every test: RUNNING_TESTS.md. A change is done when the CPU suites pass
(`python3 -I -m unittest discover -s tests` and `-s tests/frontend`) and, for device code, the GPU suite its claim is about passes on the A100.

| Status | What (result dir under `results/`) |
|---|---|
| Implemented + measured | **Goal 3, executor (`2026-11-10-a100-merged`, main + #13/#22, PR #23):** 9-workload panel against the fastest PyTorch variant per row (eager, whole-step and whole-run CUDA graphs, `torch.compile`, batched), 7 reps, A100-SXM4-40GB, every row gated on loss and weights. Geomean 1.090x. **It holds on small dependent workloads: 2.2-3.8x faster on the four small-MLP rows** (h72 d2 at 1/8/32 runs: 32.6/32.3/46.3 us a step against 72.6/78.8/104.0; the 8-width hetero row 3.85x). **It does not hold on GEMM-bound rows:** MLP h256 d4 0.50x, MLP h1024 0.34x, GPT tiny 0.70x, GPT small 0.73x, GPT 162M (TF32 tape, 3 steps) 0.54x (118 vs 63 ms). The bf16s 162M training step is still **about 6x slower than torch bf16 eager (293 vs 47 ms) and 9-10x slower than its CUDA graph** (`2026-11-06-a100-bf16-storage`, README). Merging costs the executor about 1% against #13/#22 alone. **Superseded:** `2026-11-09-a100-exec-vs-torch` and `2026-11-09-a100-exec-vs-graph` were measured on the pre-#13 executor (the "loses to a whole-step graph 1.8-4.8x on every workload" numbers); kept as the record of that executor, not main's. Step overlap within one `kvm.cu` run (goal 5a, `--overlap 0|1|2`, bitwise-equal weights): a small win of 1.045-1.063x on the A100 at 1 run, and the VM's own timestamps show it comes from taking `DW2` off the chain, not from pipelining steps (`2026-11-09-a100-step-overlap`). `kvm.cu` MLP timeline from the VM's own per-instruction globaltimer/`%smid` stamps (`--timeline`, `bench/mlp_vs_torch/plot_timeline.py`, `2026-11-06-a100-kvm-cu`; an earlier `-DKVM_TIMELINE` build is recorded in `2026-11-09-a100-timeline`). Full 162M demo (train + serve + Doom, one launch) passes all checks, 13.66 s, 1.95M tasks, trace kept as `trace_compact.npz` (`2026-11-06-demo-large-trace`, `assets/timeline-162m-*.png`, `assets/demo-162m.mp4`); at the time the executor was 15-18x slower than PyTorch at 162M, admission 604 ms of 1042 ms (`2026-11-06-gpt162-vs-torch`; tiny/small in `2026-11-06-vs-pytorch`, pre-merge). Timeline figure (goal 4) from the executor's own per-task trace: `demo/viz/timeline.py` -> `assets/timeline-*.png`, data `results/2026-11-06-demo-tiny/trace.csv.gz`; replay video `demo/viz/make_video.py` (`assets/demo-tiny.mp4`). Tiny demo (train windows + serve + Doom, one launch) all checks pass (`2026-11-06-demo-tiny`): 106 mostly idle workers, training+serving done in 345 ms, the 4.2 s span is Doom. `kvm.cu` 1.50x over best baseline at 1 run, 1.13x at 32 (README; predates loop opcode; TF32 against compiled batched graphs it is 0.70x at 32 runs, `2026-11-06-a100-kvm-cu`). Executor: 8,609 checks 0 failed on sm_86 and the A100 suites green on the merged tree (`2026-11-10-a100-merged/tests`), 2.1M-task DAG via 65,536 slots, negatives and sanitizers (`2026-11-04-a100-exec-recycling`, `2026-11-04-a100-sched-integrated`). **sm_86 litmus verdict (`2026-11-10-a100-merged/tests/litmus_sm86.log`, RTX 3090 and 3080 Ti):** every property reads 0 violations on the production headers, every mutant is detected, every probe and e2e scenario passes, 3 of 3 full passes finish. Exit 1 anyway, from one control only: the relaxed-CAS `mp` control, which consumer Ampere does not exhibit (an undetected control counts as a failure); the A100 detects it (0 violations in 2,000,000 rounds on production, the `weakpub` mutant caught with 63). Scheduler shares, EDF, inheritance, overload isolation (`2026-11-03-a100-sched`). Real Doom frame-exact vs CPU oracle in every recorded run, 35 Hz held beside a saturating load (`2026-11-03-a100-doom`, `2026-11-04-a100-sched-integrated`, `2026-11-04-a100-adaptive`). GPT/extra op packs (`2026-11-04-a100-ops-perf`; `test_ops` 5,403 passed on the A100 in `2026-11-10-a100-merged`). 24 model-zoo bundles vs golden (`2026-11-04-a100-models`); GPT golden gap = 1xTF32 noise through AdamW (`2026-11-04-a100-gpt-golden`). Multi-GPU code on one GPU: 14 pass, 4 skip (`2026-11-03-dist-singlebox`). 162M step 1093 -> 845 ms f32 -> 293 ms bf16s after fusion (-50% edges), 128-row attention tiles and the bf16 tensor-core GEMM (`2026-11-06-a100-combined`, `-a100-bf16`, `-a100-bf16-storage`, `-a100-attention`, `-a100-fusion`, `2026-11-10-a100-bf16-attention-storage`); 162M GPT learns a 256-token stream and tracks PyTorch to 4e-4 over 300 steps (`2026-11-06-a100-learning-162m`); serving golden at 162M and PyTorch references for the model zoo (`2026-11-06-a100-correctness`). Claims audit vs Flapping Airplanes (`2026-11-04-claims-audit`); scale-out is simulation only (`2026-11-04-scaleout-sim`). |
| Code only | Multi-GPU / multi-node runs (no hardware), RDMA (stub, simulated in `2026-11-04-scaleout-sim`), 1T-parameter section (arithmetic and DES simulation), affinity hint (no benefit measured, off), generic training (`frontend/kvmfe/generic.py`, merged from feat/integration: CPU-tested against numpy float64, its A100 run `tests/generic/run.sh` on the landing branch: 3 of 4 zoo models pass, `encoder_reg` fails its 2e-4 golden bound (`2026-11-09-landing`)), the integration device changes (staged admission, racecheck-clean hand-off, arbiter knobs: see the PR text for the GPU status). |
| Not built | Class-aware or sliced admission, device-side yield, dynamic shapes / control flow, fp16, TMA/wgmma/tcgen05, a multi-GPU run. Goal 3 is not met for GEMM-bound work (GPT, wide MLP): see the panel row above. |

Known failures and honest negatives (do not paper over them):
- Goal 3 holds for the executor only on small dependent MLPs (2.2-3.8x over PyTorch's fastest variant); it loses on wide MLPs (0.34-0.50x), GPT tiny/small (0.70-0.73x) and the 162M GPT (0.54x on the TF32 tape; about 6x behind bf16 eager and 9-10x behind its graph on the bf16s step). `2026-11-09-a100-exec-vs-torch` and `-exec-vs-graph` describe the pre-#13 executor and are superseded by `2026-11-10-a100-merged`; the earlier `bench/vm-wins` win over R sequential graphs did not reproduce on that pre-#13 main (R=32: 6615 vs 3565 us/step), and the merged panel's R=32 row now reads 46.3 vs 104.0 us.
- Root-caused 2026-11-09 (`2026-11-09-a100-flake-rootcause`), no open flakes of these three kinds: `recycle` long-running admission failed because
  `window_done` compared `==` on a wrapping table (fixed in 15994b4; 25 alternated A100 runs: pre-fix 10 failed, fixed 0); `test_ops` T=1 attention
  backward is deterministic rounding noise around an exact 0, so its fixed bound was a magic number (4.1e-5 seen over 2000 seeds, above the old 3e-5):
  now bounded per element from the data (worst 0.15 of the bound); the scalar `versions` `at_least` reader may legitimately see the next version's data
  (fix f595945; 600 alternated A100 runs: pre-fix 15 failed, fixed 0). Both suites now name the failing window and raw mark in their verdict line.
- Matmul is still below cuBLAS: 0.34-0.51x in the original pack (best 51.0 vs 105.6 TFLOP/s at 4096^3; the 70% target was missed), 0.56-0.73x TF32 with the 256x128 tiles (`2026-11-09-a100-matmul-parity`); attention 15-30% below
  torch fp32 SDPA (`2026-11-04-a100-ops-perf`).
- Nothing has run on more than one GPU, over NVLink, or over a network.
- The bandwidth arbiter gave no latency win (7% p99 on a synthetic chain, none with real tenants); it is a throttle.
- The single admission FIFO (1.9 us/task, 0.52M tasks/s) is the tail-latency limit: a big training record puts every other class behind
  100-300 ms; gating admission buys deadlines by shedding 38% of serve requests (`2026-11-04-a100-adaptive`).
- The adaptive controller did not beat a hand-set fixed priority configuration; the dynamic scheduler is 1.7-1.8x worse on serve p99 than a
  global queue with serve priority (`2026-11-04-a100-sched-integrated`).
- RTX 5070 Laptop (sm_120): `kvm.cu` passes but never clearly beats the batched graph, 0.73-1.12x (`2026-11-03-rtx5070-laptop-sm120`).
- `kvm.cu` (closed in the closeout PR): timeline (`--timeline`, `%smid`) and exact weight equality are in; on the A100 batching alone gives 9.5x of the 10.7x at 32 runs, and against compiled torch bmm/vmap graphs the TF32 VM loses at 32 runs (0.70x; `results/2026-11-06-a100-kvm-cu`).

## How the best do it (read before designing)

- **Hazy Research's [Megakernels](https://github.com/HazyResearch/Megakernels)**, the public ancestor: its lead author, Ben
  Spector, leads Flapping Airplanes, and "kittens" most likely means ThunderKittens, the library it builds on. `include/` is the VM:
  a controller warp (instruction fetch, a page allocator over 16 KiB shared-memory pages, semaphores, a timing store) beside
  loader, storer and launcher warps, and consumer warps that take extra registers from the others (`increase_registers`).
  `megakernels/` has the Python scheduler that builds each SM's instruction queue ahead of time, a Python reference VM and a diff
  test. `include/controller/timings_store.cuh` records per-worker, per-instruction timestamps, the kind of data their picture
  plots. Blogs: [Look Ma, No Bubbles!](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles) (a
  Llama-1B forward pass in one kernel, under 680 µs on a B200, 78% of an H100's memory bandwidth) and
  [We Bought the Whole GPU, So We're Damn Well Going to Use the Whole GPU](https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main),
  which their post 5/5 echoes.
- **[open-flapping-airplanes](https://github.com/Bobliuuu/open-flapping-airplanes)**, version 1: one persistent kernel trains a
  162M GPT (forward, backward and AdamW as tasks), serves it and runs Doom on one B200. Read
  [b200-run/README.md](https://github.com/Bobliuuu/open-flapping-airplanes/blob/main/b200-run/README.md) for matched baselines,
  and [llm/b200/gemm_tcgen05/README.md](https://github.com/Bobliuuu/open-flapping-airplanes/blob/main/llm/b200/gemm_tcgen05/README.md)
  before any Blackwell work: the occupancy API reports 1 block per SM for any kernel with a `tcgen05` instruction, and a block must
  allocate TMEM once and relinquish the permit at once, or co-resident blocks serialize or deadlock.
- **[Mirage Persistent Kernel](https://github.com/mirage-project/mirage)** (CMU) compiles a model graph into a megakernel task
  graph: the reference for generating tapes instead of writing them by hand.

## Build, run, check

```bash
nvcc -O3 -arch=sm_80 -std=c++17 -DKTILE=32 -DKKC=64 -DKSTG=4 kvm.cu -o kvm   # -arch: the target GPU's own
./kvm --selftest              # tile GEMM, all 4 transpose combinations against CPU fp32
./kvm --runs 8 --steps 300    # the VM against 3 baselines; must end in PASS
KVM_HOST=user@host ./run_remote.sh --runs 8 --steps 300   # rsync, build and run on a GPU box under flock
```

A change is done when `--selftest` passes and the run ends in `PASS` on the GPU its claim is about. Without a CUDA toolkit or
root, `micromamba create -p "$HOME/.local/cuda-12.9" -c conda-forge cuda-nvcc=12.9 cuda-cudart-dev=12.9 cuda-version=12.9`
gives a working `nvcc` in minutes (the pip `nvidia-cuda-nvcc-cu12` wheel ships `ptxas` but no `nvcc`).

## Measuring

- Correctness first: a number from a run that did not pass is not a result.
- Bound every result. Per step, the floor is the largest of FLOPs at peak, bytes at bandwidth, and the dependency chain's depth
  times the cost of one hop. Report how far each run is from it and what the gap is made of.
- Time every baseline eager and in CUDA graphs before claiming a win: open-flapping-airplanes beat Megatron eager (62.4 s) and lost
  to it graphed (43.1 s).
- Shared GPU boxes move absolute times between runs: quote ratios measured in one process, best of several, with the spread.
- Keep raw output under `logs/<UTC time>-<what>/` (git-ignored). Commit the result files a README cites, each with the GPU name and
  SM count, driver, CUDA version, build flags, command and commit.

## Working here

- **This repository is public.** Only code, README files, this file and `results/` are published. Everything else stays local,
  and `.gitignore` enforces it: all of `docs/` (session notes, per-task docs, plans, drafts), `logs/`, profiler output, `.env`
  and keys, and agent and editor folders. A new kind of private file gets its `.gitignore` line in the same change.
  `run_remote.sh` honors `.gitignore`, so ignored files never reach a GPU box either.
- Tracked files never carry email addresses, credentials, host addresses, personal paths or anything from other private
  projects. Hosts and keys come from the environment (`KVM_HOST`, `KVM_KEY`). Read `git diff --cached` for them before each
  commit.
- The owner pushes straight to `main`. Run `git fetch` and read `git log HEAD..origin/main` before you start and before you push;
  work on a branch, open a pull request, and never force-push or rewrite others' commits. Other agents may share this folder: use
  your own `git worktree`, and never `git stash`.
- Ask the user before spending money on GPU time, and state the price.

# User Preferences

- When dealing with bugs or reliability,  after you finishing all the code, use independent agent for edge case tests. The key is as clear context as possible while still having enough understanding of the code.
- write structured, production-quality code. readability, encapsulation, abstraction are key. Make sure you tried to cover all the edge cases, fall back, error handling, logging, and all other important things that speed up feedback loop and increase code quality. This doesn't mean more is better, always avoid overindexing, you don't need 1000 lines for a simple problem. Purposefully delete redundant code whenever you saw  them, keep the overall codebase as clean as you can.
- Document what you did consistently under docs\ with a .md file. The file format is "parent-directory-name-yyyy-mm-dd-hr-min-what-i-did.md". Do not create a docs entry for prompt wording/taste-only edits unless the user asks for documentation or the change alters a durable runtime contract.
- I prefer you dig deep when you approach a problem, use first principle. When come up ideal solutions, go in depth in trying a route that is the most elegant, failing miserably trying a robust solution (working without bugs, solve the problems 100% [can't be 90% or even 99%] with the least moving parts, following first principle) is still better than come up a working average solution. Being average is FAILURE.
- be proactive in exploring and trying good solutions, don't just wait for me to prompt you what to do, the code you hand to me should be the best of best you can make it to be instead of just following what I told you, unless you are not sure the goal of the project. When this dones right, this means: you should fix all related possible bugs when you fix one bug; You should create code that is easy to run but meticulously engineered; anything else should also follows this spirit.
- Keep loggings as clear and as detailed as possible, lack of logging is lack of clarity. This will lead to degrade of your performnce. Make sure to never log less, but you should improve logging clarity based on your improved understanding of the project. But do impelment loggings at different levels, and by default show all logs.
- For any of your fallback, safety net, error handling, make sure they work perfectly as intended, otherwise they are not solutions, they are bugs.
- For any logic that is outdated or not used, make sure you remove all the code associated, even it is thousands lines of code. you have to keep the code base clean rather than a bunch of slop.
- Also when you fix bugs, think outside of the box. Does your fix only address that one particular error? Or does it fix that types or even that categories of errors in a whole? Does it also prevent anything related from happening when we develop further? Is it safe by design? And if that fix doesn't pass these checks, it is not a fix, it is a cover up.
- For any sticky, or package related, or unclear but tracable bugs, go down to the source package code, search official docs, do whatever you can to gather core context, then write better logging systems - you want full cycle visibility and then you will find bugs much quicker, and much more accurate.
- When you write test, think about how user would use it. If they would use it from front end, then test from front end. Your test is not for you, it is to test when an actual user use it, what is that experience like. Of course, you test suite should be broader than that and go down each rabbit hole, but you have to test in the exact way the user would use it and from the exact place the user would use it.
- You should always have a copy of the debug logging somewhere local or your own console - even primarily you should save them to servers or db or cloud. You should do this under any circumstances, for example console output, cloud output etc, this doesn't mean you should save production logs (only apply this principle to dev logs). this is because you need to establish a strong and instant "feedback loop" which means you make edits, you run tests, you see everything happened and happening at a place you have instant access to, and you either knows exactly where and what is broken, or can conclude everything worked as expected. Try to save these loggings somewhere inside the current work directory with meaningful and intuitive timestamps, folder and file names.
- For any auxillary, temperary, or unneeded files, delete all of them after you don't need them to maintain a clean repo.
- Create appropriate .md files with all capital letters in its name as the operational entry point for this projects and constantly maintain the up-to-date content.
- Be careful with subagent and critic agents. Think of it this way, every time you use it, it incurs new additional cost, and if you do it too often, usually the return is less than the investment. You want to use it when you think the return is greater than the investment.
- I hate reading a lot. Use charts, diagrams, concise languages that is 100% on point and high-signal. 0 BS tolerance. Always try to articulate things using first principle, explain why, explain following the first principle, shallow level information is so bad because it doesn't create understanding, it creates generic answer. I want the understanding, the process, it usually takes you to ask 1 million whys, but that is first principle. Find the root, the then torso and limbs. Avoid generic information, and information overload at all cost.

Also when you prepare responses, don't just limit to my exact question, curiously find related high-value info and data that help me understand better. You answer needs to satisfy a curious Freak Boss who hates boring and generic stuff. 

Go/research even deeper for anything that might be/or seemingly ambiguous, conflicting, paradox, or confusing, or having forseable infomation/research gaps or anything that hurts understanding. Make sure you proactively address these stuff before I point them out. You will be punished if I have to point it out.

Also when you are trying to figure out information, always start from how the best does it, and only after you figured it out move to the remaining.
- For high-stake operations, always start from small size experiment, get the architecture/setup right first, and by design making sure the result is transferable to larger size, and natively handle common problems such as losing data, backup, tracing, and etc that fits the operation (you don't need to over do this, because it will cost resources and time), and don't let stupidity gets in your way, espicially don't make simple mistakes.
- When you update AGENTS.md (this file), do not touch "User Preference" section. Not event a character.
- While you working, for each session, you MUST create/use 1 Markdown file under `docs/` as your central source of truth. This is not for me, this is for you, make sure you can better manage context and progress. You should keep it high signal, put conflicts, ambiguity, to-do, progress, findings, thought, mistakes, and everything (there gotta be a lot more) that helps you track into there. At completion, move the doc into docs/archives/.
- When multiple agents working within the same session, using the source of truth doc is mision critial to avoid doing duplicate works, blocking works, and draw boundaries. The source-of-truth should be the collaborative ground in this case, clearly indicating who did who.
- Unless I clearly told you to pivot, do not take what I said too literally and easily deviate substantially from your original goal. Most times what I input is just new information that will help your work, do not understsand it as completely new instructions.
- When you report to me, make sure to only tell me things that are meaningful to me and not waste my bandwidth. you should also mention things I didn't ask explicitly ask, but help me understand the situation
- You are expected to remind me, correct me, or challenge me if you smell something off or even worse (such as my instruction is off, you find evidence to deny my instructions). I HATE 'Yes Man'. But do not stop working while doing so.
