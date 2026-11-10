# KVM closeout: goal 1-5 evidence matrix (2026-11-09)

Claim -> result directory -> verdict. `pass` = a committed, correct, measured result exists and supports the claim as worded in AGENTS.md; `partial` = a result
exists but supports only part of the claim, or contradicts it in some regime; `missing` = no result. Every result named here is on this branch (`perf/kvm-cu`
and `results/coopt-sh` were merged into it; `run/demo162` and `perf/combined` were already ancestors of main). Verdicts follow the stricter reading.

| # | Claim (AGENTS.md) | Result dir(s) | Verdict | Why |
|---|---|---|---|---|
| 1a | One launch runs a whole training run on every SM (`kvm.cu`, MLP) | `2026-11-06-a100-kvm-cu` (`s1`, `f16a`, `f16b`) | pass | 300 steps x 1-32 runs, one persistent launch, 108 SMs |
| 1b | Same for the generic executor, GPT incl. 162M | `2026-11-06-demo-large-trace`, `2026-11-06-a100-learning-162m`, `2026-11-06-demo-tiny` | pass | 1.95M tasks, err 0, one launch; 162M trains (loss falls) |
| 2a | Final weights bitwise equal to the same tape as ordinary kernels (`kvm.cu`) | `2026-11-06-a100-kvm-cu` | pass | check is now exact equality (counts differing elements, 0 in every run, all programs) |
| 2b | Loss within stated tolerance of an independent reference | `2026-11-06-a100-kvm-cu` (CPU fp32, 2e-5), `2026-11-06-a100-correctness`, `2026-11-04-a100-models`, `2026-11-04-a100-gpt-golden`, `2026-11-06-a100-combined` | pass | executor: golden gap is TF32 noise; PyTorch reference added in `a100-correctness` |
| 2c | Executor weights bitwise equal to the executor's own ordinary-kernel form | `2026-11-04-a100-exec-recycling`, `2026-11-06-a100-combined` | partial | version/slot coherence exact; not stated as a one-kernel vs per-op-kernel weight diff for the executor |
| 3a | `kvm.cu` faster than per-op launches, CUDA graph, batched (vmap-like) | `2026-11-06-a100-kvm-cu`, README | pass | 1.50x at 1 run, 1.63x at 8, 1.13x at 32 over the best of these |
| 3b | `kvm.cu` faster than a framework (eager, whole-run graph, `torch.compile`) | `2026-11-06-a100-kvm-cu/f16b` | partial | TF32 like-for-like: 1.50x / 1.51x / **0.70x (loses)** at 1 / 8 / 32 runs; fp16 operands vs torch fp16: 2.01x / 2.24x / 1.25x. The 32-run win is a precision switch, not the VM. |
| 3c | Executor faster than the strongest conventional execution | `2026-11-06-gpt162-vs-torch`, `2026-11-06-vs-pytorch`, `2026-11-06-a100-combined` | missing (negative) | measured and **lost**: 6-12x slower than torch at 162M, 3.2-12.6x slower than a CUDA graph at tiny/small. Goal 3 does not hold for the executor. |
| 4a | Timeline from the VM's own per-instruction timestamps (`kvm.cu`) | `2026-11-06-a100-kvm-cu/{s1,f16a,f16b}` (`timeline_*.csv`, `.png`; `%globaltimer` + `%smid`) | pass | caveat: 1 us timer ticks; profiled run is slower |
| 4b | Timeline of the executor drawn like theirs (rows = SMs, op-class colours) | `assets/timeline-*.png`; `2026-11-06-demo-tiny`, `2026-11-06-demo-large-trace`, `demo-large-bf16s` in `a100-combined` | pass | 106 SM rows, all tasks drawn, from `trace.csv` |
| 5a | Step overlapping the next | `2026-11-09-closeout/overlap`, `2026-11-07-a100-coopt-sh/pbt-overlap` | partial | executor window 40 vs 1: 1.41x (85.4 -> 60.5 ms for 40 steps), confounded with per-step host admission; isolated overlap inside kvm.cu: none (0.8-1.0x) |
| 5b | Many runs filling one GPU | `2026-11-06-a100-kvm-cu`, `2026-11-07-a100-coopt-sh` | partial | works (R=32 fused in one launch) but a batched graph also fills the GPU: the VM is 1.13x over it in TF32 and loses to compiled bmm; the in-kernel successive-halving / PBT results are the co-opt a graph cannot do |
| 5c | Different workloads sharing the GPU | `2026-11-06-demo-tiny`, `2026-11-06-demo-large-trace`, `2026-11-04-a100-sched-integrated`, `2026-11-04-a100-adaptive`, `2026-11-03-a100-sched`, `2026-11-03-a100-doom` | pass | train + serve + real Doom in one launch, 35 Hz held, Doom frame-exact; honest negatives on tail latency (FIFO) and adaptive control |

## Goal 5 detail

**In-kernel population control (a co-optimization sequential kernels cannot express)**, `results/2026-11-07-a100-coopt-sh` (merged from `results/coopt-sh`):
rung boundaries rank runs on device and retire them (`alive[r]`), later tiles skip. Small-model data-dependent retire 2.0x vs the best valid torch graph baseline,
big model 0.88x (loses); in-kernel PBT with held-out eval verified by forced replay. This is the "many runs filling one GPU, with data-dependent control" result.

**Step overlap**: measured in `results/2026-11-09-closeout/overlap/README.md` (1.41x, confounded) and the negative `pbt-overlap` result: per-column optimizer-to-forward overlap in
`kvm.cu` is correct but 0.8-1.0x of no-overlap, because single-buffered H1/dY keep the WAR on DW2 on the critical path.

## Minimum remaining work to call KVM finished

1. Goal 3 for the executor: either win somewhere against a whole-step CUDA graph (needs the admission FIFO fix, `perf/admission-bottleneck`, plus bigger tiles `perf/combined2`; both unmerged) or
   state in the README headline that goal 3 is `kvm.cu`-only and the executor loses. The README already says so; the matrix keeps it `missing (negative)`.
2. Goal 3b at 32 runs in TF32: `kvm.cu` loses to compiled bmm. Either close it (tile efficiency, ~10 TFLOP/s) or keep the stated loss.
3. Goal 5a: a positive, measured step-overlap win (double-buffered H1/dY), or keep it as a documented negative.
4. Merge the remaining unmerged branches whose results the README cites (`perf/combined2`, `perf/sched-tail`, `perf/admission-bottleneck`) or mark them dropped.
5. Hardware: nothing has run on more than one GPU or H100/B200 (no hardware). Out of scope for closeout unless boxes appear.
6. Known open failures: `recycle` long-running admission flake (2 of 3 runs), `test_ops` T=1 attention-backward report (AGENTS.md).
