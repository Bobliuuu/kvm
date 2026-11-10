# Executor hot path: PR #6 and the admission branches rebuilt on #1's hand-off (A100, Modal), 2026-11-09

**Verdict.** On `gpt_162m` the executor is now **3.1x faster than main in f32** (1-step tape, 249 ms vs 774 ms per step, ab3) and runs
the bf16s tape again at 127 ms (main rejects it, see "bf16 storage" below). It is still **3.6x slower than a whole-step CUDA graph in f32**
(68.4 ms; eager 71 ms) and **2.8x in bf16s** (graph 44.7 ms, autocast eager 57.6 ms). At small sizes it is 4-7% faster than main
(`gpt_tiny` 833 vs 871 us/step) and 4-8x faster than torch eager, but it does **not** beat a whole-step CUDA graph at any size measured
(tiny 1.9x slower, small 2.4x). The target "beat a whole-step graph at small sizes" is not met; the cost breakdown below says why.

| | |
|---|---|
| Hardware | Modal `A100-40GB` request. ab0/ab1 ran on A100-SXM4-40GB; ab2 and ab3 were given an A100 80GB PCIe (same sm_80, 108 SMs). Every table compares variants **inside one container**; never compare numbers across tables. |
| Code | branch `perf/exec-handoff-port`; variants are git refs staged by `bench/exec_vs_graph/stage_variants.py` |
| Build | `nvcc -O3 -arch=sm_80 -std=c++17 -Iinclude --fmad=false -Xcompiler -ffp-contract=off -lineinfo -diag-suppress 177 kvm_exec.cu` (ab0/ab1 add `-DKVM_LEAN`: no comm/extra packs), built on a CPU container into the volume |
| Workloads | `bench/vs_pytorch/gen.py`, TF32, a100 tiling: `gpt_tiny` / `gpt_small` 50 steps; `gpt_162m` f32 and bf16s 1-step tapes (270,914 / 58,634 tasks) |
| Method | `bench/exec_vs_graph/modal_ab.py`: per round torch (eager, whole-step graph, compile + graph) then every executor variant REPS launches in a rotated order; us per optimizer step, medians over all launches |
| Raw | `raw/ab*/` (table.txt, summary.json, per-round JSON, ab.log, trace summaries) |

## What was ported (onto #1's completion hand-off, which stays)

| Piece | From | State |
|---|---|---|
| Ticket claim for scheduler-off pops (reserve on `avail`, fetch-add `head`, no CAS retries) | #6 / perf/admission-bottleneck | **default** (`--claim cas` restores the head CAS; the scheduler always uses the CAS: its accounting needs the scanned cell) |
| Chunked release: edges + window accounting first, descriptors decoded 4 x blockDim at a time and released as written (indegree add-then-check) | perf/chunked-release 2656be5 | on #1's staged copy of the record |
| One-record window sizing in the `kvm_exec` CLI | perf/admission-v2 72b2660 | default when no sizing flag is given |
| Continuation / chaining: the highest-priority released successor runs next on the same CTA unless a queued task outranks it | #6 (continuation) + perf/chunked-release 566de39 (chain rule) | default; `--no-cont`, `--cont-strict`, `--cont-above` |
| First successor id loaded before the op; argument words prefetched with the descriptor in the pop | #6 / new | default |
| Host stream prep without per-edge set inserts / adjacency copies | #6 | default |
| `KVM_LEAN` build flag; `KVM_CRIT_PRIO=1` critical-path priorities at bundle compile | #6; 566de39 | opt-in |
| #6's admitter staging (per-chunk int4 staging) and completion-path rewrite | #6 / perf/completion-path | **not taken**: #1's bulk staged copy and its completion hand-off (post-task bookkeeping on thread 32 during the next pop) cover the same costs; ab0 shows #1 alone already at 845 vs #6's 826 us/step on tiny |

## Results

### ab3: final defaults (A100-SXM4-40GB, full-ops builds, one container, 2 rounds x 3 launches)

| us/step (median) | main d12c206 | perf/chunked-release | **this branch** (ticket, continuation) | this branch --claim cas | this branch --no-cont | torch eager | torch graph | compile+graph |
|---|---|---|---|---|---|---|---|---|
| gpt_tiny | 871 | 857 | **833** | 846 | 841 | 3,441-3,925 | 433 | 204 |
| gpt_small | 2,003 | 2,107 | **1,861** | 2,167 | 1,742 | 6,057-6,501 | 764 | 429 |
| gpt_162m f32 (ms) | 774 | 273 | **249** | 462 | 248 | 71-85 | 68.4 | failed |
| gpt_162m bf16s (ms) | rejected | 139 | **127** | 138 | 133 | 57.6 | 44.7 | failed |

The branch was measured at 93f727c (device code identical to the PR head). `gpt_small` is noisy in this container: the torch graph's own
median moved between 650 and 835 us inside the round, and executor medians spread up to 25%; tiny and the 162M rows spread under 2%.
The torch.compile + graph baseline fails at 162M in this torch build (`IndexError` in the capture), so only eager and graph are given there.
Continuation is worth nothing measurable here (`--no-cont` within noise everywhere; #6 had measured 6-9% on its own tree); it stays on
because it costs nothing and the gates pass with it.


### ab2 (A100 80GB PCIe, full-ops builds; before ticket-by-default, so the port's default row is the CAS)

| us/step (median) | main d12c206 | perf/chunked-release | port (CAS) | port --claim ticket | port --no-cont | port --cont-strict | torch eager | torch graph | compile+graph |
|---|---|---|---|---|---|---|---|---|---|
| gpt_tiny | 732 | 700 | 675 | **632** | 713 | 689 | 2,631-3,052 | 335 | 155 |
| gpt_small | 1,830 | 1,765 | 1,778 | **1,696** | 1,768 | 1,771 | 4,942-5,688 | 648 | 334 |
| gpt_162m f32 (ms) | 726 | 243 | 456 | **230** | 486 | 487 | 71-72 | 70-71 | failed (torch IndexError) |
| gpt_162m bf16s (ms) | rejected | 133 | rejected | rejected | rejected | rejected | 46 | 43 | failed |

"rejected": `bad buffer shape or dtype` at bundle load (bf16 storage lost on main, fixed by #10 and on this branch since 93f727c).

### ab0 / ab1 (A100-SXM4-40GB, `-DKVM_LEAN` builds)

| us/step (median) | main | #6 tip 7801809 | #6 --no-cont | port bbc4502+ (CAS) | port --claim ticket | port --no-cont | torch graph | compile+graph |
|---|---|---|---|---|---|---|---|---|
| gpt_tiny (ab0) | 845 | 826 | 875 | | | | 432 | 201 |
| gpt_small (ab0) | 1,795 | 1,740 | 1,749 | | | | 834 | 426 |
| gpt_tiny (ab1) | 844 | | | 830 | 828 | 831 | 431 | 205 |
| gpt_small (ab1) | 1,797 | | | 1,749 | 1,741 | 1,780 | | |

The ticket claim is worth nothing on the lean builds (ab1) and 6-14% on the full-ops builds (ab2): the full GPT pack is a 255-register kernel at
one CTA per SM, where a CAS retry storm of ~107 pollers costs more. Not isolated further.

## Correctness

All on A100-SXM4-40GB (Modal), prebuilt binaries (`tests/gpu/build.sh`, CPU container), raw logs in `raw/gate2`, `raw/gate3`, `raw/gate4`.

| Gate | Tree | Result |
|---|---|---|
| CPU suites `tests` / `tests/frontend` | PR head | 153 OK / 124 OK (15 skipped) |
| `test_exec` (all suites) | 93f727c | **8609 checks, 1 failed**: `front` (below); every other suite passes |
| `front` with its race removed, 20 runs (`-DKVM_LEAN` build) | ea48d76 | 20 of 20 PASS |
| `test_exec` (all suites) after merging main (#15, #16) | 27de80d | **8609 checks, 0 failed** (`raw/gate5`) |
| compute-sanitizer memcheck / racecheck / synccheck on `smoke` | 93f727c | 0 errors / 0 hazards / 0 errors |
| `-DKVM_BREAK_FENCE --expect-fail versions` | 93f727c | 118 violations detected (NEGATIVE TEST PASS) |
| `-DKVM_BREAK_PARK_RECHECK --expect-fail litmus` | 93f727c | detected (NEGATIVE TEST PASS) |
| litmus (`tests/litmus/run_litmus.sh`, incl. `queue(ticket)` and mutant `popticket`) | ea48d76 | **LITMUS PASS**: 16 properties / scenarios / probes, 8 mutants and controls detected, 0 failures; `queue(ticket)` 1.0M rounds 0 violations, mutant `popticket` 2385 violations (detected), `popplain` livelocks (counted as detected, as before). The new e2e controls `KVM_BREAK_CHUNK_RELEASE` (dag, bigrecord) and `KVM_BREAK_CONT` (versions) were **not detected**: the e2e scenarios never let a predecessor finish before its successor is decoded, nor continue into a version-waiting successor. They are reported, not counted |

`front` failure: the suite submitted the same 100 chains twice (classes 1 and 0) onto ONE counter per chain, with no edge between the two
submissions, so the two copies of a chain race on `x = x + 1` once both are admitted; 1 of 100 counters came out wrong. The faster admission
makes the overlap likelier; it is a test race, not an executor bug. The test now gives each submission its own counters (ea48d76) and passes
20 of 20 runs, and the full suite passes on the merged tree (27de80d).

## bf16 storage regression on main (found here, fixed by #10)

Merge #8 took #1's `kvm_exec.cu`, which predates real bf16 storage (d48f3e6): main allocated every buffer at 4 bytes, rejected dtype 2 and lost
`--dump-bin`, so every bf16s bundle failed at load (`[kvm_exec][E] bad buffer shape or dtype`, ab2). The host code was restored verbatim from
1c2111b in PR #10 (merged a99fdca) and on this branch.
The #15 merge (7ce1932) then reverted #10's lines in `kvm_exec.cu` (its only change to that file), so main lost bf16s again; this branch keeps
the restored file when it merges main (27de80d). The bf16s rows of ab3 ran on this branch's code (93f727c).

## Why a whole-step CUDA graph still wins at small sizes

`gpt_tiny` is a chain of 28 dependent levels per step with 3-8 ready tasks per level; 107 workers are 1.4% busy (trace, ab1). Per level the
executor pays the op (a single 256-thread CTA, 6-14 us for ops that run 3-5 us as graph kernels: matmul_tile 13 us, adamw 14 us,
gradnorm_partial 7 us), the release (fence, indegree atomic, enqueue) and the pop (mean ready-to-start 9 us). A graph edge costs about 1.3 us.
Continuation removes the pop on the chain but the op bodies and the release stay: at ~22-30 us per level x 28 levels the step cannot reach the
graph's 335-433 us. Beating it needs op bodies at kernel speed (multi-CTA ops, not this executor change) or fewer levels (fusion in the
frontend), not cheaper dispatch.

## Not done / honest negatives

- Small sizes: no win over a whole-step graph (above). The executor beats torch eager 4-8x there.
- 162M: 3.6x (f32) / 2.8x (bf16s) slower than a whole-step torch graph; the remaining gap is tile speed (matmul about 0.5x cuBLAS, owned by the ops work) and idle workers, not measured further here.
- `KVM_CRIT_PRIO` is ported but not measured on this branch.
- The chunked-release and continuation negative controls are not caught by any existing scenario (litmus row above); a targeted e2e scenario for each is the follow-up.
- Class-aware / sliced admission (perf/class-admission: latency ring + sliced bulk ingestion; perf/sched-tail: express records, strict class tier) is NOT ported: it targets tail latency of mixed classes with the scheduler on, not step time, and conflicts with #1's admitter and KvmFront. It is the next item for the admission-FIFO tail-latency limit.
- Multi-GPU, the Doom/serve demo and the scheduler-on paths with the new admitter were not timed (the exec suite covers them functionally).
