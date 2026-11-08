# admission-v2 round 1: merge perf/admission-bottleneck + perf/completion-path onto main (A100, 2026-11-07)

Branch `perf/admission-v2` = origin/main (6c1391e) + `perf/admission-bottleneck` (ticket-based ready-queue claim, batched admitter loads)
+ `perf/completion-path` (cheaper task end). Both merged without conflicts. No new kernel code in this round.

gpt_162m, 1 step, `kvm_exec --quiet`, same container (Modal A100 80GB PCIe, driver 610.57, nvcc 12.8.93), executors built from main and
from this tree, bundles generated once (`gen.py --tiling a100`, f32 and bf16s), 5 interleaved reps (`raw/run.log`, `tests/modal/admv2.sh`):

| bundle | main ms/step (5 reps) | this branch ms/step (5 reps) | speedup |
|---|---|---|---|
| f32 (270,914 tasks) | 798.9 800.5 801.7 801.8 800.4 | 441.7 443.3 444.8 447.6 448.7 | 1.80x |
| bf16s (58,634 tasks) | 265.3 265.9 266.3 268.7 269.7 | 214.9 217.4 217.9 219.2 219.9 | 1.23x |

Absolute numbers differ from the README's (845 / 293 ms, SXM4-40GB); ratios are only valid inside this container.
After the change the bf16s step has 41% worker utilisation, 9.2 s summed task time over 107 workers (86 ms if perfectly packed) against a 210 ms span;
matmul_mma_bf16 is 43% of task time. Correctness of this merge is not yet re-run (see later commits).

## Round 2: the host-drained windows were the idle workers (r2/, r3/)

Cause: `kvm_exec`'s default `live_cap` 16384 / `ring_words` 65536 cut the 58k-task bundle into ~65 windows, and `submit_stream` submits a window only after
the windows holding its producers have COMPLETED (host polling `win_done`): a drain per window. Trace of the default run: 41.5% utilisation. Same executor with the
bundle as one record (`--live-cap 65536 --ring-words 16777216`): 71.9% (`r2/`), 217 -> 150 ms. Fix (`kvm_exec.cu` main): when no sizing flag is given and a single
bundle fits, size live_cap / max_tasks / ring / succ ring so it is one record.

Final A/B (`r3/ab.txt`, A100-SXM4-40GB, one container, 5 interleaved reps, main binary vs this branch, same bundles; `kvm_exec --quiet` kernel_ms):

| bundle | main | this branch |
|---|---|---|
| bf16s, 58,634 tasks | 289.4 291.8 295.4 290.2 294.4 (median 291.8) | 151.5 150.4 151.0 149.0 150.6 (median 150.6) = **1.94x** |
| f32, 270,914 tasks | 850.5 846.6 861.5 844.6 854.1 (median 850.5) | 336.0 327.9 329.9 323.9 326.5 (median 327.9) = **2.59x** |

Breakdown for bf16s (default, A100-80GB PCIe container): 267 (main) -> 217 (ticket claim + completion path, merged branches) -> 150 (window sizing; SXM4 numbers).
Still 3.2x slower than torch autocast eager (47 ms) and 4.8x slower than its CUDA graph (31 ms) from the README. Remaining in the 150 ms run: the first 19 ms are idle
(one record is admitted completely before any task is released; `r2` trace), then ~72-90% busy, with a matmul-dominated body (mean task 160 us).

## Correctness on this tree

| check | result |
|---|---|
| executor suite (`suites/exec.log`) | 8613 checks, 0 failed, PASS |
| litmus | inside the exec suite: "litmus: PASS (24.0s, 0 new failures)" (`suites/exec.log`). The separate litmus suite (mutants) did not finish within ~2 h (build of 9 executors, then no output) and was stopped; not counted |
| 162M step-1 gradients vs PyTorch float64, bf16s (`r3/num_bf16s`) | loss +2.67e-05, worst tensor 1.69e-02 (gpt.h11.attn.k.w), identical to main's 1.69e-02 |
| same, f32 (`r3/num_f32`) | worst tensor 1.46e-03 = torch-tf32's own 1.46e-03 |
| not re-run | sanitizers, zoo, demo (Doom + serve), 300-step learning curve vs torch |
