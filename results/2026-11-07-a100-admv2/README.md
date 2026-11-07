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
