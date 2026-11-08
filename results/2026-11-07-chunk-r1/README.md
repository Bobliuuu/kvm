# chunked release (perf/chunked-release), A100-SXM4-40GB, Modal, 2026-11-07
Change (include/kvm_exec.cuh): admission edges first, then descriptors in 256-task chunks; each task released as soon as its chunk is written
(indegree ADD of the true count; old+ind==0 => admitter releases, else last predecessor's decrement does), next-chunk PCIe prefetch. indeg memset to 0.
Same container, 5 interleaved reps (ab.txt). ex_new_a = admission-v2 round 1 (windowed, NOT the later one-record sizing build).
| gpt_162m | main | admv2 r1 (windowed) | chunked release (one record) |
|---|---|---|---|
| bf16s, 58,634 tasks | 286-292 ms | 214-217 | 133.8-137.1 (median 135.3) |
| f32, 270,914 tasks | 822-836 | 446-460 | 254.4-263.5 (median 257.7) |
(admv2 final one-record, other container: bf16s 150.6, f32 327.9.) Utilisation 70.7% (trace_b16.txt); admitter idle 1.4 ms, copy+decode 147 ns/task.
Gates: exec suite 8613 checks 0 failed incl. litmus (exec.log); gradients vs fp64: bf16s worst tensor 1.69e-02 (same as before), f32 1.46e-03 (= torch-tf32) (r2_custom.log).
Coarsening sweep (sweep.txt): bigger elementwise tiles are SLOWER: bf16s 49.5k tasks 217 ms, 51.8k tasks 190 ms vs 135 ms base; f32 267k tasks 272 vs 258. Keep a100 preset.
Not run: separate litmus-mutant suite, 300-step loss curve, Doom/serve demo, sanitizers.
