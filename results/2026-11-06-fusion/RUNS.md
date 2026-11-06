# Runs (perf/fusion)
| what | where | script | result |
|---|---|---|---|
| bench cpu:0 / a100nm:1 / a100:1, mlp_ce 200 + xfmr_odd 50, 5 reps + torch | shared A100-40GB, one lock hold, private scratch | bench_job.sh | logs/bench-o2-*.txt, README table |
| test_extra (op pack incl. big-tile wrappers) | same | suite_job.sh | 2568 passed, 1 stale expectation (fixed to flag 64) |
| GPU vs CPU golden, executor functional suite | same | gate_job.sh | pending, see below |
