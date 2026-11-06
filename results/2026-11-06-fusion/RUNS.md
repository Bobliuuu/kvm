# Runs (perf/fusion)
| what | where | script | result |
|---|---|---|---|
| bench cpu:0 / a100nm:1 / a100:1, mlp_ce 200 + xfmr_odd 50, 5 reps + torch | shared A100-40GB, one lock hold, private scratch | bench_job.sh | logs/bench-o2-*.txt, README table |
| test_extra (op pack incl. big-tile wrappers) | same | suite_job.sh | 2568 passed, 1 stale expectation (fixed to flag 64) |
| GPU vs CPU golden (mlp 3 steps; xfmr 1 step, same script with steps=1: golden VM times out at 3) for cpu:0, a100nm:1, a100:1; executor functional suite + litmus | same | gate_job.sh | mlp PASS in all three with identical error numbers (weights 2.0e-4 rel L2); xfmr identical numbers in all three (weights 7.7e-5, loss 2.4e-6; compare.py's 'loss decreases' check cannot hold for 1 step); exec suite 8607 checks 0 failed, litmus PASS (logs/gate-*.txt) |
