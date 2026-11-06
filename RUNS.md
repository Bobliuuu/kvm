# Run log (generality: device-side control)

| date (UTC) | what | tree | result | logs |
|---|---|---|---|---|
| 2026-11-06 | `tests/ctl/run.sh` on A100 (sm_80): 10 bundles, loops with data-dependent trip counts (0..12 of 12; GEMM-body loop 1/7/12 of 14) and both branch directions, one `kvm_exec` build, `--golden` vs CPU VM | 69e90f8 (+docs) | 10/10 PASS, executor `pred_skipped` count equals the expected skipped tasks in every case | `results/2026-11-06-generality-ctl/ctl-summary.txt` |
| 2026-11-06 | `tests/gpu/build.sh` + `test_exec` functional suite, A100, tree with predicated tasks | 69e90f8 | 8607 checks, 0 failed, 0 suites failed (incl. dag, versions, recycle, litmus suites) | `results/2026-11-06-generality-ctl/reg-functional.txt` |
| 2026-11-06 | `test_exec bench` hot-path cost, flags==0, 3 runs each, base = origin/feat/integration 1ca85d1 vs new | 69e90f8 | independent noop @108 blocks 1.415 us/task both; dependent hop 8.60 (base) vs 8.64 us (new), +0.4%, within run-to-run noise | `results/2026-11-06-generality-ctl/reg-bench.txt` |
