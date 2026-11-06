# Run log (generality: device-side control)

| date (UTC) | what | tree | result | logs |
|---|---|---|---|---|
| 2026-11-06 | `tests/ctl/run.sh` on A100 (sm_80): 10 bundles, loops with data-dependent trip counts (0..12 of 12; GEMM-body loop 1/7/12 of 14) and both branch directions, one `kvm_exec` build, `--golden` vs CPU VM | 69e90f8 (+docs) | 10/10 PASS, executor `pred_skipped` count equals the expected skipped tasks in every case | `results/2026-11-06-generality-ctl/ctl-summary.txt` |
