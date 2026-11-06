# Runs log (feat/generality)

Box runs use a private scratch dir, nvcc builds outside the GPU lock, short lock holds for runs. Modal was not used (launch denied by the
permission layer; the coordinator's authorization is not a user approval).

| date | what | where | result |
|---|---|---|---|
| 2026-11-06 | python suites (frontend 93, runtime 150 tests incl. packs) | remote CPU | OK |
| 2026-11-06 | `tests/generic/run_box.sh` zoo vs CPU golden on A100 | see results/2026-11-06-generality | pending (lock queue) |
| 2026-11-06 | litmus weakpub/mp (serial build via `KVM_JOBS=1`) | A100 | pending (ptxas ~30 GB, box memory saturated by other agents) |

# Run log (generality: device-side control)

| date (UTC) | what | tree | result | logs |
|---|---|---|---|---|
| 2026-11-06 | `tests/ctl/run.sh` on A100 (sm_80): 10 bundles, loops with data-dependent trip counts (0..12 of 12; GEMM-body loop 1/7/12 of 14) and both branch directions, one `kvm_exec` build, `--golden` vs CPU VM | 69e90f8 (+docs) | 10/10 PASS, executor `pred_skipped` count equals the expected skipped tasks in every case | `results/2026-11-06-generality-ctl/ctl-summary.txt` |
