# perf/combined (main + fusion-162m + attention-162m + bf16-a100) on the A100: results (2026-11-06/07) -- PENDING, partial

Status: merge committed (7a758c7); results below are filled in as runs finish. Nothing here is a final claim yet.

| item | state |
|---|---|
| CPU suites (`tests` 153, `tests/frontend` 81) | green at 7a758c7 |
| exec suite (Modal A100-40GB, `tests/modal/run_suite.py`) | 8586 checks, 0 failed (`modal/exec.log`) |
| `--gpt` bundle suite (Modal) | GPT PASS (`modal/gpt.log`) |
| litmus (Modal) | running |
| test_bf16, test_ops, model zoo (24 + bf16 cases), 162M step-1 gradients vs float64 | queued on the shared lab A100 (GPU lock contention) |
| step time f32 / bf16s vs main (interleaved) | queued on the lab A100 |
| demo large (bf16s) + trace | running on Modal |

Modal runs: A100-SXM4-40GB, driver 610.57, nvcc 12.8.93, flags of `tests/gpu/build.sh`.
