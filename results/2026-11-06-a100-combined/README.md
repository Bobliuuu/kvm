# perf/combined on the A100: main + fusion-162m + attention-162m + bf16-a100 (2026-11-06/07)

**Verdict.** The four branches compose. gpt_162m, 1 step, `kvm_exec`, median of 7 interleaved rounds in one container (best-worst in brackets):
**main 1092.9 ms -> combined f32 845.5 ms (1.29x) -> combined bf16s 293.3 ms (3.73x)**. Tasks 318,139 -> 270,914 -> 58,634; edges 6,889,225 -> 2,251,649 -> 480,777.
All correctness suites pass except the two known weight-check failures. One regression is open: the merged executor is ~6% slower than main's on the same (main) bundle.

## Setup

| | |
|---|---|
| GPU | Modal NVIDIA A100-SXM4-40GB (suites, demo, step times: `steps/env.txt`, driver 610.57.04, nvcc 12.8.93, 108 SMs, max SM clock 1410 MHz). `grads/` and `bisect/` ran on an A100-SXM4-80GB container (same driver; recorded in their `env.txt`). Ratios are only taken inside one container. |
| Code | branch `perf/combined`; merge commit 7a758c7 (conflict resolution below), results added on top |
| Executor build | `nvcc -O3 -arch=sm_80 -std=c++17 -Iinclude --fmad=false -Xcompiler -ffp-contract=off -lineinfo -diag-suppress 177 kvm_exec.cu` (main built the same way) |
| Runner | `tests/modal/run_suite.py` (`--suite exec\|litmus\|gpt\|sanitize` or `--suite custom --cmd`), commands in `steps/cmd.sh`, `bisect/cmd.sh` |
| Model | gpt_162m: 12 layers, d768, 12 heads, seq 1024, batch 4, vocab 50304, tiling a100 |

## Merge

Conflicts: `bench/vs_pytorch/{gen,grads162,torch_bench}`, `tests/gpt162/torch_ref` took the bf16 side (a superset; `gen.py --reuse` default set to `step`).
`lower.py`, `nn.py`, `plan.py` are unions (planner keeps fusion's `step` horizon and bf16's dtype-sized accounting). `harness.cuh` uses one `KVM_HARNESS_NO_FULL` guard (`test_bf16` renamed to it).
Composition fix: bf16 weight-gradient tiles are 128 rows tall (`MMA_TILE`), fusion's optimizer chunks rounded to 64; `storage.py` now tags `adamw`/`grad_clip` with `tile_rows=128` when a bf16 matmul lowers to the MMA tile, and `lower._opt_chunks` honours it (f32 unchanged). Tests added in `tests/frontend`.
Attention stays fp32 in bf16s and uses 128-row tasks; bf16 weight shadows, bf16 saved pre-activations and n-outer tile order are all active in the bf16s bundle.

## Results

| check | result | files |
|---|---|---|
| CPU suites | `tests` 153 OK, `tests/frontend` 81 OK | |
| executor suite | 8586 checks, 0 failed | `modal/exec.log` |
| litmus | 15 properties/scenarios passed, 7 mutants detected, 0 failures | `modal/litmus.log` |
| `--gpt` bundle suite | GPT PASS | `modal/gpt.log` |
| smoke under compute-sanitizer (memcheck, racecheck, synccheck) | 0 errors | `sanitizers/exec_smoke_sanitize.log` |
| attention cases under memcheck / racecheck / synccheck (`test_ops`, `TEST_ONLY=attention`, 1941 checks) | 0 errors, 0 hazards | `sanitizers/attn_*.log` |
| `test_bf16` (685) and its memcheck / racecheck / synccheck (429, small mode) | 685/0; sanitizers 0 errors | `ops/` |
| `test_ops` | 3224 passed, 0 failed | `ops/test_ops.log` |
| model zoo vs golden, all 28 cases (24 + bf16/bf16s) incl. sanitizers on `san_*` | 26 PASS, 2 FAIL (known, not hidden): `vit_bf16s` weights rel L2 0.90 (zero-gradient key bias, same in the emulation); `san_bf16s` weights 1.73e-2 vs 1e-2 | `zoo/zoo_cases_all.log` |
| 162M step 1 vs PyTorch float64, f32 | loss +9.6e-6 (torch-tf32 same), worst tensor 1.46e-3 = torch-tf32 1.46e-3 | `grads/f32/` |
| 162M step 1 vs PyTorch float64, bf16s | loss +2.7e-5, worst tensor 1.69e-2 (gpt.h11.attn.k.w), groups 5e-3 to 1.2e-2; torch autocast-bf16 worst 3.36e-2. Same as the bf16 branch (1.7e-2) | `grads/bf16s/` |
| demo large, bf16s (`remote_demo.sh` preset large, tiling a100, lr 3e-4, 2 warmup) | ALL CHECKS PASSED: 392,630 tasks in 5.158 s, err_code 0; 22,935 version-carrying tasks exact, 0 violations; losses 10.979 -> 10.924; Doom 700/700 frames raw/fingerprint/state identical; 164.9 frames/s against 35 Hz, 0 of 698 tick gaps over 28.6 ms (p99 13.4 ms). Training steps 1-6 end at 0.64, 1.00, 1.43, 1.77, 4.73 (Doom overlap), 5.07 s: about 0.34-0.36 s per step when Doom is not running | `demo-large-bf16s/` (`trace_compact.npz` = every task) |

Serving is SKIP in the demo (no golden at this size); coherence is covered by the version check.

## Step time (`steps/ab_runs.txt`, 7 rounds, one process per launch, one GPU, executors built in the same container)

| run | tasks | edges | ms/step best / median / worst |
|---|---|---|---|
| main executor, main bundle | 318,139 | 6,889,225 | 1089.0 / 1092.9 / 1113.1 |
| combined executor, combined f32 bundle | 270,914 | 2,251,649 | 843.6 / 845.5 / 850.6 |
| combined executor, combined bf16s bundle | 58,634 | 480,777 | 288.3 / 293.3 / 299.4 |
| combined executor, **main's** bundle | 318,139 | 6,889,225 | 1149.4 / 1160.2 / 1165.1 |

The fusion branch alone reported 868 ms and attention alone 905 ms; combined f32 is 845 ms. Still far from PyTorch (59-70 ms tf32, 31 ms autocast+graph at this size): README goal 3 is not met at 162M.

## Open regression: the merged executor is slower on identical work

Same bundle (main's), 7 interleaved rounds, executors from main / main+attention / main+bf16 / combined (`bisect/ab_runs.txt`, A100-80GB container):

| executor | median ms | vs main |
|---|---|---|
| main | 1073.6 | 1.000 |
| + attention-162m | 1091.4 | +1.7% |
| + bf16-a100 | 1119.3 | +4.3% |
| combined | 1137.4 | +5.9% |

Attribution so far: the cost is spread over both branches (attention +1.7%, bf16 +4.3%, roughly additive). The mechanism is not proven. Candidates: the bf16 branch raises the dynamic shared memory of the persistent kernel from 72 to 96 KiB (`KVM_GPT_MMA_SMEM`, `include/kvm_ops_gpt.cuh`) and adds op 122 to the in-kernel dispatch; the attention branch adds larger 128-row tasks to it. Register / stack numbers of main vs combined from `-Xptxas -v` are in `ptxas/` if that run finished. A fix needs a tile that fits 72 KiB or a per-op smem opt-in, which is not cheap; it is documented, not fixed. The net effect on the real (combined) bundles is still -23% (f32) and -73% (bf16s).

## Not done / limits

* Executor stays 8-9x slower than PyTorch's best at this size (293 ms vs 31 ms).
* The ~6% executor regression above.
* Everything ran on Modal; the lab box was not used (its GPU lock had ~18 waiters).
