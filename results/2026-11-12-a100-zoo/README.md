# The model zoo on the merged executor, on an A100 (2026-11-12)

The top README listed 2 of the zoo's 28 cases as failing their weight check: `vit_bf16s` and `san_bf16s`. The check itself was since fixed:

- `eeac44e`: the bf16 weight gate is an AdamW replay on the GPU's own gradients.
- `4d09833`: the encoder key bias is judged as structurally zero.

No A100 re-run had been recorded. This is that re-run, on this tree (`4bbdaab`).

## Setup

- **GPU:** NVIDIA A100-SXM4-40GB (sm_80), driver 590.48 (CUDA 13.1).
- **Executor:** this tree's `kvm_exec.cu`, as a full build: every op pack, including the extra pack the zoo uses, so no `-DKVM_LEAN`. Built with:
  `nvcc 12.8.93 -O3 -arch=sm_80 -std=c++17 -Iinclude --fmad=false -Xcompiler -ffp-contract=off -diag-suppress 177 -Xptxas --split-compile=16`.
- **Run on the GPU machine:**
  1. Bundles and CPU goldens: `examples/models/prebuild.sh`, for `cases.txt` and then `cases_bf16s.txt`.
  2. Runs and checks: `KVM_EXEC_BIN=<that binary> examples/models/run_box.sh`, once per list (`--cases examples/models/cases_bf16s.txt` for the second). It runs compute-sanitizer (memcheck, racecheck and synccheck) on the `san_*` cases.
- **Raw files:**
  - `runs_cases.txt` and `runs_bf16s.txt`: the runner's logs.
  - `cases/` and `cases_bf16s/`: one folder per case, with `check.txt`, `check.json` and `times.txt`. The two lists share some case names, so each list has its own folder.

## Result

Every case passes `check.py`, and every sanitizer run exits 0.

| list | cases | pass | `san_*` cases | sanitizer runs (all exit 0) |
|---|---|---|---|---|
| `cases.txt` | 28 | 28 | 9 | 27: 18 memcheck/synccheck summaries with 0 errors, 9 racecheck summaries with 0 hazards |
| `cases_bf16s.txt` | 8 | 8 | 2 | 6: 4 with 0 errors, 2 with 0 hazards |

**The two cases the README listed**, from their `check.txt` in `cases/`:

- **`vit_bf16s`:** the parameters whose gradient is structurally zero (the encoder key bias among them) are judged at the noise level. The largest GPU gradient there is 2.8e-4, against a scale of 0.81.
- **`san_bf16s`:** the weights still differ from the golden by 1.74e-2, which is what the old gate failed (1.73e-2 against 1e-2). AdamW replayed on the GPU's own gradients reproduces the GPU's weights to 5.27e-8, and the same replay on the golden's gradients reproduces the golden's weights to 5.28e-8. So the difference is the bf16 gradients' rounding carried through AdamW, not a fault.
