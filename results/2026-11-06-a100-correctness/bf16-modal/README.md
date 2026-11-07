# bf16 zoo weight check (item b), re-run on a Modal A100

| | |
|---|---|
| GPU | NVIDIA A100-SXM4-40GB (Modal), sm_80, driver 610.57.04, CUDA 12.8.93 |
| Tree | `origin/perf/bf16-a100` (real bf16 storage), `tests/modal/run_suite.py` from `origin/tools/modal-suite`, `kvm_exec` built `nvcc -O3 -arch=sm_80 -std=c++17 -Iinclude --fmad=false -Xcompiler -ffp-contract=off -lineinfo -diag-suppress 177` |
| Command | `modal run --detach tests/modal/run_suite.py --suite custom --tag bf16zoo --cmd "source tests/modal/zoo_bf16.sh"` (script: build, then `gen_bundle.py` + `kvm_exec --dump-buffers` of `vit_bf16 vit_bf16s san_bf16s llama_bf16s` from `examples/models/cases_bf16s.txt`); log `bf16zoo/custom.log`. Dumps (370 MB) not committed |
| Checked with | `examples/models/check.py` of this branch, `--mode lowp`; `meta.json` completed with `invariant_params` by this branch's frontend (the bf16 branch's `gen_bundle` predates it) |

The Modal run reproduces the lab-box numbers digit for digit (vit_bf16s weights 0.90 before the fix, san_bf16s 1.73e-2, llama_bf16s 8.54e-3): the failures were the check, not noise of one machine.

| case | before | after (`check_*.txt`) |
|---|---|---|
| vit_bf16, vit_bf16s | FAIL weights 0.98 / 0.90 (key bias `vit.h0.attn.k.b`) | PASS: key bias judged in absolute terms (envelope 3.0e-2, measured 2.0e-2 / 1.4e-2; moments 1e-7 vs 7.8e-4); live weights 2.2e-3 / 1.8e-3 |
| san_bf16s | FAIL weights 1.73e-2 vs 1e-2 | PASS by the optimizer replay (below) |
| llama_bf16s | PASS 8.5e-3 | PASS |

## san_bf16s: AdamW amplification, not a defect (`examples/models/adam_amplification.py`, `amplification_*.json`)

AdamW is replayed in float64 from the common initial weights on each run's own raw per-step gradients (kept by `reuse none`), with the hyper-parameters, schedule, loss scale and clip read from the workload:

| san_bf16s (2 steps) | worst tensor rel L2 |
|---|---|
| replay(golden gradients) vs golden weights (validates the replay) | 5.3e-8 |
| **replay(GPU gradients) vs GPU weights** | **5.3e-8**: the optimizer did exactly the right thing with the gradients it got |
| GPU vs golden weights (the old gate) | 1.73e-2 (`llama.h0.attn.o.w`) |
| GPU vs golden gradients, step 1 / step 2 | 1.7e-3 / 2.1e-2 |
| weight difference after step 1 alone, replayed | 1.77e-2: a 1.7e-3 gradient difference becomes 1.8e-2 in the weights (AdamW divides by sqrt(v): elements whose gradient is below the rounding noise move by +-lr or not) |
| step-2 gradient difference explained by the step-1 weight difference alone (exact float64 gradient of the step-2 batch at the two replayed weight sets, no rounding) | predicted 2.03e-2 against observed 2.12e-2 |

So the divergence is created at step 1 by gradient rounding noise (within the 6e-3 step-1 gradient gate, 4.8e-3 measured) and the step-2 gradient difference is that weight difference fed back, not a second error. The 10-step Llama (8.5e-3) has the same structure.
A control with i.i.d. noise of the measured relative size on the golden gradients gave only 1.2e-3 (a first version of the script): the real rounding differences are not i.i.d. (they concentrate on small-gradient elements, where AdamW's step is a sign function), so that control does *not* reproduce the spread and is not used; the replay and the feedback prediction are the evidence.

Fix of the check (`check.py`, lowp): the `weights rel L2 <= 1e-2` gate (no basis: the divergence has no fixed size, 1.7e-2 at 2 steps, 8.5e-3 at 10) is replaced by (1) the replay of AdamW on the GPU's own gradients must reproduce the GPU weights (<= 1e-5, measured <= 1.6e-7 on four cases) and its twin on the golden validates the replay, plus the existing gates on the gradients themselves (every step's loss, step-1 activations and gradients). The golden-vs-GPU weight difference is printed with the per-step gradient differences. The old gate stays only for bundles whose `meta.json` has no model config.
Limit: the replay checks the optimizer, not the later-step gradients' accuracy beyond the loss gate (1e-3 on every step's loss).
