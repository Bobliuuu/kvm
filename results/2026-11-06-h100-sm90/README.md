# H100 (sm_90a): executor and op packs build and pass unmodified

Base commit: 6d31aba (`fix/window-done-wrap`); this branch adds only `tests/modal/run_h100.py` (no device code changed, so the A100 / sm_80 build is untouched and was not re-run).
GPU: NVIDIA H100 80GB HBM3 (Modal), driver 610.57.04. Toolchain: `nvidia/cuda:12.4.1-devel-ubuntu22.04` (nvcc 12.4). Arch flag: `-arch=sm_90a`.

| Binary | Build | Result | A100 reference |
|---|---|---|---|
| `test_exec` core (`tests/gpu/build_core.sh`, `KVM_ARCH=sm_90a`) | about 1 min | 8586 checks, 0 failed, 0 suites failed (`logs/core.log`) | |
| `test_exec` full (`tests/gpu/build.sh`, `KVM_ARCH=sm_90a`, all packs) | about 66 min on 16 CPU cores | 8586 checks, 0 failed (`logs/test_exec_full.log`) | 8582 on an older commit |
| `tests/gpu_ops/test_ops.cu` (`-O2 -arch=sm_90a --fmad=false -Xcompiler -ffp-contract=off`) | about 15 min | 1396 passed, 0 failed (`logs/test_ops.log`) | 1396 |
| `tests/gpu_ops/test_extra.cu` (same flags) | about 73 min | 2089 passed, 0 failed (`logs/test_extra.log`) | 2089 |

Nothing needed fixing: dynamic shared memory already opts in via `cudaFuncSetAttribute`, and the cp.async / mma code is valid on sm_90a. The logs are tails (last 30-80 lines).

Command (repo root): `modal run --detach tests/modal/run_h100.py --stage core|build|runbin:<bins> --tag <name>`. `build` compiles all three heavy binaries on a CPU-only container into a Modal volume (compile times are long, so no GPU is held for them); `runbin` runs them on an H100. Logs come back with `modal volume get kvm-h100-logs /<tag>/<file>`.

Not done: no Hopper-specific kernel (wgmma / TMA) and no H100 matmul TFLOP/s vs cuBLAS; time went to compile latency and Modal log plumbing. The matmul remains the sm_80 `mma` path running on H100 (unmeasured here).

Modal spend estimate (not read from the bill): about 1.5 H100-hours including some wasted early attempts (two runs stopped before producing logs, one run whose log never reached the volume) plus about 3 hours of 16-core CPU for builds: roughly 8 USD. Over the "few dollars" target because of those wasted attempts.
