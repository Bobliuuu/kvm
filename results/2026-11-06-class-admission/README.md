# Class-aware / sliced admission (2026-11-06) - STATUS: partial

Branch `perf/class-admission` (base `fix/window-done-wrap` 6d31aba). Design: `include/kvm_exec.cuh` (admitter), `kvm_exec.cu` (host), tests `tests/gpu/test_exec.cu` suite `latlane`.

## What was verified
- GPU: NVIDIA A100-SXM4-40GB, 108 SMs, driver 580.105.08 (CUDA 13.0), nvcc 12.8, `-O3 -arch=sm_80 -std=c++17 --fmad=false -ffp-contract=off`, command `./test_exec latlane` (build `tests/gpu/build.sh`).
- `latlane`: 327 checks, 0 failed (bitwise-correct results, every task exactly once, per-lane admission order, 60 lat records admitted inside a bulk record's admission, lat-ring wrap, a lat task parked on a version of a half-admitted bulk record, a lat record admitted while a bulk record waits for window room, errors from a late slice and from the lat lane).

## NOT yet verified (the shared GPU was held by another job for hours, so these runs were still queued)
- Full `test_exec` default suites (8582+ checks), compute-sanitizer memcheck/racecheck/synccheck, the mutant (`run_exec_tests.sh --negative-lat`), `test_adaptive admit_gran`, and the adaptive before/after batch (`adaptive/run_adaptive.sh lat`, `lat_big`). No latency ratios are claimed.
