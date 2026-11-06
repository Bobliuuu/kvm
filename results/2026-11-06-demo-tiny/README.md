# demo tiny on the A100: `demo/run_demo.sh tiny`

One persistent kernel runs, concurrently: GPT (`gpt_tiny`) training in 20 windows of 10 steps, 5 inference requests on versioned weights,
and real Doom (700 frames), all driven only by `MANIFEST.json`. The check script compares against the CPU reference VM and the CPU Doom engine.

| | |
|---|---|
| GPU | NVIDIA A100-SXM4-40GB, 108 SMs (grid 108) |
| Driver / CUDA | 580.105.08 / nvcc 12.8.93 |
| Build | `nvcc -O3 -arch=sm_80 -std=c++17 -rdc=true -DKVM_WITH_GAME --fmad=false -Xcompiler -ffp-contract=off -lineinfo` |
| Flags | model gpt_tiny, 200 steps, window 10, request every 50, 700 frames, tiling a100, exact 3xTF32 GEMMs (`--precise`), sched dynamic |
| Command | `KVM_HOST=<alias> KVM_KEY=<key> KVM_REMOTE=<private scratch dir> demo/run_demo.sh tiny --lock-wait 14400` |
| Code | commit 5dcf3d9 (main + window_done wraparound fix); the GPU stage was re-run by hand from the generated `gpu.sh` after the first wrapper hit its 900 s GPU-lock wait (box shared) |

Results (`check.log`, `report.json`, `run.json` here; raw logs in the git-ignored `logs/20261106-tiny-demo/`):

| Check | Result |
|---|---|
| executor | PASS: 26565/26565 tasks in 4234.6 ms, err_code 0, 26 admissions, 1 ring wrap |
| version coherence | PASS: 184 version-carrying tasks, 184 exact, 0 violations |
| loss | PASS: 200 step losses 4.1460 -> 0.0042 (a memorisable bigram stream); max relative deviation from the CPU golden 3.78e-06 (tolerance 2e-03) |
| serving | PASS: 20/20 token buffers equal the golden decode of the same weights version |
| Doom bytes vs CPU | PASS: 700/700 frames raw-identical, fingerprint-identical and state-identical |
| 35 Hz cadence | PASS: 174.2 frames/s, tick gap p50/p99/max 5.61/10.13/10.62 ms, 0 of 698 gaps over 28.57 ms (8 early ticks are late by construction: boot tick 208 ms) |
