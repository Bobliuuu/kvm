# Runs: real bf16 storage (feat/generality-bf16)

Box: shared A100-SXM4-40GB (sm_80), private scratch, short `flock ~/.gpu.lock` holds, nvcc outside the lock. The box was memory-starved by
other users' ptxas (about 32 GB each): the full `test_extra` build (every op in one kernel) was OOM-killed three times, so the
bf16-storage op tests live in a light binary (`tests/gpu_ops/test_bf16.cu`, own kernel with only the ops under test).

| date | what | result | log |
|---|---|---|---|
| 2026-11-06 | `tests/gpu_ops/test_bf16.cu` (cast_f32_bf16 vs independent RNE incl. NaN/denormal/random bits, cast_bf16_f32 exact widening, matmul_tile_bf16s bitwise == matmul_tile_bf16 on widened operands over ta/tb x A/B/both bf16 x epilogue x padded ld, error paths) | 14 passed, 0 failed | `test_bf16-a100.txt` |
| 2026-11-06 | CPU: `python3 -I -m unittest discover -s tests` / `-s tests/frontend` (cluster box, no GPU) | 145 OK / 86 OK (incl. 8 new `test_bf16_storage`) | - |
