# Runs: real bf16 storage (feat/generality-bf16)

Box: shared A100-SXM4-40GB (sm_80), private scratch, short `flock ~/.gpu.lock` holds, nvcc outside the lock. The box was memory-starved by
other users' ptxas (about 32 GB each): the full `test_extra` build (every op in one kernel) was OOM-killed three times, so the
bf16-storage op tests live in a light binary (`tests/gpu_ops/test_bf16.cu`, own kernel with only the ops under test).

| date | what | result | log |
|---|---|---|---|
| 2026-11-06 | `tests/gpu_ops/test_bf16.cu` (cast_f32_bf16 vs independent RNE incl. NaN/denormal/random bits, cast_bf16_f32 exact widening, matmul_tile_bf16s bitwise == matmul_tile_bf16 on widened operands over ta/tb x A/B/both bf16 x epilogue x padded ld, error paths) | 14 passed, 0 failed | `test_bf16-a100.txt` |
| 2026-11-06 | CPU: `python3 -I -m unittest discover -s tests` / `-s tests/frontend` (cluster box, no GPU) | 145 OK / 86 OK (incl. 8 new `test_bf16_storage`) | - |
| 2026-11-06 | end-to-end: generic MLP 64-256-256-10 (GELU, CE, SGD, 3 steps, a100 tiling) as f32 / bf16 (emulated) / bf16-real bundles through `kvm_exec --golden --golden-tol 5e-3` (`tests/bf16/gen_bundle.py`) | all 3 run to completion, device error 0. f32 golden OK (worst rel L2 1.6e-3). Both bf16 modes miss the 5e-3 tolerance by the same 6% (worst 5.33e-3, same 3 activation buffers, identical error pattern: 1xTF32 elementwise + bf16 rounding vs the CPU golden's f32). | `e2e-mlp-*.txt` |
| 2026-11-06 | bf16 (emulated) vs bf16-real dumps compared by buffer name (`tests/bf16/compare_dumps.py`) | weights and losses bit-identical (9 buffers, 0 differ); f32 vs bf16-real differ (as expected) | `compare-bf16-vs-real.txt` |
| 2026-11-06 | memory plan, reuse=shape, same MLP (`memory-plan.txt`) | per-tensor: bf16 tensors take 2 B/element (150,016 elements = 300 KB vs 600 KB as f32). NET pool is NOT smaller here (1.21 MB vs 1.04 MB): the f32 matmul output still exists next to its bf16 copy and the bf16 weight copies are extra. A bf16-output matmul epilogue would be needed for a net saving. | `memory-plan.txt` |
