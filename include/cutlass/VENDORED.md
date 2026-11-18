# Vendored CUTLASS headers

`include/cutlass/` and `include/cute/` hold an unmodified subset of NVIDIA CUTLASS 4.8.1
(commit `b2f249acd20392beeb16f5e5a31f4f1a4e38b3bd`), under its 3-clause BSD license (`LICENSE.txt` beside this file).

**What uses it:** `kvm_ops_gpt.cuh`'s `mm_cut_mainloop`, the 256x128 tile's mainloop for rounded operands. It runs CUTLASS 2.x's sm80
`MmaMultistage` at the tile's own configuration: 256x128x16, 4x2 warps of 64x64, 4 stages, the tile's 96 KiB. On an A100 it gives the
tile's own loop's bits, and it runs 10-20% faster on rounded operands.

**Why vendored:** the executor builds with `-Iinclude` alone, so the headers live where that flag finds them, and a key's hash
covers them like every other header.

**The subset:** the 173 headers a translation unit including `kvm_ops_gpt.cuh` reaches, listed by `nvcc -M` and copied as they are.
To refresh it from a CUTLASS checkout:

```
nvcc -M -arch=sm_80 -std=c++17 -Iinclude -I$CUTLASS/include tests/gpu_ops/test_ops.cu | tr ' \\' '\n\n' | grep "$CUTLASS/include" | sort -u
```

Copy each listed file to the same path under `include/`, together with CUTLASS's `LICENSE.txt`.
