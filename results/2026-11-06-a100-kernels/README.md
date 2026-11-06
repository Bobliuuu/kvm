# A/B on A100-40GB: base fc0ced7 vs K's edits (2a8724c): in-smem TF32 rounding (tile_round) + split-accumulator attention (Acc3/mma3s)

Authoritative data: `modal/` (Modal A100-40GB, uncontended, tight [min..max]); script tests/gpu_ops/modal_bench.py, app run logged in RUNS.md.
Correctness (both builds): test_ops 1396 passed / 0 failed; all bench checks PASS.

Lambda A100 data (`{head,base}.*.txt`, `n4`, `velt`) is contaminated: that GPU is shared outside the lock, timings were bimodal (same binary 3 ms vs 6.7 ms for 4096^3).
Do not draw conclusions from it (an earlier revert of Acc3 based on it was wrong and was undone).

Modal results, full-dispatch ms (base / head / base-over-head speedup):

| bench | base | head | speedup |
|---|---|---|---|
| attn fwd B=4 T=1024 H=16 Hkv=16 hd=64 | 0.707 | 0.669 | 1.06x |
| attn bwd B=4 T=1024 H=16 Hkv=16 hd=64 | 2.302 | 2.244 | 1.03x |
| attn fwd B=4 T=2048 H=16 Hkv=4 hd=64 | 2.492 | 2.339 | 1.07x |
| attn bwd B=4 T=2048 H=16 Hkv=4 hd=64 | 6.808 | 6.581 | 1.03x |
| attn fwd B=2 T=2048 H=8 Hkv=8 hd=128 | 1.454 | 1.369 | 1.06x |
| attn bwd B=2 T=2048 H=8 Hkv=8 hd=128 | 3.885 | 3.911 | 0.99x |
| matmul TF32 NN 4096x4096x4096 | 5.151 | 6.302 | 0.82x |
| matmul TF32 TN 4096x4096x4096 | 4.396 | 5.145 | 0.85x |
| matmul TF32 NT 4096x4096x4096 | 3.635 | 4.613 | 0.79x |
| matmul 3xTF32 NN 4096x4096x4096 | 6.758 | 6.736 | 1.00x |
| matmul TF32 NN 2048x2048x2048 | 0.599 | 0.701 | 0.85x |
| matmul TF32 NN 8192x768x768 | 0.384 | 0.443 | 0.87x |
| matmul 128tile TF32 NN 4096x4096x4096 | 3.507 | 3.365 | 1.04x |
| matmul 128tile TF32 TN 4096x4096x4096 | 3.101 | 3.349 | 0.93x |
| matmul 128tile TF32 NT 4096x4096x4096 | 3.013 | 3.132 | 0.96x |
| matmul 128tile 3xTF32 NN 4096x4096x4096 | 5.672 | 5.636 | 1.01x |
| matmul 128tile TF32 NN 2048x2048x2048 | 0.513 | 0.561 | 0.91x |
| matmul 128tile TF32 NN 8192x768x768 | 0.299 | 0.319 | 0.94x |
| matmul 128tile TF32 NN 1024x1024x1024 | 0.093 | 0.098 | 0.95x |
| matmul 128tile TF32 NN 4096x3072x768 | 0.587 | 0.622 | 0.94x |
| matmul 128tile TF32 NN 512x512x512 | 0.058 | 0.060 | 0.97x |
| matmul 256x128tile TF32 NN 4096x4096x4096 | 3.428 | 3.416 | 1.00x |
| matmul 256x128tile TF32 TN 4096x4096x4096 | 2.830 | 2.988 | 0.95x |
| matmul 256x128tile TF32 NT 4096x4096x4096 | 2.676 | 2.995 | 0.89x |
| matmul 256x128tile 3xTF32 NN 4096x4096x4096 | 4.944 | 4.955 | 1.00x |
| matmul 256x128tile TF32 NN 2048x2048x2048 | 0.620 | 0.672 | 0.92x |
| matmul 256x128tile TF32 NN 8192x768x768 | 0.325 | 0.344 | 0.94x |
| matmul 256x128tile TF32 NN 1024x1024x1024 | 0.158 | 0.169 | 0.93x |
| matmul 256x128tile TF32 NN 4096x3072x768 | 0.661 | 0.703 | 0.94x |
| matmul 256x128tile TF32 NN 512x512x512 | 0.099 | 0.105 | 0.94x |
| matmul TF32 NN 1024x1024x1024 | 0.099 | 0.114 | 0.87x |
| matmul TF32 NN 4096x3072x768 | 0.741 | 0.849 | 0.87x |
| matmul TF32 NN 512x512x512 | 0.022 | 0.024 | 0.92x |

Conclusions:
- Acc3 split-accumulator attention: KEPT. fwd +6-7%, bwd +3% (hd128 bwd neutral). Small but real.
- tile_round (round TF32 once in smem): REVERTED. 1xTF32 matmul is 5-21% slower (64x64 tile 0.79-0.87x), neutral for 3xTF32.
- Best clean TF32 numbers remain: 256x128 tile ~2.7-3.4 ms for 4096^3 = 40-51 TFLOP/s (cuBLAS 105.6).
