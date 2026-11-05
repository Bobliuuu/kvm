#!/usr/bin/env python3
"""Vendor-library reference points on the same GPU for the op-efficiency question (does a 'no hand-written kernels' VM keep up with cuBLAS/cuDNN?).
TF32 matmul 4096^3 (cuBLAS), layernorm 16384x1024 fwd (bytes moved = 2 x 64 MiB), best of 20."""
import torch, torch.nn.functional as F
torch.backends.cuda.matmul.allow_tf32 = True
def best(fn, n=20):
    for _ in range(3): fn()
    b = 1e9
    for _ in range(n):
        a, e = torch.cuda.Event(True), torch.cuda.Event(True); a.record(); fn(); e.record(); torch.cuda.synchronize(); b = min(b, a.elapsed_time(e))
    return b / 1e3
n = 4096; A = torch.randn(n, n, device='cuda'); B = torch.randn(n, n, device='cuda'); t = best(lambda: A @ B)
print(f'cublas TF32 matmul {n}^3: {2*n**3/t/1e12:.1f} TFLOP/s ({t*1e3:.2f} ms)')
x = torch.randn(16384, 1024, device='cuda'); w = torch.ones(1024, device='cuda'); b0 = torch.zeros(1024, device='cuda')
t = best(lambda: F.layer_norm(x, (1024,), w, b0)); print(f'torch layernorm fwd 16384x1024: {2*x.numel()*4/t/1e9:.0f} GB/s ({t*1e6:.1f} us)')
