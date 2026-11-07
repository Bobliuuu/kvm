#!/usr/bin/env python3
"""Reference throughput on the same shapes as bench_ops.cu, from PyTorch (cuBLAS GEMM, FlashAttention / mem-efficient SDPA,
fused ATen norm/AdamW/CE kernels). Prints 'REF' lines: median [min..max] ms over 7 repetitions and the same metric as BENCH.
Run on the GPU box under the GPU lock:  flock -w 900 ~/.gpu.lock python3 -I ref_torch.py"""
import statistics
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

dev = "cuda"
REPS, ITERS = 7, 10


def timeit(fn, iters=ITERS):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(REPS):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(iters):
            fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) / iters)
    return statistics.median(ts), min(ts), max(ts)


def rep(name, t, work, unit):
    med, lo, hi = t
    print(f"REF    {name:<52s} {med:8.3f} ms [{lo:.3f}..{hi:.3f}]  {work / med * 1e-6:9.1f} {unit}", flush=True)


def mm(M, N, K, ta, tb, mode):
    torch.backends.cuda.matmul.allow_tf32 = mode == "tf32"
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    dt = torch.bfloat16 if mode == "bf16" else torch.float32
    a = torch.randn((K, M) if ta else (M, K), device=dev, dtype=dt)
    b = torch.randn((N, K) if tb else (K, N), device=dev, dtype=dt)
    A = a.t() if ta else a
    B = b.t() if tb else b
    rep(f"matmul {mode} {'T' if ta else 'N'}{'T' if tb else 'N'} {M}x{N}x{K}", timeit(lambda: A @ B), 2.0 * M * N * K, "GFLOP/s")


def attn(B, T, H, Hkv, hd, dt, name):
    q = torch.randn(B, H, T, hd, device=dev, dtype=dt, requires_grad=True)
    k = torch.randn(B, Hkv, T, hd, device=dev, dtype=dt, requires_grad=True)
    v = torch.randn(B, Hkv, T, hd, device=dev, dtype=dt, requires_grad=True)
    kk = k.repeat_interleave(H // Hkv, 1) if H != Hkv else k
    vv = v.repeat_interleave(H // Hkv, 1) if H != Hkv else v
    be = SDPBackend.FLASH_ATTENTION if dt != torch.float32 else SDPBackend.EFFICIENT_ATTENTION
    fl = 2.0 * B * H * T * T * hd
    with sdpa_kernel(be):
        f = lambda: F.scaled_dot_product_attention(q, kk, vv, is_causal=True)
        rep(f"attn fwd {name} B={B} T={T} H={H} Hkv={Hkv} hd={hd}", timeit(f), fl, "GFLOP/s")
        o = F.scaled_dot_product_attention(q, kk, vv, is_causal=True)
        do = torch.randn_like(o)
        g = lambda: torch.autograd.grad(o, (q, k, v), do, retain_graph=True)
        rep(f"attn bwd {name} B={B} T={T} H={H} Hkv={Hkv} hd={hd}", timeit(g, 5), fl * 2.5, "GFLOP/s")


def mem():
    R, d = 16384, 1024
    x = torch.randn(R, d, device=dev); w = torch.randn(d, device=dev); b = torch.randn(d, device=dev)
    rep("layernorm fwd 16384x1024", timeit(lambda: F.layer_norm(x, (d,), w, b), 20), 2.0 * R * d * 4, "GB/s")
    n = 1 << 24
    p, g, m, v = (torch.randn(n, device=dev) for _ in range(4)); v.abs_()
    st = torch.tensor([1.0], device=dev)
    rep("adamw 16M params (torch._fused_adamw_)", timeit(lambda: torch._fused_adamw_([p], [g], [m], [v], [], [st], lr=1e-3, beta1=.9, beta2=.99, weight_decay=.01, eps=1e-8, amsgrad=False, maximize=False), 20), 7.0 * n * 4, "GB/s")
    N, V = 8192, 50304
    lg = torch.randn(N, V, device=dev, requires_grad=True); tg = torch.randint(0, V, (N,), device=dev)
    rep("softmax_ce fwd 8192x50304 (F.cross_entropy)", timeit(lambda: F.cross_entropy(lg, tg), 10), 1.0 * N * V * 4, "GB/s (1 read counted)")
    loss = F.cross_entropy(lg, tg)
    rep("softmax_ce bwd 8192x50304 (autograd)", timeit(lambda: torch.autograd.grad(loss, lg, retain_graph=True), 10), 2.0 * N * V * 4, "GB/s (read+write)")
    x = torch.randn(1 << 26, device=dev)
    rep("gelu fwd 64M (F.gelu tanh)", timeit(lambda: F.gelu(x, approximate="tanh"), 10), 2.0 * x.numel() * 4, "GB/s")


if __name__ == "__main__":
    print("device:", torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).multi_processor_count, "SMs, torch", torch.__version__, flush=True)
    import os
    if os.environ.get("REF_ATTN_ONLY"):   # attention at the 162M shape (B4 T1024 H12 hd64), fp32 and TF32-allowed efficient-attention, bf16 flash
        for tf32 in (False, True):
            torch.backends.cuda.matmul.allow_tf32 = tf32
            attn(4, 1024, 12, 12, 64, torch.float32, "tf32" if tf32 else "fp32")
        attn(4, 1024, 12, 12, 64, torch.bfloat16, "bf16")
        raise SystemExit
    for mode in ("tf32", "fp32", "bf16"):
        for (M, N, K, ta, tb) in [(4096, 4096, 4096, 0, 0), (4096, 4096, 4096, 1, 0), (4096, 4096, 4096, 0, 1), (2048, 2048, 2048, 0, 0),
                                  (8192, 768, 768, 0, 0), (1024, 1024, 1024, 0, 0), (4096, 3072, 768, 0, 0), (512, 512, 512, 0, 0)]:
            if mode == "fp32" and (ta or tb):
                continue
            mm(M, N, K, ta, tb, mode)
    for dt, nm in ((torch.float32, "fp32"), (torch.bfloat16, "bf16")):
        attn(4, 1024, 16, 16, 64, dt, nm); attn(4, 2048, 16, 4, 64, dt, nm); attn(2, 2048, 8, 8, 128, dt, nm)
    mem()
