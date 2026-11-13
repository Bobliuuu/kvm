"""Floor micro-benchmarks for the 162M GPT step on one A100 (torch + cuBLAS + SDPA + CUDA graph). One process, ~1 min.
Measures: realized bf16 and TF32 matmul peak, HBM copy bandwidth, cuBLAS time of every GEMM of the step at its exact shape,
SDPA flash causal fwd+bwd at the model's attention shape, and the per-edge cost of a CUDA-graph dependency chain."""
import json, time, torch
torch.backends.cuda.matmul.allow_tf32 = True
dev = 'cuda'; out = {}
def tm(f, n=20, w=5):
    for _ in range(w): f()
    torch.cuda.synchronize(); e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(n): f()
    e1.record(); torch.cuda.synchronize(); return e0.elapsed_time(e1) / n   # ms
def best(f, k=5, **kw): return min(tm(f, **kw) for _ in range(k))
# realized peaks
for dt, name in ((torch.bfloat16, 'bf16'), (torch.float32, 'tf32')):
    a = torch.randn(8192, 8192, device=dev, dtype=dt); b = torch.randn(8192, 8192, device=dev, dtype=dt)
    ms = best(lambda: a @ b, n=10); out[f'peak_{name}_8192_tflops'] = 2 * 8192**3 / ms / 1e9
    del a, b
# HBM bandwidth (read+write copy of 1 GiB, and read-only sum)
x = torch.empty(256 * 2**20, device=dev, dtype=torch.float32); y = torch.empty_like(x)
ms = best(lambda: y.copy_(x), n=20); out['hbm_copy_GBps'] = 2 * x.numel() * 4 / ms / 1e6
ms = best(lambda: x.sum(), n=20); out['hbm_read_GBps'] = x.numel() * 4 / ms / 1e6
del x, y
# GEMMs of one step at exact shapes: T=4096 tokens, d=768, ffn 3072, vocab 50304; bf16 operands, fp32 accumulate
T, d, F, V = 4096, 768, 3072, 50304
def gemm_set(M, K, N):   # fwd C[M,N]=A[M,K]B[K,N]; dgrad [M,K]=dC[M,N]B^T; wgrad [K,N]=A^T dC
    A = torch.randn(M, K, device=dev, dtype=torch.bfloat16); B = torch.randn(K, N, device=dev, dtype=torch.bfloat16); C = torch.randn(M, N, device=dev, dtype=torch.bfloat16)
    f = best(lambda: A @ B); g = best(lambda: C @ B.t()); w = best(lambda: A.t() @ C)
    return f, g, w
shapes = {'qkvo(768x768) x4/layer x12': (T, d, d, 48), 'up(768->3072) x12': (T, d, F, 12), 'down(3072->768) x12': (T, F, d, 12), 'lm_head(768->50304) x1': (T, d, V, 1)}
tot = 0; flops = 0; rows = {}
for k, (M, K, N, mult) in shapes.items():
    f, g, w = gemm_set(M, K, N); t = (f + g + w) * mult; tot += t; fl = 3 * 2 * M * K * N * mult; flops += fl
    rows[k] = {'fwd_ms': f, 'dgrad_ms': g, 'wgrad_ms': w, 'step_ms': t, 'tflops': fl / t / 1e9}
out['cublas_gemms'] = rows; out['cublas_gemm_total_ms'] = tot; out['gemm_flops'] = flops
# SDPA flash causal, bf16, B=4 H=12 T=1024 hd=64, fwd+bwd, 12 layers
q, k_, v = (torch.randn(4, 12, 1024, 64, device=dev, dtype=torch.bfloat16, requires_grad=True) for _ in range(3))
do = torch.randn(4, 12, 1024, 64, device=dev, dtype=torch.bfloat16)
F_ = torch.nn.functional
fw = best(lambda: F_.scaled_dot_product_attention(q, k_, v, is_causal=True))
o = F_.scaled_dot_product_attention(q, k_, v, is_causal=True)
def fb():
    q.grad = k_.grad = v.grad = None; o.backward(do, retain_graph=True)
bw = best(fb)
out['sdpa_per_layer_ms'] = {'fwd': fw, 'bwd': bw}; out['sdpa_step_ms'] = 12 * (fw + bw)
# CUDA graph: chain of N dependent tiny kernels -> per-edge cost; and 1 kernel launched in a chain via eager
z = torch.zeros(1, device=dev)
def chain(n):
    for _ in range(n): z.add_(1)
s = torch.cuda.Stream()
with torch.cuda.stream(s):
    chain(3); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s): chain(1000)
    g.replay(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20): g.replay()
    torch.cuda.synchronize(); out['graph_edge_us'] = (time.perf_counter() - t0) / 20 / 1000 * 1e6
t0 = time.perf_counter()
for _ in range(5): chain(1000)
torch.cuda.synchronize(); out['eager_launch_chain_us'] = (time.perf_counter() - t0) / 5 / 1000 * 1e6
out['device'] = torch.cuda.get_device_name(); out['sm_count'] = torch.cuda.get_device_properties(0).multi_processor_count; out['torch'] = torch.__version__; out['cuda'] = torch.version.cuda
print(json.dumps(out, indent=1))
