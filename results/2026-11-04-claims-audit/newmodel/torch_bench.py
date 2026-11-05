#!/usr/bin/env python3
"""Same training problems as gen.py's mlp_ce / xfmr_odd in PyTorch, for a timing ratio against one kvm_exec launch.
  python3 torch_bench.py CASE STEPS [REPS]  -> eager / whole-step CUDA graph / torch.compile (default mode), us per step.
Same shapes, SGD lr 0.05, 1xTF32 GEMMs (allow_tf32), tanh GELU, mean CE. Not bit-identical init (timing only)."""
import sys, time, torch, torch.nn.functional as F
torch.backends.cuda.matmul.allow_tf32 = True
case, steps = sys.argv[1], int(sys.argv[2]); reps = int(sys.argv[3]) if len(sys.argv) > 3 else 5
dev = 'cuda'; g = torch.Generator(device=dev).manual_seed(0)
P = lambda *s, std=0.1: (torch.randn(*s, device=dev, generator=g) * std).requires_grad_()
V = 100
if case == 'mlp_ce':
    B, D, H, TOK = 3, 48, 72, 96
    ps = [P(V, D), P(D, H, std=.2), P(1, H, std=0), P(H, H, std=.2), P(1, H, std=0), P(H, V, std=.2)]
    def fwd(ids):
        x = F.embedding(ids, ps[0]); x = F.gelu(x @ ps[1] + ps[2], approximate='tanh'); x = F.gelu(x @ ps[3] + ps[4], approximate='tanh'); return x @ ps[5]
elif case == 'xfmr_odd':
    Bt, T, D, Hh, L, FF = 2, 64, 48, 3, 3, 96; TOK = Bt * T; hd = D // Hh
    ps = [P(V, D, std=.02)]
    for _ in range(L): ps += [P(1, D, std=0), P(D, D, std=.02), P(D, D, std=.02), P(D, D, std=.02), P(D, D, std=.01), P(1, D, std=0), P(D, FF, std=.02), P(FF, D, std=.01)]
    ps += [P(1, D, std=0), P(D, V, std=.02)]
    for p in ps:
        if p.shape[0] == 1 and p.shape[1] == D: p.data.fill_(1.0)
    pos = torch.arange(T, device=dev); inv = 10000 ** (-torch.arange(0, hd, 2, device=dev) / hd); ang = pos[:, None] * inv[None]; cs, sn = ang.cos(), ang.sin()
    def rope(x):  # x: (B,T,H,hd)
        a, b = x[..., 0::2], x[..., 1::2]; c, s = cs[None, :, None], sn[None, :, None]
        return torch.stack([a * c - b * s, a * s + b * c], -1).flatten(-2)
    rms = lambda x, g: x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * g
    def fwd(ids):
        x = F.embedding(ids, ps[0]).view(Bt, T, D)
        for l in range(L):
            g1, wq, wk, wv, wo, g2, w1, w2 = ps[1 + 8 * l: 9 + 8 * l]
            h = rms(x, g1); sp = lambda t: t.view(Bt, T, Hh, hd)
            q, k, v = rope(sp(h @ wq)), rope(sp(h @ wk)), sp(h @ wv)
            a = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True).transpose(1, 2).reshape(Bt, T, D)
            x = x + a @ wo; h = rms(x, g2); x = x + F.gelu(h @ w1, approximate='tanh') @ w2
        return (rms(x, ps[-2]) @ ps[-1]).view(TOK, V)
else: raise SystemExit('case')
ids = torch.randint(0, V, (TOK,), device=dev, generator=g); tgt = torch.randint(0, V, (TOK,), device=dev, generator=g)
def step():
    loss = F.cross_entropy(fwd(ids), tgt)
    gr = torch.autograd.grad(loss, ps)
    with torch.no_grad():
        torch._foreach_add_(ps, gr, alpha=-0.05)
def timeit(fn, n):
    best = 1e9
    for _ in range(reps):
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(n): fn()
        torch.cuda.synchronize(); best = min(best, time.perf_counter() - t)
    return best / n * 1e6
for _ in range(5): step()
res = {'eager': timeit(step, steps)}
s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3): step()
torch.cuda.current_stream().wait_stream(s)
# whole-step graph: autograd.grad inside capture needs grads=detached -> capture fwd+bwd+update with static input
gph = torch.cuda.CUDAGraph()
with torch.cuda.graph(gph): step()
res['cuda_graph_step'] = timeit(gph.replay, steps)
try:
    cstep = torch.compile(step); [cstep() for _ in range(3)]
    res['torch_compile'] = timeit(cstep, steps)
except Exception as e: res['torch_compile'] = f'failed: {type(e).__name__}: {str(e)[:100]}'
print(case, 'tokens', TOK, 'us/step (best of %d):' % reps, {k: (round(v, 1) if isinstance(v, float) else v) for k, v in res.items()}, torch.__version__)
