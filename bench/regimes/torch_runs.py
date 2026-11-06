"""PyTorch baselines for R independent runs of the audit's `mlp_ce` problem (same shapes as newmodel/gen.py + torch_bench.py).

Each variant advances ALL R runs by one SGD step per call; `step_us` is wall time per step for the whole set (best of reps).
  seq_graph    R separate whole-step CUDA graphs replayed one after another on one stream
  multi_graph  the same R graphs, each replayed on its own stream (the GPU may overlap them)
  batched      vmap-like: parameters stacked [R, ...], bmm model, ONE whole-step CUDA graph for all R runs
  eager        plain PyTorch, R runs in a Python loop (only used for small R)
Runs are independent: batched computes sum_r mean-CE_r, whose gradient w.r.t. run r's parameters is run r's own gradient.
SGD lr 0.05, TF32 on, tanh GELU, mean CE, TOK = 96 tokens, V = 100, D = 48, H = 72.
"""
import time, torch, torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = True
V, D, H, TOK, LR = 100, 48, 72, 96, 0.05
dev = 'cuda'


def _params(r, gen):
    mk = lambda *s, std=0.1: (torch.randn(r, *s, device=dev, generator=gen) * std)
    return [mk(V, D), mk(D, H, std=.2), mk(1, H, std=0), mk(H, H, std=.2), mk(1, H, std=0), mk(H, V, std=.2)]


def _data(r, gen):
    return (torch.randint(0, V, (r, TOK), device=dev, generator=gen), torch.randint(0, V, (r, TOK), device=dev, generator=gen))


def _single_step_fn(ps, ids, tgt):
    ps = [p.requires_grad_() for p in ps]
    def step():
        x = F.embedding(ids, ps[0]); x = F.gelu(x @ ps[1] + ps[2], approximate='tanh'); x = F.gelu(x @ ps[3] + ps[4], approximate='tanh')
        loss = F.cross_entropy(x @ ps[5], tgt)
        gr = torch.autograd.grad(loss, ps)
        with torch.no_grad():
            torch._foreach_add_(ps, gr, alpha=-LR)
    return step


def _graphed(step):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): step()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): step()
    return g


class Variants:
    def __init__(self, r, seed=0):
        self.r = r
        gen = torch.Generator(device=dev).manual_seed(seed)
        self.fns = {}
        # batched
        P = [p.clone().requires_grad_() for p in _params(r, gen)]
        ids, tgt = _data(r, gen)
        # embedding for all runs: gather rows with a flat index
        flat = (ids + torch.arange(r, device=dev)[:, None] * V).reshape(-1)
        def bstep():
            x = P[0].reshape(r * V, D)[flat].reshape(r, TOK, D)
            x = F.gelu(torch.baddbmm(P[2], x, P[1]), approximate='tanh'); x = F.gelu(torch.baddbmm(P[4], x, P[3]), approximate='tanh')
            logits = torch.bmm(x, P[5]).reshape(r * TOK, V)
            loss = F.cross_entropy(logits, tgt.reshape(-1), reduction='sum') / TOK
            gr = torch.autograd.grad(loss, P)
            with torch.no_grad():
                torch._foreach_add_(P, gr, alpha=-LR)
        for _ in range(3): bstep()
        self.batched_g = _graphed(bstep); self.fns['batched'] = self.batched_g.replay
        # R separate runs
        steps, graphs, streams = [], [], []
        for i in range(r):
            pi = [p.clone() for p in _params(1, gen)]; pi = [p[0].contiguous() for p in pi]
            idi, tgi = _data(1, gen)
            st = _single_step_fn(pi, idi[0], tgi[0]); steps.append(st)
        for _ in range(3):
            for st in steps: st()
        graphs = [_graphed(st) for st in steps]
        streams = [torch.cuda.Stream() for _ in range(r)]
        def seq():
            for g in graphs: g.replay()
        def multi():
            cur = torch.cuda.current_stream()
            for g, s in zip(graphs, streams):
                s.wait_stream(cur)
                with torch.cuda.stream(s): g.replay()
            for s in streams: cur.wait_stream(s)
        self.fns['seq_graph'], self.fns['multi_graph'] = seq, multi
        if r <= 4:
            def eager():
                for st in steps: st()
            self.fns['eager'] = eager
        self._keep = (P, ids, tgt, flat, bstep, steps, graphs)   # graphs replay on raw pointers: the tensors they read must outlive __init__
        torch.cuda.synchronize()

    def time(self, name, n):
        fn = self.fns[name]
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(n): fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t) / n * 1e6
