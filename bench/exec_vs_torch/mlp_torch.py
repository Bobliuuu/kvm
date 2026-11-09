"""PyTorch side of the executor-vs-PyTorch MLP comparison: the SAME problem as a gen_mlp.py bundle, trained from the bundle's own bytes.

Everything the executor workload fixes is read from its workload.json: initial weights (counter-based generator, bit-identical), the token
stream (stream_ids / stream_tgt, one batch per step), SGD lr 0.05 (gen_mlp.py), no clipping, mean cross-entropy, tanh GELU, 1xTF32 GEMMs.
Every variant runs the same step body, including the loss log (index_copy into a device-side step counter), so the work per step is identical:

  eager         R separate runs, plain PyTorch, a Python loop over runs
  seq_graph     (R=1: THE whole-step CUDA graph) R separate whole-step CUDA graphs replayed one after another on one stream
  multi_graph   the same R graphs, each on its own stream (the GPU may overlap them)
  batched       vmap-like: parameters stacked [R, ...], bmm model, ONE whole-step CUDA graph for all R runs (needs identical shapes)
  compile       R=1: torch.compile(max-autotune-no-cudagraphs) on the forward+loss, backward and SGD update eager
  compile_graph R=1: the compiled forward/backward captured with the update into one whole-step CUDA graph
  batched_compile  the batched model compiled, then captured in one whole-step CUDA graph

A variant exposes reset() (state back to the initial weights, step counter 0), run(n) (advance every run n steps) and losses() ([step][run]).
"""
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'bench' / 'vs_pytorch'))
from torch_bench import init_values, self_check_init  # noqa: E402

LR = 0.05
COMPILE_MODE = 'max-autotune-no-cudagraphs'


def single_loss(ps, ids, tgt):
    x = F.embedding(ids, ps[0])
    for i in range(1, len(ps) - 1, 2):
        x = F.gelu(x @ ps[i] + ps[i + 1], approximate='tanh')
    return F.cross_entropy(x @ ps[-1], tgt)


def batched_loss(ps, ids, tgt):
    """ps carry a leading run axis; returns the per-run mean CE [R]. The gradient of its sum w.r.t. run r's parameters is run r's own gradient."""
    r, tok = ids.shape
    x = ps[0][torch.arange(r, device=ids.device)[:, None], ids]
    for i in range(1, len(ps) - 1, 2):
        x = F.gelu(torch.baddbmm(ps[i + 1], x, ps[i]), approximate='tanh')
    logits = torch.bmm(x, ps[-1])
    return F.cross_entropy(logits.reshape(r * tok, -1), tgt.reshape(-1), reduction='none').view(r, tok).mean(1)


class Workload:
    """Initial weights and token stream of a gen_mlp.py bundle directory."""

    def __init__(self, wl_dir, dev='cuda'):
        spec = json.loads((Path(wl_dir) / 'workload.json').read_text())
        bufs = {b['name']: b for b in spec['buffers']}
        names = ['w.mlp.wte']
        depth = sum(1 for n in bufs if n.startswith('w.mlp.fc') and n.endswith('.w'))
        for i in range(depth):
            names += [f'w.mlp.fc{i}.w', f'w.mlp.fc{i}.b']
        names.append('w.mlp.head.w')
        self.names = names
        self.init = [torch.from_numpy(init_values(bufs[n], wl_dir).reshape(bufs[n]['rows'], bufs[n]['cols'])).to(dev) for n in names]
        self.steps = sum(1 for n in bufs if n.startswith('loss.'))
        ids = torch.from_numpy(init_values(bufs['stream_ids'], wl_dir).astype('int64')).to(dev)
        tgt = torch.from_numpy(init_values(bufs['stream_tgt'], wl_dir).astype('int64')).to(dev)
        self.ids, self.tgt = ids.view(self.steps, -1), tgt.view(self.steps, -1)
        self.tok = self.ids.shape[1]
        self.dev = dev


class Variant:
    """One execution style over R runs. `body` is the whole step (select batch, forward, backward, SGD, log) on `ps` (list of tensors)."""

    def __init__(self, wl, r, name, batched, loss_fn):
        self.wl, self.r, self.name, self.batched = wl, r, name, batched
        dev = wl.dev
        lead = (r,) if batched else ()
        n_sets = 1 if batched else r
        # one parameter set per separate run; one stacked set when batched
        self.sets = [[(w.expand(*lead, *w.shape).clone() if batched else w.clone()).requires_grad_() for w in wl.init] for _ in range(n_sets)]
        self.ctr = torch.zeros(1, dtype=torch.int64, device=dev)
        self.logs = [torch.zeros(wl.steps, r if batched else 1, device=dev) for _ in range(n_sets)]
        self.loss_fn = loss_fn
        self.bodies = [self._body(ps, log) for ps, log in zip(self.sets, self.logs)]
        self.graphs = None
        self.streams = None

    def _body(self, ps, log):
        wl, ctr, loss_fn, r, batched = self.wl, self.ctr, self.loss_fn, self.r, self.batched

        def body():
            ids, tgt = wl.ids[:1], wl.tgt[:1]          # [1, tok]: the executor bundle trains on batch 0 at every step (see the README)
            if batched:
                per = loss_fn(ps, ids.expand(r, -1), tgt.expand(r, -1))
                total = per.sum()
            else:
                per = loss_fn(ps, ids.view(-1), tgt.view(-1)).view(1)
                total = per.sum()
            gr = torch.autograd.grad(total, ps)
            with torch.no_grad():
                torch._foreach_add_(ps, gr, alpha=-LR)
                log.index_copy_(0, ctr, per.detach().view(1, -1))
        return body

    def warm(self, n=3):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(n):
                for b in self.bodies:
                    b()
                self.ctr.add_(1)
                self.ctr.remainder_(self.wl.steps)
        torch.cuda.current_stream().wait_stream(s)
        self.reset()

    def capture(self):
        self.graphs = []
        for b in self.bodies:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                b()
            self.graphs.append(g)
        self.reset()

    def reset(self):
        with torch.no_grad():
            for ps in self.sets:
                for p, w in zip(ps, self.wl.init):
                    p.copy_(w.expand_as(p))
            for log in self.logs:
                log.zero_()
            self.ctr.zero_()
        torch.cuda.synchronize()

    def losses(self):
        """[step][run] losses of the last run(): per run, the SGD trajectory."""
        torch.cuda.synchronize()
        if self.batched:
            return self.logs[0].cpu().tolist()
        return torch.cat(self.logs, 1).cpu().tolist()

    def weights(self, run=0):
        ps = self.sets[0] if self.batched else self.sets[run]
        return [(p[run] if self.batched else p).detach().cpu().double().numpy() for p in ps]


class Graphed(Variant):
    """Variants whose step (all runs) replays graphs; the counter advances in its own tiny graph so the replay loop is pure replay."""

    def capture(self):
        super().capture()
        self.ctr_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.ctr_graph):
            self.ctr.add_(1)
        self.reset()


class SeqGraphV(Graphed):
    def run(self, n):
        for _ in range(n):
            for g in self.graphs:
                g.replay()
            self.ctr_graph.replay()


class MultiGraphV(Graphed):
    def run(self, n):
        if self.streams is None:
            self.streams = [torch.cuda.Stream() for _ in self.graphs]
        cur = torch.cuda.current_stream()
        for _ in range(n):
            for g, s in zip(self.graphs, self.streams):
                s.wait_stream(cur)
                with torch.cuda.stream(s):
                    g.replay()
            for s in self.streams:
                cur.wait_stream(s)
            self.ctr_graph.replay()


class OneGraphV(Graphed):
    def run(self, n):
        g, c = self.graphs[0], self.ctr_graph
        for _ in range(n):
            g.replay()
            c.replay()


class EagerV(Variant):
    def run(self, n):
        for _ in range(n):
            for b in self.bodies:
                b()
            self.ctr.add_(1)


def build(wl, r, name):
    """Construct, warm up (compile, autotune) and capture one variant; returns it reset to the initial state."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if name == 'eager':
        v = EagerV(wl, r, name, False, single_loss)
        v.warm()
    elif name in ('seq_graph', 'multi_graph'):
        v = (SeqGraphV if name == 'seq_graph' else MultiGraphV)(wl, r, name, False, single_loss)
        v.warm(); v.capture()
    elif name == 'batched':
        v = OneGraphV(wl, r, name, True, batched_loss)
        v.warm(); v.capture()
    elif name == 'batched_compile':
        v = OneGraphV(wl, r, name, True, torch.compile(batched_loss, mode=COMPILE_MODE, fullgraph=True))
        v.warm(); v.capture()
    elif name == 'compile':
        v = EagerV(wl, r, name, False, torch.compile(single_loss, mode=COMPILE_MODE, fullgraph=True))
        v.warm()
    elif name == 'compile_graph':
        v = OneGraphV(wl, r, name, False, torch.compile(single_loss, mode=COMPILE_MODE, fullgraph=True))
        v.warm(); v.capture()
    else:
        raise ValueError(name)
    return v


def default_variants(r):
    """The variants meaningful at R runs."""
    out = ['seq_graph', 'multi_graph', 'batched', 'batched_compile']
    if r == 1:
        out = ['eager', 'compile', 'compile_graph', 'seq_graph', 'batched_compile']
    elif r <= 8:
        out = ['eager'] + out
    return out


if __name__ == '__main__':
    self_check_init()
    print('ok')
