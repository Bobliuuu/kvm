#!/usr/bin/env python3
"""Train the frontend GPT for N steps in PyTorch (eager, CUDA graph, torch.compile variants) from the executor's own initial
weights and token stream, and time each. Everything the executor workload fixes is read from its workload.json:

  python3 torch_bench.py WORKLOAD_DIR [--modes eager,graph,compile_mat,compile_ro,compile_graph] [--reps 7] [--out RESULT.json]
                         [--precision tf32|precise|bf16] [--losses-only]

--precision bf16 is torch.autocast(bfloat16): fp32 master weights, matmuls / attention in bf16, norms, softmax and the loss in fp32
(a fp32 residual stream, unlike the executor's 'bf16s' precision mode which stores the stream in bf16 too).

Model (kvmfe.nn.GPT): pre-LayerNorm (eps 1e-5), RoPE (rotate-half, theta 1e4), causal attention scale hd^-0.5, tanh-GELU MLP,
untied bias-free head, mean cross-entropy. Optimizer: AdamW lr 3e-3, betas (0.9, 0.95), eps 1e-8, weight decay 0.01 on matrices
only (not on embeddings, biases, norm gains), global grad-norm clip 1.0 (coef = min(1, 1/(norm + 1e-6)), as the executor's grad_clip).
Initial weights come from the executor's counter-based generator (reimplemented with numpy; self-checked against runtime/initgen.py).
"""
import argparse
import json
import math
import statistics
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'frontend'))
sys.path.insert(0, str(ROOT / 'runtime'))
import initgen  # noqa: E402
from kvmfe.nn import GPT, ModelConfig  # noqa: E402

OPT = dict(lr=3e-3, betas=(0.9, 0.95), eps=1e-8)
WD, CLIP = 0.01, 1.0


# ------------------------------------------------------------------ initial values (identical to the executor's buffers)
def seeded_np(seed, n, draws, bias, coef):
    M = np.uint64(0xFFFFFFFFFFFFFFFF)
    j = np.arange(n * draws, dtype=np.uint64)
    with np.errstate(over='ignore'):
        z = np.uint64(seed) + (j + np.uint64(1)) * np.uint64(initgen.GOLDEN)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    u = (z >> np.uint64(40)).astype(np.float32) * np.float32(1.0 / 16777216.0)
    if draws == 4:
        u = u.reshape(n, 4)
        acc = (u[:, 0] + u[:, 1]) + (u[:, 2] + u[:, 3])
    else:
        acc = u
    return (acc - np.float32(bias)) * np.float32(coef)


def init_values(buf, wl_dir):
    init = buf['init']
    n = buf['rows'] * buf['cols']
    if init['kind'] == 'fill':
        return np.full(n, init['value'], dtype=np.float32)
    if init['kind'] == 'seeded':
        draws, bias, coef = initgen.params(init.get('dist', 'normal'), init['scale'])
        return seeded_np(init['seed'], n, draws, bias, coef).astype(np.float32)
    if init['kind'] == 'file':
        return np.fromfile(Path(wl_dir) / init['path'], dtype='<f4' if buf.get('dtype', 'f32') == 'f32' else '<i4')
    if init['kind'] == 'values':
        return np.array(init['values'])
    raise SystemExit(f'unsupported init {init["kind"]} for {buf["name"]}')


def self_check_init():
    d, b, c = initgen.params('normal', 0.02)
    ref = np.array(initgen.seeded_values(12345, 257, d, b, c), dtype=np.float32)
    assert np.array_equal(ref, seeded_np(12345, 257, d, b, c)), 'numpy initgen differs from runtime/initgen.py'


# ------------------------------------------------------------------ the model
class TorchGPT:
    def __init__(self, cfg, specs, dev, autocast=False):
        self.cfg, self.specs, self.autocast = cfg, specs, autocast
        hd = cfg.d_model // cfg.heads
        inv = cfg.rope_theta ** (-2.0 * torch.arange(hd // 2, dtype=torch.float64) / hd)
        ang = torch.arange(cfg.seq, dtype=torch.float64)[:, None] * inv[None]
        self.cos = ang.cos().float().to(dev)           # [T, hd/2]
        self.sin = ang.sin().float().to(dev)

    def rope(self, x):                                  # x [B, H, T, hd]
        h = x.shape[-1] // 2
        x1, x2 = x[..., :h], x[..., h:]
        return torch.cat([x1 * self.cos - x2 * self.sin, x1 * self.sin + x2 * self.cos], -1)

    def loss(self, p, ids, tgt):
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.autocast):
            return self._loss(p, ids, tgt)

    def _loss(self, p, ids, tgt):
        c = self.cfg
        B, T, H, d = c.batch, c.seq, c.heads, c.d_model
        hd = d // H
        x = F.embedding(ids, p['gpt.wte'])              # [B*T, d]
        for i in range(c.layers):
            n = f'gpt.h{i}'
            h = F.layer_norm(x, (d,), p[f'{n}.n1.g'], p[f'{n}.n1.b'], 1e-5)
            sp = lambda k: F.linear(h, p[f'{n}.attn.{k}.w'].t(), p[f'{n}.attn.{k}.b']).view(B, T, H, hd).transpose(1, 2)
            q, k, v = self.rope(sp('q')), self.rope(sp('k')), sp('v')
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B * T, d)
            x = x + F.linear(a, p[f'{n}.attn.proj.w'].t(), p[f'{n}.attn.proj.b'])
            h = F.layer_norm(x, (d,), p[f'{n}.n2.g'], p[f'{n}.n2.b'], 1e-5)
            h = F.gelu(F.linear(h, p[f'{n}.mlp.fc1.w'].t(), p[f'{n}.mlp.fc1.b']), approximate='tanh')
            x = x + F.linear(h, p[f'{n}.mlp.fc2.w'].t(), p[f'{n}.mlp.fc2.b'])
        h = F.layer_norm(x, (d,), p['gpt.nf.g'], p['gpt.nf.b'], 1e-5)
        return F.cross_entropy(h @ p['gpt.head.w'], tgt)


# ------------------------------------------------------------------ one training run = N steps from the initial state
class Trainer:
    def __init__(self, wl_dir, precision, dev='cuda'):          # dev='cpu' only for the offline correctness check (eager)
        spec = json.loads((Path(wl_dir) / 'workload.json').read_text())
        info = json.loads((Path(wl_dir) / 'info.json').read_text())
        self.cfg = ModelConfig(**info['model'])
        self.steps = info['steps']
        bufs = {b['name']: b for b in spec['buffers']}
        specs = GPT(self.cfg).specs
        self.init = {n: torch.from_numpy(init_values(bufs['w.' + n], wl_dir).reshape(s.rows, s.cols)).to(dev) for n, s in specs.items()}
        # biases / norm params are stored [1, n] in the executor; PyTorch wants vectors
        self.shape = {n: (v.shape if specs[n].rows > 1 else (v.shape[1],)) for n, v in self.init.items()}
        self.params = {n: v.reshape(self.shape[n]).clone().requires_grad_(True) for n, v in self.init.items()}
        stream = lambda n: torch.from_numpy(init_values(bufs[n], wl_dir).astype(np.int64)).to(dev).view(self.steps, -1)
        self.ids, self.tgt = stream('stream_ids'), stream('stream_tgt')
        self.model = TorchGPT(self.cfg, specs, dev, autocast=precision == 'bf16')
        dec = [self.params[n] for n, s in specs.items() if s.decay]
        nod = [self.params[n] for n, s in specs.items() if not s.decay]
        self.opt = torch.optim.AdamW([{'params': dec, 'weight_decay': WD}, {'params': nod, 'weight_decay': 0.0}], fused=True,
                                     capturable=True, **OPT)
        self.plist = list(self.params.values())
        self.loss_fn = self.model.loss
        self.cudagraph_trees = False
        self.dev = dev
        torch.backends.cuda.matmul.allow_tf32 = precision == 'tf32'
        torch.backends.cudnn.allow_tf32 = precision == 'tf32'
        torch.set_float32_matmul_precision('high' if precision == 'tf32' else 'highest')

    def reset(self):
        with torch.no_grad():
            for n, p in self.params.items():
                p.copy_(self.init[n].reshape(self.shape[n]))
                p.grad = None
            for st in self.opt.state.values():
                for t in st.values():
                    t.zero_()

    def step(self, ids, tgt):
        if self.cudagraph_trees:                    # reduce-overhead reuses output memory across steps unless told where a step starts
            torch.compiler.cudagraph_mark_step_begin()
        loss = self.loss_fn(self.params, ids, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.plist, CLIP, foreach=True)
        self.opt.step()
        for p in self.plist:
            p.grad = None
        return loss.detach().clone() if self.cudagraph_trees else loss.detach()   # a kept loss must not alias the graph pool

    # ---- the four execution styles; each returns run() -> list of N loss tensors (device), and leaves the state at the initial value
    def eager(self):
        def run():
            return [self.step(self.ids[s], self.tgt[s]) for s in range(self.steps)]
        return run

    def graphed(self):
        """One whole training step (forward, backward, clip, AdamW, data select, loss log) in a single CUDA graph, replayed N times."""
        ctr = torch.zeros(1, dtype=torch.int64, device=self.dev)
        log = torch.zeros(self.steps, device=self.dev)

        def body():
            ids, tgt = self.ids.index_select(0, ctr).view(-1), self.tgt.index_select(0, ctr).view(-1)
            loss = self.loss_fn(self.params, ids, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.plist, CLIP, foreach=True)
            self.opt.step()
            log.index_copy_(0, ctr, loss.detach().view(1))
            ctr.add_(1)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                for p in self.plist:
                    p.grad = None
                body()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        for p in self.plist:
            p.grad = None
        with torch.cuda.graph(g):
            body()
        self.reset()
        ctr.zero_()

        def run():
            ctr.zero_()
            for _ in range(self.steps):
                g.replay()
            return list(log.unbind(0))
        return run

    def compiled(self, mode, graph=False):
        self.loss_fn = torch.compile(self.model.loss, mode=mode, fullgraph=True)
        self.cudagraph_trees = mode == 'reduce-overhead'
        if graph:
            for _ in range(3):                       # compile + autotune outside the graph, then capture the whole step
                for s in range(2):
                    self.step(self.ids[s], self.tgt[s])
            self.reset()
            return self.graphed()
        for s in range(3):                           # warm-up: compile (and autotune); state restored by the caller
            self.step(self.ids[s], self.tgt[s])
        self.reset()
        return self.eager()


def timed(trainer, run, reps):
    ts, losses = [], None
    for r in range(reps + 1):                        # rep 0 is warm-up (also the loss record)
        trainer.reset()
        if trainer.dev != 'cuda':
            t0 = time.time()
            out = run()
            if r == 0:
                losses = [float(x) for x in out]
            ts.append((time.time() - t0) * 1e3)
            continue
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        out = run()
        e1.record()
        torch.cuda.synchronize()
        if r == 0:
            losses = [float(x) for x in out]
        else:
            ts.append(e0.elapsed_time(e1))
    return ts, losses


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('workload_dir')
    ap.add_argument('--modes', default='eager,graph,compile_mat,compile_ro,compile_graph')
    ap.add_argument('--reps', type=int, default=7)
    ap.add_argument('--precision', choices=['tf32', 'precise', 'bf16'], default='tf32')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out')
    a = ap.parse_args()
    self_check_init()
    on_gpu = a.device == 'cuda'
    result = {'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0) if on_gpu else 'cpu',
              'sms': torch.cuda.get_device_properties(0).multi_processor_count if on_gpu else 0, 'precision': a.precision, 'reps': a.reps, 'modes': {}}
    for mode in a.modes.split(','):
        torch.manual_seed(0)
        t = Trainer(a.workload_dir, a.precision, a.device)
        t0 = time.time()
        try:
            run = {'eager': t.eager, 'graph': t.graphed,
                   'compile_mat': lambda: t.compiled('max-autotune-no-cudagraphs'),
                   'compile_ro': lambda: t.compiled('reduce-overhead'),
                   'compile_graph': lambda: t.compiled('max-autotune-no-cudagraphs', graph=True)}[mode]()
            ts, losses = timed(t, run, a.reps)
        except Exception as e:                       # a mode that cannot run is reported, not hidden
            result['modes'][mode] = {'error': f'{type(e).__name__}: {str(e)[:400]}'}
            print(f'{mode}: FAILED {type(e).__name__}: {str(e)[:300]}', flush=True)
            continue
        n = t.steps
        result['modes'][mode] = {'steps': n, 'ms_total': ts, 'ms_per_step_best': min(ts) / n, 'ms_per_step_median': statistics.median(ts) / n,
                                 'ms_per_step_worst': max(ts) / n, 'tokens_per_s_best': t.cfg.tokens / (min(ts) / n / 1e3), 'losses': losses,
                                 'setup_s': time.time() - t0}
        print(f'{mode}: best {min(ts) / n * 1e3:.1f} us/step  median {statistics.median(ts) / n * 1e3:.1f}  worst {max(ts) / n * 1e3:.1f}  '
              f'loss {losses[0]:.4f} -> {losses[-1]:.4f}  (setup {time.time() - t0:.0f}s)', flush=True)
        del t
        if on_gpu:
            torch.cuda.empty_cache()
    if a.out:
        Path(a.out).write_text(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
