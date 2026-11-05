#!/usr/bin/env python3
"""Audit: push models the repo never shipped through the frontend -> compiler -> bundle (+ CPU golden).

  python3 gen.py OUT_DIR CASE [--steps N]      CASE: xfmr_odd | mlp_ce | resmlp_rms | bad_head | bad_dims
Uses only the public frontend API (Builder, Module, build_training, lower_program, to_workload); nothing in kvmfe is edited.
"""
import argparse, json, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'frontend'))
from kvmfe.emit import GraphPolicy, lower_program, to_workload
from kvmfe.lower import Tiling
from kvmfe.nn import GPT, Linear, Module, Norm, ModelConfig
from kvmfe.optim import OptimConfig
from kvmfe.pipeline import compile_training
from kvmfe.train import TrainConfig, build_training


class MLPNet(Module):
    """embedding -> [linear+gelu]*k -> linear -> softmax-CE. No attention, no norm, no rope (not in the repo)."""
    def __init__(self, cfg, hidden, depth, norm=None):
        super().__init__('mlp')
        self.cfg, self.depth = cfg, depth
        self.wte = self.add('wte', cfg.vocab, cfg.d_model, ('normal', 0.1))
        dims = [cfg.d_model] + [hidden] * depth
        self.fcs = [Linear(f'mlp.fc{i}', dims[i], dims[i + 1], True, 0.2) for i in range(depth)]
        self.norms = [Norm(f'mlp.n{i}', dims[i + 1], norm) for i in range(depth)] if norm else []
        self.head = Linear('mlp.head', hidden, cfg.vocab, False, 0.2)
        self.absorb(*self.fcs, *self.norms, self.head)

    def __call__(self, b, ids, w):
        x = b.embedding(ids, w[self.wte], self.cfg.seq)
        for i, fc in enumerate(self.fcs):
            h = fc(b, x, w, gelu=True)
            if self.norms:                           # residual+norm only when shapes agree
                h = self.norms[i](b, h, w)
            x = h
        return self.head(b, x, w)


def build(case, steps):
    tl = Tiling.preset('cpu')
    sgd = TrainConfig(steps=steps, optim=OptimConfig(kind='sgd', lr=0.05), clip=0)
    if case == 'xfmr_odd':      # 3 layers, 3 heads (head dim 16), rmsnorm, no bias, vocab 100, ffn x2, seq 64 -> not a shipped preset
        cfg = ModelConfig('xfmr_odd', vocab=100, d_model=48, layers=3, heads=3, seq=64, batch=2, ffn_mult=2, norm='rmsnorm', bias=False)
        c = compile_training(cfg, sgd, tl, reuse='none'); return c.spec, c.low.data_files
    if case == 'bad_head':      # head dim 24: must be refused cleanly
        c = compile_training(ModelConfig('x', d_model=48, heads=2, seq=32), sgd, tl); return c.spec, c.low.data_files
    cfg = ModelConfig('mlp', vocab=100, d_model=48, layers=1, heads=3, seq=32, batch=3)   # ModelConfig validates ATTENTION shape even for an MLP (finding); only batch/seq/vocab/d are read
    if case == 'bad_dims':
        net = MLPNet(cfg, hidden=7, depth=1)
    elif case == 'mlp_ce':
        net = MLPNet(cfg, hidden=72, depth=2)
    elif case == 'resmlp_rms':
        net = MLPNet(cfg, hidden=72, depth=3, norm='rmsnorm')
    else:
        raise SystemExit(f'unknown case {case}')
    prog = build_training(net, sgd)
    low = lower_program(prog.b.ops, tl, 'none', 0, prefix='act')
    return to_workload(low, {'train': GraphPolicy()}), low.data_files


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('case'); ap.add_argument('--steps', type=int, default=3); ap.add_argument('--no-golden', action='store_true', help='skip the (slow, pure-Python) CPU golden: timing bundles')
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    spec, files = build(a.case, a.steps)
    for p, raw in files.items():
        (out / p).parent.mkdir(parents=True, exist_ok=True); (out / p).write_bytes(raw)
    (out / 'workload.json').write_text(json.dumps(spec, separators=(',', ':')))
    subprocess.run([sys.executable, str(ROOT / 'runtime/compile_workload.py'), *([] if a.no_golden else ['--emit-golden']), str(out / 'workload.json'), str(out / 'bundle')], check=True)
    ops = sorted({t['op'] for g in spec['graphs'] for t in g['tasks']})
    print(a.case, 'tasks', sum(len(g['tasks']) for g in spec['graphs']), 'ops', ops)
