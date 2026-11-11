#!/usr/bin/env python3
"""Bundle generator for the executor-vs-PyTorch benchmarks: the audit's `mlp_ce` model (embedding -> depth x (linear+GELU) -> linear -> CE, SGD lr 0.05,
96 tokens, vocab 100, d 48) with the tiling and width as parameters. The audit built it with the `cpu` tiling preset (8-row tiles, 138
tasks per step); the `a100` preset (128-row tiles, one tile per op at 96 tokens) is the executor's intended configuration.

  python3 -I gen_mlp.py OUT_DIR --steps N [--tiling cpu|a100] [--hidden 72] [--depth 2] [--no-golden]
Uses only the public frontend API (same calls as results/2026-11-04-claims-audit/newmodel/gen.py, whose MLPNet it imports).
"""
import argparse, importlib.util, json, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'frontend'))
spec = importlib.util.spec_from_file_location('audit_gen', ROOT / 'results/2026-11-04-claims-audit/newmodel/gen.py')
g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)

ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--steps', type=int, default=3)
ap.add_argument('--tiling', default='a100'); ap.add_argument('--hidden', type=int, default=72); ap.add_argument('--depth', type=int, default=2)
ap.add_argument('--no-golden', action='store_true')
ap.add_argument('--golden-timeout', type=float, default=None, help="seconds the pure-Python CPU golden may take (compile_workload.py's default "
                'when unset); wide or deep shapes on a loaded host need more')
a = ap.parse_args()
tl = g.Tiling.preset(a.tiling)
sgd = g.TrainConfig(steps=a.steps, optim=g.OptimConfig(kind='sgd', lr=0.05), clip=0)
cfg = g.ModelConfig('mlp', vocab=100, d_model=48, layers=1, heads=3, seq=32, batch=3)
from kvmfe import nn as _nn


class Net(g.MLPNet):   # the audit's MLPNet predates the training-program protocol (batch_streams / losses) of the integrated tree
    def batch_streams(self, micros):
        return _nn.token_streams(self.cfg, micros)

    def losses(self, b, batch, w, grad_scale=1.0):
        return [('loss', b.ce_loss(self(b, batch['ids'], w), batch['tgt'], grad_scale))]


net = Net(cfg, hidden=a.hidden, depth=a.depth)
prog = g.build_training(net, sgd)
low = g.lower_program(prog.b.ops, tl, 'none', 0, prefix='act')
workload = g.to_workload(low, {'train': g.GraphPolicy()})
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
for p, raw in low.data_files.items():
    (out / p).parent.mkdir(parents=True, exist_ok=True); (out / p).write_bytes(raw)
(out / 'workload.json').write_text(json.dumps(workload, separators=(',', ':')))
golden = [] if a.no_golden else ['--emit-golden', *(['--golden-timeout', str(a.golden_timeout)] if a.golden_timeout else [])]
subprocess.run([sys.executable, str(ROOT / 'runtime/compile_workload.py'), *golden, str(out / 'workload.json'), str(out / 'bundle')], check=True)
print('mlp tiling', a.tiling, 'hidden', a.hidden, 'depth', a.depth, 'steps', a.steps, 'tasks', sum(len(x['tasks']) for x in workload['graphs']), 'tasks/step', sum(len(x['tasks']) for x in workload['graphs']) // a.steps)
