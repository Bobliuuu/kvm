#!/usr/bin/env python3
"""Task counts of the audit newmodel cases per tiling preset / fusion switch (CPU only, no golden).
  python3 count.py CASE STEPS TILING FUSE(0/1) [OUT_DIR]   -> prints tasks, tasks/step and a per-device histogram; with OUT_DIR also writes a bundle"""
import json, os, subprocess, sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'frontend')); sys.path.insert(0, str(ROOT / 'results/2026-11-04-claims-audit/newmodel'))
from kvmfe.emit import GraphPolicy, lower_program, to_workload
from kvmfe.lower import Tiling
from kvmfe.nn import ModelConfig
from kvmfe.optim import OptimConfig
from kvmfe.pipeline import compile_training
from kvmfe.train import TrainConfig, build_training
import gen


def build(case, steps, tiling, fuse):
    tl = Tiling.preset(tiling[:-2], mm_merge=0) if tiling.endswith('nm') else Tiling.preset(tiling)   # 'a100nm' = a100 without GEMM tile merging
    sgd = TrainConfig(steps=steps, optim=OptimConfig(kind='sgd', lr=0.05), clip=0)
    if case == 'xfmr_odd':
        cfg = ModelConfig('xfmr_odd', vocab=100, d_model=48, layers=3, heads=3, seq=64, batch=2, ffn_mult=2, norm='rmsnorm', bias=False)
        c = compile_training(cfg, sgd, tl, reuse='none', fuse=fuse); return c.spec, c.low.data_files
    from kvmfe.generic import CrossEntropy, GeneratedData
    from kvmfe.train import batch_tokens
    cfg = ModelConfig('mlp', vocab=100, d_model=48, layers=1, heads=3, seq=32, batch=3)
    net = gen.MLPNet(cfg, hidden=72, depth=2)
    def make(m):
        i, t = batch_tokens(cfg, m); return {'ids': i, 'y': t}
    data = GeneratedData(make, {'ids': (96, 1, 'i32'), 'y': (96, 1, 'i32')}, inputs=('ids',), target='y')
    c = compile_training(net, sgd, tl, reuse='none', fuse=fuse, loss=CrossEntropy(), data=data, steps=steps)
    return c.spec, c.low.data_files


if __name__ == '__main__':
    case, steps, tiling, fuse = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
    spec, files = build(case, steps, tiling, fuse)
    n = sum(len(g['tasks']) for g in spec['graphs'])
    h = Counter(t['op'] for g in spec['graphs'] for t in g['tasks'])
    print(f'{case} steps={steps} tiling={tiling} fuse={fuse} tasks={n} per_step={n/steps:.1f}')
    print('  ' + ' '.join(f'{k}:{v}' for k, v in h.most_common()))
    if len(sys.argv) > 5:
        out = Path(sys.argv[5]); out.mkdir(parents=True, exist_ok=True)
        for p, raw in files.items():
            (out / p).parent.mkdir(parents=True, exist_ok=True); (out / p).write_bytes(raw)
        (out / 'workload.json').write_text(json.dumps(spec, separators=(',', ':')))
        subprocess.run([sys.executable, str(ROOT / 'runtime/compile_workload.py'), *(['--emit-golden'] if os.environ.get('GOLDEN') else []), str(out / 'workload.json'), str(out / 'bundle')], check=True, stdout=subprocess.DEVNULL)
