#!/usr/bin/env python3
"""Task-count / critical-path effect of the fusion pass for every zoo model (CPU only, cost model of kvmfe.analyze).

  python3 examples/models/fusion_report.py [--tiling cpu|a100] [--steps N]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'frontend'))
from kvmfe import models, nn  # noqa: E402
from kvmfe.analyze import CostModel, analyze  # noqa: E402
from kvmfe.lower import Tiling  # noqa: E402
from kvmfe.pipeline import compile_training  # noqa: E402
from kvmfe.train import TrainConfig  # noqa: E402

CASES = ['gpt_tiny', 'gpt_small', 'llama_tiny', 'llama_gqa_tied', 'moe_tiny', 'mlp_tiny', 'vit_tiny']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tiling', default='cpu')
    ap.add_argument('--steps', type=int, default=1)
    a = ap.parse_args()
    print(f'{"model":16} {"tasks":>16} {"crit path (tasks)":>20} {"crit path (model us)":>22}   fused patterns')
    for name in CASES:
        cfg = nn.PRESETS.get(name) or models.PRESETS[name]
        row = []
        for fuse in (False, True):
            c = compile_training(cfg, TrainConfig(steps=a.steps), Tiling.preset(a.tiling), with_spec=False, fuse=fuse)
            r = analyze(c.low, CostModel(), 108)
            row.append((r['tasks'], r['critical_path_tasks'], r['critical_path_us'], {k: v for k, v in r['by_op'].items() if 'silu_mul' in k or 'dgelu' in k}))
        (t0, p0, u0, _), (t1, p1, u1, ops) = row
        print(f'{name:16} {t0:>7,} -> {t1:>6,} ({(1 - t1 / t0):5.1%}) {p0:>6} -> {p1:>5} {u0:>10,.0f} -> {u1:>8,.0f}   {ops}')


if __name__ == '__main__':
    main()
