#!/usr/bin/env python3
"""Per model: CPU float64 VM vs the independent numpy oracle (tests/frontend/oracle_models.py): step-0 loss difference and the worst
autodiff-vs-finite-difference gradient error over a spot check of every parameter tensor. python3 examples/models/oracle_report.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'frontend'), str(ROOT / 'tests/frontend')]
import numpy as np  # noqa: E402
import oracle_models as om  # noqa: E402
from kvmfe import models  # noqa: E402
from kvmfe.lower import Tiling  # noqa: E402
from kvmfe.pipeline import build_model, compile_training  # noqa: E402
from kvmfe.train import TrainConfig  # noqa: E402

CASES = {n: models.PRESETS[n] for n in ('mlp_tiny', 'vit_tiny', 'llama_tiny', 'llama_gqa_tied', 'moe_tiny')}
CASES['moe_drop_k1'] = models.LlamaConfig('moe_drop_k1', layers=1, heads=4, kv_heads=4, ffn=32, batch=2, experts=4, top_k=1, capacity_factor=0.5)


def main():
    print('| model | params | loss CPU f64 VM vs numpy oracle (abs) | worst autodiff vs finite-difference grad error (spot check of every tensor) |')
    print('|---|---|---|---|')
    for name, cfg in CASES.items():
        model = build_model(cfg)
        c = compile_training(cfg, TrainConfig(steps=1), Tiling.preset('cpu'), reuse='none')
        mem = c.run()
        p, data = om.params(model), om.streams(model)
        want = om.forward(model, p, data)
        dl = max(abs(mem.matrix(f'{n}.0')[0][0] - v) for n, v in want.items())
        rng, worst = np.random.default_rng(0), 0.0
        for pn in model.specs:
            b = mem.bufs[c.low.plan.buf(c.program.grads[pn])]
            g = np.array(b.a).reshape(b.rows, b.cols)
            for idx in {np.unravel_index(np.abs(g).argmax(), g.shape), tuple(int(rng.integers(0, d)) for d in g.shape)}:
                fd = om.fd_grad(model, p, data, pn, idx)
                worst = max(worst, abs(g[idx] - fd) / max(abs(fd), 0.05 * np.abs(g).max(), 1e-8))
        n = sum(s.rows * s.cols for s in model.specs.values())
        print(f'| {name} | {n:,} | {dl:.1e} | {worst:.1e} |', flush=True)


if __name__ == '__main__':
    main()
