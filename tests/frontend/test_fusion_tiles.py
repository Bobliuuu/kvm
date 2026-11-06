"""Perf/fusion golden: the task-count reductions (a100 tiling instead of the tiny cpu tiling, fusion pass, GEMM tile merging) must
not change the result. Each variant is run on the CPU golden VM and the final weights and losses must equal the baseline
(cpu tiling, no fusion) BITWISE; the variants must also have fewer tasks."""
import unittest

import _path  # noqa: F401
from kvmfe.generic import MLP, CrossEntropy, GeneratedData
from kvmfe.lower import Tiling
from kvmfe.nn import ModelConfig
from kvmfe.optim import OptimConfig
from kvmfe.pipeline import compile_training
from kvmfe.train import TrainConfig, batch_tokens

SGD = OptimConfig(kind='sgd', lr=0.05)
ADAM = OptimConfig(kind='adamw', lr=0.01)


def state(c, mem):
    return {k: v.a[:] for k, v in mem.bufs.items() if k.startswith('w.') or k.startswith('loss.')}


def mlp(tiling, fuse, steps=2, opt=SGD, **kw):
    cfg = ModelConfig('m', vocab=40, d_model=32, layers=1, heads=2, seq=32, batch=3)
    net = MLP('mlp', [32, 72, 72, 40], act='gelu', bias=True)
    make = lambda m: {'x': [[((r * 7 + c * 3 + m) % 11) / 11.0 - 0.5 for c in range(32)] for r in range(96)],
                      'y': [[(r * 5 + m) % 40] for r in range(96)]}
    data = GeneratedData(make, {'x': (96, 32, 'f32'), 'y': (96, 1, 'i32')}, inputs=('x',), target='y')
    return compile_training(net, loss=CrossEntropy(), data=data, optimizer=opt, steps=steps, tiling=tiling, reuse='none', fuse=fuse, **kw)


def xfmr(tiling, fuse, steps=2, opt=SGD, **kw):
    cfg = ModelConfig('x', vocab=40, d_model=32, layers=2, heads=2, seq=32, batch=2, ffn_mult=2, norm='rmsnorm', bias=False)
    return compile_training(cfg, TrainConfig(steps=steps, optim=opt, clip=0), tiling, reuse='none', fuse=fuse, **kw)


class FusionGolden(unittest.TestCase):
    def check(self, build, variants):
        base = build(Tiling.preset('cpu'), False)
        want = state(base, base.run())
        for tiling, fuse in variants:
            c = build(tiling, fuse)
            self.assertLess(len(c.low.tasks), len(base.low.tasks), (tiling, fuse))
            self.assertEqual(state(c, c.run()), want, (tiling, fuse))
            self.assertEqual(state(c, c.run(order='random', seed=3)), want, (tiling, fuse, 'random order'))

    def test_mlp_a100_tiling_and_fusion(self):
        self.check(mlp, [(Tiling.preset('a100'), False), (Tiling.preset('a100'), True)])

    def test_xfmr_a100_tiling_and_fusion(self):
        self.check(xfmr, [(Tiling.preset('a100'), False), (Tiling.preset('a100'), True)])


if __name__ == '__main__':
    unittest.main()
