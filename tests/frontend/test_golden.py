"""The oracle tests: emitted GPT workload on the CPU golden VM vs an independent float64 implementation."""
import unittest

import _path  # noqa: F401
import oracle_gpt as ref_gpt
from kvmfe.lower import Tiling
from kvmfe.nn import ModelConfig, PRESETS
from kvmfe.optim import OptimConfig
from kvmfe.pipeline import compile_training
from kvmfe.refvm import execute
from kvmfe.train import TrainConfig


def run(cfg, steps, reuse='shape', gap=0, order='program', seed=0, tiling='cpu'):
    c = compile_training(cfg, TrainConfig(steps=steps), Tiling.preset(tiling), reuse, gap)
    return c, execute(c.spec, order=order, seed=seed)


class Golden(unittest.TestCase):
    def test_20_step_loss_curve_matches_independent_reference(self):
        cfg = PRESETS['gpt_tiny']
        _, mem = run(cfg, 20)
        got = [mem.matrix(f'loss.{s}')[0][0] for s in range(20)]
        want, hist = ref_gpt.train_reference(cfg, 20, OptimConfig())
        for s, (a, b) in enumerate(zip(got, want)):
            self.assertAlmostEqual(a, b, delta=1e-9 * max(1.0, abs(b)), msg=f'step {s}')
        self.assertLess(got[-1], got[0] - 0.5)             # it actually learns the bigram chain
        for n, rows in hist[-1].items():                   # final weights too
            for r, ref in zip(mem.matrix(f'w.{n}'), rows):
                for x, y in zip(r, ref):
                    self.assertAlmostEqual(x, y, delta=1e-9)

    def test_rmsnorm_no_bias_two_layers(self):
        cfg = ModelConfig(name='t', vocab=32, d_model=32, layers=2, heads=2, seq=32, batch=1, norm='rmsnorm', bias=False)
        _, mem = run(cfg, 3)
        want, _ = ref_gpt.train_reference(cfg, 3, OptimConfig())
        for s in range(3):
            self.assertAlmostEqual(mem.matrix(f'loss.{s}')[0][0], want[s], delta=1e-9)

    def test_batch_two_sequences(self):
        cfg = ModelConfig(name='t', vocab=32, d_model=32, layers=1, heads=2, seq=32, batch=2)
        _, mem = run(cfg, 2)
        want, _ = ref_gpt.train_reference(cfg, 2, OptimConfig())
        for s in range(2):
            self.assertAlmostEqual(mem.matrix(f'loss.{s}')[0][0], want[s], delta=1e-9)


class BigTiles(unittest.TestCase):
    """128x128 / 256x128 GEMM tiles and multi-block attention tasks give the independent reference's result on the CPU golden VM,
    in program order and in random dependency-respecting orders (their edges are complete)."""
    cfg = ModelConfig(name='t', vocab=32, d_model=32, layers=1, heads=2, seq=128, batch=2)   # 256 tokens, ffn 128: 128/256 tiles apply

    def check(self, **kw):
        tiling = Tiling.preset('cpu', mm_min_tiles=1, mm_min_eff=0.5, **kw)
        c = compile_training(self.cfg, TrainConfig(steps=2), tiling)
        flags = {t['args'].get('flags', 0) & 96 for g in c.spec['graphs'] for t in g['tasks'] if t['op'] == 'matmul_tile'}
        want, _ = ref_gpt.train_reference(self.cfg, 2, OptimConfig())
        for order in ('program', 'random'):
            mem = execute(c.spec, order=order, seed=3)
            for s in range(2):
                self.assertAlmostEqual(mem.matrix(f'loss.{s}')[0][0], want[s], delta=1e-9, msg=f'{kw} {order} step {s}')
        return c, flags

    def test_mm128(self):
        _, flags = self.check(mm_tile=128)
        self.assertIn(32, flags)

    def test_mm256_and_attention_span(self):
        c, flags = self.check(mm_tile=256, attn_span=2)
        self.assertIn(64, flags)
        base = compile_training(self.cfg, TrainConfig(steps=2), Tiling.preset('cpu'))
        self.assertLess(len(c.low.tasks), len(base.low.tasks))

    def test_attention_span_four(self):
        self.check(attn_span=4)


class Dataflow(unittest.TestCase):
    """Edges found by the analysis are complete: any topological order gives bitwise identical state."""

    def same(self, a, b):
        self.assertEqual(a.bufs.keys(), b.bufs.keys())
        for k in a.bufs:
            self.assertEqual(a.bufs[k].a, b.bufs[k].a, k)

    def test_random_orders_equal_program_order(self):
        cfg = ModelConfig(name='t', vocab=32, d_model=32, layers=2, heads=2, seq=32, batch=1)
        _, base = run(cfg, 3)
        for seed in (1, 2):
            self.same(base, run(cfg, 3, order='random', seed=seed)[1])

    def test_memory_policy_does_not_change_results(self):
        cfg = PRESETS['gpt_tiny']
        _, base = run(cfg, 3)
        for reuse, gap in (('none', 0), ('shape', 25)):
            _, m = run(cfg, 3, reuse, gap, order='random', seed=3)
            for s in range(3):
                self.assertEqual(base.matrix(f'loss.{s}'), m.matrix(f'loss.{s}'))
            for k in base.bufs:
                if k.startswith('w.'):
                    self.assertEqual(base.bufs[k].a, m.bufs[k].a, k)

    def test_tiling_does_not_change_results(self):
        cfg = PRESETS['gpt_tiny']
        _, a = run(cfg, 2, tiling='cpu')
        _, b = run(cfg, 2, tiling='a100')
        for s in range(2):
            self.assertAlmostEqual(a.matrix(f'loss.{s}')[0][0], b.matrix(f'loss.{s}')[0][0], delta=1e-12)


class Oracle(unittest.TestCase):
    def test_reference_gradients_match_finite_differences(self):
        cfg = ModelConfig(name='t', vocab=16, d_model=16, layers=1, heads=1, seq=32, batch=1)
        p = ref_gpt.init_params(cfg)
        ids, tgt = ref_gpt.batch_tokens(cfg, 0)
        _, G = ref_gpt.loss_and_grads(cfg, p, ids, tgt)
        for name, (i, j) in (('gpt.h0.attn.q.w', (3, 5)), ('gpt.h0.mlp.fc1.w', (2, 7)), ('gpt.wte', (int(ids[0]), 4)),
                             ('gpt.nf.g', (0, 2)), ('gpt.head.w', (1, 3)), ('gpt.h0.attn.k.b', (0, 1))):
            h = 1e-6
            old = p[name][i][j]
            p[name][i][j] = old + h
            lp, _ = ref_gpt.loss_and_grads(cfg, p, ids, tgt)
            p[name][i][j] = old - h
            lm, _ = ref_gpt.loss_and_grads(cfg, p, ids, tgt)
            p[name][i][j] = old
            self.assertAlmostEqual(G[name][i][j], (lp - lm) / (2 * h), delta=1e-6, msg=name)


if __name__ == '__main__':
    unittest.main()
