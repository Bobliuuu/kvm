"""Policy-gradient / Q-learning losses and the audit's never-shipped models (token MLPs without attention, odd transformer) through the
public API only, against float64 numpy oracles (forward + central finite differences) and on the CPU golden VM."""
import sys
import unittest
from pathlib import Path

import numpy as np

import _path  # noqa: F401
import oracle_models as om
from kvmfe.generic import MLP, ArrayData, CrossEntropy, PolicyGradient, QLearning, TrainableModule
from kvmfe.ir import FrontendError
from kvmfe.lower import Tiling
from kvmfe.nn import ModelConfig
from kvmfe.optim import OptimConfig
from kvmfe.pipeline import compile_training
from kvmfe.train import TrainConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'generic'))
import zoo  # noqa: E402

CPU = Tiling.preset('cpu')


def build(case, steps=None):
    return compile_training(case.module, loss=case.loss, data=case.data, optimizer=case.optimizer, steps=steps or case.steps,
                            tiling=CPU, reuse='none')


def first(model):
    out = {}
    for s in model.batch_streams(1):
        a = np.array(s.values[:s.rows * s.cols], dtype=np.float32 if s.dtype == 'f32' else np.int64).reshape(s.rows, s.cols)
        out[s.name] = a.astype(np.float64) if s.dtype == 'f32' else a
    return out


def fwd_pg(p, d):
    h = np.tanh(d['x'] @ p['pg.fc0.w'] + p['pg.fc0.b'])
    lg = h @ p['pg.fc1.w'] + p['pg.fc1.b']
    m = lg.max(1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(lg - m).sum(1))
    nll = lse - lg[np.arange(len(lg)), d['act'][:, 0]]
    return float((d['adv'][:, 0] * nll).mean())


def fwd_dqn(p, d):
    h = np.maximum(d['x'] @ p['dqn.fc0.w'] + p['dqn.fc0.b'], 0)
    q = h @ p['dqn.fc1.w'] + p['dqn.fc1.b']
    return float((((q * d['mask']).sum(1) - d['y'][:, 0]) ** 2).mean())


class Oracle(unittest.TestCase):
    def check(self, name, fwd):
        case = zoo.CASES[name]()
        comp = build(case, steps=1)
        mem = comp.run()
        model = TrainableModule(case.module, case.loss, case.data)
        p, data = om.params(model), first(model)
        want = fwd(p, data)
        self.assertAlmostEqual(mem.matrix('loss.0')[0][0], want, delta=1e-9 * max(1, abs(want)), msg=f'{name} loss')
        rng = np.random.default_rng(0)
        for pname in model.specs:
            buf = mem.bufs[comp.low.plan.buf(comp.program.grads[pname])]
            g = np.array(buf.a, dtype=np.float64).reshape(buf.rows, buf.cols)
            for idx in {np.unravel_index(np.abs(g).argmax(), g.shape), tuple(int(rng.integers(0, k)) for k in g.shape)}:
                old, h = p[pname][idx], 1e-5
                p[pname][idx] = old + h
                lp = fwd(p, data)
                p[pname][idx] = old - h
                lm = fwd(p, data)
                p[pname][idx] = old
                fd = (lp - lm) / (2 * h)
                err = abs(g[idx] - fd) / max(abs(fd), 0.05 * np.abs(g).max(), 1e-8)
                self.assertLess(err, 2e-6, f'{name} d loss / d {pname}{idx}: autodiff {g[idx]:.8g} vs finite difference {fd:.8g}')

    def test_policy_gradient(self):
        self.check('pg_policy', fwd_pg)

    def test_q_learning(self):
        self.check('dqn_q', fwd_dqn)

    def test_policy_gradient_with_unit_advantage_is_cross_entropy(self):
        rows = [[0.1 * i, -0.2 * i, 0.3] for i in range(8)]
        act = [i % 3 for i in range(8)]
        run = lambda loss, arrays, tgt: compile_training(
            MLP('m', [3, 5, 3], act='tanh'), loss=loss, data=ArrayData(8, arrays, dtypes={'act': 'i32'}, target=tgt),
            optimizer=OptimConfig(kind='sgd', lr=0.1), steps=2, tiling=CPU, reuse='none').run()
        a = run(PolicyGradient(), {'x': rows, 'act': [[v] for v in act], 'adv': [[1.0]] * 8}, ('act', 'adv'))
        b = run(CrossEntropy(), {'x': rows, 'act': [[v] for v in act]}, 'act')
        for k in a.bufs:
            if k.startswith('w.'):
                self.assertEqual(a.bufs[k].a, b.bufs[k].a, k)
            elif k.startswith('loss.'):
                self.assertEqual(a.bufs[k].a[0], b.bufs[k].a[0], k)

    def test_rl_losses_learn(self):
        for name, steps in (('pg_policy', 25), ('dqn_q', 25)):
            c = build(zoo.CASES[name](), steps=steps)
            mem = c.run()
            ls = [mem.matrix(f'loss.{s}')[0][0] for s in range(steps)]
            self.assertLess(ls[-1], ls[0], (name, ls))


class Refusals(unittest.TestCase):
    def test_target_count_must_match_loss(self):
        with self.assertRaisesRegex(FrontendError, 'needs 2 target'):
            compile_training(MLP('m', [3, 3], act='relu'), loss=PolicyGradient(),
                             data=ArrayData(2, {'x': [[0.0] * 3] * 2, 'y': [[0]] * 2}, dtypes={'y': 'i32'}),
                             optimizer=OptimConfig(kind='sgd'), steps=1, tiling=CPU)

    def test_action_range_and_dtypes(self):
        arrays = {'x': [[0.0] * 3] * 2, 'act': [[0], [5]], 'adv': [[1.0], [1.0]]}
        with self.assertRaisesRegex(FrontendError, 'action ids outside'):
            compile_training(MLP('m', [3, 3], act='relu'), loss=PolicyGradient(),
                             data=ArrayData(2, arrays, dtypes={'act': 'i32'}, target=('act', 'adv')),
                             optimizer=OptimConfig(kind='sgd'), steps=1, tiling=CPU)

    def test_qlearning_mask_shape(self):
        with self.assertRaisesRegex(FrontendError, 'QLearning'):
            compile_training(MLP('m', [3, 4], act='relu'), loss=QLearning(),
                             data=ArrayData(2, {'x': [[0.0] * 3] * 2, 'mask': [[1.0, 0.0]] * 2, 'y': [[0.0]] * 2}, target=('mask', 'y')),
                             optimizer=OptimConfig(kind='sgd'), steps=1, tiling=CPU)


class AuditModels(unittest.TestCase):
    """The audit's newmodel/gen.py cases with no workaround: every one goes through compile_training."""

    def test_token_mlps_without_attention_train_and_fall(self):
        for name in ('mlp_ce', 'resmlp_rms', 'bad_dims'):
            c = build(zoo.CASES[name](), steps=4)
            mem = c.run()
            ls = [mem.matrix(f'loss.{s}')[0][0] for s in range(4)]
            self.assertLess(ls[-1], ls[0], (name, ls))

    def test_xfmr_odd_via_model_config(self):
        cfg = ModelConfig('xfmr_odd', vocab=100, d_model=48, layers=3, heads=3, seq=64, batch=2, ffn_mult=2, norm='rmsnorm', bias=False)
        c = compile_training(cfg, TrainConfig(steps=2, optim=OptimConfig(kind='sgd', lr=0.05), clip=0), CPU, reuse='none')
        mem = c.run()
        self.assertLess(mem.matrix('loss.1')[0][0], mem.matrix('loss.0')[0][0])

    def test_model_config_without_attention_blocks_needs_no_attention_geometry(self):
        ModelConfig('embed_head', vocab=100, d_model=40, layers=0, heads=2, seq=48)      # was: head dim 20 / seq % 32 refusal
        with self.assertRaisesRegex(FrontendError, 'head dim'):
            ModelConfig('bad_head', d_model=48, heads=2, seq=32)
        with self.assertRaisesRegex(FrontendError, 'generic'):
            ModelConfig('bad_head', d_model=48, heads=2, seq=32)


if __name__ == '__main__':
    unittest.main()
