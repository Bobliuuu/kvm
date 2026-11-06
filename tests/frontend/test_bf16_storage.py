"""Real bf16 storage (KvmBuf dtype 2): a generic MLP whose weight copies and between-layer activations live in 2-byte buffers must
equal the emulated bf16 path (bf16 values in fp32 containers) bit for bit, on the frontend reference VM and through the runtime
compiler's checked CPU golden (runtime/ref_extra.py). The memory plan must count 2 bytes per bf16 element."""
import sys
import unittest

import numpy as np

import _path  # noqa: F401
from kvmfe import _runtime  # noqa: F401  (puts runtime/ on sys.path)
from kvmfe.generic import MLP, MSE, ArrayData, CrossEntropy
from kvmfe.ir import Builder, FrontendError, Tensor
from kvmfe.lower import Tiling
from kvmfe.optim import OptimConfig
from kvmfe.pipeline import compile_training
from kvmfe.train import TrainConfig

CPU = Tiling.preset('cpu')


def make(precision, act='gelu', loss='ce', steps=3, norm=None, residual=False):
    rng = np.random.default_rng(3)
    x = rng.normal(size=(24, 20)).tolist()
    dims = [20, 24, 24, 5]
    if loss == 'ce':
        data = ArrayData(8, {'x': x, 'y': rng.integers(0, 5, size=24).tolist()}, dtypes={'y': 'i32'})
        lf = CrossEntropy()
    else:
        data = ArrayData(8, {'x': x, 'y': rng.normal(size=(24, 5)).tolist()})
        lf = MSE()
    return compile_training(MLP('m', dims, act=act, norm=norm, residual=residual, precision=precision, std=0.3), TrainConfig(steps=steps),
                            CPU, loss=lf, data=data, optimizer=OptimConfig(kind='sgd', lr=0.05), reuse='shape')


def state(comp):
    mem = comp.run()
    return {k: v.a[:] for k, v in mem.bufs.items() if k.startswith(('w.', 'loss.'))}


class Equivalence(unittest.TestCase):
    def check(self, **kw):
        emu, real = make('bf16', **kw), make('bf16-real', **kw)
        a, b = state(emu), state(real)
        self.assertEqual(a.keys(), b.keys())
        for k in a:
            self.assertEqual(a[k], b[k], k)
        return emu, real

    def test_gelu_ce(self):
        self.check()

    def test_tanh_mse(self):
        self.check(act='tanh', loss='mse')

    def test_residual_rmsnorm(self):
        self.check(act='gelu', residual=True, norm='rmsnorm')

    def test_real_differs_from_f32(self):
        f32 = state(make('f32'))
        real = state(make('bf16-real'))
        self.assertNotEqual(f32['w.m.fc0.w'], real['w.m.fc0.w'])

    def test_storage_is_two_bytes_and_planned_as_such(self):
        emu, real = self.check()
        dts = {b['name']: b.get('dtype', 'f32') for b in real.spec['buffers']}
        self.assertIn('bf16', dts.values())
        self.assertNotIn('bf16', {b.get('dtype', 'f32') for b in emu.spec['buffers']})
        bf = [n for n, d in dts.items() if d == 'bf16']
        ops = {t['op'] for g in real.spec['graphs'] for t in g['tasks']}
        self.assertTrue({'cast_f32_bf16', 'matmul_tile_bf16s'} <= ops, ops)
        self.assertNotIn('matmul_tile_bf16s', {t['op'] for g in emu.spec['graphs'] for t in g['tasks']})
        # bytes: every bf16 slot counts 2 bytes per element in the plan
        plan = real.low.plan
        slot_bytes = sum(b.rows * b.cols * 2 for b in plan.buffers.values() if b.dtype == 'bf16')
        self.assertGreater(slot_bytes, 0)
        self.assertEqual(sorted(bf), sorted(n for n, b in plan.buffers.items() if b.dtype == 'bf16' and n in dts))


class RuntimeGolden(unittest.TestCase):
    def test_compiler_checks_and_golden_run_match_the_frontend_vm(self):
        import compile_workload as cw
        import workload_vm as wvm
        real = make('bf16-real', steps=2)
        compiled = cw.compile_spec(real.spec, data_dir=None) if not real.low.data_files else None
        if compiled is None:
            import tempfile
            from pathlib import Path
            with tempfile.TemporaryDirectory() as d:
                for p, raw in real.low.data_files.items():
                    (Path(d) / p).parent.mkdir(parents=True, exist_ok=True)
                    (Path(d) / p).write_bytes(raw)
                compiled = cw.compile_spec(real.spec, data_dir=d)
                golden = wvm.run_golden(compiled)
        else:
            golden = wvm.run_golden(compiled)
        self.assertIn('bf16', set(golden['dtypes'].values()))
        want = real.run()
        for k, v in golden['buffers'].items():
            if k.startswith('w.'):
                np.testing.assert_allclose(v, want.bufs[k].a, rtol=1e-5, atol=1e-6, err_msg=k)

    def test_compiler_rejects_wrong_dtypes(self):
        import compile_workload as cw
        base = {'buffers': [
            {'name': 'x', 'rows': 4, 'cols': 4, 'init': {'kind': 'fill', 'value': 1}},
            {'name': 'y', 'rows': 4, 'cols': 4, 'dtype': 'bf16', 'init': {'kind': 'fill', 'value': 1}}],
            'graphs': [{'name': 'g', 'tasks': []}]}

        def spec(op, **args):
            s = {**base, 'graphs': [{'name': 'g', 'tasks': [{'id': 't', 'op': op, 'args': args}]}]}
            return s
        ok = spec('cast_f32_bf16', x='x', out='y', e0=0, e1=16)
        cw.compile_spec(ok)                                                    # f32 -> bf16 storage: accepted
        for bad in (spec('cast_f32_bf16', x='y', out='y', e0=0, e1=16),         # source must be f32
                    spec('cast_bf16_f32', x='x', out='x', e0=0, e1=16),         # source must be bf16
                    spec('round_bf16', x='y', out='y', e0=0, e1=16)):           # f32-only op refuses bf16 storage
            with self.assertRaises(cw.CompileError):
                cw.compile_spec(bad)


class Builder_(unittest.TestCase):
    def test_refusals(self):
        b = Builder('t')
        x = Tensor('x', 4, 4, 'input', buf='x', init=('fill', 1.0))
        w = Tensor('w', 4, 4, 'param', buf='w', init=('fill', 1.0), rg=True)
        xb = b.cast(x, 'bf16', real=True)
        self.assertEqual(xb.dtype, 'bf16')
        with self.assertRaisesRegex(FrontendError, 'need bf16=True'):
            b.linear(xb, w)
        with self.assertRaisesRegex(FrontendError, 'needs an f32'):
            b.cast(xb, 'bf16')
        with self.assertRaisesRegex(FrontendError, 'widens a bf16'):
            b.cast(x, 'f32')
        self.assertEqual(b.cast(xb, 'f32').dtype, 'f32')
        self.assertEqual(b.linear(xb, w, bf16=True).dtype, 'f32')


if __name__ == '__main__':
    unittest.main()
