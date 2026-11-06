"""Generic training API (kvmfe.generic + pipeline.compile_training(module, loss=, data=, optimizer=)) on the CPU golden VM.

Models the repo never shipped (tests/generic/zoo.py) are checked against independent float64 numpy oracles written here (forward +
central finite differences, so autodiff is the thing under test), then trained for a few steps in program order and in random
dataflow order, then pushed through the MANIFEST path the demo host runner consumes. The GPT path must be untouched: a GPT run through
the generic API equals the built-in one. The same models run on the A100 in tests/generic/run.sh."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

import _path  # noqa: F401
import oracle_models as om
from kvmfe import demo, run_manifest
from kvmfe.generic import MLP, MSE, ArrayData, Conv1D, CrossEntropy, GeneratedData, TrainableModule
from kvmfe.ir import FrontendError
from kvmfe.lower import Tiling
from kvmfe.nn import GPT, ModelConfig
from kvmfe.optim import OptimConfig
from kvmfe.pipeline import compile_training
from kvmfe.train import TrainConfig, batch_stream

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'generic'))
import zoo  # noqa: E402

CPU = Tiling.preset('cpu')


def build(case, **kw):
    return compile_training(case.module, loss=case.loss, data=case.data, optimizer=case.optimizer, steps=kw.pop('steps', case.steps),
                            tiling=CPU, reuse=kw.pop('reuse', 'none'), **kw)


# ---- independent float64 forward oracles (loss of the first microbatch) -------------------------------------------------------
def fwd_reg_mlp(p, d):
    h = np.tanh(d['x'] @ p['reg.fc0.w'] + p['reg.fc0.b'])
    h = np.tanh(h @ p['reg.fc1.w'] + p['reg.fc1.b'])
    out = h @ p['reg.fc2.w'] + p['reg.fc2.b']
    return float(((out - d['y']) ** 2).mean())


def fwd_res_rms(p, d):
    x = d['x']
    for i in range(3):
        y = om.lin(x, p[f'rm.fc{i}.w'], p[f'rm.fc{i}.b'], gelu=True, res=x if x.shape[1] == p[f'rm.fc{i}.w'].shape[1] else None)
        x = om.rmsnorm(y, p[f'rm.n{i}.g'])
    return om.ce(om.lin(x, p['rm.fc3.w'], p['rm.fc3.b']), d['y'])


def conv(x, p, name, taps, length):
    out = np.zeros((x.shape[0], p[f'{name}.w0'].shape[1])) + p[f'{name}.b']
    for j in range(taps):
        s = j - taps // 2
        for r in range(x.shape[0]):
            t = r % length + s
            if 0 <= t < length:
                out[r] += x[r + s] @ p[f'{name}.w{j}']
    return out


def silu(x):
    return x / (1 + np.exp(-x))


def fwd_cnn(p, d, length=12):
    h = np.maximum(conv(d['x'], p, 'cnn.c1', 3, length), 0)
    h = silu(conv(h, p, 'cnn.c2', 5, length))
    pooled = h.reshape(-1, length, h.shape[1]).mean(1)
    return om.ce(pooled @ p['cnn.head.w'] + p['cnn.head.b'], d['y'])


def fwd_encoder(p, d, seq=32):
    x = d['x'] @ p['enc.embed.w'] + p['enc.embed.b']
    h = om.layernorm(x, p['enc.n1.g'], p['enc.n1.b'])
    lin = lambda t, n: t @ p[f'enc.{n}.w'] + p[f'enc.{n}.b']
    x = x + lin(om.attention(lin(h, 'q'), lin(h, 'k'), lin(h, 'v'), 2, 2, seq, False), 'o')
    x = x + om.lin(om.layernorm(x, p['enc.n2.g'], p['enc.n2.b']), p['enc.fc1.w'], p['enc.fc1.b'], gelu=True) @ p['enc.fc2.w'] + p['enc.fc2.b']
    pooled = om.layernorm(x, p['enc.nf.g'], p['enc.nf.b']).reshape(-1, seq, x.shape[1]).mean(1)
    return float(((lin(pooled, 'head') - d['y']) ** 2).mean())


ORACLES = {'reg_mlp': fwd_reg_mlp, 'res_rms_mlp': fwd_res_rms, 'cnn1d': fwd_cnn, 'encoder_reg': fwd_encoder}


def first_batch(model):
    return {s.name: np.array(s.values[:s.rows * s.cols], dtype=np.float32 if s.dtype == 'f32' else np.int64).reshape(s.rows, s.cols)
            .astype(np.float64 if s.dtype == 'f32' else np.int64) for s in model.batch_streams(1)}


class Oracle(unittest.TestCase):
    """Step-0 loss and a gradient spot check of every parameter tensor against central finite differences of the numpy oracle."""

    def check(self, name):
        case = zoo.CASES[name]()
        comp = build(case, steps=1)
        mem = comp.run()
        model = TrainableModule(case.module, case.loss, case.data)
        p, data = om.params(model), first_batch(model)
        if case.loss.target_dtype == 'i32':
            data['y'] = data['y'][:, 0]
        fwd = ORACLES[name]
        want = fwd(p, data)
        self.assertAlmostEqual(mem.matrix('loss.0')[0][0], want, delta=1e-9 * max(1, abs(want)), msg=f'{name} loss')
        rng = np.random.default_rng(0)
        for pname in model.specs:
            buf = mem.bufs[comp.low.plan.buf(comp.program.grads[pname])]
            g = np.array(buf.a, dtype=np.float64).reshape(buf.rows, buf.cols)
            picks = {np.unravel_index(np.abs(g).argmax(), g.shape), tuple(int(rng.integers(0, k)) for k in g.shape)}
            for idx in picks:
                old = p[pname][idx]
                h = 1e-5
                p[pname][idx] = old + h
                lp = fwd(p, data)
                p[pname][idx] = old - h
                lm = fwd(p, data)
                p[pname][idx] = old
                fd = (lp - lm) / (2 * h)
                if np.abs(g).max() < 1e-9:
                    self.assertLess(abs(fd), 1e-9, f'{name} d loss / d {pname}{idx} should be ~0')
                    continue
                err = abs(g[idx] - fd) / max(abs(fd), 0.05 * np.abs(g).max(), 1e-8)
                self.assertLess(err, 2e-6, f'{name} d loss / d {pname}{idx}: autodiff {g[idx]:.8g} vs finite difference {fd:.8g}')

    def test_regression_mlp_mse(self):
        self.check('reg_mlp')

    def test_residual_rmsnorm_mlp_ce(self):
        self.check('res_rms_mlp')

    def test_conv1d_cnn_ce(self):
        self.check('cnn1d')

    def test_transformer_encoder_mse(self):
        self.check('encoder_reg')

    def test_mse_scalar_oracle_and_nonsquare_tiles(self):
        """MSE with an output wider than a row tile and a row count that is not a tile multiple (cpu tiling: 8-row blocks)."""
        rng = np.random.default_rng(3)
        x, y = rng.normal(size=(19, 4)), rng.normal(size=(19, 37))
        mod = MLP('m', [4, 37], act='relu')
        comp = compile_training(mod, loss=MSE(), data=ArrayData(19, {'x': x.tolist(), 'y': y.tolist()}),
                                optimizer=OptimConfig(kind='sgd', lr=0.01), steps=1, tiling=CPU)
        mem = comp.run()
        p = om.params(TrainableModule(mod, MSE(), ArrayData(19, {'x': x.tolist(), 'y': y.tolist()})))
        xr, yr = x.astype(np.float32).astype(np.float64), y.astype(np.float32).astype(np.float64)
        want = float((((xr @ p['m.fc0.w'] + p['m.fc0.b']) - yr) ** 2).mean())
        self.assertAlmostEqual(mem.matrix('loss.0')[0][0], want, delta=1e-9)


class Training(unittest.TestCase):
    def test_every_case_trains_in_any_dataflow_order(self):
        """Complete dataflow edges: a random topological order gives the same final buffers as program order, for all four models."""
        from kvmfe.refvm import execute
        for name, make in zoo.CASES.items():
            comp = build(make())
            a = comp.run()
            b = execute(comp.spec, order='random', seed=5, data_files=comp.low.data_files)
            for k, v in a.bufs.items():
                self.assertEqual(v.a, b.bufs[k].a, f'{name}: buffer {k} differs between program and random order')

    def test_losses_fall(self):
        specs = {'reg_mlp': 40, 'cnn1d': 40}
        for name, steps in specs.items():
            case = zoo.CASES[name]()
            comp = build(case, steps=steps)
            mem = comp.run()
            ls = [mem.matrix(f'loss.{s}')[0][0] for s in range(steps)]
            self.assertLess(np.mean(ls[-5:]), 0.8 * np.mean(ls[:5]), f'{name} loss did not fall: {ls[:3]} ... {ls[-3:]}')

    def test_accumulation_equals_big_batch(self):
        """Two microbatches of 6 samples per step == one microbatch of 12 (the gradient is the mean either way)."""
        rng = np.random.default_rng(1)
        x, y = rng.normal(size=(24, 5)).tolist(), rng.normal(size=(24, 2)).tolist()
        runs = []
        for batch, accum in ((12, 1), (6, 2)):
            comp = compile_training(MLP('m', [5, 7, 2], act='tanh'), TrainConfig(steps=2, accum=accum), CPU,
                                    loss=MSE(), data=ArrayData(batch, {'x': x, 'y': y}), optimizer=OptimConfig(kind='sgd', lr=0.1))
            mem = comp.run()
            runs.append({k: v.a[:] for k, v in mem.bufs.items() if k.startswith('w.')})
        for k in runs[0]:
            np.testing.assert_allclose(runs[0][k], runs[1][k], rtol=1e-9, atol=1e-12, err_msg=k)

    def test_gpt_through_the_generic_api_equals_the_builtin_path(self):
        cfg = ModelConfig('gpt_tiny')
        built_in = compile_training(cfg, TrainConfig(steps=2), CPU).run()
        ids, tgt = batch_stream(cfg, 2)
        n = cfg.tokens
        data = ArrayData(n, {'ids': ids, 'tgt': tgt}, dtypes={'ids': 'i32', 'tgt': 'i32'}, inputs=('ids',), target='tgt')
        generic = compile_training(GPT(cfg), TrainConfig(steps=2), CPU, loss=CrossEntropy(), data=data).run()
        for s in range(2):
            self.assertEqual(built_in.matrix(f'loss.{s}')[0][0], generic.matrix(f'loss.{s}')[0][0])
        for k, v in built_in.bufs.items():
            if k.startswith('w.'):
                self.assertEqual(v.a, generic.bufs[k].a, k)


class Manifest(unittest.TestCase):
    def test_manifest_bundle_runs_on_the_host_runner_semantics(self):
        """write_training_demo -> MANIFEST.json -> the generic runner (kvmfe.run_manifest, the spec of demo/kvm_run) == direct run."""
        case = zoo.CASES['reg_mlp']()
        with tempfile.TemporaryDirectory() as tmp:
            m = demo.write_training_demo(tmp, case.module, loss=case.loss, data=case.data, optimizer=case.optimizer, steps=4, window=2,
                                         tiling='cpu', reuse='none')
            self.assertEqual([a['name'] for a in m['admissions']], ['train_w0', 'train_w1'])
            self.assertEqual(m['admissions'][1]['base_versions'], {'weights': 4})
            self.assertEqual(set(m['classes']), {'train'})
            seen = {}
            vm, order = run_manifest.run_manifest(tmp, after_each=lambda adm, v: seen.update(
                {adm['steps'][0] + s: v.buffers[f'loss.{s}'][0] for s in range(2)}))
            self.assertEqual(order, ['train_w0', 'train_w1'])
            self.assertEqual(vm.versions['weights'], 8)
            direct = build(zoo.CASES['reg_mlp'](), steps=4).run()
            want = [direct.matrix(f'loss.{s}')[0][0] for s in range(4)]
            np.testing.assert_allclose([seen[s] for s in range(4)], want, rtol=2e-6)
            self.assertLess(want[-1], want[0])

    def test_window_must_fit(self):
        case = zoo.CASES['reg_mlp']()
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(FrontendError, 'window 5'):
            demo.write_training_demo(tmp, case.module, loss=case.loss, data=case.data, steps=4, window=5, tiling='cpu')


class Refusals(unittest.TestCase):
    """Bad input is a clean FrontendError that names the problem (never a silent miscompile or a bare KeyError)."""

    def compile(self, module, loss, data, **kw):
        return compile_training(module, loss=loss, data=data, steps=1, tiling=CPU, **kw)

    def test_mse_target_shape(self):
        data = ArrayData(4, {'x': [[0.0] * 3] * 4, 'y': [[0.0] * 5] * 4})
        with self.assertRaisesRegex(FrontendError, 'MSE: target stream .* f32 4x2'):
            self.compile(MLP('m', [3, 2]), MSE(), data)

    def test_ce_target_needs_int_classes_in_range(self):
        ok = {'x': [[0.0] * 3] * 4}
        with self.assertRaisesRegex(FrontendError, 'CrossEntropy: target stream .* i32 column'):
            self.compile(MLP('m', [3, 2]), CrossEntropy(), ArrayData(4, {**ok, 'y': [[0.0]] * 4}))
        with self.assertRaisesRegex(FrontendError, r'class ids outside \[0, 2\)'):
            self.compile(MLP('m', [3, 2]), CrossEntropy(), ArrayData(4, {**ok, 'y': [0, 1, 2, 1]}, dtypes={'y': 'i32'}))

    def test_array_data_validation(self):
        with self.assertRaisesRegex(FrontendError, 'different sample counts'):
            ArrayData(2, {'x': [[1.0]] * 3, 'y': [[1.0]] * 2})
        with self.assertRaisesRegex(FrontendError, 'row 1 has 2 values'):
            ArrayData(2, {'x': [[1.0], [1.0, 2.0]], 'y': [[1.0]] * 2})
        with self.assertRaisesRegex(FrontendError, 'finite'):
            ArrayData(2, {'x': [[float('nan')]] * 2, 'y': [[1.0]] * 2})
        with self.assertRaisesRegex(FrontendError, 'integer'):
            ArrayData(2, {'x': [[1.0]] * 2, 'y': [[1.5]] * 2}, dtypes={'y': 'i32'})
        with self.assertRaisesRegex(FrontendError, 'missing from arrays'):
            ArrayData(2, {'x': [[1.0]] * 2})
        with self.assertRaisesRegex(FrontendError, 'batch must be'):
            ArrayData(0, {'x': [[1.0]], 'y': [[1.0]]})

    def test_generated_data_shape_is_checked(self):
        data = GeneratedData(lambda m: {'x': [[1.0, 2.0]] * 3, 'y': [[1.0]] * 3}, {'x': (4, 2, 'f32'), 'y': (4, 1, 'f32')})
        with self.assertRaisesRegex(FrontendError, r'make\(0\)\[.x.\] is 3x2, declared 4x2'):
            self.compile(MLP('m', [2, 1]), MSE(), data)

    def test_module_contract(self):
        data = ArrayData(2, {'x': [[1.0]] * 2, 'y': [[1.0]] * 2})
        with self.assertRaisesRegex(FrontendError, 'non-empty `.specs`'):
            self.compile(object(), MSE(), data)
        with self.assertRaisesRegex(FrontendError, 'must be a generic.Loss'):
            self.compile(MLP('m', [1, 1]), 'mse', data)
        with self.assertRaisesRegex(FrontendError, 'both loss= and data='):
            compile_training(MLP('m', [1, 1]), loss=MSE(), tiling=CPU)
        with self.assertRaisesRegex(FrontendError, 'belong to the generic form'):
            compile_training(ModelConfig('gpt_tiny'), steps=3, tiling=CPU)

    def test_block_arguments(self):
        with self.assertRaisesRegex(FrontendError, 'taps must be odd'):
            Conv1D('c', 2, 2, 4, 8)
        with self.assertRaisesRegex(FrontendError, 'unknown activation'):
            MLP('m', [2, 2, 2], act='swish')
        with self.assertRaisesRegex(FrontendError, 'dims must be'):
            MLP('m', [3])

    def test_shape_mismatch_inside_the_module_is_reported_by_the_builder(self):
        data = ArrayData(2, {'x': [[1.0] * 4] * 2, 'y': [[1.0]] * 2})
        with self.assertRaisesRegex(FrontendError, 'inner dims'):
            self.compile(MLP('m', [3, 1]), MSE(), data)


class Optimizers(unittest.TestCase):
    def test_adamw_with_schedule_and_clip_on_a_generic_module(self):
        from kvmfe.train import Schedule
        case = zoo.CASES['res_rms_mlp']()
        comp = compile_training(case.module, TrainConfig(steps=3, schedule=Schedule('cosine', warmup=1), clip=0.5), CPU, reuse='none',
                                loss=case.loss, data=case.data, optimizer=case.optimizer)
        mem = comp.run()
        self.assertTrue(all(np.isfinite(mem.matrix(f'loss.{s}')[0][0]) for s in range(3)))
        self.assertEqual(mem.matrix('opt_step')[0][0], 3)


if __name__ == '__main__':
    unittest.main()
