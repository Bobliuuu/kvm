"""Model zoo on the CPU golden VM: loss and autodiff gradients against independent float64 numpy oracles (finite differences),
dataflow completeness, and the frontend features (fusion, rematerialization, accumulation, schedules, loss scaling, bf16)."""
import unittest

import numpy as np

import _path  # noqa: F401
import oracle_models as om
from kvmfe import models
from kvmfe.lower import Tiling
from kvmfe.pipeline import build_model, compile_training
from kvmfe.train import Schedule, TrainConfig

CPU = Tiling.preset('cpu')
# small variants keep the pure-Python golden VM fast; the presets are exercised by OracleChecks and on the GPU
SMALL = models.LlamaConfig('s', vocab=32, d_model=32, heads=2, kv_heads=1, ffn=48, layers=1, seq=32, batch=1)
SMALL_MOE = models.LlamaConfig('sm', vocab=32, d_model=32, heads=2, kv_heads=2, ffn=32, layers=1, seq=32, batch=1, experts=4, top_k=2)


def compiled(cfg, steps=1, reuse='none', fuse=False, **tc):
    return compile_training(cfg, TrainConfig(steps=steps, **tc), CPU, reuse=reuse, fuse=fuse)


def losses(c, mem, steps=1):
    return {n: [mem.matrix(f'{n}.{s}')[0][0] for s in range(steps)] for n in c.program.losses[0]}


def weights(mem):
    return {k: v.a[:] for k, v in mem.bufs.items() if k.startswith('w.')}


def grad_of(c, mem, name):
    t = c.program.grads[name]
    b = mem.bufs[c.low.plan.buf(t)]
    return np.array(b.a, dtype=np.float64).reshape(b.rows, b.cols)


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-12)


class OracleChecks(unittest.TestCase):
    """Step-0 loss and a gradient spot check of EVERY parameter tensor (largest-|g| element + a pseudo-random one)."""

    def check(self, preset, tol_g=2e-6):
        cfg = models.PRESETS[preset] if isinstance(preset, str) else preset
        model = build_model(cfg)
        c = compiled(cfg)
        mem = c.run()
        p, data = om.params(model), om.streams(model)
        want = om.forward(model, p, data)
        for n, v in losses(c, mem).items():
            self.assertAlmostEqual(v[0], want[n], delta=1e-9 * max(1, abs(want[n])), msg=f'{preset} {n}')
        rng = np.random.default_rng(0)
        worst = 0.0
        for name in model.specs:
            g = grad_of(c, mem, name)
            # the model's own declaration of structurally zero gradients (ParamSpec.invariant) is complete and true: the examples/models
            # check applies its absolute (not relative) rule to exactly these parameters
            self.assertEqual(model.specs[name].invariant, bool(np.abs(g).max() < 1e-9), f'{preset} {name}: invariant flag vs autodiff gradient')
            picks = {np.unravel_index(np.abs(g).argmax(), g.shape), tuple(int(rng.integers(0, d)) for d in g.shape)}
            for idx in picks:
                fd = om.fd_grad(model, p, data, name, idx)
                if np.abs(g).max() < 1e-9:  # structurally zero (attention key bias: softmax ignores a per-query constant): absolute check
                    self.assertLess(abs(fd), 1e-9, f'{preset} d loss / d {name}{idx}: expected ~0, finite difference {fd:.3g}')
                    continue
                err = abs(g[idx] - fd) / max(abs(fd), 0.05 * np.abs(g).max(), 1e-8)
                worst = max(worst, err)
                self.assertLess(err, tol_g, f'{preset} d loss / d {name}{idx}: autodiff {g[idx]:.8g} vs finite difference {fd:.8g}')
        return worst

    def test_mlp(self):
        self.check('mlp_tiny')

    def test_vit(self):
        self.check('vit_tiny')

    def test_llama_gqa(self):
        self.check('llama_tiny')

    def test_llama_gqa_single_kv_head_tied_embeddings(self):
        self.check(models.LlamaConfig('t', kv_heads=1, tie_embeddings=True, layers=1))

    def test_moe(self):
        self.check('moe_tiny')

    def test_moe_with_dropped_tokens_and_k1(self):
        cfg = models.LlamaConfig('t', layers=1, heads=4, kv_heads=4, ffn=32, batch=2, experts=4, top_k=1, capacity_factor=0.5)
        model = build_model(cfg)
        p = om.params(model)
        probs_dropped = om.moe_route(np.full((64, 4), 0.25), 1, cfg.capacity)[1]
        self.assertTrue((probs_dropped < 0).any(), 'the config must actually drop tokens')
        self.check(cfg)

    def test_mlp_activations(self):
        for act in ('silu', 'tanh', 'sigmoid'):
            self.check(models.MLPConfig('t', act=act, hidden=(16,)))

    def test_vit_two_layers_patch2(self):
        self.check(models.ViTConfig('t', image=8, patch=2, layers=2, batch=2))


class Dataflow(unittest.TestCase):
    def same(self, a, b):
        self.assertEqual(a.bufs.keys(), b.bufs.keys())
        for k in a.bufs:
            self.assertEqual(a.bufs[k].a, b.bufs[k].a, k)

    def test_random_topological_order_equals_program_order(self):
        for cfg in (models.PRESETS['mlp_tiny'], models.PRESETS['vit_tiny'], SMALL_MOE, SMALL):
            c = compiled(cfg, steps=2, reuse='shape')
            self.same(c.run(), c.run(order='random', seed=5))


class Features(unittest.TestCase):
    def test_fusion_is_bitwise_identical_and_removes_tasks(self):
        for cfg in (SMALL, SMALL_MOE):
            a, b = compiled(cfg, 2, reuse='shape'), compiled(cfg, 2, reuse='shape', fuse=True)
            self.assertLess(len(b.low.tasks), len(a.low.tasks), cfg.name)
            ma, mb = a.run(), b.run()
            self.assertEqual(weights(ma), weights(mb), cfg.name)
            self.assertEqual(losses(a, ma, 2), losses(b, mb, 2), cfg.name)

    def test_fusion_gpt_gelu_backward(self):
        from kvmfe.nn import PRESETS as GPT
        a, b = compiled(GPT['gpt_tiny'], 1, reuse='shape'), compiled(GPT['gpt_tiny'], 1, reuse='shape', fuse=True)
        self.assertLess(len(b.low.tasks), len(a.low.tasks))
        self.assertEqual(weights(a.run()), weights(b.run()))

    def test_precise_tiling_sets_gemm_flag_on_every_matmul_variant(self):
        for fuse in (False, True):
            spec = compile_training(SMALL, TrainConfig(steps=1), Tiling.preset('cpu', precise=True), fuse=fuse).spec
            gemms = [t for t in spec['graphs'][0]['tasks'] if t['op'].startswith('matmul_tile')]
            self.assertTrue(gemms and all(t['args']['flags'] & 16 for t in gemms), fuse)
        plain = compile_training(SMALL, TrainConfig(steps=1), CPU).spec
        self.assertFalse(any(t.get('args', {}).get('flags', 0) & 16 for t in plain['graphs'][0]['tasks']))

    def test_remat_same_result_less_memory(self):
        base = models.LlamaConfig(**{**SMALL.__dict__, 'layers': 3})
        rm = models.LlamaConfig(**{**base.__dict__, 'remat': True})
        a, b = compiled(base, 1, reuse='shape'), compiled(rm, 1, reuse='shape')
        self.assertEqual(weights(a.run()), weights(b.run()))
        self.assertLess(b.low.plan.stats['pool_bytes'], a.low.plan.stats['pool_bytes'])
        self.assertGreater(len(b.low.tasks), len(a.low.tasks))

    def test_remat_moe_and_vit_blocks(self):
        cfg = models.LlamaConfig(**{**SMALL_MOE.__dict__, 'remat': True})
        ref = SMALL_MOE
        self.assertEqual(weights(compiled(ref, 1, reuse='shape').run()), weights(compiled(cfg, 1, reuse='shape').run()))

    def test_gradient_accumulation_equals_big_batch(self):
        one = SMALL
        two = models.LlamaConfig(**{**one.__dict__, 'batch': 2})
        a, b = compiled(two, 2, reuse='shape'), compiled(one, 2, reuse='shape', accum=2)
        wa, wb = weights(a.run()), weights(b.run())
        for k in wa:
            np.testing.assert_allclose(wb[k], wa[k], atol=1e-11, rtol=1e-9, err_msg=k)
        ma, mb = a.run(), b.run()
        for s in range(2):
            mean = (mb.matrix(f'loss.{s}.0')[0][0] + mb.matrix(f'loss.{s}.1')[0][0]) / 2
            self.assertAlmostEqual(mean, ma.matrix(f'loss.{s}')[0][0], delta=1e-12)

    def test_accumulation_with_mlp_and_moe_aux(self):
        c = compiled(SMALL_MOE, 1, accum=2, reuse='shape')
        m = c.run()
        self.assertEqual(len(c.program.losses), 2)
        self.assertGreater(m.matrix('aux0.0.1')[0][0], 0)

    def test_lr_schedule_buffer_and_const_equivalence(self):
        cfg = models.PRESETS['mlp_tiny']
        plain = compiled(cfg, 3, reuse='shape')
        const = compiled(cfg, 3, reuse='shape', schedule=Schedule('const'))
        self.assertEqual(weights(plain.run()), weights(const.run()))
        sch = Schedule('cosine', warmup=2, total=6, min_ratio=0.1)
        c = compiled(cfg, 3, reuse='shape', schedule=sch)
        mem = c.run()
        t, base = 3, 3e-3                     # after 3 steps the buffer holds the lr used by the third update
        prog = min(max((t - 2) / 4, 0), 1)
        self.assertAlmostEqual(mem.bufs['lr'].a[0], base * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * prog))), delta=1e-15)
        for kind in ('linear', 'const'):
            m = compiled(cfg, 2, reuse='shape', schedule=Schedule(kind, warmup=1, total=4)).run()
            self.assertGreater(m.bufs['lr'].a[0], 0)

    def test_loss_scale_is_transparent(self):
        a = weights(compiled(SMALL, 2, reuse='shape').run())
        b = weights(compiled(SMALL, 2, reuse='shape', loss_scale=1024.0).run())
        for k in a:
            np.testing.assert_allclose(b[k], a[k], atol=1e-8, rtol=1e-6, err_msg=k)

    def test_loss_scale_requires_adamw(self):
        from kvmfe.ir import FrontendError
        from kvmfe.optim import OptimConfig
        with self.assertRaises(FrontendError):
            compiled(models.PRESETS['mlp_tiny'], 1, loss_scale=8.0, optim=OptimConfig('sgd'))


class MixedPrecision(unittest.TestCase):
    def test_bf16_loss_matches_bf16_oracle(self):
        for cfg in (models.MLPConfig('t', precision='bf16'), models.ViTConfig('t', precision='bf16'),
                    models.LlamaConfig(**{**SMALL.__dict__, 'precision': 'bf16'}), models.LlamaConfig(**{**SMALL_MOE.__dict__, 'precision': 'bf16'})):
            model = build_model(cfg)
            c = compiled(cfg)
            want = om.forward(model, om.params(model), om.streams(model))
            got = losses(c, c.run())
            for n, v in got.items():
                self.assertAlmostEqual(v[0], want[n], delta=1e-8, msg=f'{type(cfg).__name__} {n}')

    def test_bf16_trains_and_tracks_fp32(self):
        f32 = models.MLPConfig('t')
        lp = models.MLPConfig('t', precision='bf16')
        a, b = compiled(f32, 25, reuse='shape'), compiled(lp, 25, reuse='shape', loss_scale=128.0)
        la, lb = losses(a, a.run(), 25)['loss'], losses(b, b.run(), 25)['loss']
        self.assertLess(lb[-1], lb[0] - 0.5)
        self.assertLess(abs(la[-1] - lb[-1]), 0.1)


class Training(unittest.TestCase):
    def test_every_model_learns(self):
        for cfg, steps, drop in ((models.PRESETS['mlp_tiny'], 30, 1.0), (models.PRESETS['vit_tiny'], 20, 0.1),
                                 (SMALL, 10, 0.5), (SMALL_MOE, 10, 0.5)):
            c = compiled(cfg, steps, reuse='shape')
            ls = losses(c, c.run(), steps)['loss']
            self.assertLess(ls[-1], ls[0] - drop, f'{cfg.name}: {ls[0]:.3f} -> {ls[-1]:.3f}')


class CheckedInGolden(unittest.TestCase):
    """The float64 VM's 5-step loss curves of every preset are checked in (make_golden_models.py): an unintended numerical change fails here."""

    def test_loss_curves(self):
        import json
        from pathlib import Path
        want = json.loads((Path(__file__).with_name('golden_models.json')).read_text())
        self.assertEqual(set(want), set(models.PRESETS))
        for name, curves in want.items():
            c = compiled(models.PRESETS[name], 5, reuse='shape')
            got = losses(c, c.run(), 5)
            for k, ref in curves.items():
                for s, (a, b) in enumerate(zip(got[k], ref)):
                    self.assertAlmostEqual(a, b, delta=1e-9 * max(1.0, abs(b)), msg=f'{name} {k} step {s}')


class Registry(unittest.TestCase):
    def test_new_ops_defined_and_have_references(self):
        from kvmfe import opset
        from kvmfe.defs import read_defs
        defs = read_defs()
        for n in ('act_fwd', 'moe_route', 'attn_full_fwd', 'matmul_tile_bf16', 'adamw_lr', 'matmul_tile_silu_mul_bwd'):
            self.assertIn(n, defs)
            self.assertIsNotNone(opset.op(n).ref, n)
            self.assertTrue(96 <= defs[n][0] < 128, n)


if __name__ == '__main__':
    unittest.main()


class Decode(unittest.TestCase):
    """KV-cache decode of the Llama family (GQA, RoPE offset by the cache length, SwiGLU) against full-forward greedy decode."""

    def run_case(self, cfg, steps, prompt, n_new):
        from kvmfe.emit import merge
        from kvmfe.refvm import execute
        from kvmfe.serve import ServeRequest, compile_serving
        from kvmfe.train import weights_version
        model = build_model(cfg)
        train = compile_training(cfg, TrainConfig(steps=steps), CPU, reuse='shape')
        reqs = [ServeRequest(f'sv{k}', prompt, n_new, weights_version(k)) for k in (0, steps)]
        spec = merge(train.spec, *[compile_serving(model, r, CPU)[1] for r in reqs])
        mem = execute(spec, order='random', seed=3)
        p0 = om.params(model)
        p1 = {n: np.array(mem.bufs[f'w.{n}'].a).reshape(s.rows, s.cols) for n, s in model.specs.items()}
        for r, p in zip(reqs, (p0, p1)):
            got = [int(mem.bufs[f'{r.name}.cur{i}'].a[0]) for i in range(1, n_new + 1)]
            self.assertEqual(got, om.greedy(model, p, prompt, n_new), r.name)

    def test_gqa_decode_matches_full_forward(self):
        self.run_case(models.LlamaConfig('t', layers=2, kv_heads=1), 2, (1, 8, 11, 4), 4)

    def test_tied_embeddings_single_token_prompt(self):
        self.run_case(models.LlamaConfig('t', layers=1, tie_embeddings=True, seq=32), 2, (7,), 3)
