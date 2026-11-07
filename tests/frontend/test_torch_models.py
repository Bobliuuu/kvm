"""Goal 2 against an independent reference: the frontend's program on the CPU float64 VM equals PyTorch autograd (examples/models/torch_models.py)
for every model family of the zoo: first-step loss(es) and the gradient of EVERY parameter tensor. Skipped without torch.

Tolerances (measured worst in examples/models/torch_report.py output; the bound is far above float64 noise and far below any real bug):
  f32 families  loss rel 1e-10, every gradient rel L2 1e-10  (both sides are float64: only summation order differs; measured worst 2e-15)
  bf16 emulation loss rel 1e-6, every gradient rel L2 1e-2   (one bf16 ulp is 3.9e-3 relative; measured worst 3.0e-3 on Llama's embedding, 1.1e-4 on the
                                                              MLP once gradients are rounded as the frontend does: where torch autograd sums unrounded
                                                              branch gradients before a cast, the frontend rounds per consumer)
"""
import sys
import unittest

import numpy as np

import _path  # noqa: F401
from kvmfe import models
from kvmfe.pipeline import build_model
from test_models import SMALL, SMALL_MOE, compiled, grad_of, losses

try:
    import torch  # noqa: F401
    sys.path.insert(0, str(_path.ROOT / 'examples' / 'models'))
    import torch_models as tm
except ImportError:                                            # pragma: no cover
    tm = None

CASES = {
    'gpt_tiny': ('gpt_tiny', 1e-10, 1e-10),
    'llama_gqa': ('llama_tiny', 1e-10, 1e-10),
    'llama_gqa_tied': ('llama_gqa_tied', 1e-10, 1e-10),
    'llama_small': (SMALL, 1e-10, 1e-10),
    'moe': ('moe_tiny', 1e-10, 1e-10),
    'moe_small_k2': (SMALL_MOE, 1e-10, 1e-10),
    'mlp': ('mlp_tiny', 1e-10, 1e-10),
    'vit': ('vit_tiny', 1e-10, 1e-10),
    'llama_bf16': (models.LlamaConfig('lb', vocab=32, d_model=32, heads=2, kv_heads=1, ffn=48, layers=1, seq=32, precision='bf16'), 1e-6, 1e-2),
    'mlp_bf16': (models.MLPConfig(name='mb', precision='bf16'), 1e-6, 1e-2),
}


def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


@unittest.skipIf(tm is None, 'torch not installed')
class TorchReference(unittest.TestCase):
    def check(self, name):
        cfg, tol_loss, tol_grad = CASES[name]
        if isinstance(cfg, str) and cfg == 'gpt_tiny':
            from kvmfe.nn import PRESETS
            cfg = PRESETS['gpt_tiny']
        elif isinstance(cfg, str):
            cfg = models.PRESETS[cfg]
        model = build_model(cfg)
        c = compiled(cfg)
        mem = c.run()
        want_loss, want_grad = tm.reference(model)
        got = {n: v[0] for n, v in losses(c, mem).items()}
        self.assertEqual(set(got), set(want_loss), name)
        for n, v in got.items():
            self.assertLess(abs(v - want_loss[n]), tol_loss * max(1.0, abs(want_loss[n])), f'{name} {n}: VM {v!r} torch {want_loss[n]!r}')
        worst = 0.0
        for n, g in want_grad.items():
            e = rel_l2(grad_of(c, mem, n), g) if np.linalg.norm(g) > 1e-12 else float(np.abs(grad_of(c, mem, n)).max())
            worst = max(worst, e)
            self.assertLess(e, tol_grad, f'{name} d loss / d {n}: relative L2 {e:.3g} against torch autograd')
        return worst

    def test_gpt(self):
        self.check('gpt_tiny')

    def test_llama_gqa(self):
        self.check('llama_gqa')

    def test_llama_gqa_tied_head(self):
        self.check('llama_gqa_tied')

    def test_llama_small(self):
        self.check('llama_small')

    def test_moe(self):
        self.check('moe')

    def test_moe_small_top2(self):
        self.check('moe_small_k2')

    def test_mlp(self):
        self.check('mlp')

    def test_vit(self):
        self.check('vit')

    def test_llama_bf16_emulation(self):
        self.check('llama_bf16')

    def test_mlp_bf16_emulation(self):
        self.check('mlp_bf16')

    def test_a_wrong_reference_is_caught(self):
        """The comparison has teeth: a perturbed torch gradient fails."""
        cfg = models.PRESETS['mlp_tiny']
        c = compiled(cfg)
        mem = c.run()
        _, want = tm.reference(build_model(cfg))
        n = next(iter(want))
        bad = want[n] * (1 + 1e-6)
        self.assertGreater(rel_l2(grad_of(c, mem, n), bad), 1e-9)


if __name__ == '__main__':
    unittest.main()
