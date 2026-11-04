"""CPU-only tests of the AdamW error model in golden_check.py (run: python3 -I -m unittest discover -s tests/gptdiff).

The model must (1) accept a sign flip of a near-zero gradient, which makes post-Adam weights differ by 2*lr, (2) reject a wrong
update (e.g. a wrong learning rate, a wrong step number, a lost gradient scale) of a well-determined gradient element.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import golden_check as gc  # noqa: E402

OPT = gc.OPT


def image_band(w0, g, coef, tau):
    lo, _, _ = gc.adam_step1(w0, [x + tau / coef for x in g], coef, 0.0, **OPT)
    hi, _, _ = gc.adam_step1(w0, [x - tau / coef for x in g], coef, 0.0, **OPT)
    return lo, hi


class AdamModel(unittest.TestCase):
    def setUp(self):
        self.w0 = [0.02, -0.01, 0.0, 0.03]
        self.g = [0.3, -0.2, 4e-6, 0.1]          # element 2 is tiny relative to max|g| = 0.3
        self.tau = gc.TOL_TF32 * 0.3

    def test_first_update_is_sign_like(self):
        w1, _, _ = gc.adam_step1(self.w0, self.g, 1.0, 0.0, **OPT)
        for w, w0, g in zip(w1, self.w0, self.g):
            self.assertAlmostEqual(w0 - w, OPT['lr'] * (1 if g > 0 else -1), delta=1.01 * OPT['eps'] / abs(g) * OPT['lr'] + 1e-15)

    def test_flip_of_tiny_gradient_is_inside_band(self):
        lo, hi = image_band(self.w0, self.g, 1.0, self.tau)
        flipped = gc.adam_step1(self.w0, [0.3, -0.2, -3e-5, 0.1], 1.0, 0.0, **OPT)[0]
        self.assertTrue(all(l <= w <= h for l, w, h in zip(lo, flipped, hi)))
        self.assertGreater(abs(flipped[2] - gc.adam_step1(self.w0, self.g, 1.0, 0.0, **OPT)[0][2]), 1.9 * OPT['lr'])

    def test_flip_of_large_gradient_is_outside_band(self):
        lo, hi = image_band(self.w0, self.g, 1.0, self.tau)
        bad = gc.adam_step1(self.w0, [-0.3, -0.2, 4e-6, 0.1], 1.0, 0.0, **OPT)[0]
        self.assertFalse(lo[0] <= bad[0] <= hi[0])

    def test_wrong_lr_is_outside_band_for_a_determined_element(self):
        lo, hi = image_band(self.w0, self.g, 1.0, self.tau)
        w = self.w0[0] - 1.01 * OPT['lr']
        self.assertFalse(lo[0] - 4e-7 * 0.023 <= w <= hi[0] + 4e-7 * 0.023)

    def test_rel_l2(self):
        self.assertAlmostEqual(gc.rel_l2([1.0, 2.0], [1.0, 2.0]), 0.0)
        self.assertAlmostEqual(gc.rel_l2([0.0, 2.0], [3.0, 4.0]), (9 + 4) ** 0.5 / 5)


if __name__ == '__main__':
    unittest.main()
