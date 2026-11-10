"""CPU-only tests of the structurally-zero gradient bound (structural_zero.py), on the real encoder_reg bundle.

Positive: the CPU golden itself and measured-size noise pass. Negative: a real attention-backward bug (a dropped term, a wrong scale), noise
above the bound, an inflated moment or a moved weight fail; a nonzero bias gradient is never classified as structurally zero.
Run: python3 -I -m unittest discover -s tests/gptdiff
"""
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'frontend'))
import structural_zero as sz  # noqa: E402


def build(case):
    d = Path(tempfile.mkdtemp(prefix='sz-'))
    subprocess.run([sys.executable, str(ROOT / 'tests/generic/gen_bundle.py'), str(d), case, '--tiling', 'a100', '--precise'],
                   check=True, stdout=subprocess.DEVNULL)
    spec = json.loads((d / 'workload.json').read_text())
    gold = json.loads((d / 'bundle/golden.json').read_text())
    return spec, gold['buffers'], gold['dtypes']


class StructuralZero(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from kvmfe.refvm import Memory
        cls.spec, cls.gold, cls.dtypes = build('encoder_reg')
        cls.sz = sz.analyse(cls.spec, cls.gold, cls.dtypes)
        mem = Memory(cls.spec, False, {})
        cls.w0 = {n: [x for r in range(b.rows) for x in b.row(r)] for n, b in mem.bufs.items()}

    def failures(self, buf):
        return [w for ok, w, _ in sz.check(self.sz, buf, self.w0) if not ok]

    def noisy(self, factor):
        """The golden with the structurally-zero gradients replaced by noise `factor` x their bound (alternating sign)."""
        b = copy.deepcopy(self.gold)
        for out, a in self.sz['zero'].items():
            mass = sz._mass(b[a['x']], self.sz['shapes'][a['x']], a['R'], a['c0'], a['c1'])
            for i, m in enumerate(mass):
                b[out][a['c0'] + i] = factor * (a['R'] + 8) * sz.EPS32 * m * (1 if i % 2 else -1)
        return b

    def test_exactly_the_key_bias_chain_is_found(self):
        self.assertEqual(sz.names(self.sz), {'act53', 'act115', 'm.enc.k.b', 'v.enc.k.b', 'w.enc.k.b'})
        self.assertEqual(sorted(self.sz['chain']), ['w.enc.k.b'])

    def test_golden_and_noise_below_bound_pass(self):
        self.assertEqual(self.failures(self.gold), [])
        self.assertEqual(self.failures(self.noisy(0.5)), [])

    def test_noise_above_bound_fails(self):
        self.assertEqual(len(self.failures(self.noisy(2.0))), 2)       # both steps' gradient buffers

    def test_dropped_term_bug_fails(self):
        """dK computed without one query row's contribution: the colsum of the damaged dK is O(|dK|), 1e2-1e3 x the bound."""
        b = copy.deepcopy(self.gold)
        for out, a in self.sz['zero'].items():
            cols = self.sz['shapes'][a['x']][1]
            for c in range(cols):
                b[a['x']][3 * cols + c] = 0.0
            b[out] = [sum(b[a['x']][r * cols + c] for r in range(a['R'])) for c in range(cols)]
        bad = self.failures(b)
        self.assertEqual(len(bad), 2)
        self.assertGreater(max(abs(x) for x in b['act53']), 1e-5)     # >> the 1e-8 noise, 1e1 x the 1.5e-6 bound

    def test_wrong_scale_bug_fails(self):
        b = copy.deepcopy(self.gold)
        a = self.sz['zero']['act53']
        cols = self.sz['shapes'][a['x']][1]
        for r in range(0, a['R'], 2):          # one sequence's dK scaled by 1.01: its rows no longer cancel the other's
            for c in range(cols):
                b[a['x']][r * cols + c] *= 1.01
        b['act53'] = [sum(b[a['x']][r * cols + c] for r in range(a['R'])) for c in range(cols)]
        self.assertTrue(any('act53' in w for w in self.failures(b)))

    def test_inflated_moment_or_moved_weight_fails(self):
        b = copy.deepcopy(self.gold)
        b['m.enc.k.b'][0] = 1e-3
        self.assertTrue(any('m.enc.k.b' in w for w in self.failures(b)))
        b = copy.deepcopy(self.gold)
        b['w.enc.k.b'][0] = 0.5
        self.assertTrue(any('w.enc.k.b' in w for w in self.failures(b)))

    def test_nonzero_bias_gradient_is_not_exempt(self):
        # act38 = colsum(gelu_bwd out), a real bias gradient: must stay under the relative bound
        self.assertNotIn('act38', sz.names(self.sz))
        self.assertNotIn('act38', self.sz['zero'])

    def test_ordinary_workload_has_no_structural_zero(self):
        spec, gold, dtypes = build('reg_mlp')
        self.assertEqual(sz.analyse(spec, gold, dtypes, reference={}), {})   # reference={} would KeyError if f64 were consulted


if __name__ == '__main__':
    unittest.main()
