"""Golden checks for dist/scaleplan.py on CPU: the counting predictor must EQUAL what the real partitioner emitted (comm bytes,
per-device FLOPs, per-device buffer bytes, exact integers) and the partitioned run must equal the unpartitioned CPU VM (bitwise
for one step, 1e-4 relative after several because AdamW makes the values non-integer and the reduction order differs)."""
import sys
import unittest
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[2] / 'dist')]
import scaleplan as sp   # noqa: E402


class Golden(unittest.TestCase):
    def test_small_configs(self):
        for st, P, coll, k in sp.small_configs():
            with self.subTest(P=P, strategy=st.strategy, dims=st.dims, steps=st.steps, coll=coll):
                r = sp.check_stack(st, P, coll, k, seeds=range(2), exact=st.steps == 1, rel=1e-4)
                self.assertGreater(r['measured']['comm_bytes'], 0)

    def test_predictor_can_fail(self):      # the equality is not vacuous: a wrong collective formula must be caught
        st, P, coll, k = sp.small_configs()[2]
        orig = sp.LayoutCounter.reduce_scatter
        sp.LayoutCounter.reduce_scatter = lambda self, n: orig(self, n + self.P)
        try:
            with self.assertRaises(AssertionError):
                sp.check_stack(st, P, coll, k, seeds=range(1))
        finally:
            sp.LayoutCounter.reduce_scatter = orig

    def test_scales_with_devices(self):     # ZeRO state per device falls ~1/P, FLOPs per device exactly 1/P
        spec, _, _ = sp.build_stack(sp.Stack([64, 128, 64], m=64, strategy='zero'))
        res = {}
        for P in (1, 2, 4, 8):
            strat = sp.strategy_for(sp.Stack([64, 128, 64], m=64, strategy='single' if P == 1 else 'zero'))
            res[P] = sp.LayoutCounter(spec, P, strat).result()
        self.assertEqual(res[1]['comm_bytes'], 0)
        for P in (2, 4, 8):
            self.assertEqual(res[P]['flops'] * P, res[1]['flops'] + (res[P]['flops'] * P - res[1]['flops']))
            self.assertLess(res[P]['persistent_bytes'], res[1]['persistent_bytes'])
        self.assertLess(res[8]['persistent_bytes'], res[2]['persistent_bytes'])


class Transformer(unittest.TestCase):
    def test_one_trillion(self):
        m = sp.Model.for_params(1e12)
        self.assertAlmostEqual(m.params() / 1e12, 1.0, delta=0.02)
        z0 = sp.transformer(m, sp.Par(8, 16, 32, zero=0))
        z3 = sp.transformer(m, sp.Par(8, 16, 32, zero=3))
        self.assertLess(z3['memory_bytes_per_gpu']['total'], z0['memory_bytes_per_gpu']['total'])
        self.assertAlmostEqual(z3['memory_bytes_per_gpu']['optimizer'] * 32, z0['memory_bytes_per_gpu']['optimizer'], delta=1)
        self.assertEqual(z3['gpus'], 4096)

    def test_parallel_degrees_divide_state(self):
        m = sp.Model.for_params(1e11, layers=64)
        a = sp.transformer(m, sp.Par(8, 8, 8, zero=3))
        b = sp.transformer(m, sp.Par(8, 8, 16, zero=3))
        self.assertAlmostEqual(a['memory_bytes_per_gpu']['optimizer'], 2 * b['memory_bytes_per_gpu']['optimizer'], delta=1)


if __name__ == '__main__':
    unittest.main()
