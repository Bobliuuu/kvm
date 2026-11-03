"""Unit tests of dataflow, memory planning, lowering and config errors."""
import unittest

import _path  # noqa: F401
from kvmfe.dataflow import build_edges
from kvmfe.ir import Builder, FrontendError, Tensor
from kvmfe.lower import Tiling, lower_op
from kvmfe.nn import GPT, ModelConfig, PRESETS
from kvmfe.plan import plan_memory


class DataflowRules(unittest.TestCase):
    def edges(self, *accesses):
        return [sorted(p) for p in build_edges(list(accesses))]

    def test_raw_war_waw(self):
        A = lambda r0, r1: ('a', r0, r1, 0, 4)
        e = self.edges(([], [A(0, 4)]),         # 0 writes rows 0-4
                       ([A(0, 2)], []),          # 1 reads rows 0-2           RAW 0
                       ([A(2, 4)], []),          # 2 reads rows 2-4           RAW 0
                       ([], [A(0, 2)]),          # 3 rewrites rows 0-2        WAR 1, WAW 0
                       ([A(2, 4)], []),          # 4 reads 2-4: untouched by 3, still RAW 0 only
                       ([A(0, 4)], []))          # 5 reads all                RAW 3 and 0
        self.assertEqual(e, [[], [0], [0], [0, 1], [0], [0, 3]])

    def test_disjoint_regions_have_no_edge(self):
        self.assertEqual(self.edges(([], [('a', 0, 2, 0, 2)]), ([], [('a', 0, 2, 2, 4)]), ([], [('b', 0, 2, 0, 2)])), [[], [], []])

    def test_in_place_task_has_no_self_edge_and_history_does_not_grow(self):
        rw = ('a', 0, 1, 0, 1)
        e = self.edges(*[([rw], [rw])] * 50)
        self.assertEqual(max(len(p) for p in e), 1)           # each depends only on its predecessor (readers pruned by overwrite)


class Planner(unittest.TestCase):
    def ops(self, n):
        b = Builder()
        ts = [b.tensor(f't{i}', 4, 4) for i in range(n)]
        for i in range(n - 1):
            b.emit('add2', {'a': ts[i], 'b': ts[i]}, {'out': ts[i + 1]})
        return b.ops

    def test_chain_reuses_two_slots(self):
        plan = plan_memory(self.ops(10), 'shape', 0)
        self.assertLessEqual(plan.stats['slots'], 2)
        self.assertGreater(plan.stats['reuse_factor'], 4)

    def test_none_and_gap(self):
        self.assertEqual(plan_memory(self.ops(10), 'none').stats['slots'], 10)
        self.assertGreater(plan_memory(self.ops(10), 'shape', 4).stats['slots'], plan_memory(self.ops(10), 'shape', 0).stats['slots'])

    def test_pinned_and_shape_mismatch(self):
        a, b = Tensor('a', 2, 2, buf='w'), Tensor('b', 2, 3, buf='w')
        bld = Builder()
        bld.emit('x', {'a': a}, {'b': b})
        with self.assertRaises(FrontendError):
            plan_memory(bld.ops)
        with self.assertRaises(FrontendError):
            plan_memory([], 'bogus')


class Lowering(unittest.TestCase):
    def test_matmul_tiles_are_64x64_with_edges_only_to_needed_producers(self):
        b = Builder()
        x, w = b.tensor('x', 130, 40), b.tensor('w', 40, 70)
        b.linear(x, w)
        tasks = lower_op(b.ops[0], Tiling())
        self.assertEqual(len(tasks), 3 * 2)                   # ceil(130/64) x ceil(70/64)
        self.assertEqual({t.args['m0'] for t in tasks}, {0, 64, 128})
        self.assertTrue(all(t.dev == 'matmul_tile' for t in tasks))

    def test_shape_errors_are_named(self):
        b = Builder()
        with self.assertRaisesRegex(FrontendError, r'linear.*inner dims'):
            b.linear(b.tensor('x', 4, 5), b.tensor('w', 6, 7))
        with self.assertRaisesRegex(FrontendError, 'head dim'):
            ModelConfig(d_model=24, heads=2)
        with self.assertRaisesRegex(FrontendError, 'multiple of 32'):
            ModelConfig(seq=48)
        with self.assertRaisesRegex(FrontendError, 'preset'):
            Tiling.preset('h100')
        with self.assertRaises(FrontendError):
            Tiling(rows=0)

    def test_162m_config_is_a_pure_config_change(self):
        cfg = PRESETS['gpt_162m']
        self.assertAlmostEqual(cfg.params / 1e6, 162.3, delta=0.5)
        self.assertEqual((cfg.layers, cfg.d_model, cfg.heads, cfg.seq), (12, 768, 12, 1024))

    def test_config_loading(self):
        self.assertEqual(ModelConfig.load('{"name": "x", "layers": 3}').layers, 3)
        with self.assertRaisesRegex(FrontendError, 'bad model config'):
            ModelConfig.load({'nope': 1})


if __name__ == '__main__':
    unittest.main()
