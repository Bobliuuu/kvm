"""The compiler must stay near-linear on frontend-sized graphs (the 162M GPT tiling is ~315k tasks, ~6.8M edges)."""
import gc
import random
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime'))
import compile_workload as cw
from test_compiler import all_ops


def synthetic(n, degree, seed=1, window=500):
    """n matmul_tile-like tasks, each depending on `degree` earlier ones, a publisher every 100 tasks, a reader every 7th."""
    g = random.Random(seed)
    bufs = [{'name': b, 'rows': 128, 'cols': 128} for b in ('a', 'b', 'o', 'bias')]
    tasks, last_pub = [], None
    for i in range(n):
        lo = max(0, i - window)
        after = set(g.sample(range(lo, i), min(degree, i - lo)))
        t = {'id': f't{i}', 'op': 'matmul_tile', 'args': dict(a='a', b='b', out='o', bias='bias', res=None, pre=None, m0=0, n0=0,
                                                              M=128, N=128, K=128, flags=0, alpha=1.0)}
        if i % 100 == 0:
            t['publishes'] = 'w'
            if last_pub is not None:
                after.add(last_pub)
            last_pub = i
        if i % 7 == 0:
            t['reads'] = [{'var': 'w', 'policy': 'at_least', 'version': i // 100}]
        if after:
            t['after'] = [f't{j}' for j in sorted(after)]
        tasks.append(t)
    return {'variables': ['w'], 'buffers': bufs, 'graphs': [{'name': 'g', 'tasks': tasks}]}


class ScaleTests(unittest.TestCase):
    def test_large_graph_compiles_correctly_and_linearly(self):
        ops = all_ops()
        times = {}
        for n in (8000, 32000):
            spec = synthetic(n, 20)
            t = time.perf_counter()
            c = cw.compile_spec(spec, ops=ops)
            times[n] = time.perf_counter() - t
            self.assertEqual(c.metadata['task_count'], n)
            self.assertGreater(c.metadata['edge_count'], 18 * n)
            self.assertEqual(len(c.successors), 4 * c.metadata['edge_count'])
        # 4x the work must cost well under 8x the time (quadratic behaviour would be ~16x); generous for noisy machines
        self.assertLess(times[32000], 8 * max(times[8000], 0.05), times)
        self.assertLess(times[32000], 30.0, times)

    def test_gc_state_is_restored(self):
        for start in (True, False):
            (gc.enable if start else gc.disable)()
            cw.compile_spec({'graphs': [{'name': 'g', 'tasks': [{'id': 'a', 'op': 'noop'}]}]})
            self.assertEqual(gc.isenabled(), start)
        gc.enable()

    def test_successor_lists_are_sorted_and_duplicate_free(self):
        c = cw.compile_spec(synthetic(3000, 10), ops=all_ops())
        for i in range(c.metadata['task_count']):
            s = list(c.successors_of(i))
            self.assertEqual(s, sorted(set(s)))


if __name__ == '__main__':
    unittest.main()
