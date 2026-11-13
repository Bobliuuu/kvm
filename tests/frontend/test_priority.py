"""Task priorities from each task's remaining path inside its step (emit.task_levels, analyze.step_ranks)."""
import re
import unittest
from collections import defaultdict

import _path  # noqa: F401
from _path import ROOT
from kvmfe.emit import PRIORITY_MAX, GraphPolicy, task_levels, to_workload
from kvmfe.ir import FrontendError
from kvmfe.lower import Tiling
from kvmfe.nn import ModelConfig
from kvmfe.pipeline import compile_training
from kvmfe.train import TrainConfig

TINY = ModelConfig(name='rank_tiny', vocab=64, d_model=32, layers=2, heads=2, seq=32, batch=2)


def lowered(steps):
    return compile_training(TINY, TrainConfig(steps=steps), Tiling.preset('cpu'), reuse='step').low


class StepLevels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.low = lowered(3)
        cls.levels = task_levels(cls.low, {'train': GraphPolicy()})

    def test_a_task_never_ranks_below_a_successor_in_its_step(self):
        # its remaining path is its own cost plus at least the successor's, and the levels floor that path
        low, lv = self.low, self.levels
        for i, preds in enumerate(low.preds):
            for j in preds:
                if low.tasks[j].op.step == low.tasks[i].op.step:
                    self.assertGreaterEqual(lv[j], lv[i], f'{low.tasks[j].name} -> {low.tasks[i].name}')

    def test_every_step_spans_all_levels(self):
        # cut at the step's end, the ranks of every step reach the top level, so steps do not crowd into a few levels however many there are
        by_step = defaultdict(set)
        for t, lv in zip(self.low.tasks, self.levels):
            by_step[t.op.step].add(lv)
        self.assertEqual(sorted(by_step), [0, 1, 2])
        for step, used in by_step.items():
            self.assertEqual((min(used), max(used)), (0, PRIORITY_MAX), f'step {step}')
            self.assertGreater(len(used), PRIORITY_MAX // 2, f'step {step} uses only {sorted(used)}')

    def test_a_longer_run_keeps_the_same_levels_a_step(self):
        low = lowered(6)
        lv = task_levels(low, {'train': GraphPolicy()})
        by_step = defaultdict(list)
        for t, x in zip(low.tasks, lv):
            by_step[t.op.step].append(x)
        self.assertEqual(sorted(by_step[1]), sorted(by_step[4]))        # the middle steps' graphs are alike, so their levels are too

    def test_ranks_change_only_the_priority_field(self):
        ranked = to_workload(self.low, {'train': GraphPolicy()})
        flat = to_workload(self.low, {'train': GraphPolicy(rank_levels=0)})
        strip = lambda w: [[{k: v for k, v in t.items() if k != 'priority'} for t in g['tasks']] for g in w['graphs']]
        self.assertEqual(strip(ranked), strip(flat))
        self.assertEqual(ranked['buffers'], flat['buffers'])
        self.assertFalse(any('priority' in t for g in flat['graphs'] for t in g['tasks']))
        got = [t.get('priority', 0) for g in ranked['graphs'] for t in g['tasks']]
        self.assertEqual(got, self.levels)


class Policies(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.low = lowered(1)

    def test_a_graph_without_ranks_keeps_its_priority_on_every_task(self):
        w = to_workload(self.low, {'train': GraphPolicy(priority=10, rank_levels=0)})
        self.assertEqual({t.get('priority') for g in w['graphs'] for t in g['tasks']}, {10})

    def test_ranks_sit_above_the_graphs_priority(self):
        w = to_workload(self.low, {'train': GraphPolicy(priority=21, rank_levels=10)})
        got = {t.get('priority') for g in w['graphs'] for t in g['tasks']}
        self.assertEqual((min(got), max(got)), (21, 31))

    def test_levels_past_the_top_bucket_are_refused(self):
        for pol in (GraphPolicy(priority=5), GraphPolicy(rank_levels=PRIORITY_MAX + 1), GraphPolicy(rank_levels=-1),
                    GraphPolicy(priority=-1, rank_levels=3)):
            with self.subTest(priority=pol.priority, rank_levels=pol.rank_levels), self.assertRaisesRegex(FrontendError, 'rank_levels'):
                to_workload(self.low, {'train': pol})

    def test_priority_max_is_the_executors_top_bucket(self):
        src = (ROOT / 'include' / 'kvm_exec.cuh').read_text()
        m = re.search(r'#define\s+KVM_NBUCKETS\s+(\d+)', src)
        self.assertIsNotNone(m, 'KVM_NBUCKETS not found in include/kvm_exec.cuh')
        self.assertEqual(PRIORITY_MAX, int(m.group(1)) - 1)


if __name__ == '__main__':
    unittest.main()
