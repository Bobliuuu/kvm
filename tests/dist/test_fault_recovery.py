"""Kill a rank mid-step in the multi-device simulator, restart from the last published version, compare bitwise.
python3 -I -m unittest discover -s tests/dist -p 'test_fault*.py' -v"""
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import ft            # noqa: E402
import sim           # noqa: E402
import verify        # noqa: E402
import workloads as wl   # noqa: E402
import partition as pt   # noqa: E402

ZERO = {'x': 'rows', 'W1': 'rows', 'W2': 'rows', 'gW1': 'rows', 'gW2': 'rows'}
STEPS = 4


def golden_steps(spec, names, steps):
    """The unpartitioned CPU VM, step after step, carrying `names` between steps."""
    import copy
    cur = copy.deepcopy(spec)
    g = None
    for _ in range(steps):
        g = verify.golden(cur)
        for b in cur['buffers']:
            if b['name'] in names:
                b['init'] = dict(kind='values', values=list(g[b['name']]))
    return g


def full(plan, state, name):
    lay = plan.layout[name]
    P = len(state)
    if lay == pt.ROWS:
        return [x for r in range(P) for x in state[r][name]]
    return state[0][name]


class Recovery(unittest.TestCase):
    def setUp(self):
        self.spec = wl.mlp()
        self.plan = pt.partition(self.spec, pt.Cluster.uniform(4, 2), {'mode': 'shard', 'layouts': ZERO, 'collective': 'ring', 'chunks': 2})
        self.names = ['W1', 'W2']
        # the step is deterministic given its input, so feed the weights back: x unchanged, W updated by the SGD tasks
        self.gold = golden_steps(self.spec, self.names, STEPS)

    def test_failure_free_equals_unpartitioned_vm_over_steps(self):
        state, log = ft.train(self.plan, self.names, STEPS)
        for n in self.names:
            self.assertEqual(full(self.plan, state, n), self.gold[n])

    def test_kill_each_rank_at_random_points_recovers_bitwise(self):
        clean, _ = ft.train(self.plan, self.names, STEPS)
        rng = random.Random(1104)
        kills_done = 0
        for trial in range(40):
            step = rng.randrange(STEPS)
            rank = rng.randrange(4)
            state, log = ft.train(self.plan, self.names, STEPS, seed=trial, kills={step: (rank, None)}, rng=rng)
            self.assertTrue(any(l[0] == 'killed' for l in log), 'the kill must actually fire')
            kills_done += 1
            self.assertEqual(state, clean, f'trial {trial}: rank {rank} killed in step {step}')
            for n in self.names:
                self.assertEqual(full(self.plan, state, n), self.gold[n])
        self.assertEqual(kills_done, 40)

    def test_repeated_failures_in_one_run(self):
        clean, _ = ft.train(self.plan, self.names, STEPS)
        rng = random.Random(7)
        state, log = ft.train(self.plan, self.names, STEPS, kills={0: (1, None), 1: (3, None), 3: (0, None)}, rng=rng)
        self.assertEqual(sum(l[0] == 'killed' for l in log), 3)
        self.assertEqual(state, clean)

    def test_rollback_is_to_the_last_version_not_further(self):
        rng = random.Random(3)
        _, log = ft.train(self.plan, self.names, STEPS, kills={2: (2, None)}, rng=rng)
        self.assertEqual([l for l in log if l[0] == 'done'][-1], ('done', STEPS))
        self.assertEqual([l[1] for l in log if l[0] == 'done'], [1, 2, 3, 4])      # steps 1-2 were not redone
        self.assertEqual([l[1] for l in log if l[0] == 'killed'], [2])

    def test_two_adjacent_losses_are_not_recoverable(self):
        with self.assertRaises(ft.CheckpointLost):
            ft.train(self.plan, self.names, STEPS, kills={1: (1, 5)}, lose_extra=(2,))

    def test_restore_from_a_wrong_version_is_detected_by_the_comparison(self):
        """The check can fail: restoring stale (version 0) weights after a step-2 failure must NOT match."""
        clean, _ = ft.train(self.plan, self.names, STEPS)
        orig = ft.Checkpoint.restore
        stale = ft.train(self.plan, self.names, 0)[0]
        def bad(self_):
            return {r: {k: list(v) for k, v in stale[r].items()} for r in range(self_.P)}
        ft.Checkpoint.restore = lambda s: bad(s) if s.version == 2 else orig(s)
        try:
            state, _ = ft.train(self.plan, self.names, STEPS, kills={2: (0, 3)})
        finally:
            ft.Checkpoint.restore = orig
        self.assertNotEqual(state, clean)


class RecoveryReplicated(unittest.TestCase):
    def test_data_parallel_fp_training_recovers_bitwise(self):
        spec = wl.regress(16)
        plan = pt.partition(spec, pt.Cluster.uniform(4, 2), {'mode': 'shard', 'layouts': {'x': 'rows', 'target': 'rows', 'pred': 'rows'}})
        clean, _ = ft.train(plan, ['w'], 3)
        rng = random.Random(5)
        for trial in range(12):
            state, log = ft.train(plan, ['w'], 3, seed=trial, kills={rng.randrange(3): (rng.randrange(4), None)}, rng=rng)
            self.assertEqual(state, clean)


if __name__ == '__main__':
    unittest.main()
