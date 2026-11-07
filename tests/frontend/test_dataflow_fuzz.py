"""Fuzz: for random region programs the edges (including through synthetic joins) imply every RAW/WAR/WAW hazard."""
import random
import unittest

import _path  # noqa: F401
from kvmfe import dataflow


def hazards(acc):
    hit = lambda a, b: a[0] == b[0] and a[1] < b[2] and b[1] < a[2] and a[3] < b[4] and b[3] < a[4]
    out = []
    for i, (ri, wi) in enumerate(acc):
        out.append({j for j in range(i) if any(hit(w, r) for w in acc[j][1] for r in ri)
                    or any(hit(w, w2) for w in acc[j][1] for w2 in wi) or any(hit(r, w) for r in acc[j][0] for w in wi)})
    return out


def ancestors(seq, preds):
    anc = []
    for k, ps in enumerate(preds):
        a = set()
        for p in ps:
            a |= anc[p] | {p}
        anc.append(a)
    return {seq[k]: {seq[x] for x in anc[k] if seq[x] >= 0} for k in range(len(seq)) if seq[k] >= 0}


class Fuzz(unittest.TestCase):
    def check(self, join_min, trials=120, as_argument=False):
        old, dataflow.JOIN_MIN = dataflow.JOIN_MIN, (10 ** 9 if as_argument else join_min)
        try:
            for trial in range(trials):
                rng = random.Random(trial)

                def rect():
                    r0, c0 = rng.randint(0, 6), rng.randint(0, 3)
                    return (rng.choice('ab'), r0, rng.randint(r0 + 1, 8), c0, rng.randint(c0 + 1, 4))
                acc = [([rect() for _ in range(rng.randint(0, 2))], [rect() for _ in range(rng.randint(0, 2))])
                       for _ in range(rng.randint(40, 120))]
                seq, preds = dataflow.build_edges(acc, join_min=join_min if as_argument else None)
                self.assertTrue(all(p < k for k, ps in enumerate(preds) for p in ps), 'preds must precede')
                got = ancestors(seq, preds)
                for i, need in enumerate(hazards(acc)):
                    self.assertLessEqual(need, got[i], f'trial {trial} task {i}')
        finally:
            dataflow.JOIN_MIN = old

    def test_without_joins(self):
        self.check(10 ** 9)

    def test_with_joins(self):
        self.check(3)

    def test_join_min_argument_overrides_the_module_default(self):
        self.check(2, as_argument=True)

    def test_equal_sets_share_one_join(self):
        # 6 writers, then 4 readers of the same cell: the readers need the same 6-task set, so exactly one join serves them
        acc = [([], [('a', i, i + 1, 0, 1)]) for i in range(6)] + [([('a', 0, 6, 0, 1)], [('b', i, i + 1, 0, 1)]) for i in range(4)]
        seq, preds = dataflow.build_edges(acc, join_min=4)
        self.assertEqual(seq.count(-1), 1)
        self.assertEqual(sum(map(len, preds)), 6 + 6 + 3)      # first reader direct, the join over the 6 writers, 3 readers on it


if __name__ == '__main__':
    unittest.main()
