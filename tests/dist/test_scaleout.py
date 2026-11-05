"""Scale-out simulator tests (CPU only).  python3 -I -m unittest discover -s tests/dist -p 'test_scaleout*.py' -v"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import verify    # noqa: E402,F401
import scaleout as so   # noqa: E402
import resilience as rs   # noqa: E402


def free_topo(**kw):
    d = dict(nvlink_gbs=1e9, nvlink_us=0, rack_gbs=1e9, rack_us=0, dc_gbs=1e9, dc_us=0, wan_gbps=1e9, wan_ms=0)
    d.update(kw)
    return so.Topology(8, 32, 4, 4, **d)


class Engine(unittest.TestCase):
    def test_pipeline_matches_the_closed_form(self):
        m = so.ModelCfg('dense', 16, 1024, 1024, 1, 4)
        for S, M in ((4, 16), (8, 64), (2, 8)):
            for infl in ('1f1b', 'gpipe'):
                r = so.run(m, so.Par(1, S, 1, M, M, inflight=infl), free_topo(), so.Hw(comm_sm_tax=0))
                self.assertAlmostEqual(r.metrics['over_pipeline_lb'], 1.0, delta=0.01 + 0.2 / M * S, msg=f'{S} {M} {infl}')

    def test_1f1b_bounds_activation_memory_gpipe_does_not(self):
        m = so.ModelCfg('dense', 16, 1024, 1024, 1, 4)
        a = so.run(m, so.Par(1, 4, 1, 32, 32, inflight='1f1b'), free_topo(), so.Hw(comm_sm_tax=0))
        b = so.run(m, so.Par(1, 4, 1, 32, 32, inflight='gpipe'), free_topo(), so.Hw(comm_sm_tax=0))
        self.assertLess(a.memory['peak_live_layers'], b.memory['peak_live_layers'])
        self.assertAlmostEqual(a.makespan_us, b.makespan_us, delta=0.01 * b.makespan_us)

    def test_cycle_is_reported(self):
        g = so.Graph()
        a = g.add(('s',), 1.0)
        b = g.add(('s',), 1.0, [a])
        g.deps[a].append(b)
        with self.assertRaisesRegex(RuntimeError, 'deadlock'):
            so.simulate(g)

    def test_streams_serialize_and_priorities_order(self):
        g = so.Graph()
        lo = g.add(('s',), 5.0, prio=1)
        hi = g.add(('s',), 5.0, prio=9)
        s, e = so.simulate(g)
        self.assertEqual((e[hi], e[lo]), (5.0, 10.0))

    def test_symmetry_reduction_equals_explicit_instantiation(self):
        m = so.ModelCfg('dense', 8, 2048, 2048, 4096, 4)
        topo = so.Topology(8, 4, 1, 1)
        par = so.Par(2, 2, 4, 4, 32, lpt=2, tp_chunks=1)
        g, _ = so.lower(m, par, topo, so.Hw())
        s0, e0 = so.simulate(g)
        g2 = so.replicate(g, par.tp, par.dp)
        s1, e1 = so.simulate(g2)
        self.assertEqual(len(g2) > len(g), True)
        self.assertAlmostEqual(max(e0), max(e1), delta=1e-6 * max(e0))

    def test_a_straggler_stalls_every_rank_through_the_collectives(self):
        m = so.ModelCfg('dense', 8, 2048, 2048, 4096, 4)
        topo = so.Topology(8, 4, 1, 1)
        par = so.Par(2, 2, 4, 4, 32, lpt=2)
        g, _ = so.lower(m, par, topo, so.Hw())
        base = max(so.simulate(so.replicate(g, 2, 4))[1])
        slow = max(so.simulate(so.replicate(g, 2, 4, slow={(1, 0): 1.5}))[1])
        self.assertGreater(slow, 1.15 * base)

    def test_topology_levels_follow_the_placement(self):
        t = so.Topology(8, 32, 4, 4)
        par = so.Par(8, 16, 32, 64, 4096)
        self.assertEqual(t.levels([par.rank(0, 0, x) for x in range(8)]), [(so.NVLINK, 8)])
        self.assertEqual(t.levels([par.rank(0, d, 0) for d in range(32)]), [(so.RACK, 32)])
        self.assertEqual(t.tier(par.rank(3, 0, 0), par.rank(4, 0, 0)), so.WAN)       # 4 stages per datacenter
        self.assertEqual(t.tier(par.rank(0, 0, 0), par.rank(1, 0, 0)), so.DC)
        bad = so.Par(8, 16, 32, 64, 4096, order=('dp', 'pp', 'tp'))
        self.assertIn(so.WAN, [x for x, _ in t.levels([bad.rank(0, d, 0) for d in range(32)])])

    def test_collective_formulas(self):
        t = so.Topology()
        lv = [(so.RACK, 8)]
        B = 8e9
        ag = so.coll_time_us(t, lv, 'ag', B)
        self.assertAlmostEqual(ag, 7 / 8 * B / (50 * 1e3) + 7 * 5.0, places=3)
        self.assertAlmostEqual(so.coll_time_us(t, lv, 'ar', B), 2 * ag, places=3)
        self.assertEqual(so.coll_time_us(t, [], 'ar', B), 0.0)


class Cost(unittest.TestCase):
    def test_default_dense_1t_numbers_vs_scaling_arithmetic(self):
        m = so.dense_1t()
        self.assertAlmostEqual(m.params() / 1e12, 1.01, delta=0.01)
        r = so.run(m, so.Par(8, 16, 32, 64, 4096, lpt=4), so.make_topology(4096, 400, 20), so.Hw())
        mem = r.memory['parts_gb']
        self.assertAlmostEqual(mem['state'], 16 * m.params() / 4096 / 1e9 * 1.0, delta=0.6)    # 16 B/param ZeRO-3 (+ embeddings on the end stages)
        self.assertTrue(r.memory['fits'])

    def test_wan_bandwidth_and_latency_are_monotone(self):
        m = so.dense_1t()
        par = so.Par(8, 16, 32, 64, 4096, lpt=4)
        t = lambda bw, ms: so.run(m, par, so.make_topology(4096, bw, ms), so.Hw()).makespan_us
        self.assertGreater(t(100, 20), t(400, 20))
        self.assertGreater(t(400, 20), t(1600, 20) - 1)
        self.assertGreater(t(400, 100), t(400, 5))

    def test_putting_dp_across_the_wan_is_much_worse_than_pp(self):
        m = so.dense_1t()
        t = so.make_topology(4096, 400, 20)
        pp_wan = so.run(m, so.Par(8, 16, 32, 64, 4096, lpt=4, window_extra=8), t, so.Hw()).makespan_us
        dp_wan = so.run(m, so.Par(8, 16, 32, 64, 4096, lpt=4, window_extra=8, order=('dp', 'pp', 'tp')), t, so.Hw()).makespan_us
        self.assertGreater(dp_wan, 1.3 * pp_wan)

    def test_message_volumes_match_the_real_partitioner(self):
        """ring all-gather of a ROWS-sharded buffer: the partitioner's byte count == the closed form the cost model uses."""
        import partition as pt
        import workloads as wl
        P, d, h = 8, 8, 16
        plan = pt.partition(wl.mlp(m=8, d=d, h=h, o=8), pt.Cluster.uniform(P, 2),
                            {'mode': 'shard', 'layouts': {'W1': 'rows', 'x': 'rows'}, 'collective': 'ring'})
        w1_bytes = d * h * 4
        self.assertGreaterEqual(plan.comm_bytes, w1_bytes * (P - 1))                   # every device receives (P-1)/P of W1: total P*(P-1)/P*|W1|
        ag_only = pt.partition({'buffers': [dict(name='W1', rows=d, cols=h, init=dict(kind='fill', value=1)), dict(name='o', rows=d, cols=h)],
                                'graphs': [dict(name='g', tasks=[dict(id='c', op='copy', args=dict(src='W1', dst='o', n=d * h))])]},
                               pt.Cluster.uniform(P, 2), {'mode': 'shard', 'layouts': {'W1': 'rows', 'o': 'repl'}, 'collective': 'ring'})
        self.assertEqual(ag_only.comm_bytes, w1_bytes * (P - 1))


class Resilience(unittest.TestCase):
    def test_stealing_never_hurts_and_helps_a_straggler(self):
        R, B = 16, 64
        speed = [1.0] * R
        speed[3] = 0.5
        st, _ = rs.step_time(R, B, 100.0, speed, steal=False)
        dy, _ = rs.step_time(R, B, 100.0, speed, steal=True, claim_us=2.0, fetch_us=5.0)
        self.assertLess(dy, 0.8 * st)
        even, _ = rs.step_time(R, B, 100.0, None, steal=True, claim_us=2.0, fetch_us=5.0)
        base, _ = rs.step_time(R, B, 100.0, None, steal=False)
        self.assertLessEqual(even, base + 1e-9)          # no stragglers: nothing is stolen, nothing is lost

    def test_fixed_order_reduction_is_bitwise_independent_of_the_assignment(self):
        R, B = 8, 32
        ref = rs.reduce_fixed(B)
        naive = set()
        for seed in range(12):
            speed = [1.0] * R
            speed[seed % R] = 0.4
            _, owner = rs.step_time(R, B, 100.0, speed, steal=True, seed=seed, jitter=0.5)
            naive.add(tuple(rs.reduce_by_completion(B, owner, R)))
            # fixed order uses only bundle ids: identical by construction -- and the owner map is a complete assignment
            self.assertEqual(sorted(owner), list(range(B)))
            self.assertEqual(rs.reduce_fixed(B), ref)
        self.assertGreater(len(naive), 1, 'summing in completion order must change bits (otherwise the rule is unnecessary)')

    def test_goodput_model_sanity(self):
        g = rs.goodput(4096, 90.0, 50000.0, 30.0, 300.0)
        self.assertTrue(0.9 < g['goodput'] < 1.0)
        self.assertLess(rs.goodput(16384, 90.0, 50000.0, 30.0, 300.0)['goodput'], g['goodput'])


if __name__ == '__main__':
    unittest.main()
