"""Partitioner + multi-device CPU simulator tests.   python3 -I -m unittest discover -s tests/dist -v

Every strategy is checked the same way: partition, run ALL shards in the simulator under several random interleavings (including
"data landed, version not yet visible"), reconstruct the logical buffers and compare with the unpartitioned CPU VM. Integer-valued
data must match bitwise; the fp workload within a tolerance.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import verify    # noqa: E402  (also sets up sys.path for runtime/ and dist/)
import sim       # noqa: E402
import workloads as wl   # noqa: E402
import partition as pt   # noqa: E402
import commops           # noqa: E402

TP = {'W1': 'cols', 'W2': 'rows', 'gW1': 'cols', 'gW2': 'rows'}
ZERO = {'x': 'rows', 'W1': 'rows', 'W2': 'rows', 'gW1': 'rows', 'gW2': 'rows'}


def shard(layouts, **kw):
    return dict({'mode': 'shard', 'layouts': layouts}, **kw)


class Strategies(unittest.TestCase):
    """The MLP forward + SGD step under each parallelism strategy (all just different initial layouts)."""

    def run_case(self, P, nodes, layouts, **kw):
        plan, rep = verify.check(wl.mlp(), pt.Cluster.uniform(P, nodes), shard(layouts, **kw), seeds=range(4))
        return plan

    def test_data_parallel_needs_no_forward_comm(self):
        self.assertEqual(self.run_case(4, 1, {'x': 'rows'}).comm_msgs, 0)

    def test_tensor_parallel_megatron(self):          # W1 column split, W2 row split -> partial sums -> all-reduce
        for coll in ('direct', 'ring', 'tree', 'auto'):
            with self.subTest(collective=coll):
                plan = self.run_case(4, 1, TP, collective=coll)
                self.assertEqual(plan.layout['y'], pt.PART)
                self.assertGreater(plan.comm_msgs, 0)

    def test_tensor_parallel_chunked_ring(self):
        self.run_case(4, 2, TP, collective='ring', chunks=2)

    def test_zero_fsdp_gathers_weights_and_scatters_grads(self):
        for nodes in (1, 2):
            with self.subTest(nodes=nodes):
                self.assertGreater(self.run_case(4, nodes, ZERO).comm_msgs, 0)

    def test_zero_with_ring_all_gather_two_nodes(self):
        self.run_case(4, 2, ZERO, collective='ring')

    def test_bf16_weight_gather(self):                # small integers are exactly representable in bf16
        self.run_case(4, 2, ZERO, wire='bf16')

    def test_dp_x_tp(self):
        self.run_case(4, 1, dict(ZERO, **TP))

    def test_all_to_all_resharding(self):             # ROWS <-> COLS, both directions, across nodes
        for frm, to in (('rows', 'cols'), ('cols', 'rows')):
            with self.subTest(frm=frm, to=to):
                spec = {'buffers': [dict(name='x', rows=8, cols=12, init=dict(kind='values', values=[float(i % 11) for i in range(96)])),
                                    dict(name='y', rows=8, cols=12)],
                        'graphs': [dict(name='g', tasks=[dict(id='c', op='copy', args=dict(src='x', dst='y', n=96))])]}
                plan, _ = verify.check(spec, pt.Cluster.uniform(4, 2), shard({'x': frm, 'y': to}), seeds=range(4))
                self.assertEqual(plan.comm_msgs, 12)                  # P*(P-1) tiles

    def test_device_counts(self):
        for P in (1, 2, 8):
            with self.subTest(P=P):
                self.run_case(P, 1, TP, collective='ring')

    def test_uneven_split_is_rejected_with_a_message(self):
        with self.assertRaisesRegex(pt.PartitionError, 'rows % 3'):
            pt.partition(wl.mlp(), pt.Cluster.uniform(3), shard({'x': 'rows'}))

    def test_unknown_layout_buffer(self):
        with self.assertRaisesRegex(pt.PartitionError, 'unknown buffer'):
            pt.partition(wl.mlp(), pt.Cluster.uniform(2), shard({'nope': 'rows'}))

    def test_replicas_stay_identical(self):           # reconstruct() raises if an R buffer differs between devices
        verify.check(wl.mlp(), pt.Cluster.uniform(4, 2), shard(dict(ZERO, **TP), collective='ring'), seeds=range(6))


class RegressionFp(unittest.TestCase):
    def test_data_parallel_training_loop_matches_within_tolerance(self):
        spec = wl.regress(16)
        plan, rep = verify.check(spec, pt.Cluster.uniform(4, 2), shard({'x': 'rows', 'target': 'rows', 'pred': 'rows'}),
                                 seeds=range(5), exact=False, rel=1e-5)
        self.assertGreater(plan.comm_msgs, 0)                     # the scalar loss/grad all-reduce
        self.assertEqual(plan.layout['w'], pt.R)


class Pipeline(unittest.TestCase):
    def setUp(self):
        self.spec, _ = wl.chain(L=4, m=8, d=8)
        self.split = [f'p{l}' for l in range(4)] + [f'h{l}' for l in range(4)] + ['x']

    def test_pipeline_stages_with_microbatches_match_bitwise(self):
        for M in (1, 2, 4):
            for nodes in (1, 4):
                with self.subTest(M=M, nodes=nodes):
                    st = {'mode': 'place', 'auto': 'pipeline', 'stages': 4, 'microbatches': M, 'split': self.split}
                    plan, _ = verify.check(self.spec, pt.Cluster.uniform(4, nodes), st, seeds=range(3))
                    self.assertGreater(plan.comm_msgs, 0)

    def test_microbatch_rewrite_equals_original(self):
        mb = pt.microbatch(self.spec, 4, self.split, {})
        a, b = verify.golden(self.spec), verify.golden(mb)
        for l in range(4):
            full = [b[f'h{l}.mb{i}'] for i in range(4)]
            self.assertEqual([x for part in full for x in part], a[f'h{l}'])

    def test_explicit_assignment_and_heft(self):
        verify.check(self.spec, pt.Cluster.uniform(4, 2), {'mode': 'place', 'assign': {'fwd.mm*': 1, 'fwd.act*': 2}}, seeds=range(3))
        verify.check(self.spec, pt.Cluster.uniform(4, 2), {'mode': 'place', 'auto': 'heft'}, seeds=range(3))

    def test_modeled_pipeline_speedup_needs_microbatches(self):
        # a compute-bound shape (slow modeled device) so the 1.4 us/task dispatch cost is small against the work
        big, _ = wl.chain(L=8, m=256, d=256, fill=True)
        split = [f'p{l}' for l in range(8)] + [f'h{l}' for l in range(8)] + ['x']
        mk = lambda P, **st: pt.estimate(pt.partition(big, pt.Cluster.uniform(P, tflops=0.5), dict({'mode': 'place'}, **st)))['makespan_us']
        one = mk(1)
        t = {M: mk(4, auto='pipeline', stages=4, microbatches=M, split=split) for M in (1, 4)}
        self.assertGreater(t[1], 0.9 * one)        # one microbatch: nothing overlaps, no gain (plus transfers)
        self.assertLess(t[4], 0.6 * one)           # four microbatches fill the pipeline
        self.assertGreater(t[4], one / 4)          # and it can never beat the ideal 4x

    def test_balance_stages_is_optimal(self):
        cuts = pt.balance_stages([5, 1, 1, 1, 5, 1, 1, 1], 2)
        sums = [sum([5, 1, 1, 1, 5, 1, 1, 1][a:b]) for a, b in cuts]
        self.assertEqual(max(sums), 8)


class CrossDeviceVersions(unittest.TestCase):
    def test_serve_on_another_device_sees_published_versions(self):
        spec = json.loads((Path(__file__).resolve().parents[2] / 'examples' / 'live_training_inference.json').read_text())
        plan, _ = verify.check(spec, pt.Cluster.uniform(2), {'mode': 'place', 'assign': {'serve.answer_fresh': 1}}, seeds=range(8))
        names = [t['op'] for t in plan.shards[0].spec['graphs'][0]['tasks']]
        self.assertIn('signal', names)                          # the weight version is mirrored to device 1 by a remote publish
        self.assertTrue(any(r['policy'] == 'at_least' and r['var'].startswith('in.')
                            for t in plan.shards[1].spec['graphs'][0]['tasks'] for r in t.get('reads', [])))

    def test_exact_reads_across_devices_are_refused(self):
        spec = json.loads((Path(__file__).resolve().parents[2] / 'examples' / 'live_training_inference.json').read_text())
        spec['graphs'][1]['tasks'][0]['reads'][0]['policy'] = 'exact'
        with self.assertRaisesRegex(pt.PartitionError, 'exact'):
            pt.partition(spec, pt.Cluster.uniform(2), {'mode': 'place', 'assign': {'serve.snapshot': 1}})


class Fuzz(unittest.TestCase):
    """Random layouts / device counts / topologies / collectives / chunking, and random task placements: always bitwise right
    (or a PartitionError that names the problem) -- never a silently wrong answer."""

    def test_random_layouts(self):
        import random
        rng = random.Random(20261103)
        names = ['x', 'W1', 'W2', 'gW1', 'gW2', 'h', 'a', 'y', 'z']
        ok = 0
        for _ in range(60):
            spec = wl.mlp(seed=rng.randint(0, 99))
            P = rng.choice([1, 2, 4, 8])
            nodes = rng.choice([n for n in (1, 2, 4) if P % n == 0])
            lay = {b: rng.choice(['repl', 'rows', 'cols']) for b in names if rng.random() < 0.6}
            st = shard(lay, collective=rng.choice(['auto', 'ring', 'tree', 'direct']), chunks=rng.choice([1, 2]))
            try:
                verify.check(spec, pt.Cluster.uniform(P, nodes), st, seeds=range(2))
                ok += 1
            except pt.PartitionError:
                pass
        self.assertGreater(ok, 40)

    def test_random_layouts_bf16_wire_within_bf16_precision(self):
        import random
        rng = random.Random(7)
        for _ in range(12):
            lay = {b: rng.choice(['repl', 'rows']) for b in ['x', 'W1', 'W2', 'gW1', 'gW2']}
            verify.check(wl.mlp(seed=rng.randint(0, 9)), pt.Cluster.uniform(4, 2), shard(lay, wire='bf16'), seeds=range(2), exact=False, rel=0.02)

    def test_random_placements(self):
        import random
        rng = random.Random(5)
        spec, _ = wl.chain(L=4, m=8, d=8)
        ids = [f'fwd.mm{l}' for l in range(4)] + [f'fwd.act{l}' for l in range(4)]
        for _ in range(12):
            assign = {i: rng.randrange(4) for i in ids}
            verify.check(spec, pt.Cluster.uniform(4, rng.choice([1, 2, 4])), {'mode': 'place', 'assign': assign}, seeds=range(2))
        for _ in range(4):
            assign = {i: rng.randrange(3) for i in ['fwd.mm1', 'fwd.mm2', 'fwd.act1']}
            verify.check(wl.mlp(), pt.Cluster.uniform(3, 3), {'mode': 'place', 'assign': {'fwd.mm1': assign['fwd.mm1'], 'fwd.mm2': assign['fwd.mm2'], 'fwd.out': assign['fwd.act1']}}, seeds=range(2))


class SimulatorCanFail(unittest.TestCase):
    """The checker must be able to fail: break the protocol and make sure the comparison notices."""

    def test_signal_before_data_is_detected(self):
        plan = pt.partition(wl.mlp(), pt.Cluster.uniform(4), shard(ZERO))
        gold = verify.golden(plan.spec)
        orig = commops.ref_ops.OPS['put_signal']

        def broken(b, a):          # publish first, copy later: a consumer that runs in between sees stale data
            commops._signal_to(b['flag'], a['var'], a['ver'])
            pending.append((b, a))
        pending = []
        commops.ref_ops.OPS['put_signal'] = commops.ref_ops.RefOp(broken, orig.extents, orig.mins)
        try:
            bad = 0
            for seed in range(30):
                pending.clear()
                try:
                    results, _ = sim.run(plan, seed)
                    for b, a in pending:
                        commops._put(b, a)
                    got = plan.reconstruct(results)
                    bad += any(got[n] != gold[n] for n in got if n in gold)
                except (sim.SimError, pt.PartitionError):       # deadlock, or replicas that no longer agree: both are detections
                    bad += 1
            self.assertGreater(bad, 0, 'a protocol that signals before the data must produce wrong results in some interleaving')
        finally:
            commops.ref_ops.OPS['put_signal'] = orig

    def test_missing_signal_deadlocks_loudly(self):
        plan = pt.partition(wl.mlp(), pt.Cluster.uniform(2), shard(ZERO))
        t = next(t for t in plan.shards[0].spec['graphs'][0]['tasks'] if t['op'] == 'put_signal')
        t['op'], t['args'] = 'put', {k: v for k, v in t['args'].items() if k in ('src', 'dst', 'src_off', 'dst_off', 'rows', 'cols', 'src_ld', 'dst_ld')}
        with self.assertRaises(sim.SimError):
            sim.run(plan, 0)


class DeterminismAndPlans(unittest.TestCase):
    def test_same_bits_for_every_schedule(self):
        plan = pt.partition(wl.mlp(), pt.Cluster.uniform(4, 2), shard(dict(ZERO, **TP), collective='ring', chunks=2))
        outs = [json.dumps(plan.reconstruct(sim.run(plan, s)[0])) for s in range(6)]
        self.assertEqual(len(set(outs)), 1)

    def test_partitioning_is_deterministic(self):
        mk = lambda: [json.dumps(s.spec, sort_keys=True) for s in pt.partition(wl.mlp(), pt.Cluster.uniform(4, 2), shard(ZERO)).shards]
        self.assertEqual(mk(), mk())

    def test_write_plan_files(self):
        plan = pt.partition(wl.mlp(), pt.Cluster.uniform(4, 2), shard(ZERO))
        gold = verify.golden(plan.spec)
        with tempfile.TemporaryDirectory() as d:
            pt.write_plan(plan, d, gold)
            for r in range(4):
                rd = Path(d) / f'rank{r}'
                for f in ('tasks.bin', 'successors.bin', 'args.bin', 'metadata.json', 'dist.txt'):
                    self.assertTrue((rd / f).exists(), f)
                txt = (rd / 'dist.txt').read_text()
                self.assertIn(f'rank {r} world 4', txt)
                self.assertIn('expect', txt)
                self.assertEqual((rd / 'tasks.bin').stat().st_size % 64, 0)
            # in-vars start at version 0 at runtime even though the compiler was told they are published
            self.assertTrue(all(l.split()[2] == '0' for l in (Path(d) / 'rank0' / 'dist.txt').read_text().splitlines()
                                if l.startswith('var ') and plan.shards[0].spec['variables'][int(l.split()[1])].startswith('in.')))

    def test_comm_ops_are_registered_and_compile_to_abi_ids(self):
        ops = commops.merged_ops()
        self.assertEqual({n: o.id for n, o in ops.items() if o.id >= 64}, {
            'put': 64, 'put_signal': 65, 'signal': 66, 'reduce2': 67, 'reduce_put_signal': 68, 'send_proxy': 69, 'pack_bf16': 70, 'unpack_bf16': 71})
        self.assertTrue(all(64 <= o.id <= 95 for n, o in ops.items() if n in commops._ref()))

    def test_every_remote_signal_chain_is_single_writer_and_increasing(self):
        plan = pt.partition(wl.mlp(), pt.Cluster.uniform(4, 2), shard(dict(ZERO, **TP), collective='ring', chunks=2))
        seen = {}
        for s in plan.shards:
            for t in s.spec['graphs'][0]['tasks']:
                if t['op'] in ('put_signal', 'reduce_put_signal', 'signal'):
                    key = (t['args']['flag'], t['args']['var'])
                    self.assertNotIn(key, seen, f'two writers for inbox slot {key}')
                    seen[key] = t['args']['ver']
                elif t['op'] == 'send_proxy':
                    key = (f"rank{t['args']['dst_rank']}", t['args']['var'])
                    self.assertNotIn(key, seen)
                    seen[key] = t['args']['ver']


class CommOpsReference(unittest.TestCase):
    def test_tile_put_with_strides(self):
        src = list(map(float, range(40)))
        dst = [0.0] * 40
        commops._put({'src': src, 'dst': dst}, dict(src_off=3, dst_off=1, rows=3, cols=4, src_ld=7, dst_ld=5))
        self.assertEqual(dst[1:5], [3, 4, 5, 6])
        self.assertEqual(dst[6:10], [10, 11, 12, 13])

    def test_bf16_round_to_nearest_even(self):
        self.assertEqual(commops.bf16_bits(1.00390625), 0x3f80)          # exactly halfway -> even
        self.assertEqual(commops.bf16_bits(1.00390625 + 1.1920929e-7), 0x3f81)
        self.assertEqual(commops.bf16_bits(-0.0), 0x8000)


if __name__ == '__main__':
    unittest.main()
