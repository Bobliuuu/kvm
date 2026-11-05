"""Generic partitioner path + CPU multi-device simulator at 16-64 devices on a hierarchical cluster (8 devices per node), integer data,
bitwise against the unpartitioned CPU VM. Slower than the rest (~1 min). These are the largest configurations CHECKED for
correctness; everything beyond is cost model only.   python3 -I -m unittest discover -s tests/dist -p 'test_scaleout_sim64.py' -v"""
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import verify    # noqa: E402
import workloads as wl   # noqa: E402
import partition as pt   # noqa: E402

TP = {'W1': 'cols', 'W2': 'rows', 'gW1': 'cols', 'gW2': 'rows'}
ZERO = {'x': 'rows', 'W1': 'rows', 'W2': 'rows', 'gW1': 'rows', 'gW2': 'rows'}


def cluster(P):
    return pt.Cluster.uniform(P, max(1, P // 8), intra=pt.Link('p2p', 450.0, 1.0), inter=pt.Link('proxy', 50.0, 5.0))


class Sim64(unittest.TestCase):
    def mlp(self, P):
        return wl.mlp(m=P, d=P, h=2 * P, o=P)

    def test_zero_fsdp_64_devices_8_nodes(self):
        plan, _ = verify.check(self.mlp(64), cluster(64), {'mode': 'shard', 'layouts': ZERO, 'collective': 'ring', 'chunks': 2}, seeds=range(2))
        self.assertGreater(plan.comm_msgs, 64 * 63)

    def test_megatron_tp_with_allreduce_variants_32_devices(self):
        for coll in ('ring', 'tree', 'direct'):
            with self.subTest(coll=coll):
                verify.check(self.mlp(32), cluster(32), {'mode': 'shard', 'layouts': TP, 'collective': coll}, seeds=range(2))

    def test_dp_x_tp_x_zero_mixed_layouts_64_devices(self):
        verify.check(self.mlp(64), cluster(64), {'mode': 'shard', 'layouts': dict(ZERO, **TP), 'collective': 'ring'}, seeds=range(2))

    def test_expert_parallel_style_all_to_all_64_devices(self):
        spec = {'buffers': [dict(name='x', rows=64, cols=64, init=dict(kind='values', values=[float(i % 13) for i in range(4096)])), dict(name='y', rows=64, cols=64)],
                'graphs': [dict(name='g', tasks=[dict(id='c', op='copy', args=dict(src='x', dst='y', n=4096))])]}
        plan, _ = verify.check(spec, cluster(64), {'mode': 'shard', 'layouts': {'x': 'rows', 'y': 'cols'}}, seeds=range(2))
        self.assertEqual(plan.comm_msgs, 64 * 63)

    def test_pipeline_16_stages_across_nodes_with_microbatches(self):
        spec, _ = wl.chain(L=16, m=16, d=8)
        split = [f'p{l}' for l in range(16)] + [f'h{l}' for l in range(16)] + ['x']
        for M in (4, 8):
            with self.subTest(M=M):
                st = {'mode': 'place', 'auto': 'pipeline', 'stages': 16, 'microbatches': M, 'split': split}
                plan, _ = verify.check(spec, pt.Cluster.uniform(16, 4), st, seeds=range(2))
                self.assertGreater(plan.comm_msgs, 0)
                est = pt.estimate(plan)
                self.assertGreater(est['makespan_us'], 0)


if __name__ == '__main__':
    t = time.time()
    unittest.main()
