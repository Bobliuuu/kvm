"""Partitioned frontend models (GPT, Llama, ViT) in the multi-device CPU simulator: pipeline placement equals the unpartitioned
program bitwise; data parallelism keeps the replicas bitwise identical and equals gradient accumulation over P microbatches on one
device (the same model compiled with accum=P) within float32 reduction-order noise."""
import copy
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / 'runtime'), str(ROOT / 'dist'), str(ROOT / 'frontend'), str(ROOT / 'examples/models')]
import gpt_dp             # noqa: E402
import partition as pt    # noqa: E402
import verify             # noqa: E402

DP_TOL = 1e-4      # max|err| / max|ref| per buffer; measured 2.8e-5 (gpt_tiny x2), 3.0e-6 (x4), 2.5e-7 (llama_tiny x2)


def _rel(got, ref):
    return max(abs(a - b) for a, b in zip(got, ref)) / max(1e-30, max(abs(b) for b in ref))


class Pipeline(unittest.TestCase):
    def check(self, model, stages, steps=2):
        import gen_bundle as zoo
        c, _ = zoo.build(model, steps, 'cpu', True)
        spec = copy.deepcopy(c.spec)
        plan = pt.partition(copy.deepcopy(spec), pt.Cluster.uniform(stages), {'mode': 'place', 'auto': 'pipeline', 'stages': stages})
        gold = verify.golden(spec)
        for seed in range(2):
            got = plan.reconstruct(verify.sim.run(plan, seed)[0])
            self.assertEqual([n for n in got if n in gold and got[n] != gold[n]], [], f'{model} x{stages} seed {seed}')
        self.assertTrue(all(len(s.spec['graphs'][0]['tasks']) > 10 for s in plan.shards), 'every stage got work')

    def test_gpt_tiny_2_and_4_stages(self):
        self.check('gpt_tiny', 2)
        self.check('gpt_tiny', 4)

    def test_llama_tiny(self):
        self.check('llama_tiny', 2)

    def test_vit_tiny(self):
        self.check('vit_tiny', 2)


class DataParallel(unittest.TestCase):
    def check(self, model, P, nodes=1):
        rank_spec, ref_spec, streams = gpt_dp.build(model, P, 2, 'cpu', True)
        plan = pt.partition(rank_spec, pt.Cluster.uniform(P, nodes), {'mode': 'dp', 'streams': streams})
        gold = verify.golden(ref_spec)
        for seed in range(2):
            got = plan.reconstruct(verify.sim.run(plan, seed)[0])     # raises if the replicas differ in any bit
            worst = max((_rel(got[n], gold[n]), n) for n in got)
            self.assertLessEqual(worst[0], DP_TOL, f'{model} x{P}: {worst}')
        return plan

    def test_gpt_tiny_2_4(self):
        self.check('gpt_tiny', 2)
        self.check('gpt_tiny', 4)

    def test_gpt_tiny_across_nodes(self):          # every gradient message through the host proxy
        plan = self.check('gpt_tiny', 2, nodes=2)
        self.assertTrue(any(r['kind'] == 'sendq' for s in plan.shards for r in s.remotes.values()))

    def test_llama_tiny(self):
        self.check('llama_tiny', 2)

    def test_rank_count_power_of_two(self):
        rank_spec, _, streams = gpt_dp.build('gpt_tiny', 2, 1, 'cpu', True)
        with self.assertRaises(pt.PartitionError):
            pt.partition(rank_spec, pt.Cluster.uniform(3), {'mode': 'dp', 'streams': streams})


class Launcher(unittest.TestCase):
    def test_advertise_and_listen(self):
        sys.path.insert(0, str(ROOT / 'dist'))
        import launch
        nodes = launch.parse_nodes('a:1,b:2')
        ranks, peers = launch.layout(nodes, 29500, launch.parse_advertise('h1:5000,h2:6000/6001', nodes))
        self.assertEqual(peers, 'h1:5000,h2:6000,h2:6001')
        argv, env = launch.build_cmd(ranks[2], 3, peers, ['x'], '', 10, {}, 41000)
        self.assertEqual(env['KVM_DIST_LISTEN_PORT'], '41001')
        self.assertEqual(launch.peers_for(ranks[1], ranks, peers, 41000), 'h1:5000,127.0.0.1:41000,127.0.0.1:41001')   # same node: loopback
        self.assertEqual(launch.peers_for(ranks[0], ranks, peers, 41000), '127.0.0.1:41000,h2:6000,h2:6001')
        with self.assertRaises(SystemExit):
            launch.parse_advertise('h1:5000', nodes)


if __name__ == '__main__':
    unittest.main()
