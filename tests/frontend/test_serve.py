import unittest

import _path  # noqa: F401
import oracle_gpt as ref_gpt
from kvmfe.emit import merge
from kvmfe.ir import FrontendError
from kvmfe.lower import Tiling
from kvmfe.nn import GPT, ModelConfig
from kvmfe.optim import OptimConfig
from kvmfe.pipeline import compile_training
from kvmfe.refvm import execute
from kvmfe.serve import ServeRequest, compile_serving
from kvmfe.train import TrainConfig, weights_version

CFG = ModelConfig(name='t', vocab=32, d_model=32, layers=2, heads=2, seq=32, batch=1)
TILING = Tiling.preset('cpu')


def serve_spec(req):
    return compile_serving(GPT(CFG), req, TILING)


class Serving(unittest.TestCase):
    def test_exact_snapshots_match_reference_decode_while_training_runs(self):
        steps, prompt, n_new = 2, (1, 8, 11), 3
        train = compile_training(CFG, TrainConfig(steps=steps), TILING)
        reqs = [ServeRequest(f'sv{k}', prompt, n_new, weights_version(k)) for k in range(steps + 1)]
        spec = merge(train.spec, *[serve_spec(r)[1] for r in reqs])
        _, hist = ref_gpt.train_reference(CFG, steps, OptimConfig())
        for seed in (0, 1, 2):                               # three different interleavings of training and serving
            mem = execute(spec, order='random', seed=seed)
            for k, r in enumerate(reqs):
                got = [int(mem.bufs[f'{r.name}.cur{i}'].a[0]) for i in range(1, n_new + 1)]
                self.assertEqual(got, ref_gpt.greedy(CFG, hist[k], prompt, n_new), f'seed {seed} weights after {k} updates')

    def test_requests_see_different_weights(self):
        outs = set()
        train = compile_training(CFG, TrainConfig(steps=3), TILING)
        reqs = [ServeRequest(f'sv{k}', (1, 8, 11), 4, weights_version(k)) for k in (0, 3)]
        mem = execute(merge(train.spec, *[serve_spec(r)[1] for r in reqs]), order='random', seed=4)
        for r in reqs:
            outs.add(tuple(int(mem.bufs[f'{r.name}.cur{i}'].a[0]) for i in range(1, 5)))
        self.assertEqual(len(outs), 2, 'untrained and trained weights should generate different text')

    def test_single_token_prompt_and_graph_metadata(self):
        low, spec = serve_spec(ServeRequest('one', (5,), 2, 0, priority=7, deadline_us=1234))
        tasks = spec['graphs'][0]['tasks']
        self.assertTrue(all(t['priority'] == 7 and t['deadline_us'] == 1234 for t in tasks))
        reads = [t for t in tasks if 'reads' in t]
        self.assertTrue(reads and all(r['reads'] == [{'var': 'weights', 'policy': 'exact', 'version': 0}] for r in reads))
        self.assertEqual(spec['variables'], ['weights'])
        execute(spec)                                         # weights are zero here, but the graph must run

    def test_at_least_policy_and_errors(self):
        _, spec = serve_spec(ServeRequest('al', (1, 2), 1, 4, policy='at_least'))
        self.assertTrue(any(r['policy'] == 'at_least' for t in spec['graphs'][0]['tasks'] for r in t.get('reads', [])))
        for bad in (ServeRequest('x', (), 1), ServeRequest('x', (99,), 1), ServeRequest('x', (1,), 0), ServeRequest('x', (1,), 1, policy='nope')):
            with self.assertRaises(FrontendError):
                serve_spec(bad)


if __name__ == '__main__':
    unittest.main()
