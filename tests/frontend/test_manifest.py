"""End to end through the compiler: demo --all -> MANIFEST.json -> generic runner on the compiler's CPU VM (float32 reference)
vs the independent float64 oracle."""
import tempfile
import unittest
from pathlib import Path

import _path  # noqa: F401
import oracle_gpt
from kvmfe import demo, run_manifest
from kvmfe.nn import PRESETS
from kvmfe.optim import OptimConfig

CFG = PRESETS['gpt_tiny']
HAS_DOOM = (Path(__file__).resolve().parents[2] / 'include' / 'kvm_ops_game.def').exists()


class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        demo.main(['--all', '--model', 'gpt_tiny', '--train-steps', '4', '--out', cls.tmp.name] + ([] if HAS_DOOM else ['--game-frames', '0']))
        cls.losses = {}

        def grab(adm, vm):
            if adm['name'].startswith('train_w'):
                a = adm['steps'][0]
                for s in range(adm['steps'][1] - a):
                    cls.losses[a + s] = vm.buffers[f'loss.{s}'][0]
        cls.vm, cls.order = run_manifest.run_manifest(cls.tmp.name, allow_opaque=True, after_each=grab)
        cls.want, cls.hist = oracle_gpt.train_reference(CFG, 4, OptimConfig())

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_windows_publish_and_versions_advance(self):
        self.assertEqual(self.vm.versions['weights'], 8)
        if HAS_DOOM:
            self.assertEqual(self.vm.versions['frame'], 8)
        self.assertLess(self.order.index('train_w0'), self.order.index('train_w1'))

    def test_loss_curve_matches_float64_oracle_within_float32_tolerance(self):
        for s in range(4):
            self.assertAlmostEqual(self.losses[s], self.want[s], delta=2e-5 * self.want[s], msg=f'step {s}')

    def test_served_tokens_match_oracle_decode_at_each_snapshot(self):
        prompt = demo.prompt_for(CFG)
        for k in (0, 2, 4):
            got = [int(self.vm.buffers[f'serve_s{k}.cur{i}'][0]) for i in range(1, 4)]
            self.assertEqual(got, oracle_gpt.greedy(CFG, self.hist[k], prompt, 3), f'weights after {k} steps')


if __name__ == '__main__':
    unittest.main()
