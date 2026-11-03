import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import _path  # noqa: F401
from kvmfe import game
from kvmfe.ir import FrontendError
from kvmfe.lower import Tiling
from kvmfe.refvm import execute

ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / 'frontend'
HAS_DOOM = any((ROOT / 'include').glob('kvm_ops_game.def'))


@unittest.skipUnless(HAS_DOOM, 'include/kvm_ops_game.def not present')
class Game(unittest.TestCase):
    def setUp(self):
        self.low, self.spec = game.compile_game(game.GameConfig(frames=4, present_rows=50), Tiling())
        self.tasks = {t['id']: t for t in self.spec['graphs'][0]['tasks']}

    def reach(self, a, b):
        stack, seen = [b], set()
        while stack:
            n = stack.pop()
            if n == a:
                return True
            for p in self.tasks[n].get('after', []):
                if p not in seen:
                    seen.add(p)
                    stack.append(p)
        return False

    def test_tick_chain_pacing_and_frame_versions(self):
        ticks = [i for i, t in self.tasks.items() if t['op'] == 'doom_tick']
        self.assertEqual(len(ticks), 4)
        for a, b in zip(ticks, ticks[1:]):
            self.assertTrue(self.reach(a, b), 'tick f+1 must depend on tick f')
        for f, i in enumerate(ticks):
            self.assertEqual(self.tasks[i]['deadline_us'], (f + 1) * 28571)
            self.assertEqual(self.tasks[i]['args']['frame'], f)
        presents = [t for t in self.tasks.values() if t['op'] == 'doom_present']
        self.assertEqual(len(presents), 4 * 4)                  # 200 rows / 50
        self.assertEqual(sorted({p['args']['row0'] for p in presents}), [0, 50, 100, 150])
        pubs = [t for t in self.tasks.values() if t.get('publishes') == 'frame']
        self.assertEqual(len(pubs), 4)
        self.assertEqual(self.spec['variables'], ['frame'])
        first_present = next(i for i, t in self.tasks.items() if t['op'] == 'doom_present')
        self.assertTrue(self.reach(ticks[0], first_present))

    def test_no_cpu_reference_is_a_clear_error(self):
        with self.assertRaisesRegex(FrontendError, 'no CPU reference'):
            execute(self.spec)

    def test_missing_defs_fail_clearly(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / 'kvm_ops.def').write_text('X(noop, 0, "")\n')
            old = os.environ.get('KVM_INCLUDE')
            os.environ['KVM_INCLUDE'] = d
            try:
                with self.assertRaisesRegex(FrontendError, 'doom_tick'):
                    game.build_game(game.GameConfig())
            finally:
                if old is None:
                    os.environ.pop('KVM_INCLUDE')
                else:
                    os.environ['KVM_INCLUDE'] = old


class Cli(unittest.TestCase):
    def run_cli(self, *args):
        r = subprocess.run([sys.executable, '-m', *args], cwd=FRONTEND, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[-800:])
        return r

    def test_demo_emits_all_workloads(self):
        with tempfile.TemporaryDirectory() as d:
            frames = ['--game-frames', '2'] if HAS_DOOM else []
            self.run_cli('kvmfe.demo', '--train-steps', '2', '--serve-every', '1', *frames, '--out', d, '--no-bundles')
            names = {p.name for p in Path(d).iterdir()}
            self.assertTrue({'train.json', 'serve_s0.json', 'serve_s2.json', 'demo_combined.json', 'frontend.json'} <= names)
            comb = json.loads((Path(d) / 'demo_combined.json').read_text())
            self.assertEqual({g['name'] for g in comb['graphs']} >= {'train', 'serve_s0', 'serve_s1', 'serve_s2'}, True)
            side = json.loads((Path(d) / 'frontend.json').read_text())
            self.assertEqual(side['loss_buffers'], ['loss.0', 'loss.1'])

    def test_inspect_reports_counts(self):
        r = self.run_cli('kvmfe.inspect', '--model', 'gpt_tiny', '--steps', '2')
        self.assertIn('critical path', r.stdout)
        self.assertIn('memory:', r.stdout)

    def test_check_defs(self):
        self.run_cli('kvmfe.check_defs')


if __name__ == '__main__':
    unittest.main()
