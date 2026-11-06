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

    def test_demo_all_emits_bundles_and_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            self.run_cli('kvmfe.demo', '--all', '--train-steps', '4', '--out', d)
            root = Path(d)
            m = json.loads((root / 'MANIFEST.json').read_text())
            names = [a['name'] for a in m['admissions']]
            self.assertEqual(sorted(names), sorted(['train_w0', 'train_w1', 'serve_s0', 'serve_s2', 'serve_s4'] + (['game'] if HAS_DOOM else [])))
            for a in m['admissions']:
                self.assertTrue((root / a['bundle_dir'] / 'metadata.json').is_file(), a['name'])
                self.assertTrue((root / a['compile']['workload']).is_file())
            self.assertIn('w.gpt.wte', m['shared_buffers'])
            self.assertEqual(m['buffers']['w.gpt.wte']['owner'], 'train')
            self.assertEqual(m['workloads']['serve_s2']['priority'], 10)
            self.assertEqual(m['workloads']['train']['variables'], ['weights'])

    def test_demo_lr_and_warmup_reach_the_device_schedule(self):
        """--lr / --warmup (the 'large' demo preset needs them: the default 3e-3 without warmup does not train gpt_162m)."""
        with tempfile.TemporaryDirectory() as d:
            self.run_cli('kvmfe.demo', '--all', '--model', 'gpt_tiny', '--train-steps', '2', '--game-frames', '0', '--lr', '1e-3', '--warmup', '2', '--out', d)
            spec = json.loads((Path(d) / 'workloads' / 'train.json').read_text())
            sched = [t['args'] for g in spec['graphs'] for t in g['tasks'] if t['op'] == 'lr_schedule']
            self.assertTrue(sched)
            self.assertEqual({(a['base'], a['warmup'], a['kind']) for a in sched}, {(1e-3, 2, 0)})

    def test_inspect_reports_counts(self):
        r = self.run_cli('kvmfe.inspect', '--model', 'gpt_tiny', '--steps', '2')
        self.assertIn('critical path', r.stdout)
        self.assertIn('memory:', r.stdout)

    def test_check_defs(self):
        self.run_cli('kvmfe.check_defs')


if __name__ == '__main__':
    unittest.main()
