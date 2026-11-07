"""Unit tests of the GPU-comparison helpers (no GPU): python3 -I -m unittest discover -s examples/models"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import check  # noqa: E402


class StepOneBuffers(unittest.TestCase):
    def test_only_pre_optimizer_step0_outputs_not_rewritten_later(self):
        tasks = [
            {'id': 's0/0:a', 'op': 'matmul_tile', 'args': {'out': 'act0', 'a': 'x'}},
            {'id': 's0/1:b', 'op': 'softmax_ce_fwd', 'args': {'rowloss': 'act1', 'lse': 'act2', 'logits': 'act0'}},
            {'id': 's0/2:g', 'op': 'adamw_lr', 'args': {'p': 'w.p', 'g': 'act3'}},
            {'id': 's0/3:late', 'op': 'matmul_tile', 'args': {'out': 'act9'}},
            {'id': 's1/0:a', 'op': 'matmul_tile', 'args': {'out': 'act0'}},       # act0 is reused by step 1: not step-1 data
        ]
        self.assertEqual(check.step1_buffers(tasks), {'act1', 'act2'})

    def test_max_rel(self):
        self.assertAlmostEqual(check.max_rel([1.0, 2.1], [1.0, 2.0]), 0.05)
        self.assertEqual(check.max_rel([0.0], [0.0]), 0.0)
        self.assertEqual(check.max_rel([0.5], [0.0]), 0.5)


class EndToEnd(unittest.TestCase):
    """A real tiny bundle: golden compared with itself passes, a perturbed GPU dump fails."""

    def test_precise_mode_detects_a_wrong_buffer(self):
        sys.path.insert(0, str(HERE.parents[1] / 'frontend'))
        import importlib.util
        spec = importlib.util.spec_from_file_location('zoo_gen_bundle', HERE / 'gen_bundle.py')
        gen_bundle = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gen_bundle)
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            c, _ = gen_bundle.build('mlp_tiny', 2, precise=True)
            (d / 'workload.json').write_text(json.dumps(c.spec))
            subprocess.run([sys.executable, str(HERE.parents[1] / 'runtime/compile_workload.py'), '--emit-golden', str(d / 'workload.json'),
                            str(d / 'bundle')], check=True, capture_output=True)
            meta = json.loads((d / 'bundle/metadata.json').read_text())
            gold = json.loads((d / 'bundle/golden.json').read_text())['buffers']
            lines = [f"buffer {i}: {' '.join(repr(float(v)) for v in gold[b['name']])}" for i, b in enumerate(meta['buffers'])]
            (d / 'dump.txt').write_text('\n'.join(lines) + '\nOK: 1/1 tasks, 1.0 ms\n')
            run = lambda: subprocess.run([sys.executable, str(HERE / 'check.py'), str(d / 'workload.json'), str(d / 'bundle'), str(d / 'dump.txt'),
                                          '--mode', 'precise'], capture_output=True, text=True)
            self.assertEqual(run().returncode, 0, run().stdout)
            i = next(i for i, b in enumerate(meta['buffers']) if b['name'].startswith('w.mlp.fc0.w'))
            bad = [f"buffer {i}: {' '.join(repr(float(v) * 1.01 + 1e-3) for v in gold[meta['buffers'][i]['name']])}" if k == i else ln
                   for k, ln in enumerate(lines)]
            (d / 'dump.txt').write_text('\n'.join(bad) + '\n')
            r = run()
            self.assertEqual(r.returncode, 1)
            self.assertIn('FAIL', r.stdout)


class InvariantParameters(unittest.TestCase):
    """ViT bf16: the key-projection bias has a structurally zero gradient, so its weight is optimizer-driven rounding noise. The check
    must (a) accept any noise inside the optimizer's travel envelope and the noise level of the moments, (b) reject a weight outside the
    envelope or moments at gradient scale, (c) still reject a wrong live weight."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        d = cls.d = Path(cls.tmp.name)
        subprocess.run([sys.executable, str(HERE / 'gen_bundle.py'), str(d), '--model', '{"family":"vit","name":"vit_bf16","precision":"bf16"}',
                        '--steps', '2', '--loss-scale', '128'], check=True, capture_output=True)
        cls.meta = json.loads((d / 'bundle/metadata.json').read_text())
        cls.gold = json.loads((d / 'bundle/golden.json').read_text())['buffers']
        info = json.loads((d / 'meta.json').read_text())
        cls.name = info['invariant_params'][0]
        cls.T = info['steps'] * info['optim']['lr']

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_check(self, edit):
        bufs = {k: list(v) for k, v in self.gold.items()}
        edit(bufs)
        lines = [f"buffer {i}: {' '.join(repr(float(v)) for v in bufs[b['name']])}" for i, b in enumerate(self.meta['buffers'])]
        (self.d / 'dump.txt').write_text('\n'.join(lines) + '\nOK: 1/1 tasks, 1.0 ms\n')
        return subprocess.run([sys.executable, str(HERE / 'check.py'), str(self.d / 'workload.json'), str(self.d / 'bundle'), str(self.d / 'dump.txt'),
                               '--mode', 'lowp'], capture_output=True, text=True)

    def test_the_declaration_is_found_and_reported(self):
        r = self.run_check(lambda b: None)
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertIn('1 structurally-zero-gradient parameters', r.stdout)
        self.assertIn('within the optimizer envelope', r.stdout)

    def test_noise_inside_the_envelope_passes_even_with_opposite_signs(self):
        def edit(b):                                                # the 'GPU' flipped every sign of the optimizer-driven weight and moment
            b[f'w.{self.name}'] = [-0.9 * self.T if x > 0 else 0.9 * self.T for x in b[f'w.{self.name}']]
            b[f'm.{self.name}'] = [-x for x in b[f'm.{self.name}']]
        r = self.run_check(edit)
        self.assertEqual(r.returncode, 0, r.stdout)

    def test_weight_outside_the_envelope_fails(self):
        r = self.run_check(lambda b: b.update({f'w.{self.name}': [1.5 * self.T] * len(b[f'w.{self.name}'])}))
        self.assertEqual(r.returncode, 1)
        self.assertIn('FAIL', r.stdout)
        self.assertIn('within the optimizer envelope', [ln for ln in r.stdout.splitlines() if 'FAIL' in ln][0])

    def test_moments_at_gradient_scale_fail(self):
        scale = max(abs(x) for k, v in self.gold.items() if k.startswith('m.') and k != f'm.{self.name}' for x in v)
        r = self.run_check(lambda b: b.update({f'm.{self.name}': [scale] * len(b[f'm.{self.name}'])}))
        self.assertEqual(r.returncode, 1)
        self.assertIn('moments at the noise level', [ln for ln in r.stdout.splitlines() if 'FAIL' in ln][0])

    def test_a_wrong_live_weight_still_fails(self):
        r = self.run_check(lambda b: b.update({'w.vit.head.w': [x * 1.05 + 1e-3 for x in b['w.vit.head.w']]}))
        self.assertEqual(r.returncode, 1)
        self.assertIn('weights rel L2', r.stdout)


if __name__ == '__main__':
    unittest.main()
