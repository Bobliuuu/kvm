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


if __name__ == '__main__':
    unittest.main()
