"""Pluggable operator packs: tools/kvm_build.sh --print-env / check_packs.py, abi.load_ops(extra=...), $KVM_EXTRA_DEFS / $KVM_EXTRA_REFS
in the host compiler and CPU VM. The GPU half (build + run on the executor) is tests/packs/run.sh."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / 'packs' / 'example_saxpy'
sys.path.insert(0, str(ROOT / 'runtime'))
import abi  # noqa: E402


def run(*cmd, env=None, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, **(env or {})}, **kw)


class Packs(unittest.TestCase):
    def test_extra_def_adds_operator_with_unique_id(self):
        ops = abi.load_ops(extra=(str(PACK / 'ops.def'),))
        self.assertEqual((ops['saxpy'].id, [a.type for a in ops['saxpy'].args]), (300, ['buf', 'buf', 'buf', 'f32', 'i32']))
        self.assertNotIn('saxpy', abi.load_ops(extra=()))

    def test_duplicate_id_or_name_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            clash = Path(d) / 'a.def'
            clash.write_text('X(mine, 32, "x:buf")\n')
            with self.assertRaisesRegex(abi.AbiError, 'duplicate'):
                abi.load_ops(extra=(str(clash),))
            r = run(sys.executable, '-I', str(ROOT / 'tools/check_packs.py'), str(clash))
            self.assertNotEqual(r.returncode, 0)
            self.assertIn('duplicate', r.stderr)

    def test_print_env_lists_defs_and_refs(self):
        r = run('bash', str(ROOT / 'tools/kvm_build.sh'), '--print-env', '--pack', str(PACK))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn(f'KVM_EXTRA_DEFS=\'{PACK}/ops.def\'', r.stdout)
        self.assertIn(f'KVM_EXTRA_REFS=\'{PACK}/ref.py\'', r.stdout)

    def test_compiler_and_cpu_vm_run_the_pack_op(self):
        env = {'KVM_EXTRA_DEFS': str(PACK / 'ops.def'), 'KVM_EXTRA_REFS': str(PACK / 'ref.py')}
        with tempfile.TemporaryDirectory() as d:
            r = run(sys.executable, str(ROOT / 'runtime/compile_workload.py'), '--emit-golden',
                    str(ROOT / 'tests/packs/workload.json'), d + '/b', env=env)
            self.assertEqual(r.returncode, 0, r.stderr)
            g = json.loads((Path(d) / 'b/golden.json').read_text())['buffers']
            x = [1, 2, 3, 4, 5, 6, 7, 8]
            self.assertEqual(g['out'], [-0.5 * (2 * v + 0.5) + v for v in x])
            r = run(sys.executable, str(ROOT / 'runtime/compile_workload.py'), str(ROOT / 'tests/packs/workload.json'), d + '/c')
            self.assertNotEqual(r.returncode, 0)             # without the pack the op is unknown
            self.assertIn('saxpy', r.stderr + r.stdout)

    def test_game_pack_is_one_flag_not_a_recipe(self):
        text = (ROOT / 'tools/kvm_build.sh').read_text()
        self.assertIn('RDC=1; NV_EXTRA+=(-DKVM_WITH_GAME)', text)


if __name__ == '__main__':
    unittest.main()
