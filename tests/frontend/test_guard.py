"""Data-dependent control flow at admission granularity (manifest_version 2 `guard`, FRAMEWORK.md "Dynamic shapes and data-dependent control flow"):
a `while` loop with a data-dependent trip count is a chain of guarded admissions, each reading a scalar the previous one wrote.

The loop body is one SGD step on a scalar, `w -= 0.25`; the loop runs `while w > 0.1`, at most MAX times (the static unroll bound)."""
import json
import tempfile
import unittest
from pathlib import Path

import _path  # noqa: F401
from kvmfe import run_manifest
from kvmfe.ir import FrontendError

MAX = 5


def write_manifest(root: Path, w0: float, guard=True, version=2, publishes=None, extra=None):
    spec = dict(variables=['v'], buffers=[dict(name='w', rows=1, cols=1, init=dict(kind='fill', value=w0)),
                                          dict(name='g', rows=1, cols=1, init=dict(kind='fill', value=1.0))],
                graphs=[dict(name='body', tasks=[dict(id='sgd', op='sgd', args=dict(w='w', g='g', lr=0.25, n=1), after=[])])])
    (root / 'workloads').mkdir()
    (root / 'workloads' / 'body.json').write_text(json.dumps(spec))
    adms = []
    for k in range(MAX):
        a = dict(name=f'iter{k}', workload='body', steps=None, after_bundles=[f'iter{k - 1}'] if k else [], when=[], at_us=None,
                 base_versions={'v': 0}, publishes=publishes or {}, compile=dict(workload='workloads/body.json', data_dir='workloads', base_versions={'v': 0}))
        if guard:
            a['guard'] = dict(buffer='w', index=0, cmp='gt', value=0.1)
        adms.append({**a, **(extra or {})})
    m = dict(manifest_version=version, buffers=dict(w=dict(rows=1, cols=1, dtype='f32', owner='body'), g=dict(rows=1, cols=1, dtype='f32', owner='body')),
             workloads=dict(body=dict(file='workloads/body.json')), admissions=adms)
    (root / 'MANIFEST.json').write_text(json.dumps(m))


def run(w0, **kw):
    with tempfile.TemporaryDirectory() as td:
        write_manifest(Path(td), w0, **kw)
        vm, order = run_manifest.run_manifest(td)
        return order, list(vm.skipped), vm.buffers['w'][0]


class GuardedLoop(unittest.TestCase):
    def test_trip_count_follows_the_data(self):
        for w0, trips in ((0.05, 0), (0.3, 1), (0.5, 2), (0.9, 4), (3.0, MAX)):     # 4 steps of -0.25 take 0.9 to -0.1; 3.0 never gets under 0.1 in MAX
            order, skipped, w = run(w0)
            self.assertEqual(order, [f'iter{k}' for k in range(trips)], f'w0={w0}')
            self.assertEqual(skipped, [f'iter{k}' for k in range(trips, MAX)])
            self.assertAlmostEqual(w, w0 - 0.25 * trips, places=6)

    def test_without_guards_every_iteration_runs(self):
        order, skipped, w = run(0.5, guard=False, version=1)
        self.assertEqual((len(order), skipped), (MAX, []))
        self.assertAlmostEqual(w, 0.5 - 0.25 * MAX, places=6)

    def test_skip_cascades_through_after_bundles_without_evaluating_later_guards(self):
        # w only decreases, so a later guard could never be true again; make the later guards true to prove the cascade is what skips them
        with tempfile.TemporaryDirectory() as td:
            write_manifest(Path(td), 0.05)
            m = json.loads((Path(td) / 'MANIFEST.json').read_text())
            for a in m['admissions'][1:]:
                a['guard'] = dict(buffer='w', index=0, cmp='lt', value=99.0)           # always true
            (Path(td) / 'MANIFEST.json').write_text(json.dumps(m))
            vm, order = run_manifest.run_manifest(td)
            self.assertEqual((order, len(vm.skipped)), ([], MAX))

    def test_a_guard_on_a_publishing_admission_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            write_manifest(Path(td), 0.5, publishes={'v': 1})
            with self.assertRaisesRegex(FrontendError, 'cannot publish'):
                run_manifest.run_manifest(td)

    def test_malformed_guard_and_old_manifest_version_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            write_manifest(Path(td), 0.5, version=1)
            with self.assertRaisesRegex(FrontendError, 'manifest_version 2'):
                run_manifest.run_manifest(td)
        with tempfile.TemporaryDirectory() as td:
            write_manifest(Path(td), 0.5, extra=dict(guard=dict(buffer='w', index=0, cmp='approx', value=1)))
            with self.assertRaisesRegex(FrontendError, 'guard must be exactly'):
                run_manifest.run_manifest(td)


if __name__ == '__main__':
    unittest.main()
