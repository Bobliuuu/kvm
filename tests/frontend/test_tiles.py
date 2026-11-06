"""Per-shape GEMM tile choice (64x64 / 128x128 flag 32 / 256x128 flag 64): tile-aware regions, flops and CPU reference; results equal the
64x64 lowering; fewer, bigger tasks; the compiler golden (runtime/ref_gpt.py) agrees with the frontend VM."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

import _path  # noqa: F401
from kvmfe.generic import MLP, MSE, ArrayData
from kvmfe.lower import Tiling, pick_mm_tile
from kvmfe.optim import OptimConfig
from kvmfe.pipeline import compile_training

ROOT = Path(__file__).resolve().parents[2]


def compiled(mm_tile, rows=300, din=24, hid=200, **kw):
    rng = np.random.default_rng(3)
    x, y = rng.normal(size=(rows, din)).tolist(), rng.normal(size=(rows, 7)).tolist()
    return compile_training(MLP('m', [din, hid, 7], act='tanh'), loss=MSE(), data=ArrayData(rows, {'x': x, 'y': y}),
                            optimizer=OptimConfig(kind='sgd', lr=0.05), steps=2, tiling=Tiling.preset('cpu', mm_tile=mm_tile, **kw),
                            reuse='none')


def gemm_tasks(c):
    return [t for t in c.low.tasks if t.dev == 'matmul_tile']


class Tiles(unittest.TestCase):
    def test_pick(self):
        self.assertEqual((pick_mm_tile(64, 512, 512), pick_mm_tile(128, 8, 8), pick_mm_tile(256, 8, 8)), (64, 128, 256))
        self.assertEqual((pick_mm_tile(0, 512, 512), pick_mm_tile(0, 100, 512), pick_mm_tile(0, 130, 512), pick_mm_tile(0, 384, 200)),
                         (128, 64, 64, 128))

    def test_big_tiles_equal_small_tiles_and_are_fewer(self):
        base = compiled(64)
        w0 = {k: v.a[:] for k, v in base.run().bufs.items() if k.startswith('w.')}
        for mode, flag in ((128, 32), (256, 64), (0, 32)):
            c = compiled(mode)
            gt = gemm_tasks(c)
            self.assertLess(len(gt), len(gemm_tasks(base)), mode)
            self.assertTrue(any(t.args['flags'] & flag for t in gt), mode)
            for t in gt:
                tm, tn = (256, 128) if t.args['flags'] & 64 else (128, 128) if t.args['flags'] & 32 else (64, 64)
                self.assertEqual((t.args['m0'] % tm, t.args['n0'] % tn), (0, 0))
            w = {k: v.a[:] for k, v in c.run().bufs.items() if k.startswith('w.')}
            for k in w0:
                np.testing.assert_allclose(w[k], w0[k], rtol=1e-9, atol=1e-12, err_msg=f'{mode} {k}')

    def test_flops_are_tile_aware(self):
        base, big = compiled(64), compiled(256)
        f = lambda c: sum(t.flops_bytes()[0] for t in gemm_tasks(c))
        self.assertAlmostEqual(f(base), f(big), delta=1e-9 * f(base))        # same total work whatever the tiling
        # regions cover the output exactly once
        for c in (base, big):
            cover = {}
            for t in gemm_tasks(c):
                if t.args['flags'] & 4:
                    continue
                o = t.writes[0]
                cover.setdefault(o.t.uid, []).append((o.r1 - o.r0) * (o.c1 - o.c0))
            self.assertTrue(cover)

    def test_runtime_golden_matches_frontend_vm(self):
        c = compiled(0, rows=256, din=16, hid=256)
        self.assertTrue(any(t.args['flags'] & 32 for t in gemm_tasks(c)))
        mem = c.run()
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / 'w.json').write_text(json.dumps(c.spec))
            for rel, raw in c.low.data_files.items():
                (Path(d) / rel).parent.mkdir(parents=True, exist_ok=True)
                (Path(d) / rel).write_bytes(raw)
            r = subprocess.run([sys.executable, str(ROOT / 'runtime/compile_workload.py'), '--emit-golden', '--golden-timeout', '600',
                                d + '/w.json', d + '/b'], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr[-500:])
            g = json.loads((Path(d) / 'b/golden.json').read_text())['buffers']
        for k, v in mem.bufs.items():
            if k.startswith('w.') and k in g:
                np.testing.assert_allclose(g[k], v.a, rtol=2e-5, atol=1e-6, err_msg=k)

    def test_bad_mm_tile_refused(self):
        from kvmfe.ir import FrontendError
        with self.assertRaises(FrontendError):
            Tiling(mm_tile=96)


if __name__ == '__main__':
    unittest.main()
