"""The gradient clip's partial sums keep their own chunking (Tiling.norm_elems): gradnorm_partial stores each chunk's sum in fp32, so the
chunks decide the norm's last bit and, on a clipped step, the coefficient that scales every gradient. A tiling that moves AdamW's chunks
for speed must leave the norm's chunks, and with them its weights, as they were."""
import unittest

import _path  # noqa: F401
from kvmfe.ir import FrontendError
from kvmfe.lower import Tiling
from kvmfe.nn import ModelConfig
from kvmfe.pipeline import compile_training
from kvmfe.train import TrainConfig

MODEL = '{"name":"clip-chunks","vocab":64,"d_model":192,"layers":1,"heads":3,"seq":32,"batch":1,"ffn_mult":4}'


def tasks(tiling, op):
    """(op args that place each task: its buffers and ranges) of every `op` task in program order."""
    c = compile_training(ModelConfig.load(MODEL), TrainConfig(steps=2), tiling, reuse='step', fuse=False)
    keep = ('g', 'p', 'partial', 'slot', 'e0', 'e1', 'nslots')
    return [tuple((k, t['args'][k]) for k in keep if k in t['args']) for g in c.spec['graphs'] for t in g['tasks'] if t['op'] == op]


class ClipChunks(unittest.TestCase):
    def test_a100_norm_keeps_the_base_chunks_while_adamw_splits(self):
        a100, base = Tiling.preset('a100'), Tiling.preset('a100', opt_elems=262144, norm_elems=0)
        self.assertEqual((a100.opt_elems, a100.norm_elems), (16384, 262144))
        self.assertEqual(tasks(a100, 'gradnorm_partial'), tasks(base, 'gradnorm_partial'))
        self.assertEqual(tasks(a100, 'gradnorm_finish'), tasks(base, 'gradnorm_finish'))
        self.assertGreater(len(tasks(a100, 'adamw')), len(tasks(base, 'adamw')))   # AdamW itself takes the finer chunks

    def test_zero_follows_opt_elems(self):
        t = Tiling.preset('a100', opt_elems=16384, norm_elems=0)
        self.assertEqual(len(tasks(t, 'gradnorm_partial')), len(tasks(t, 'adamw')))   # one partial a chunk, as before norm_elems

    def test_norm_elems_is_validated(self):
        for bad in (-1, 1.5, '262144'):
            with self.assertRaises(FrontendError):
                Tiling.preset('a100', norm_elems=bad)


if __name__ == '__main__':
    unittest.main()
