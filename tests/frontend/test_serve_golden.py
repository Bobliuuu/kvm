"""demo/serve_golden.py (PyTorch decode golden for serving at sizes the CPU VM cannot run) against the independent numpy decode oracle.

Skipped without torch. The oracle (oracle_gpt.greedy) is a separate float64 implementation, so agreement here ties the PyTorch golden's
forward (layernorm, RoPE, causal attention, GELU, head) to the same model the frontend builds; the 162M run is then the same code at scale.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

import _path  # noqa: F401
import oracle_gpt as ref_gpt
from kvmfe.nn import GPT, ModelConfig
from kvmfe.optim import OptimConfig

try:
    import torch  # noqa: F401
    sys.path.insert(0, str(_path.ROOT / 'demo'))
    import serve_golden as sg
except ImportError:                                            # pragma: no cover
    sg = None

CFG = ModelConfig(name='t', vocab=32, d_model=32, layers=2, heads=2, seq=32, batch=1)
PROMPT = (1, 8, 11)


def as_arrays(cfg, params):
    return {n: np.array(params[n], dtype=np.float64).reshape(s.rows, s.cols) for n, s in GPT(cfg).specs.items()}


@unittest.skipIf(sg is None, 'torch not installed')
class ServeGolden(unittest.TestCase):
    def test_initial_weights_are_the_frontends(self):
        init = sg.init_weights(CFG)
        for n, v in as_arrays(CFG, ref_gpt.init_params(CFG)).items():
            np.testing.assert_array_equal(init[n].astype(np.float64), v, err_msg=n)

    def test_decode_matches_numpy_oracle_untrained_and_trained(self):
        _, hist = ref_gpt.train_reference(CFG, 2, OptimConfig())
        for k in (0, 2):
            toks = ref_gpt.greedy(CFG, hist[k], PROMPT, 4)
            r = sg.check_request(CFG, as_arrays(CFG, hist[k]), PROMPT, toks)
            self.assertTrue(r['ok'], r)
            self.assertEqual(r['exact'], 4, f'weights after {k} updates')

    def test_wrong_tokens_and_wrong_weights_are_caught(self):
        _, hist = ref_gpt.train_reference(CFG, 2, OptimConfig())
        toks = ref_gpt.greedy(CFG, hist[2], PROMPT, 4)
        w0 = as_arrays(CFG, hist[0])
        self.assertTrue(sg.check_request(CFG, w0, PROMPT, ref_gpt.greedy(CFG, hist[0], PROMPT, 4))['ok'])
        self.assertFalse(sg.check_request(CFG, w0, PROMPT, toks)['ok'], 'tokens of version 2 accepted against version-0 weights')
        bad = list(toks)
        bad[1] = (bad[1] + 1) % CFG.vocab
        self.assertFalse(sg.check_request(CFG, as_arrays(CFG, hist[2]), PROMPT, bad)['ok'])

    def test_near_tie_is_accepted_only_inside_the_tf32_tolerance(self):
        w = as_arrays(CFG, ref_gpt.init_params(CFG))
        ref = sg.logits(CFG, w, list(PROMPT), 'f64')[-1]
        runner_up = int(ref.argsort(descending=True)[1])
        r = sg.check_request(CFG, w, PROMPT, [runner_up])
        gap = r['tokens'][0]['gap']
        self.assertEqual(r['ok'], gap <= r['tokens'][0]['tie_tol'])
        self.assertFalse(sg.check_request(CFG, w, PROMPT, [runner_up], precise=True)['ok'], 'precise (3xTF32) runs allow no near ties')


if __name__ == '__main__':
    unittest.main()
