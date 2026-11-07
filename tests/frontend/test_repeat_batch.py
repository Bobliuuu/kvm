import unittest

import _path  # noqa: F401
from kvmfe.nn import ModelConfig
import math

from kvmfe.train import batch_stream, batch_tokens, stream_entropy


class RepeatBatch(unittest.TestCase):
    def test_default_stream_changes_every_step(self):
        cfg = ModelConfig(vocab=64, seq=32, batch=2)
        self.assertNotEqual(batch_tokens(cfg, 0), batch_tokens(cfg, 1))

    def test_repeat_batch_is_step_zero_every_step(self):
        cfg = ModelConfig(vocab=64, seq=32, batch=2, repeat_batch=True)
        base = ModelConfig(vocab=64, seq=32, batch=2)
        for s in (0, 1, 7):
            self.assertEqual(batch_tokens(cfg, s), batch_tokens(base, 0))
        ids, _ = batch_stream(cfg, 3)
        n = cfg.tokens
        self.assertEqual(ids[:n], ids[n:2 * n])
        self.assertEqual(ids[:n], ids[2 * n:])

    def test_json_config_carries_the_flag(self):
        self.assertTrue(ModelConfig.load('{"name":"x","repeat_batch":true}').repeat_batch)


class SubsetStream(unittest.TestCase):
    CFG = dict(vocab=4096, seq=32, batch=2, data_vocab=64)

    def test_default_is_unchanged_and_subset_is_confined(self):
        base = ModelConfig(vocab=64, seq=32, batch=2)
        self.assertEqual(batch_tokens(base, 3), batch_tokens(ModelConfig(vocab=64, seq=32, batch=2, data_vocab=0), 3))
        ids, tgt = batch_stream(ModelConfig(**self.CFG, data_noise=0.2), 20)
        self.assertLess(max(ids + tgt), 64)

    def test_noiseless_subset_is_the_bigram_and_targets_follow_ids(self):
        cfg = ModelConfig(**self.CFG)
        ids, tgt = batch_tokens(cfg, 0)
        self.assertTrue(all(y == (5 * x + 3) % 64 for x, y in zip(ids, tgt)))
        self.assertEqual(stream_entropy(cfg), 0.0)
        noisy = ModelConfig(**self.CFG, data_noise=0.3)
        i, t = batch_tokens(noisy, 5)
        self.assertEqual(batch_tokens(noisy, 5), (i, t))                       # reproducible
        self.assertEqual(i[1:cfg.seq], t[:cfg.seq - 1])                       # the next input is the last target
        self.assertGreater(sum(y != (5 * x + 3) % 64 for x, y in zip(i, t)), 0)

    def test_entropy_matches_the_noise_model(self):
        cfg = ModelConfig(**self.CFG, data_noise=0.1)
        h = stream_entropy(cfg)
        hit = 0.9 + 0.1 / 64
        self.assertAlmostEqual(h, -hit * math.log(hit) - 63 * (0.1 / 64) * math.log(0.1 / 64), places=12)
        self.assertTrue(0 < h < math.log(64))
        self.assertAlmostEqual(stream_entropy(ModelConfig(**self.CFG, data_noise=1.0)), math.log(64), places=12)

    def test_bad_stream_is_rejected(self):
        for bad in (dict(data_vocab=5000), dict(data_noise=1.5)):
            with self.assertRaises(Exception):
                ModelConfig(**{**self.CFG, **bad})


if __name__ == '__main__':
    unittest.main()
