import unittest

import _path  # noqa: F401
from kvmfe.nn import ModelConfig
from kvmfe.train import batch_stream, batch_tokens


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


if __name__ == '__main__':
    unittest.main()
