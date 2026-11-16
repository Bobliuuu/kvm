"""Decode-step lowering options of compile_serving (serve.py): bf16 / transposed serving weights, bf16 storage with storage-aware
decode ops, the one-row GEMV with folded norms and SwiGLU, and split-context decode attention. Every combination generates the
float64 oracle's greedy tokens on the reference VM in random task orders."""
import collections
import unittest

import _path  # noqa: F401
import oracle_models as orc
from kvmfe.emit import merge
from kvmfe.ir import FrontendError
from kvmfe.lower import Tiling
from kvmfe.models import Llama, LlamaConfig
from kvmfe.pipeline import compile_training
from kvmfe.refvm import Memory, execute
from kvmfe.serve import ServeRequest, compile_serving
from kvmfe.train import TrainConfig, weights_version

TILING = Tiling.preset('cpu')
PROMPT, N_NEW = (3, 17, 5, 40), 4


def cfg(precision):
    return LlamaConfig(name='dec', vocab=64, d_model=64, layers=2, heads=4, kv_heads=2, ffn=96, seq=16, precision=precision)


def with_serving_copy(train_spec, sspec, transposed):
    """The bf16 serving copy a publish step makes from the trainer's fp32 master (rounded; transposed for 'bf16t'), filled in from the
    master's initial values under its own name, so the merged spec keeps the trainer's fp32 master."""
    m0 = Memory(train_spec)
    ren = {}
    for buf in sspec['buffers']:
        if buf.get('dtype') == 'bf16' and buf['name'].startswith('w.'):
            src = m0.bufs[buf['name']]
            R, C = src.rows, src.cols
            vals = [float(src.a[i]) for i in range(R * C)]
            if transposed:
                vals = [vals[r * C + c] for c in range(C) for r in range(R)]
            ren[buf['name']] = 's' + buf['name']
            buf['name'] = 's' + buf['name']
            buf['init'] = {'kind': 'values', 'values': vals}
    for g in sspec['graphs']:
        for t in g['tasks']:
            for k, v in list((t.get('args') or {}).items()):
                if v in ren:
                    t['args'][k] = ren[v]
    return merge(train_spec, sspec)


def decode_tasks(spec, step='dec0'):
    ts = [t for g in spec['graphs'] for t in g['tasks']]
    first = next(i for i, t in enumerate(ts) if f':{step}.' in t['id'])
    return ts[first:]


def depth(tasks):
    idx = {t['id']: i for i, t in enumerate(tasks)}
    lv = {}
    for i, t in enumerate(tasks):
        lv[i] = 1 + max((lv[idx[d]] for d in t.get('after', []) if d in idx), default=-1)
    return max(lv.values()) + 1


class DecodeLowering(unittest.TestCase):
    CASES = [  # precision, reuse, weights, gemv_rows, attn_splits
        ('bf16', 'shape', 'bf16', 0, 0),
        ('bf16', 'none', 'bf16t', 0, 0),
        ('bf16s', 'none', 'bf16', 0, 0),
        ('bf16s', 'none', 'bf16t', 32, 0),
        ('bf16s', 'none', 'bf16t', 16, 3),
        ('bf16s', 'none', 'bf16t', 32, 16),   # more splits than keys: empty splits
    ]

    def test_every_option_matches_the_oracle(self):
        for prec, reuse, weights, rows, splits in self.CASES:
            with self.subTest(precision=prec, reuse=reuse, weights=weights, gemv_rows=rows, attn_splits=splits):
                c = cfg(prec)
                model = Llama(c)
                train = compile_training(c, TrainConfig(steps=1), TILING)
                _, sspec = compile_serving(model, ServeRequest('sv', PROMPT, N_NEW, weights_version(0)), TILING, reuse=reuse,
                                           weights=weights, gemv_rows=rows, attn_splits=splits)
                spec = with_serving_copy(train.spec, sspec, weights == 'bf16t')
                want = orc.greedy(model, orc.params(model), PROMPT, N_NEW)
                for seed in (0, 1):
                    mem = execute(spec, order='random', seed=seed)
                    self.assertEqual([int(mem.bufs[f'sv.cur{i}'].a[0]) for i in range(1, N_NEW + 1)], want, f'seed {seed}')

    def test_gemv_folds_norms_and_swiglu(self):
        c = cfg('bf16s')
        req = ServeRequest('sv', (1,), 1, 0, context=5)
        _, base = compile_serving(Llama(c), req, TILING, reuse='none', weights='bf16t')
        _, spec = compile_serving(Llama(c), req, TILING, reuse='none', weights='bf16t', gemv_rows=32)
        ops = collections.Counter(t['op'] for t in decode_tasks(spec))
        self.assertNotIn('matmul_tile_bf16', ops)
        self.assertNotIn('rmsnorm_fwd', ops)
        self.assertNotIn('silu_mul_fwd', ops)
        pro = collections.Counter(t['args']['pro'] for t in decode_tasks(spec) if t['op'] == 'gemv_bf16')
        self.assertTrue(pro[1] and pro[2], pro)   # norms before q/k/v, gate/up and the head; the SwiGLU before down
        self.assertLess(depth(decode_tasks(spec)), depth(decode_tasks(base)))

    def test_split_attention_tasks_and_buffer(self):
        c = cfg('bf16s')
        _, spec = compile_serving(Llama(c), ServeRequest('sv', (1,), 1, 0, context=9), TILING, reuse='none', weights='bf16t', attn_splits=4)
        ops = collections.Counter(t['op'] for t in decode_tasks(spec))
        self.assertEqual(ops['attn_decode'], 0)
        self.assertEqual(ops['attn_decode_part'], c.layers * c.kv_heads * 4)
        self.assertEqual(ops['attn_decode_merge'], c.layers * c.kv_heads)
        parts = [b for b in spec['buffers'] if b['name'].endswith('.attnpart')]
        self.assertEqual(len(parts), c.layers)
        self.assertTrue(all(b['rows'] == 4 * c.heads and b['cols'] == c.d_model // c.heads + 2 for b in parts))

    def test_context_presizes_the_cache(self):
        c = cfg('bf16s')
        _, spec = compile_serving(Llama(c), ServeRequest('sv', (7,), 2, 0, context=10), TILING)
        bufs = {b['name']: b for b in spec['buffers']}
        self.assertEqual(bufs['sv.lens']['init'], {'kind': 'fill', 'value': 10})
        self.assertEqual(bufs['sv.kc0']['rows'], c.kv_heads * (10 + 2))
        self.assertEqual(bufs['sv.kc0'].get('dtype'), 'bf16')   # bf16s: the cache is bf16 storage
        mem = execute(spec)                                     # zero cache rows: the graph must still run
        self.assertEqual(int(mem.bufs['sv.lens'].a[0]), 12)

    def test_option_errors(self):
        m = Llama(cfg('bf16s'))
        for kw in (dict(weights='fp8'), dict(gemv_rows=32), dict(weights='bf16t', gemv_rows=32), dict(weights='bf16t', reuse='none', gemv_rows=24),
                   dict(attn_splits=-1)):
            with self.subTest(**kw), self.assertRaises(FrontendError):
                compile_serving(m, ServeRequest('sv', (1,), 1, 0, context=3), TILING, **kw)
        for bad in (ServeRequest('sv', (1, 2), 1, 0, context=3), ServeRequest('sv', (1,), 1, 0, context=-1)):
            with self.subTest(req=bad), self.assertRaises(FrontendError):
                compile_serving(m, bad, TILING)


if __name__ == '__main__':
    unittest.main()
