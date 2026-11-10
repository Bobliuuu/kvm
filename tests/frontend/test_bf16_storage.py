"""Real bf16 storage (precision 'bf16s') on gpt_tiny: stored tensors are bf16 buffers, the workload compiles under the runtime dtype
checks, the CPU golden is order-independent, the loss curve tracks fp32 and the bf16 emulation, and bf16 bytes are half of fp32 bytes."""
import dataclasses
import struct
import sys
import unittest

import _path  # noqa: F401
from kvmfe import storage
from kvmfe.emit import lower_program
from kvmfe.lower import Tiling
from kvmfe.nn import GPT, PRESETS
from kvmfe.pipeline import compile_training
from kvmfe.train import TrainConfig, build_training

sys.path.insert(0, str(_path.ROOT / 'runtime'))
import compile_workload as cw  # noqa: E402

BASE = PRESETS['gpt_tiny']
TILING = Tiling.preset('cpu')


def comp(prec, steps, **kw):
    return compile_training(dataclasses.replace(BASE, precision=prec), TrainConfig(steps=steps), TILING, **kw)


def losses(c, mem, steps):
    return [mem.bufs[f'loss.{s}'].a[0] for s in range(steps)]


class Bf16Storage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.steps = 20
        cls.f32, cls.emu, cls.sto = comp('f32', cls.steps), comp('bf16', cls.steps), comp('bf16s', cls.steps)
        cls.mem = cls.sto.run()

    def test_buffers_are_bf16_and_values_representable(self):
        bf = [b for b in self.sto.spec['buffers'] if b.get('dtype') == 'bf16']
        self.assertTrue(bf)
        names = {b['name'] for b in bf}
        self.assertTrue(any(n.startswith('wb.') for n in names), 'weight shadows')
        for n in names:
            for v in self.mem.bufs[n].a:
                self.assertEqual(struct.unpack('<I', struct.pack('<f', v))[0] & 0xFFFF, 0, f'{n} holds a non-bf16 value {v}')

    def test_masters_stay_fp32(self):
        for b in self.sto.spec['buffers']:
            if b['name'].startswith(('w.', 'm.', 'v.')):
                self.assertNotEqual(b.get('dtype'), 'bf16', b['name'])

    def test_optimizer_chunks_align_to_the_mma_tile_and_f32_keeps_the_64_row_tile(self):
        from kvmfe.opset import MMA_TILE
        def rows(prec):
            ops = build_training(GPT(dataclasses.replace(BASE, precision=prec)), TrainConfig(steps=1)).b.ops
            ops = storage.apply(ops)[0] if prec == 'bf16s' else ops
            return {op.attrs.get('tile_rows') for op in ops if op.kind in ('adamw', 'grad_clip')}
        self.assertEqual(rows('bf16s'), {MMA_TILE[0]})
        self.assertEqual(rows('f32'), {None})

    def test_runtime_dtype_checks_accept_workload(self):
        cw.compile_spec(self.sto.spec, data_dir=None)

    def test_order_independent(self):
        from kvmfe.refvm import execute
        a = execute(self.sto.spec, order='random', seed=3)
        self.assertEqual(losses(self.sto, a, self.steps), losses(self.sto, self.mem, self.steps))

    def test_loss_tracks_fp32_and_emulation(self):
        lf, le, ls = (losses(c, c.run(), self.steps) for c in (self.f32, self.emu, self.sto))
        self.assertLess(ls[-1], ls[0] - 0.5)
        for a, b in zip(ls, le):
            self.assertAlmostEqual(a, b, delta=2e-2)
        for a, b in zip(ls, lf):
            self.assertAlmostEqual(a, b, delta=5e-2)
        self.assertAlmostEqual(ls[0], le[0], delta=1e-3)

    def test_bf16_bytes_are_half(self):
        plan = self.sto.low.plan
        n = sum(b.rows * b.cols for b in plan.buffers.values() if b.dtype == 'bf16')
        self.assertGreater(n, 0)
        self.assertEqual(plan.stats['persistent_bytes'] + plan.stats['pool_bytes'],
                         sum(b.rows * b.cols * (2 if b.dtype == 'bf16' else 4) for b in plan.buffers.values()))

    def test_no_op_reads_bf16_without_support(self):
        import ref_extra   # runtime/: the device ops' bf16-storage arguments (the same table the CPU golden and the compiler use)
        ok = {'matmul_tile_bf16', 'matmul_mma_bf16', 'round_bf16'}
        aware = {op: set(args) for op, (args, _) in ref_extra._BF16_STORAGE.items()}
        bf = {b['name'] for b in self.sto.spec['buffers'] if b.get('dtype') == 'bf16'}
        for g in self.sto.spec['graphs']:
            for t in g['tasks']:
                if t['op'] in ok:
                    continue
                for k, v in t.get('args', {}).items():
                    if k in aware.get(t['op'], ()):
                        continue
                    self.assertNotIn(v, bf, f"{t['op']} arg {k}")

    def test_attention_reads_and_writes_bf16_storage(self):
        # causal attention is storage-aware: no fp32 scratch (widen before / narrow after) around it, and attention_bwd reads the bf16 saved output
        ops = build_training(GPT(dataclasses.replace(BASE, precision='bf16s')), TrainConfig(steps=1)).b.ops
        new, _ = storage.apply(ops)
        att = [op for op in new if op.kind in ('attention', 'attention_bwd')]
        self.assertTrue(att)
        for op in att:
            for n, t in {**op.ins, **op.outs}.items():
                self.assertFalse(t.name.endswith('.f32'), f'{op.kind} {n} goes through fp32 scratch {t.name}')
        fwd = [op for op in att if op.kind == 'attention']
        self.assertTrue(all(op.outs['out'].dtype == 'bf16' for op in fwd))
        self.assertTrue(all(op.ins['o'].dtype == 'bf16' for op in att if op.kind == 'attention_bwd'))
        names = {b['name']: b.get('dtype') for b in self.sto.spec['buffers']}
        for g in self.sto.spec['graphs']:
            for t in g['tasks']:
                if t['op'] == 'attn_fwd':
                    self.assertEqual(names[t['args']['o']], 'bf16')

    def test_no_bf16_to_bf16_operand_copies(self):
        # the output of a model cast is already bf16: an operand copy of it would be a full-tensor round trip (206M elements/step at 162M)
        ops = build_training(GPT(dataclasses.replace(BASE, precision='bf16s')), TrainConfig(steps=1)).b.ops
        new, st = storage.apply(ops)
        cast_out = {op.outs['out'].uid for op in ops if op.kind == 'cast' and op.attrs.get('dtype') == 'bf16'}
        for op in new:
            if op.kind == 'cast' and op.name.startswith('narrow.') and op.outs['out'].name.endswith('.b16'):
                self.assertNotIn(op.ins['x'].uid, cast_out, op.name)
        self.assertGreater(st['redirected'], 0)

    def test_weights_false_has_no_shadows(self):
        ops = build_training(GPT(dataclasses.replace(BASE, precision='bf16s')), TrainConfig(steps=1)).b.ops
        new, st = storage.apply(ops, weights=False)
        self.assertEqual(st['shadows'], 0)
        low = lower_program(new, TILING, 'shape', 0, 'act')
        self.assertFalse(any(n.startswith('wb.') for n in low.plan.buffers))


class Bf16StorageModelZoo(unittest.TestCase):
    """The same storage pass on the other model families (Llama: RMSNorm / RoPE / GQA / SwiGLU, MLP, ViT: bidirectional attention):
    the loss curve of real bf16 storage tracks the bf16 emulation (which rounds the same values in fp32 containers)."""

    def run_family(self, make):
        steps, out = 6, {}
        for prec in ('f32', 'bf16', 'bf16s'):
            c = compile_training(make(prec), TrainConfig(steps=steps, loss_scale=128.0 if prec != 'f32' else 1.0), TILING)
            out[prec] = (c, losses(c, c.run(), steps))
        self.assertTrue(any(b.get('dtype') == 'bf16' for b in out['bf16s'][0].spec['buffers']))
        cw.compile_spec(out['bf16s'][0].spec, data_dir=None)     # accepted by the compiler's dtype checks
        for a, b, f in zip(out['bf16s'][1], out['bf16'][1], out['f32'][1]):
            self.assertAlmostEqual(a, b, delta=5e-3)
            self.assertAlmostEqual(a, f, delta=5e-2)

    def test_llama(self):
        from kvmfe.models import LlamaConfig
        self.run_family(lambda p: LlamaConfig(name='llama_t', precision=p))

    def test_mlp(self):
        from kvmfe.models import MLPConfig
        self.run_family(lambda p: MLPConfig(name='mlp_t', precision=p))

    def test_vit(self):
        from kvmfe.models import ViTConfig
        self.run_family(lambda p: ViTConfig(name='vit_t', precision=p))


if __name__ == '__main__':
    unittest.main()
