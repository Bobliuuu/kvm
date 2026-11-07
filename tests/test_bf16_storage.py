"""bf16 STORAGE (ABI dtype 2) in the compiler and the CPU VM: buffers, init rounding, round_bf16 as narrow/widen, matmul_tile_bf16 on
bf16-stored operands and outputs, and the dtype checks that keep every other op off bf16 buffers."""
import struct
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime'))
import compile_workload as cw
import workload_vm as vm
from ref_extra import bf16_round
from test_compiler import task
from test_workload_vm import run_one


def bufs(**kw):
    return [dict(name=n, rows=r, cols=c, dtype=d, init=init) for n, (r, c, d, init) in kw.items()]


def fill(v):
    return {'kind': 'fill', 'value': v}


def values(vs):
    return {'kind': 'values', 'values': vs}


def is_bf16(x):
    return struct.unpack('<I', struct.pack('<f', x))[0] & 0xFFFF == 0


class Bf16Buffers(unittest.TestCase):
    def test_dtype_code_and_init_rounding(self):
        spec = {'buffers': bufs(x=(1, 4, 'bf16', values([1.0, 1 + 2 ** -10, 3.14159265, -1e-3]))), 'graphs': [{'name': 'g', 'tasks': [task('t', 'noop')]}]}
        c = cw.compile_spec(spec)
        self.assertEqual(cw.DTYPES['bf16'], 2)
        self.assertEqual(c.buffers[0].values, [bf16_round(v) for v in (1.0, 1 + 2 ** -10, 3.14159265, -1e-3)])
        self.assertEqual(c.buffers[0].values[1], 1.0)
        self.assertEqual(c.buffers[0].values[2], 3.140625)

    def test_seeded_init_rejected(self):
        spec = {'buffers': bufs(x=(1, 4, 'bf16', {'kind': 'seeded', 'seed': 1, 'scale': 1.0})), 'graphs': [{'name': 'g', 'tasks': [task('t', 'noop')]}]}
        with self.assertRaises(cw.CompileError):
            cw.compile_spec(spec)

    def test_f32_ops_reject_bf16_buffers(self):
        spec = {'buffers': bufs(x=(1, 4, 'bf16', fill(1.0)), y=(1, 4, 'f32', fill(0.0))),
                'graphs': [{'name': 'g', 'tasks': [task('t', 'gelu_fwd', {'x': 'x', 'out': 'y', 'e0': 0, 'e1': 4})]}]}
        with self.assertRaisesRegex(cw.CompileError, "needs a f32 buffer"):
            cw.compile_spec(spec)


class Bf16Ops(unittest.TestCase):
    def test_round_bf16_narrow_widen(self):
        src = [1.0, 1 + 2 ** -10, 3.14159265, 1e10, -2.5e-5, 0.1]
        out = run_one('round_bf16', {'x': 'x', 'out': 'n', 'e0': 0, 'e1': 6},
                      bufs(x=(1, 6, 'f32', values(src)), n=(1, 6, 'bf16', fill(0.0))))
        self.assertEqual(list(out['n']), [bf16_round(v) for v in src])
        self.assertTrue(all(is_bf16(v) for v in out['n']))
        back = run_one('round_bf16', {'x': 'n', 'out': 'w', 'e0': 0, 'e1': 6},
                       bufs(n=(1, 6, 'bf16', values(src)), w=(1, 6, 'f32', fill(0.0))))
        self.assertEqual(list(back['w']), [bf16_round(v) for v in src])   # widen is exact

    def _mm(self, dta, dtb, dto, dtr=None, dtp=None, flags=0):
        M, N, K = 5, 7, 9
        a = [((i * 7) % 13 - 6) * 0.1234567 for i in range(M * K)]
        b = [((i * 5) % 11 - 5) * 0.0876543 for i in range(K * N)]
        r = [((i * 3) % 7 - 3) * 0.3 for i in range(M * N)]
        bs = bufs(a=(M, K, dta, values(a)), b=(K, N, dtb, values(b)), o=(M, N, dto, fill(0.0)))
        args = {'a': 'a', 'b': 'b', 'out': 'o', 'bias': None, 'res': None, 'pre': None, 'm0': 0, 'n0': 0, 'M': M, 'N': N, 'K': K, 'flags': flags, 'alpha': 1.0}
        if dtr:
            bs += bufs(r=(M, N, dtr, values(r)))
            args['res'] = 'r'
        if dtp:
            bs += bufs(p=(M, N, dtp, fill(0.0)))
            args['pre'] = 'p'
        got = run_one('matmul_tile_bf16', args, bs)
        ra = [bf16_round(v) for v in a]
        rb = [bf16_round(v) for v in b]
        rr = [bf16_round(v) for v in r]
        return got, ra, rb, rr, (M, N, K)

    def test_matmul_bf16_storage_matches_rounded_reference(self):
        for dta, dtb, dto in (('bf16', 'bf16', 'bf16'), ('bf16', 'f32', 'f32'), ('f32', 'bf16', 'bf16'), ('f32', 'f32', 'bf16')):
            got, ra, rb, rr, (M, N, K) = self._mm(dta, dtb, dto)
            for m in range(M):
                for n in range(N):
                    want = sum(ra[m * K + k] * rb[k * N + n] for k in range(K))
                    v = got['o'][m * N + n]
                    if dto == 'bf16':
                        self.assertTrue(is_bf16(v))
                        self.assertAlmostEqual(v, want, delta=abs(want) * 2 ** -8 + 1e-6)
                    else:
                        self.assertAlmostEqual(v, want, delta=1e-5)

    def test_matmul_bf16_residual_and_pre_storage(self):
        got, ra, rb, rr, (M, N, K) = self._mm('bf16', 'bf16', 'bf16', 'bf16', 'bf16', flags=8)
        self.assertTrue(all(is_bf16(v) for v in got['o']) and all(is_bf16(v) for v in got['p']))
        for m in range(M):
            for n in range(N):
                pre = sum(ra[m * K + k] * rb[k * N + n] for k in range(K))
                self.assertAlmostEqual(got['p'][m * N + n], pre, delta=abs(pre) * 2 ** -8 + 1e-6)

    def test_unrounded_input_values_are_rounded_at_init(self):
        got, ra, rb, rr, (M, N, K) = self._mm('bf16', 'bf16', 'f32')
        # operands are bf16 at init, so the f32 result equals the product of the rounded values
        self.assertAlmostEqual(got['o'][0], sum(ra[k] * rb[k * N] for k in range(K)), delta=1e-5)


if __name__ == '__main__':
    unittest.main()
