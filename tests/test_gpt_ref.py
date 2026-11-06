"""CPU reference of the GPT operator library: hand-checked values, finite-difference gradient checks, end-to-end step."""
import math
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime'))
import ref_gpt
import ref_ops
import workload_vm as vm
from compile_workload import CompileError, compile_spec
from test_compiler import all_ops, task, wl


def vals(name, rows, cols, v, dtype='f32'):
    return {'name': name, 'rows': rows, 'cols': cols, 'dtype': dtype, 'init': {'kind': 'values', 'values': list(v)}}


def zeros(name, rows, cols=1, dtype='f32'):
    return {'name': name, 'rows': rows, 'cols': cols, 'dtype': dtype}


def run(buffers, *tasks):
    """Run tasks (op, args) one after another on the reference VM; return the final buffers."""
    ts = [task(f't{i}', op, args, after=[f't{i - 1}'] if i else []) for i, (op, args) in enumerate(tasks)]
    runtime, _ = vm.run_spec(wl(ts, buffers=buffers), deterministic=True, ops=all_ops())
    return runtime.buffers


def rnd(n, seed, lo=-1.0, hi=1.0):
    g = random.Random(seed)
    return [ref_ops.f32(g.uniform(lo, hi)) for _ in range(n)]


def numgrad(f, x, eps=1e-3):
    """Central finite differences of scalar f(list) w.r.t. each element of x (double precision inside f)."""
    g = []
    for i in range(len(x)):
        a, b = list(x), list(x)
        a[i] += eps
        b[i] -= eps
        g.append((f(a) - f(b)) / (2 * eps))
    return g


def close(test, got, want, tol, msg=''):
    scale = max(1.0, max(abs(w) for w in want))
    for i, (g, w) in enumerate(zip(got, want)):
        test.assertLessEqual(abs(g - w), tol * scale, f'{msg} element {i}: {g} vs {w}')


class PhiloxTests(unittest.TestCase):
    def test_random123_known_answer(self):
        self.assertEqual(ref_gpt.philox4x32_10([0, 0, 0, 0], [0, 0]), [0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8])
        self.assertEqual(ref_gpt.philox4x32_10([0xFFFFFFFF] * 4, [0xFFFFFFFF] * 2), [0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD])

    def test_uniform_range_and_keying(self):
        u = [ref_gpt.philox_uniform(5, 0, r) for r in range(2000)]
        self.assertTrue(all(0 <= x < 1 for x in u))
        self.assertAlmostEqual(sum(u) / len(u), 0.5, delta=0.03)
        self.assertNotEqual(ref_gpt.philox_uniform(5, 0, 1), ref_gpt.philox_uniform(5, 1, 1))
        self.assertNotEqual(ref_gpt.philox_uniform(5, 0, 1), ref_gpt.philox_uniform(6, 0, 1))


class ElementwiseTests(unittest.TestCase):
    def test_embedding_forward_backward(self):
        wte = [1, 2, 3, 4, 5, 6]
        b = run([vals('ids', 3, 1, [2, 0, 2], 'i32'), vals('wte', 3, 2, wte), vals('wpe', 2, 2, [10, 20, 30, 40]), zeros('out', 3, 2),
                 vals('dout', 3, 2, [1, 1, 2, 2, 3, 3]), zeros('dt', 3, 2), zeros('dp', 2, 2)],
                ('embedding_fwd', dict(ids='ids', wte='wte', wpe='wpe', out='out', r0=0, r1=3, T=2, d=2)),
                ('embedding_bwd', dict(ids='ids', dout='dout', dtab='dt', N=3, mod=0, v0=0, v1=3, d=2, acc=0)),
                ('embedding_bwd', dict(ids=None, dout='dout', dtab='dp', N=3, mod=2, v0=0, v1=2, d=2, acc=0)))
        self.assertEqual(list(b['out']), [15, 26, 31, 42, 15, 26])
        self.assertEqual(list(b['dt']), [2, 2, 0, 0, 4, 4])          # token 2 appears in rows 0 and 2
        self.assertEqual(list(b['dp']), [4, 4, 2, 2])                # positions: rows {0,2} -> 0, row {1} -> 1

    def test_embedding_rejects_bad_token(self):
        with self.assertRaises(RuntimeError) as cm:
            run([vals('ids', 1, 1, [3], 'i32'), vals('wte', 3, 1, [1, 2, 3]), zeros('out', 1)],
                ('embedding_fwd', dict(ids='ids', wte='wte', wpe=None, out='out', r0=0, r1=1, T=1, d=1)))
        self.assertIn('outside [0,3)', str(cm.exception))

    def test_gelu_matches_gradient(self):
        x = rnd(8, 1, -3, 3)
        b = run([vals('x', 1, 8, x), zeros('y', 1, 8), vals('dy', 1, 8, [1] * 8), zeros('dx', 1, 8), vals('r', 1, 8, x)],
                ('gelu_fwd', dict(x='x', out='y', e0=0, e1=8)), ('gelu_bwd', dict(x='x', dy='dy', dx='dx', e0=0, e1=8, acc=0)),
                ('residual_add', dict(a='x', b='y', out='r', e0=0, e1=8)))
        self.assertAlmostEqual(b['y'][0], 0.5 * x[0] * (1 + math.tanh(0.7978845608028654 * (x[0] + 0.044715 * x[0] ** 3))), places=6)
        close(self, list(b['dx']), numgrad(lambda v: sum(ref_gpt._gelu(t) for t in v), x, 1e-4), 2e-3)
        close(self, list(b['r']), [p + q for p, q in zip(x, b['y'])], 1e-6)

    def test_zero_counter_colsum_select_gather(self):
        b = run([vals('x', 2, 3, [1, 2, 3, 4, 5, 6]), zeros('cs', 3), vals('c', 1, 3, [1, 2, 3], 'i32'), vals('z', 1, 3, [9, 9, 9]),
                 vals('src', 1, 6, [10, 11, 12, 13, 14, 15], 'i32'), vals('step', 1, 1, [1], 'i32'), zeros('win', 1, 2, 'i32'),
                 vals('idx', 2, 1, [1, 0], 'i32'), zeros('rows', 2, 3)],
                ('colsum', dict(x='x', out='cs', R=2, c0=0, c1=3, acc=0)), ('colsum', dict(x='x', out='cs', R=2, c0=1, c1=2, acc=1)),
                ('counter_inc', dict(x='c', i0=1, i1=3, delta=5)), ('zero_buf', dict(x='z', e0=1, e1=3)),
                ('select_window', dict(src='src', step='step', dst='win', i0=0, i1=2, stride=2, off=1)),
                ('gather_rows', dict(src='x', idx='idx', dst='rows', r0=0, r1=2, d=3)))
        self.assertEqual(list(b['cs']), [5, 7 + 7, 9])
        self.assertEqual(list(b['c']), [1, 7, 8])
        self.assertEqual(list(b['z']), [9, 0, 0])
        self.assertEqual(list(b['win']), [13, 14])                   # base = step*stride + off = 3
        self.assertEqual(list(b['rows']), [4, 5, 6, 1, 2, 3])

    def test_rope_backward_inverts_forward_and_position_offset(self):
        x = rnd(2 * 2 * 4, 2)
        b = run([vals('x', 2, 8, x), zeros('y', 2, 8), zeros('z', 2, 8), vals('pos', 1, 1, [3], 'i32'), zeros('y3', 2, 8)],
                ('rope_fwd', dict(x='x', out='y', pos=None, r0=0, r1=2, T=2, H=2, hd=4, theta=10000.0)),
                ('rope_bwd', dict(x='y', out='z', pos=None, r0=0, r1=2, T=2, H=2, hd=4, theta=10000.0)),
                ('rope_fwd', dict(x='x', out='y3', pos='pos', r0=0, r1=2, T=2, H=2, hd=4, theta=10000.0)))
        close(self, list(b['z']), x, 1e-6)
        self.assertEqual(list(b['y'][:4]), x[:4])                    # position 0 is the identity rotation
        self.assertNotEqual(list(b['y3'][:4]), x[:4])


class MatmulNormTests(unittest.TestCase):
    def test_matmul_tile_flags_epilogue_and_tile_clipping(self):
        A = [1, 2, 3, 4, 5, 6]            # 2x3
        B = [1, 0, 0, 1, 1, 1]            # 3x2
        base = [vals('a', 2, 3, A), vals('b', 3, 2, B), vals('at', 3, 2, [1, 4, 2, 5, 3, 6]), vals('bt', 2, 3, [1, 0, 1, 0, 1, 1]),
                zeros('o', 2, 2), zeros('o2', 2, 2), zeros('o3', 2, 2), vals('bias', 1, 2, [100, 200]), vals('res', 2, 2, [1, 1, 1, 1]), zeros('pre', 2, 2)]
        common = dict(m0=0, n0=0, M=2, N=2, K=3, alpha=1.0, bias=None, res=None, pre=None)
        b = run(base,
                ('matmul_tile', dict(a='a', b='b', out='o', flags=0, **common)),
                ('matmul_tile', dict(a='at', b='bt', out='o2', flags=3, **common)),
                ('matmul_tile', dict({**common, 'bias': 'bias', 'res': 'res', 'pre': 'pre', 'alpha': 2.0}, a='a', b='b', out='o3', flags=0)),
                ('matmul_tile', dict(a='a', b='b', out='o', flags=4, **common)))
        self.assertEqual(list(b['o2']), [4, 5, 10, 11])
        self.assertEqual(list(b['o']), [8, 10, 20, 22])              # computed once, then accumulated (flag 4)
        self.assertEqual(list(b['pre']), [108, 210, 120, 222])
        self.assertEqual(list(b['o3']), [109, 211, 121, 223])

    def test_matmul_tile_validation(self):
        bufs = [zeros('a', 2, 2), zeros('o', 2, 2)]
        for kw, needle in [({'flags': 32}, 'flags'), ({'m0': 2}, 'bad M/N/K'), ({'K': 0}, 'bad M/N/K')]:
            args = dict(a='a', b='a', out='o', bias=None, res=None, pre=None, m0=0, n0=0, M=2, N=2, K=2, flags=0, alpha=1.0)
            args.update(kw)
            with self.subTest(kw=kw), self.assertRaises(CompileError) as cm:
                compile_spec(wl([task('m', 'matmul_tile', args)], buffers=bufs), ops=all_ops())
            self.assertIn(needle, str(cm.exception))
        args = dict(a='a', b='a', out='o', bias=None, res=None, pre=None, m0=0, n0=0, M=2, N=2, K=3, flags=0, alpha=1.0)
        with self.assertRaises(CompileError) as cm:
            compile_spec(wl([task('m', 'matmul_tile', args)], buffers=bufs), ops=all_ops())
        self.assertIn("'a'", str(cm.exception))                       # A is 2x3 by the arguments but the buffer is 2x2

    def norm_case(self, rms):
        R, d = 3, 5
        x, g, be, dy = rnd(R * d, 3), rnd(d, 4, 0.5, 1.5), rnd(d, 5), rnd(R * d, 6)
        eps = 1e-5
        fwd = 'rmsnorm_fwd' if rms else 'layernorm_fwd'
        bwd = 'rmsnorm_bwd_dx' if rms else 'layernorm_bwd_dx'
        bufs = [vals('x', R, d, x), vals('g', 1, d, g), vals('be', 1, d, be), zeros('y', R, d), zeros('mean', R), zeros('rstd', R),
                vals('dy', R, d, dy), zeros('dx', R, d), zeros('dg', 1, d), zeros('db', 1, d)]
        fargs = dict(x='x', gamma='g', out='y', rstd='rstd', r0=0, r1=R, d=d, eps=eps)
        bargs = dict(x='x', dy='dy', gamma='g', rstd='rstd', dx='dx', r0=0, r1=R, d=d, acc=0)
        if not rms:
            fargs.update(beta='be', mean='mean')
            bargs.update(mean='mean')
        b = run(bufs, (fwd, fargs), (bwd, bargs),
                ('norm_bwd_params', dict(x='x', dy='dy', mean=None if rms else 'mean', rstd='rstd', dgamma='dg',
                                         dbeta=None if rms else 'db', R=R, d=d, c0=0, c1=d, acc=0)))

        def forward(xv, gv, bv):
            out = []
            for r in range(R):
                row = xv[r * d:(r + 1) * d]
                mu = 0.0 if rms else sum(row) / d
                rs = 1 / math.sqrt(sum((t - mu) ** 2 for t in row) / d + eps)
                out += [(t - mu) * rs * gv[c] + (0 if rms else bv[c]) for c, t in enumerate(row)]
            return out

        loss = lambda xv, gv, bv: sum(p * q for p, q in zip(forward(xv, gv, bv), dy))
        close(self, list(b['y']), forward(x, g, be), 1e-5, 'forward')
        close(self, list(b['dx']), numgrad(lambda v: loss(v, g, be), x), 3e-3, 'dx')
        close(self, list(b['dg']), numgrad(lambda v: loss(x, v, be), g), 3e-3, 'dgamma')
        if not rms:
            close(self, list(b['db']), numgrad(lambda v: loss(x, g, v), be), 3e-3, 'dbeta')

    def test_layernorm(self):
        self.norm_case(False)

    def test_rmsnorm(self):
        self.norm_case(True)


class AttentionTests(unittest.TestCase):
    B, T, H, Hkv, hd = 1, 136, 2, 1, 16         # two attention tasks (128 + 8 rows), GQA group of 2

    def setUp(self):
        B, T, H, Hkv, hd = self.B, self.T, self.H, self.Hkv, self.hd
        R = B * T
        self.q, self.k, self.v = rnd(R * H * hd, 10), rnd(R * Hkv * hd, 11), rnd(R * Hkv * hd, 12)
        self.do = rnd(R * H * hd, 13)
        self.scale = 1 / math.sqrt(hd)

    def buffers(self):
        B, T, H, Hkv, hd = self.B, self.T, self.H, self.Hkv, self.hd
        R = B * T
        return [vals('q', R, H * hd, self.q), vals('k', R, Hkv * hd, self.k), vals('v', R, Hkv * hd, self.v), zeros('o', R, H * hd),
                zeros('lse', B * H * T), vals('do', R, H * hd, self.do), zeros('delta', B * H * T), zeros('dq', R, H * hd),
                zeros('dk', R, Hkv * hd), zeros('dv', R, Hkv * hd)]

    def tasks(self):
        B, T, H, Hkv, hd, sc = self.B, self.T, self.H, self.Hkv, self.hd, self.scale
        g = dict(b=0, T=T, H=H, hd=hd, scale=sc)
        out = []
        for h in range(H):
            for q0 in (0, ref_gpt.ATTN_ROWS):
                out.append(('attn_fwd', dict(q='q', k='k', v='v', o='o', lse='lse', h=h, q0=q0, Hkv=Hkv, **g)))
        out.append(('attn_bwd_delta', dict(o='o', do='do', delta='delta', r0=0, r1=B * T, T=T, H=H, hd=hd)))
        for h in range(H):
            for q0 in (0, ref_gpt.ATTN_ROWS):
                out.append(('attn_bwd_dq', dict(q='q', k='k', v='v', do='do', lse='lse', delta='delta', dq='dq', h=h, q0=q0, Hkv=Hkv, **g)))
        for kvh in range(Hkv):
            for k0 in (0, ref_gpt.ATTN_ROWS):
                out.append(('attn_bwd_dkv', dict(q='q', k='k', v='v', do='do', lse='lse', delta='delta', dk='dk', dv='dv', kvh=kvh, k0=k0, **g)))
        return out

    def forward(self, q, k, v):
        T, H, Hkv, hd = self.T, self.H, self.Hkv, self.hd
        out = []
        for t in range(T):
            for h in range(H):
                kvh = h // (H // Hkv)
                s = [sum(q[t * H * hd + h * hd + c] * k[j * Hkv * hd + kvh * hd + c] for c in range(hd)) * self.scale for j in range(t + 1)]
                m = max(s)
                p = [math.exp(x - m) for x in s]
                out += [sum(p[j] * v[j * Hkv * hd + kvh * hd + c] for j in range(t + 1)) / sum(p) for c in range(hd)]
        return out         # order (t, h, c) == row-major [T, H*hd]

    def test_forward_and_gradients_match_finite_differences(self):
        b = run(self.buffers(), *self.tasks())
        close(self, list(b['o']), self.forward(self.q, self.k, self.v), 1e-5, 'o')
        loss = lambda o: sum(x * y for x, y in zip(o, self.do))
        for name, x, got, idx in [('dq', self.q, b['dq'], (0, 37, 700, 2559)), ('dk', self.k, b['dk'], (0, 100, 1000, 1279)),
                                  ('dv', self.v, b['dv'], (3, 500, 1279))]:
            for i in idx:
                def f(val, i=i, x=x, name=name):
                    xs = list(x)
                    xs[i] = val
                    args = {'dq': (xs, self.k, self.v), 'dk': (self.q, xs, self.v), 'dv': (self.q, self.k, xs)}[name]
                    return loss(self.forward(*args))
                num = (f(x[i] + 1e-3) - f(x[i] - 1e-3)) / 2e-3
                self.assertAlmostEqual(got[i], num, delta=2e-3 * max(1.0, abs(num)), msg=f'{name}[{i}]')

    def test_decode_matches_last_row_of_forward(self):
        T, H, Hkv, hd, Tmax = self.T, self.H, self.Hkv, self.hd, 144
        Bn = 1
        o_full = self.forward(self.q, self.k, self.v)
        n = T - 1                                         # cache T-1 tokens, then append the last one and decode it
        kc, vc = [0.0] * (Hkv * Tmax * hd), [0.0] * (Hkv * Tmax * hd)
        for h in range(Hkv):
            for t in range(n):
                for c in range(hd):
                    kc[(h * Tmax + t) * hd + c] = self.k[t * Hkv * hd + h * hd + c]
                    vc[(h * Tmax + t) * hd + c] = self.v[t * Hkv * hd + h * hd + c]
        last = lambda a, w: a[(T - 1) * w:T * w]
        bufs = [vals('kc', Hkv * Tmax, hd, kc), vals('vc', Hkv * Tmax, hd, vc), vals('kn', 1, Hkv * hd, last(self.k, Hkv * hd)),
                vals('vn', 1, Hkv * hd, last(self.v, Hkv * hd)), vals('lens', 1, 1, [n], 'i32'), vals('q', 1, H * hd, last(self.q, H * hd)),
                zeros('o', 1, H * hd)]
        tasks = [('kv_append', dict(knew='kn', vnew='vn', kc='kc', vc='vc', lens='lens', b0=0, b1=1, n_new=1, Hkv=Hkv, hd=hd, Tmax=Tmax))]
        tasks += [('attn_decode', dict(q='q', kc='kc', vc='vc', lens='lens', o='o', b=0, h=h, H=H, Hkv=Hkv, hd=hd, Tmax=Tmax, scale=self.scale))
                  for h in range(H)]
        b = run(bufs, *tasks)
        close(self, list(b['o']), last(o_full, H * hd), 1e-5)

    def test_kv_overflow_and_bad_head_size_rejected(self):
        bufs = [zeros('kc', 2, 4), zeros('vc', 2, 4), zeros('kn', 1, 4), zeros('vn', 1, 4), vals('lens', 1, 1, [2], 'i32')]
        with self.assertRaises(RuntimeError) as cm:
            run(bufs, ('kv_append', dict(knew='kn', vnew='vn', kc='kc', vc='vc', lens='lens', b0=0, b1=1, n_new=1, Hkv=1, hd=4, Tmax=2)))
        self.assertIn('exceeds Tmax', str(cm.exception))
        args = dict(q='kn', k='kn', v='kn', o='kn', lse='lens', b=0, h=0, q0=0, T=1, H=1, Hkv=1, hd=12, scale=1.0)
        with self.assertRaises(CompileError):
            compile_spec(wl([task('a', 'attn_fwd', args)], buffers=bufs), ops=all_ops())


class LossOptimizerTests(unittest.TestCase):
    def test_cross_entropy_forward_reduce_backward(self):
        V, N = 5, 4
        lg, tgt = rnd(N * V, 20, -2, 2), [1, 4, -1, 0]
        b = run([vals('lg', N, V, lg), vals('tgt', N, 1, tgt, 'i32'), zeros('rl', N), zeros('lse', N), zeros('loss', 2), zeros('dl', N, V)],
                ('softmax_ce_fwd', dict(logits='lg', tgt='tgt', rowloss='rl', lse='lse', r0=0, r1=N, V=V, ignore=-1)),
                ('softmax_ce_reduce', dict(rowloss='rl', tgt='tgt', loss='loss', N=N, ignore=-1)),
                ('softmax_ce_bwd', dict(logits='lg', tgt='tgt', lse='lse', loss='loss', dlogits='dl', r0=0, r1=N, V=V, ignore=-1, scale=1.0)))

        def mean_loss(x):
            tot = 0.0
            for r, t in enumerate(tgt):
                if t >= 0:
                    row = x[r * V:(r + 1) * V]
                    tot += math.log(sum(math.exp(c) for c in row)) - row[t]
            return tot / 3
        self.assertAlmostEqual(b['loss'][0], mean_loss(lg), places=5)
        self.assertAlmostEqual(b['loss'][1], 1 / 3, places=6)
        close(self, list(b['dl']), numgrad(mean_loss, lg), 1e-3)
        self.assertEqual(list(b['dl'][2 * V:3 * V]), [0.0] * V)         # ignored row

    def test_all_ignored_gives_zero_loss(self):
        b = run([vals('tgt', 2, 1, [-1, -1], 'i32'), zeros('rl', 2), vals('loss', 1, 2, [9, 9])],
                ('softmax_ce_reduce', dict(rowloss='rl', tgt='tgt', loss='loss', N=2, ignore=-1)))
        self.assertEqual(list(b['loss']), [0, 0])

    def test_adamw_first_step_and_clip_coefficient(self):
        p0, g0 = [1.0, -2.0], [0.5, -0.25]
        lr, b1, b2, eps, wd = 0.1, 0.9, 0.999, 1e-8, 0.01
        b = run([vals('p', 1, 2, p0), vals('g', 1, 2, g0), zeros('m', 1, 2), zeros('v', 1, 2), vals('gs', 1, 1, [0.5]), vals('step', 1, 1, [1], 'i32')],
                ('adamw', dict(p='p', g='g', m='m', v='v', gscale='gs', step='step', e0=0, e1=2, lr=lr, b1=b1, b2=b2, eps=eps, wd=wd)))
        for i in range(2):
            g = g0[i] * 0.5
            m, v = (1 - b1) * g, (1 - b2) * g * g
            want = p0[i] - lr * wd * p0[i] - lr * (m / (1 - b1)) / (math.sqrt(v / (1 - b2)) + eps)
            self.assertAlmostEqual(b['p'][i], want, places=5)
            self.assertAlmostEqual(b['m'][i], m, places=6)

    def test_adamw_requires_positive_step(self):
        with self.assertRaises(RuntimeError) as cm:
            run([zeros('p', 1, 1), zeros('g', 1, 1), zeros('m', 1, 1), zeros('v', 1, 1), vals('step', 1, 1, [0], 'i32')],
                ('adamw', dict(p='p', g='g', m='m', v='v', gscale=None, step='step', e0=0, e1=1, lr=0.1, b1=0.9, b2=0.99, eps=1e-8, wd=0.0)))
        self.assertIn('step counter', str(cm.exception))

    def test_gradnorm_partial_and_finish(self):
        g = [3.0, 4.0, 0.0, 12.0]
        for mx, want in [(1.0, 1.0 / (13.0 + 1e-6)), (100.0, 1.0), (0.0, 1.0)]:
            b = run([vals('g', 1, 4, g), zeros('part', 1, 2), zeros('out', 1, 2)],
                    ('gradnorm_partial', dict(g='g', partial='part', slot=0, e0=0, e1=2)),
                    ('gradnorm_partial', dict(g='g', partial='part', slot=1, e0=2, e1=4)),
                    ('gradnorm_finish', dict(partial='part', nslots=2, out='out', max_norm=mx)))
            self.assertAlmostEqual(b['out'][1], 13.0, places=5)
            self.assertAlmostEqual(b['out'][0], want, places=5)


class SamplingTests(unittest.TestCase):
    LOGITS = [0.1, 2.0, 2.0, -1.0, 0.5, 1.5, 3.0, 3.0]

    def sample(self, temp, topk, seed=3, ctr=0, rows=1):
        bufs = [vals('lg', rows, 8, self.LOGITS * rows), zeros('out', rows, 1, 'i32'), vals('ctr', 1, 1, [ctr], 'i32')]
        return list(run(bufs, ('sample', dict(logits='lg', out='out', ctr='ctr', r0=0, r1=rows, V=8, temp=temp, topk=topk, seed=seed)))['out'])

    def test_argmax_lowest_index_on_ties_and_temp_zero(self):
        b = run([vals('lg', 1, 8, self.LOGITS), zeros('o', 1, 1, 'i32')], ('argmax', dict(logits='lg', out='o', r0=0, r1=1, V=8)))
        self.assertEqual(b['o'][0], 6)
        self.assertEqual(self.sample(0.0, 0), [6])

    def test_sampling_is_deterministic_and_keyed_by_seed_ctr_row(self):
        draws = {tuple(self.sample(1.0, 0, seed=s, ctr=c, rows=16)) for s in (1, 2) for c in (0, 1)}
        self.assertEqual(len(draws), 4)
        self.assertEqual(self.sample(1.0, 0, rows=16), self.sample(1.0, 0, rows=16))
        self.assertGreater(len(set(self.sample(1.0, 0, rows=64))), 3)

    def test_matches_inverse_cdf_by_hand(self):
        w = [math.exp(x - 3.0) for x in self.LOGITS]
        target = ref_gpt.philox_uniform(3, 0, 0) * sum(w)
        run_, want = 0.0, None
        for j, x in enumerate(w):
            run_ += x
            if run_ > target:
                want = j
                break
        self.assertEqual(self.sample(1.0, 0), [want])

    def test_topk_keeps_ties_at_threshold(self):
        picks = set(self.sample(1.0, 2, seed=s, rows=64)[i] for s in range(5) for i in range(64))
        self.assertEqual(picks, {6, 7})                              # the two 3.0 logits
        self.assertTrue({1, 2, 6, 7} >= set(self.sample(1.0, 3, rows=64)))      # top-3 threshold 2.0 keeps both tied 2.0s


class EndToEndTests(unittest.TestCase):
    def test_linear_classifier_trains_through_the_gpt_ops(self):
        """matmul_tile -> softmax_ce -> matmul_tile (transposed) backward -> adamw over 6 unrolled steps; loss must fall."""
        N, D, V, steps = 8, 4, 3, 6
        x = rnd(N * D, 30)
        tgt = [i % V for i in range(N)]
        bufs = [vals('x', N, D, x), vals('tgt', N, 1, tgt, 'i32'), {'name': 'w', 'rows': D, 'cols': V, 'init': {'kind': 'seeded', 'seed': 5, 'scale': 0.1}},
                zeros('lg', N, V), zeros('rl', N), zeros('lse', N), zeros('loss', 2), zeros('dl', N, V), zeros('gw', D, V),
                zeros('m', D, V), zeros('v', D, V), vals('step', 1, 1, [1], 'i32')]
        mm = dict(bias=None, res=None, pre=None, m0=0, n0=0, alpha=1.0)
        ts, losses = [], []
        for s in range(steps):
            ts += [('matmul_tile', dict(a='x', b='w', out='lg', M=N, N=V, K=D, flags=0, **mm)),
                   ('softmax_ce_fwd', dict(logits='lg', tgt='tgt', rowloss='rl', lse='lse', r0=0, r1=N, V=V, ignore=-1)),
                   ('softmax_ce_reduce', dict(rowloss='rl', tgt='tgt', loss='loss', N=N, ignore=-1)),
                   ('softmax_ce_bwd', dict(logits='lg', tgt='tgt', lse='lse', loss='loss', dlogits='dl', r0=0, r1=N, V=V, ignore=-1, scale=1.0)),
                   ('matmul_tile', dict(a='x', b='dl', out='gw', M=D, N=V, K=N, flags=1, **mm)),
                   ('adamw', dict(p='w', g='gw', m='m', v='v', gscale=None, step='step', e0=0, e1=D * V, lr=0.2, b1=0.9, b2=0.99, eps=1e-8, wd=0.0)),
                   ('counter_inc', dict(x='step', i0=0, i1=1, delta=1))]
        spec = wl([task(f't{i}', op, a, after=[f't{i - 1}'] if i else []) for i, (op, a) in enumerate(ts)], buffers=bufs)
        compiled = compile_spec(spec, ops=all_ops())
        self.assertEqual(compiled.metadata['task_count'], 7 * steps)
        runtime, trace = vm.run_spec(spec, deterministic=True, ops=all_ops())
        self.assertEqual(len(trace), 7 * steps)
        final = runtime.buffers['loss'][0]
        first = run(bufs, *ts[:3])['loss'][0]                       # loss of the untrained weights
        self.assertLess(final, first * 0.8)
        self.assertEqual(runtime.buffers['step'][0], steps + 1)


class AttentionBlockSizeTests(unittest.TestCase):
    def test_python_block_sizes_match_the_device_constant(self):
        import re
        m = re.search(r'#define KVM_GPT_ATTN_ROWS (\d+)', (ROOT / 'include' / 'kvm_ops_gpt.cuh').read_text())
        self.assertIsNotNone(m)
        sys.path.insert(0, str(ROOT / 'frontend'))
        from kvmfe import opset
        self.assertEqual(int(m.group(1)), ref_gpt.ATTN_ROWS)
        self.assertEqual(int(m.group(1)), opset.ATTN_BLOCK)


if __name__ == '__main__':
    unittest.main()
