"""CPU reference semantics (float64) of every device op the frontend emits: the golden path the GPU run is diffed against.

Signatures come from include/kvm_ops*.def (defs.py); this module only adds what a table cannot say: what an op computes (written
independently of the CUDA code, following the contract in include/kvm_ops_gpt.def) and a FLOP estimate for the cost model.
Buffers are dense row-major 2-D arrays (`Buf`); index buffers hold integers as numbers. `None` is an absent optional buffer.
"""
from __future__ import annotations

import ctypes
import math
from dataclasses import dataclass
from typing import Callable

from functools import lru_cache

from .defs import default_include_dir, read_defs
from .ir import FrontendError

MM_TILE = 64        # matmul_tile computes a 64 x 64 output tile (flags: 1 transA, 2 transB, 4 accumulate, 8 gelu)
ATTN_Q = 32         # attn_fwd / attn_bwd_dq: 32 query rows per task; attn_bwd_dkv: 32 keys per task
F_TA, F_TB, F_ACC, F_GELU, F_PRECISE = 1, 2, 4, 8, 16
F_BIG, F_HUGE = 32, 64   # matmul_tile flags: 32 = 128x128 output tile, 64 = 256x128 (rows x cols; two 128x128 passes). Default 64x64.


def mm_dims(fl):
    """(rows, cols) of the output tile a matmul_tile* task with flags `fl` computes."""
    return (256, 128) if fl & F_HUGE else (128, 128) if fl & F_BIG else (MM_TILE, MM_TILE)


class Buf:
    """rows x cols dense store; `a` is the flat row-major list. Optionally rounds every write to float32."""
    __slots__ = ('a', 'rows', 'cols', 'f32')

    def __init__(self, a, rows, cols, f32=False):
        self.a, self.rows, self.cols, self.f32 = a, rows, cols, f32

    def row(self, r, c0=0, c1=None):
        o = r * self.cols
        return self.a[o + c0:o + (self.cols if c1 is None else c1)]

    def put(self, r, c0, vals):
        o = r * self.cols + c0
        self.a[o:o + len(vals)] = [ctypes.c_float(v).value for v in vals] if self.f32 else vals


@dataclass(frozen=True)
class DeviceOp:
    name: str
    id: int
    sig: tuple
    ref: Callable | None
    flops: Callable                 # scalar args -> FLOPs of one task
    fixed_us: float | None = None   # declared duration overriding the model (opaque engines)

    def arg_types(self):
        return dict(self.sig)


REF: dict[str, tuple] = {}          # name -> (ref, flops, fixed_us)


def impl(name, flops=None, fixed_us=None):
    def deco(fn):
        if name in REF:
            raise ValueError(f'duplicate reference for {name}')
        REF[name] = (fn, flops, fixed_us)
        return fn
    return deco


def op(name: str) -> DeviceOp:
    return _op(name, str(default_include_dir()))


@lru_cache(maxsize=None)
def _op(name: str, include_dir: str) -> DeviceOp:
    defs = read_defs(include_dir)
    if name not in defs:
        raise FrontendError(f'op {name!r} is not defined in include/kvm_ops*.def; defined: {sorted(defs)}')
    ref, flops, fixed = REF.get(name, (None, None, None))
    return DeviceOp(name, defs[name][0], defs[name][1], ref, flops or (lambda a: 0.0), fixed)


def _span(a, lo, hi, n):
    return range(a[lo], min(a[hi], n))


# ------------------------------------------------------------------------------------------------ control / data movement
impl('noop', lambda a: 0.0)(lambda b, a: None)


@impl('zero_buf', lambda a: a['e1'] - a['e0'])
def _zero(b, a):
    x = b['x']
    x.a[a['e0']:a['e1']] = [0.0] * (a['e1'] - a['e0'])


@impl('counter_inc', lambda a: 1.0)
def _counter_inc(b, a):
    for i in range(a['i0'], a['i1']):
        b['x'].a[i] += a['delta']


@impl('embedding_fwd', lambda a: (a['r1'] - a['r0']) * a['d'])
def _embedding_fwd(b, a):
    wte, wpe = b['wte'], b['wpe']
    for r in range(a['r0'], a['r1']):
        i = int(b['ids'].a[r])
        v = wte.row(i)[:a['d']] if 0 <= i < wte.rows else [0.0] * a['d']
        if wpe is not None:
            v = [x + y for x, y in zip(v, wpe.row(r % a['T'])[:a['d']])]
        b['out'].put(r, 0, v)


@impl('embedding_bwd', lambda a: a['N'] * a['d'])
def _embedding_bwd(b, a):
    keys = [int(b['ids'].a[r]) if b['ids'] is not None else r % a['mod'] for r in range(a['N'])]
    for key in range(a['v0'], a['v1']):
        acc = [0.0] * a['d']
        for r, k in enumerate(keys):
            if k == key:
                acc = [x + y for x, y in zip(acc, b['dout'].row(r)[:a['d']])]
        if a['acc']:
            acc = [x + y for x, y in zip(acc, b['dtab'].row(key)[:a['d']])]
        b['dtab'].put(key, 0, acc)


@impl('argmax', lambda a: (a['r1'] - a['r0']) * a['V'])
def _argmax(b, a):
    for r in range(a['r0'], a['r1']):
        row = b['logits'].row(r)[:a['V']]
        b['out'].a[r] = float(row.index(max(row)))


# ------------------------------------------------------------------------------------------------ matmul
def _mm_flops(a):
    tm, tn = mm_dims(a['flags'])
    return 2.0 * max(0, min(tm, a['M'] - a['m0'])) * max(0, min(tn, a['N'] - a['n0'])) * a['K']


def _mm_acc(A, B, fl, m0, m1, n0, n1, K):
    """alpha-free product tile of op(A) @ op(B) for rows m0:m1, cols n0:n1 as a list of rows. numpy float64 (BLAS) when available:
    the sum order differs from a sequential loop only at the 1e-16 level, and is deterministic for given shapes."""
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is None:
        bcols = [[(B.a[n * B.cols + k] if fl & F_TB else B.a[k * B.cols + n]) for k in range(K)] for n in range(n0, n1)]
        return [[sum(x * y for x, y in zip([(A.a[k * A.cols + m] if fl & F_TA else A.a[m * A.cols + k]) for k in range(K)], bc)) for bc in bcols]
                for m in range(m0, m1)]
    a = np.array(A.a, dtype=np.float64).reshape(A.rows, A.cols)
    b = np.array(B.a, dtype=np.float64).reshape(B.rows, B.cols)
    a = a[:K, m0:m1].T if fl & F_TA else a[m0:m1, :K]
    b = b[n0:n1, :K].T if fl & F_TB else b[:K, n0:n1]
    return (a @ b).tolist()


@impl('matmul_tile', _mm_flops)
def _matmul_tile(b, a):
    M, N, K, fl = a['M'], a['N'], a['K'], a['flags']
    A, B, out = b['a'], b['b'], b['out']
    tm, tn = mm_dims(fl)
    m1, n1 = min(a['m0'] + tm, M), min(a['n0'] + tn, N)
    prod = _mm_acc(A, B, fl, a['m0'], m1, a['n0'], n1, K)
    for i, m in enumerate(range(a['m0'], m1)):
        vals = []
        for j, n in enumerate(range(a['n0'], n1)):
            v = a['alpha'] * prod[i][j]
            if b['bias'] is not None:
                v += b['bias'].a[n]
            if b['pre'] is not None:
                b['pre'].a[m * b['pre'].cols + n] = ctypes.c_float(v).value if out.f32 else v
            if fl & F_GELU:
                v = _gelu(v)
            if b['res'] is not None:
                v += b['res'].a[m * b['res'].cols + n]
            if fl & F_ACC:
                v += out.a[m * out.cols + n]
            vals.append(v)
        out.put(m, a['n0'], vals)


@impl('colsum', lambda a: a['R'] * (a['c1'] - a['c0']))
def _colsum(b, a):
    x, o = b['x'], b['out']
    for c in range(a['c0'], a['c1']):
        s = sum(x.a[r * x.cols + c] for r in range(a['R']))
        o.a[c] = s + (o.a[c] if a['acc'] else 0.0)


# ------------------------------------------------------------------------------------------------ elementwise
_GC = math.sqrt(2.0 / math.pi)


def _gelu(x):
    return 0.5 * x * (1.0 + math.tanh(_GC * (x + 0.044715 * x ** 3)))


def _gelu_d(x):
    th = math.tanh(_GC * (x + 0.044715 * x ** 3))
    return 0.5 * (1.0 + th) + 0.5 * x * (1.0 - th * th) * _GC * (1.0 + 3 * 0.044715 * x * x)


@impl('gelu_fwd', lambda a: 15.0 * (a['e1'] - a['e0']))
def _gelu_fwd(b, a):
    b['out'].a[a['e0']:a['e1']] = [_gelu(x) for x in b['x'].a[a['e0']:a['e1']]]


@impl('gelu_bwd', lambda a: 20.0 * (a['e1'] - a['e0']))
def _gelu_bwd(b, a):
    e0, e1 = a['e0'], a['e1']
    v = [_gelu_d(x) * g + (d if a['acc'] else 0.0) for x, g, d in zip(b['x'].a[e0:e1], b['dy'].a[e0:e1], b['dx'].a[e0:e1])]
    b['dx'].a[e0:e1] = v


@impl('residual_add', lambda a: a['e1'] - a['e0'])
def _residual_add(b, a):
    e0, e1 = a['e0'], a['e1']
    b['out'].a[e0:e1] = [x + y for x, y in zip(b['a'].a[e0:e1], b['b'].a[e0:e1])]


@impl('rope_fwd', lambda a: 6.0 * (a['r1'] - a['r0']) * a['H'] * a['hd'])
def _rope_fwd(b, a):
    _rope(b, a, 1.0)


@impl('rope_bwd', lambda a: 6.0 * (a['r1'] - a['r0']) * a['H'] * a['hd'])
def _rope_bwd(b, a):
    _rope(b, a, -1.0)


def _rope(b, a, sign):
    hd, H, half = a['hd'], a['H'], a['hd'] // 2
    p0 = int(b['pos'].a[0]) if b['pos'] is not None else 0
    for r in range(a['r0'], a['r1']):
        x = b['x'].row(r)
        o = list(x)
        for h in range(H):
            for i in range(half):
                ang = (r % a['T'] + p0) * a['theta'] ** (-2.0 * i / hd)
                c, s = math.cos(ang), sign * math.sin(ang)
                x1, x2 = x[h * hd + i], x[h * hd + i + half]
                o[h * hd + i], o[h * hd + i + half] = x1 * c - x2 * s, x1 * s + x2 * c
        b['out'].put(r, 0, o[:b['out'].cols])


# ------------------------------------------------------------------------------------------------ norms
@impl('layernorm_fwd', lambda a: 8.0 * (a['r1'] - a['r0']) * a['d'])
def _layernorm_fwd(b, a):
    g, beta, d = b['gamma'].row(0), b['beta'].row(0) if b['beta'] is not None else None, a['d']
    for r in range(a['r0'], a['r1']):
        x = b['x'].row(r)[:d]
        mu = sum(x) / d
        rstd = 1.0 / math.sqrt(sum((v - mu) ** 2 for v in x) / d + a['eps'])
        b['out'].put(r, 0, [(v - mu) * rstd * g[c] + (beta[c] if beta else 0.0) for c, v in enumerate(x)])
        b['mean'].a[r], b['rstd'].a[r] = mu, rstd


@impl('rmsnorm_fwd', lambda a: 6.0 * (a['r1'] - a['r0']) * a['d'])
def _rmsnorm_fwd(b, a):
    g, d = b['gamma'].row(0), a['d']
    for r in range(a['r0'], a['r1']):
        x = b['x'].row(r)[:d]
        rstd = 1.0 / math.sqrt(sum(v * v for v in x) / d + a['eps'])
        b['out'].put(r, 0, [v * rstd * g[c] for c, v in enumerate(x)])
        b['rstd'].a[r] = rstd


def _norm_dx(b, a, rms):
    g, d = b['gamma'].row(0), a['d']
    for r in range(a['r0'], a['r1']):
        mu = 0.0 if rms else b['mean'].a[r]
        rstd = b['rstd'].a[r]
        xh = [(v - mu) * rstd for v in b['x'].row(r)[:d]]
        gc = [dy * g[c] for c, dy in enumerate(b['dy'].row(r)[:d])]
        s1, s2 = (0.0 if rms else sum(gc) / d), sum(u * h for u, h in zip(gc, xh)) / d
        dx = [rstd * (u - s1 - h * s2) for u, h in zip(gc, xh)]
        if a['acc']:
            dx = [x + y for x, y in zip(dx, b['dx'].row(r)[:d])]
        b['dx'].put(r, 0, dx)


impl('layernorm_bwd_dx', lambda a: 12.0 * (a['r1'] - a['r0']) * a['d'])(lambda b, a: _norm_dx(b, a, False))
impl('rmsnorm_bwd_dx', lambda a: 10.0 * (a['r1'] - a['r0']) * a['d'])(lambda b, a: _norm_dx(b, a, True))


@impl('norm_bwd_params', lambda a: 6.0 * a['R'] * (a['c1'] - a['c0']))
def _norm_bwd_params(b, a):
    rms = b['mean'] is None
    for c in range(a['c0'], a['c1']):
        dg = db = 0.0
        for r in range(a['R']):
            xh = (b['x'].a[r * b['x'].cols + c] - (0.0 if rms else b['mean'].a[r])) * b['rstd'].a[r]
            dy = b['dy'].a[r * b['dy'].cols + c]
            dg += dy * xh
            db += dy
        b['dgamma'].a[c] = dg + (b['dgamma'].a[c] if a['acc'] else 0.0)
        if b['dbeta'] is not None:
            b['dbeta'].a[c] = db + (b['dbeta'].a[c] if a['acc'] else 0.0)


# ------------------------------------------------------------------------------------------------ attention (causal, GQA-capable)
def attn_span(x):
    """(first row/key, number of 32-row blocks) of an attention task's q0/k0 argument: bits 16.. hold the block count - 1."""
    return x & 0xFFFF, (x >> 16) + 1


def _attn_fl(a):
    r0, n = attn_span(a.get('q0', a.get('k0', 0)))
    return 4.0 * ATTN_Q * n * (r0 + ATTN_Q * (n + 1) / 2) * a['hd']


def _head(buf, r, h, hd):
    return buf.row(r, h * hd, (h + 1) * hd)


def _probs(q, K, scale, lse=None):
    s = [scale * sum(x * y for x, y in zip(q, k)) for k in K]
    if lse is None:
        m = max(s)
        z = sum(math.exp(v - m) for v in s)
        return [math.exp(v - m) / z for v in s], m + math.log(z)
    return [math.exp(v - lse) for v in s], lse


@impl('attn_fwd', _attn_fl)
def _attn_fwd(b, a):
    T, H, Hkv, hd, bb, h = a['T'], a['H'], a['Hkv'], a['hd'], a['b'], a['h']
    kvh = h // (H // Hkv)
    q0, n = attn_span(a['q0'])
    for t in range(q0, min(q0 + n * ATTN_Q, T)):
        K = [_head(b['k'], bb * T + j, kvh, hd) for j in range(t + 1)]
        p, lse = _probs(_head(b['q'], bb * T + t, h, hd), K, a['scale'])
        o = [0.0] * hd
        for j, pj in enumerate(p):
            o = [x + pj * y for x, y in zip(o, _head(b['v'], bb * T + j, kvh, hd))]
        b['o'].put(bb * T + t, h * hd, o)
        b['lse'].a[(bb * H + h) * T + t] = lse


@impl('attn_bwd_delta', lambda a: 2.0 * (a['r1'] - a['r0']) * a['H'] * a['hd'])
def _attn_bwd_delta(b, a):
    T, H, hd = a['T'], a['H'], a['hd']
    for r in range(a['r0'], a['r1']):
        for h in range(H):
            b['delta'].a[(r // T * H + h) * T + r % T] = sum(x * y for x, y in zip(_head(b['do'], r, h, hd), _head(b['o'], r, h, hd)))


def _ds(b, a, bb, h, kvh, t, j_range):
    """Per query row t: (p_j, ds_j) for keys j in j_range, from the stored lse and delta."""
    T, H, hd = a['T'], a['H'], a['hd']
    q = _head(b['q'], bb * T + t, h, hd)
    do = _head(b['do'], bb * T + t, h, hd)
    lse, delta = b['lse'].a[(bb * H + h) * T + t], b['delta'].a[(bb * H + h) * T + t]
    out = []
    for j in j_range:
        p = math.exp(a['scale'] * sum(x * y for x, y in zip(q, _head(b['k'], bb * T + j, kvh, hd))) - lse)
        dp = sum(x * y for x, y in zip(do, _head(b['v'], bb * T + j, kvh, hd)))
        out.append((j, p, p * (dp - delta)))
    return q, do, out


@impl('attn_bwd_dq', lambda a: 2.5 * _attn_fl(a))
def _attn_bwd_dq(b, a):
    T, hd, bb, h = a['T'], a['hd'], a['b'], a['h']
    kvh = h // (a['H'] // a['Hkv'])
    q0, n = attn_span(a['q0'])
    for t in range(q0, min(q0 + n * ATTN_Q, T)):
        _, _, terms = _ds(b, a, bb, h, kvh, t, range(t + 1))
        dq = [0.0] * hd
        for j, p, ds in terms:
            dq = [x + a['scale'] * ds * y for x, y in zip(dq, _head(b['k'], bb * T + j, kvh, hd))]
        b['dq'].put(bb * T + t, h * hd, dq)


@impl('attn_bwd_dkv', lambda a: 2.5 * _attn_fl(a))
def _attn_bwd_dkv(b, a):
    T, H, hd, bb, kvh = a['T'], a['H'], a['hd'], a['b'], a['kvh']
    Hkv = b['k'].cols // hd                       # the op has no Hkv argument (15-word task limit): derived from k
    G = H // Hkv
    k0, n = attn_span(a['k0'])
    k1 = min(k0 + n * ATTN_Q, T)
    dk = {j: [0.0] * hd for j in range(k0, k1)}
    dv = {j: [0.0] * hd for j in range(k0, k1)}
    for h in range(kvh * G, (kvh + 1) * G):
        for t in range(k0, T):
            q, do, terms = _ds(b, a, bb, h, kvh, t, range(k0, min(k1, t + 1)))
            for j, p, ds in terms:
                dv[j] = [x + p * y for x, y in zip(dv[j], do)]
                dk[j] = [x + a['scale'] * ds * y for x, y in zip(dk[j], q)]
    for j in range(k0, k1):
        b['dk'].put(bb * T + j, kvh * hd, dk[j])
        b['dv'].put(bb * T + j, kvh * hd, dv[j])


@impl('kv_append', lambda a: 0.0)
def _kv_append(b, a):
    nn, Hkv, hd, Tmax = a['n_new'], a['Hkv'], a['hd'], a['Tmax']
    for bb in range(a['b0'], a['b1']):
        for t in range(nn):
            for h in range(Hkv):
                dst = (bb * Hkv + h) * Tmax + int(b['lens'].a[bb]) + t
                b['kc'].put(dst, 0, b['knew'].row(bb * nn + t, h * hd, (h + 1) * hd))
                b['vc'].put(dst, 0, b['vnew'].row(bb * nn + t, h * hd, (h + 1) * hd))


@impl('attn_decode', lambda a: 4.0 * a['Tmax'] * a['hd'] / 8)
def _attn_decode(b, a):
    H, Hkv, hd, Tmax, bb, h = a['H'], a['Hkv'], a['hd'], a['Tmax'], a['b'], a['h']
    kvh, n = h // (H // Hkv), int(b['lens'].a[a['b']]) + 1
    base = (bb * Hkv + kvh) * Tmax
    p, _ = _probs(_head(b['q'], bb, h, hd), [b['kc'].row(base + j) for j in range(n)], a['scale'])
    o = [0.0] * hd
    for j, pj in enumerate(p):
        o = [x + pj * y for x, y in zip(o, b['vc'].row(base + j))]
    b['o'].put(bb, h * hd, o)


# ------------------------------------------------------------------------------------------------ loss and optimizers
@impl('softmax_ce_fwd', lambda a: 6.0 * (a['r1'] - a['r0']) * a['V'])
def _ce_fwd(b, a):
    for r in range(a['r0'], a['r1']):
        x = b['logits'].row(r)[:a['V']]
        m = max(x)
        lse = m + math.log(sum(math.exp(v - m) for v in x))
        t = int(b['tgt'].a[r])
        b['lse'].a[r] = lse
        b['rowloss'].a[r] = 0.0 if t == a['ignore'] else lse - x[t]


@impl('softmax_ce_reduce', lambda a: a['N'])
def _ce_reduce(b, a):
    cnt = sum(1 for r in range(a['N']) if int(b['tgt'].a[r]) != a['ignore'])
    s = sum(b['rowloss'].a[r] for r in range(a['N']) if int(b['tgt'].a[r]) != a['ignore'])
    b['loss'].a[0] = s / cnt if cnt else 0.0
    b['loss'].a[1] = 1.0 / max(cnt, 1) if cnt else 0.0


@impl('softmax_ce_bwd', lambda a: 6.0 * (a['r1'] - a['r0']) * a['V'])
def _ce_bwd(b, a):
    k = b['loss'].a[1] * a['scale']
    for r in range(a['r0'], a['r1']):
        t = int(b['tgt'].a[r])
        if t == a['ignore']:
            b['dlogits'].put(r, 0, [0.0] * a['V'])
            continue
        lse = b['lse'].a[r]
        b['dlogits'].put(r, 0, [(math.exp(x - lse) - (c == t)) * k for c, x in enumerate(b['logits'].row(r)[:a['V']])])


@impl('adamw', lambda a: 12.0 * (a['e1'] - a['e0']))
def _adamw(b, a):
    t = int(b['step'].a[0])
    gs = b['gscale'].a[0] if b['gscale'] is not None else 1.0
    c1, c2 = 1 - a['b1'] ** t, 1 - a['b2'] ** t
    e0, e1 = a['e0'], a['e1']
    for i in range(e0, e1):
        g = b['g'].a[i] * gs
        m = a['b1'] * b['m'].a[i] + (1 - a['b1']) * g
        v = a['b2'] * b['v'].a[i] + (1 - a['b2']) * g * g
        p = b['p'].a[i]
        b['m'].a[i], b['v'].a[i] = m, v
        b['p'].a[i] = p - a['lr'] * a['wd'] * p - a['lr'] * (m / c1) / (math.sqrt(v / c2) + a['eps'])


@impl('gradnorm_partial', lambda a: 2.0 * (a['e1'] - a['e0']))
def _gradnorm_partial(b, a):
    b['partial'].a[a['slot']] = sum(x * x for x in b['g'].a[a['e0']:a['e1']])


@impl('gradnorm_finish', lambda a: float(a['nslots']))
def _gradnorm_finish(b, a):
    norm = math.sqrt(sum(b['partial'].a[:a['nslots']]))
    b['out'].a[0] = min(1.0, a['max_norm'] / (norm + 1e-6)) if a['max_norm'] > 0 else 1.0     # clip coefficient = adamw gscale
    b['out'].a[1] = norm


@impl('select_window', lambda a: float(a['i1'] - a['i0']))
def _select_window(b, a):
    base = int(b['step'].a[0]) * a['stride'] + a['off']
    b['dst'].a[a['i0']:a['i1']] = b['src'].a[base + a['i0']:base + a['i1']]


@impl('sgd', lambda a: 2.0 * a['n'])
def _sgd(b, a):
    b['w'].a[:a['n']] = [w - a['lr'] * g for w, g in zip(b['w'].a[:a['n']], b['g'].a[:a['n']])]


# opaque engine ops (no CPU reference here; v1 measured ~6.7 ms per Doom tic with 255 helper threads)
for _n, _us in (('doom_tick', 6700.0), ('doom_present', 30.0), ('doom_status', 5.0)):
    REF[_n] = (None, lambda a: 0.0, _us)


# ================================================================================================ extra pack (kvm_ops_extra.def)
# Independent float64 implementations of the comments in include/kvm_ops_extra.def (runtime/ref_extra.py is the float32 twin).
def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def _silu(x):
    return x * _sigmoid(x)


def _silu_d(x):
    s = _sigmoid(x)
    return s * (1.0 + x * (1.0 - s))


ACT_FN = {0: (lambda x: x if x > 0 else 0.0, lambda x: 1.0 if x > 0 else 0.0), 1: (_silu, _silu_d),
          2: (math.tanh, lambda x: 1.0 - math.tanh(x) ** 2), 3: (_sigmoid, lambda x: _sigmoid(x) * (1.0 - _sigmoid(x)))}


@impl('act_fwd', lambda a: 8.0 * (a['e1'] - a['e0']))
def _act_fwd(b, a):
    f = ACT_FN[a['kind']][0]
    b['out'].a[a['e0']:a['e1']] = [f(x) for x in b['x'].a[a['e0']:a['e1']]]


@impl('act_bwd', lambda a: 10.0 * (a['e1'] - a['e0']))
def _act_bwd(b, a):
    d, e0, e1 = ACT_FN[a['kind']][1], a['e0'], a['e1']
    b['dx'].a[e0:e1] = [d(x) * g + (o if a['acc'] else 0.0) for x, g, o in zip(b['x'].a[e0:e1], b['dy'].a[e0:e1], b['dx'].a[e0:e1])]


@impl('silu_mul_fwd', lambda a: 10.0 * (a['e1'] - a['e0']))
def _silu_mul_fwd(b, a):
    e0, e1 = a['e0'], a['e1']
    b['out'].a[e0:e1] = [_silu(g) * u for g, u in zip(b['gate'].a[e0:e1], b['up'].a[e0:e1])]


@impl('silu_mul_bwd', lambda a: 20.0 * (a['e1'] - a['e0']))
def _silu_mul_bwd(b, a):
    e0, e1 = a['e0'], a['e1']
    for i in range(e0, e1):
        g, u, dy = b['gate'].a[i], b['up'].a[i], b['dy'].a[i]
        vg, vu = dy * u * _silu_d(g), dy * _silu(g)
        b['dgate'].a[i] = vg + (b['dgate'].a[i] if a['acc'] else 0.0)
        b['dup'].a[i] = vu + (b['dup'].a[i] if a['acc'] else 0.0)


@impl('softmax_rows_fwd', lambda a: 6.0 * (a['r1'] - a['r0']) * a['d'])
def _softmax_rows_fwd(b, a):
    for r in range(a['r0'], a['r1']):
        s = [a['scale'] * x for x in b['x'].row(r)[:a['d']]]
        m = max(s)
        e = [math.exp(v - m) for v in s]
        z = sum(e)
        b['out'].put(r, 0, [v / z for v in e])


@impl('softmax_rows_bwd', lambda a: 6.0 * (a['r1'] - a['r0']) * a['d'])
def _softmax_rows_bwd(b, a):
    for r in range(a['r0'], a['r1']):
        p, dp = b['p'].row(r)[:a['d']], b['dp'].row(r)[:a['d']]
        dot = sum(x * y for x, y in zip(p, dp))
        v = [a['scale'] * x * (y - dot) for x, y in zip(p, dp)]
        if a['acc']:
            v = [x + y for x, y in zip(v, b['dx'].row(r)[:a['d']])]
        b['dx'].put(r, 0, v)


@impl('gather_rows_scaled', lambda a: (a['r1'] - a['r0']) * a['d'])
def _gather_rows_scaled(b, a):
    src, o = b['src'], a['idx0']
    for r in range(a['r0'], a['r1']):
        j = int(b['idx'].a[o + r])
        s = b['scale'].a[o + r] if b['scale'] is not None else 1.0
        v = [s * x for x in src.row(j)[:a['d']]] if 0 <= j < src.rows else [0.0] * a['d']
        if a['acc']:
            v = [x + y for x, y in zip(v, b['dst'].row(r)[:a['d']])]
        b['dst'].put(r, 0, v)


@impl('scatter_add_rows', lambda a: a['N'] * a['d'])
def _scatter_add_rows(b, a):
    o, d = a['idx0'], a['d']
    for v in range(a['v0'], a['v1']):
        acc = [0.0] * d
        for r in range(a['N']):
            if int(b['idx'].a[o + r]) == v:
                s = b['scale'].a[o + r] if b['scale'] is not None else 1.0
                acc = [x + s * y for x, y in zip(acc, b['src'].row(r)[:d])]
        if a['acc']:
            acc = [x + y for x, y in zip(acc, b['dst'].row(v)[:d])]
        b['dst'].put(v, 0, acc)


@impl('row_dot', lambda a: 2.0 * (a['r1'] - a['r0']) * a['d'])
def _row_dot(b, a):
    A, o = b['a'], a['idx0']
    for r in range(a['r0'], a['r1']):
        j = int(b['idx'].a[o + r]) if b['idx'] is not None else r
        v = sum(x * y for x, y in zip(A.row(j)[:a['d']], b['b'].row(r)[:a['d']])) if 0 <= j < A.rows else 0.0
        b['out'].a[o + r] = v + (b['out'].a[o + r] if a['acc'] else 0.0)


@impl('moe_route', lambda a: 4.0 * a['N'] * a['E'] * a['k'])
def _moe_route(b, a):
    N, E, k, C = a['N'], a['E'], a['k'], a['C']
    cnt, eidx = [0] * E, [-1.0] * (E * C)
    for t in range(N):
        row = b['probs'].row(t)[:E]
        for j, e in enumerate(sorted(range(E), key=lambda e: (-row[e], e))[:k]):
            b['sel'].a[t * k + j] = float(e)
            if cnt[e] < C:
                b['slot'].a[t * k + j] = float(e * C + cnt[e])
                eidx[e * C + cnt[e]] = float(t)
                cnt[e] += 1
            else:
                b['slot'].a[t * k + j] = -1.0
    b['eidx'].a[:E * C] = eidx


def _moe_choices(b, a, t):
    k = a['k']
    sel = [int(b['sel'].a[t * k + j]) for j in range(k)]
    return sel, [int(b['slot'].a[t * k + j]) for j in range(k)], [b['probs'].row(t)[e] for e in sel]


@impl('moe_gates_fwd', lambda a: 4.0 * a['N'] * a['k'])
def _moe_gates_fwd(b, a):
    gs = b['gs']
    gs.a[:] = [0.0] * len(gs.a)
    for t in range(a['N']):
        sel, slot, p = _moe_choices(b, a, t)
        z = sum(p) if a['renorm'] else 1.0
        for s, pj in zip(slot, p):
            if s >= 0:
                gs.a[s] = pj / z


@impl('moe_gates_bwd', lambda a: 8.0 * a['N'] * a['k'])
def _moe_gates_bwd(b, a):
    N, E, k = a['N'], a['E'], a['k']
    dP = b['dprobs']
    if not a['acc']:
        dP.a[:N * E] = [0.0] * (N * E)
    for t in range(N):
        sel, slot, p = _moe_choices(b, a, t)
        dg = [b['dgs'].a[s] if s >= 0 else 0.0 for s in slot]
        if a['renorm']:
            z = sum(p)
            w = sum(x * y for x, y in zip(dg, p)) / (z * z)
            dp = [x / z - w for x in dg]
        else:
            dp = dg
        for e, v in zip(sel, dp):
            dP.a[t * E + e] += v


def _moe_f(b, a):
    cnt = [0] * a['E']
    for i in range(a['N'] * a['k']):
        cnt[int(b['sel'].a[i])] += 1
    return [c / (a['N'] * a['k']) for c in cnt]


@impl('moe_balance_fwd', lambda a: 2.0 * a['N'] * a['E'])
def _moe_balance_fwd(b, a):
    N, E = a['N'], a['E']
    f = _moe_f(b, a)
    pm = [sum(b['probs'].row(t)[e] for t in range(N)) / N for e in range(E)]
    b['loss'].a[0] = a['coef'] * E * sum(x * y for x, y in zip(f, pm))


@impl('moe_balance_bwd', lambda a: 2.0 * a['N'] * a['E'])
def _moe_balance_bwd(b, a):
    N, E = a['N'], a['E']
    f = _moe_f(b, a)
    for t in range(N):
        for e in range(E):
            b['dprobs'].a[t * E + e] = a['coef'] * E * f[e] / N + (b['dprobs'].a[t * E + e] if a['acc'] else 0.0)


def _full_head_flops(a):
    return 4.0 * ATTN_Q * a['T'] * a['hd']


@impl('attn_full_fwd', _full_head_flops)
def _attn_full_fwd(b, a):
    T, H, Hkv, hd, bb, h = a['T'], a['H'], a['Hkv'], a['hd'], a['b'], a['h']
    kvh = h // (H // Hkv)
    K = [_head(b['k'], bb * T + j, kvh, hd) for j in range(T)]
    for t in range(a['q0'], min(a['q0'] + ATTN_Q, T)):
        p, lse = _probs(_head(b['q'], bb * T + t, h, hd), K, a['scale'])
        o = [0.0] * hd
        for j, pj in enumerate(p):
            o = [x + pj * y for x, y in zip(o, _head(b['v'], bb * T + j, kvh, hd))]
        b['o'].put(bb * T + t, h * hd, o)
        b['lse'].a[(bb * H + h) * T + t] = lse


@impl('attn_full_bwd_dq', lambda a: 2.5 * _full_head_flops(a))
def _attn_full_bwd_dq(b, a):
    T, hd, bb, h = a['T'], a['hd'], a['b'], a['h']
    kvh = h // (a['H'] // a['Hkv'])
    q0, n = attn_span(a['q0'])
    for t in range(q0, min(q0 + n * ATTN_Q, T)):
        _, _, terms = _ds(b, a, bb, h, kvh, t, range(T))
        dq = [0.0] * hd
        for j, p, ds in terms:
            dq = [x + a['scale'] * ds * y for x, y in zip(dq, _head(b['k'], bb * T + j, kvh, hd))]
        b['dq'].put(bb * T + t, h * hd, dq)


@impl('attn_full_bwd_dkv', lambda a: 2.5 * _full_head_flops(a))
def _attn_full_bwd_dkv(b, a):
    T, H, hd, bb, kvh = a['T'], a['H'], a['hd'], a['b'], a['kvh']
    G = H // (b['k'].cols // hd)
    k1 = min(a['k0'] + ATTN_Q, T)
    dk = {j: [0.0] * hd for j in range(a['k0'], k1)}
    dv = {j: [0.0] * hd for j in range(a['k0'], k1)}
    for h in range(kvh * G, (kvh + 1) * G):
        for t in range(T):
            q, do, terms = _ds(b, a, bb, h, kvh, t, range(a['k0'], k1))
            for j, p, ds in terms:
                dv[j] = [x + p * y for x, y in zip(dv[j], do)]
                dk[j] = [x + a['scale'] * ds * y for x, y in zip(dk[j], q)]
    for j in range(k0, k1):
        b['dk'].put(bb * T + j, kvh * hd, dk[j])
        b['dv'].put(bb * T + j, kvh * hd, dv[j])


def bf16_round(x):
    """Round to the nearest bfloat16 (ties to even), as a float64 value; inf/nan unchanged."""
    import struct
    if x != x or x in (math.inf, -math.inf):
        return x
    bits = struct.unpack('<I', struct.pack('<f', x))[0]
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack('<f', struct.pack('<I', bits))[0]


@impl('round_bf16', lambda a: float(a['e1'] - a['e0']))
def _round_bf16(b, a):
    b['out'].a[a['e0']:a['e1']] = [bf16_round(x) for x in b['x'].a[a['e0']:a['e1']]]


@impl('matmul_tile_bf16', _mm_flops)
def _matmul_tile_bf16(b, a):
    rb = dict(b)
    for n in ('a', 'b'):
        rb[n] = Buf([bf16_round(x) for x in b[n].a], b[n].rows, b[n].cols)
    _matmul_tile(rb, a)


@impl('adamw_lr', lambda a: 12.0 * (a['e1'] - a['e0']))
def _adamw_lr(b, a):
    _adamw(b, {**a, 'lr': b['lr'].a[0]})


@impl('lr_schedule', lambda a: 1.0)
def _lr_schedule(b, a):
    t = int(b['step'].a[0])
    warm = t / a['warmup'] if a['warmup'] > 0 and t <= a['warmup'] else 1.0
    prog = min(max((t - a['warmup']) / max(a['total'] - a['warmup'], 1), 0.0), 1.0)
    mr = a['min_ratio']
    f = {0: 1.0, 1: mr + (1 - mr) * 0.5 * (1 + math.cos(math.pi * prog)), 2: 1.0 - (1 - mr) * prog}[a['kind']]
    b['lr'].a[0] = a['base'] * warm * f


@impl('scale_flat', lambda a: float(a['e1'] - a['e0']))
def _scale_flat(b, a):
    b['out'].a[a['e0']:a['e1']] = [a['alpha'] * x for x in b['x'].a[a['e0']:a['e1']]]


@impl('add_rowtile', lambda a: float((a['r1'] - a['r0']) * a['d']))
def _add_rowtile(b, a):
    for r in range(a['r0'], a['r1']):
        b['out'].put(r, 0, [x + y for x, y in zip(b['x'].row(r)[:a['d']], b['pos'].row(r % a['T'])[:a['d']])])


def _tile_nofx(b, a, out):
    """matmul_tile without epilogue into `out` (flags 1, 2 only)."""
    _matmul_tile({'a': b['a'], 'b': b['b'], 'out': out, 'bias': None, 'res': None, 'pre': None}, {**a, 'flags': a['flags'] & (3 | F_BIG | F_HUGE)})


def _tile_cells(a):
    tm, tn = mm_dims(a['flags'])
    return [(m, n) for m in range(a['m0'], min(a['m0'] + tm, a['M'])) for n in range(a['n0'], min(a['n0'] + tn, a['N']))]


@impl('matmul_tile_dgelu', lambda a: _mm_flops(a) + 20.0 * (lambda d: d[0] * d[1])(mm_dims(a['flags'])))
def _matmul_tile_dgelu(b, a):
    out, pre = b['out'], b['pre']
    _tile_nofx(b, a, out)
    for m, n in _tile_cells(a):
        out.a[m * out.cols + n] *= _gelu_d(pre.a[m * pre.cols + n])


@impl('matmul_tile_silu_mul', lambda a: _mm_flops(a) + 10.0 * (lambda d: d[0] * d[1])(mm_dims(a['flags'])))
def _matmul_tile_silu_mul(b, a):
    out, up, gate = b['out'], b['up'], b['gate']
    _tile_nofx(b, a, up)
    for m, n in _tile_cells(a):
        out.a[m * out.cols + n] = _silu(gate.a[m * gate.cols + n]) * up.a[m * up.cols + n]


@impl('matmul_tile_silu_mul_bwd', lambda a: _mm_flops(a) + 20.0 * (lambda d: d[0] * d[1])(mm_dims(a['flags'])))
def _matmul_tile_silu_mul_bwd(b, a):
    dg, du, gate, up = b['dgate'], b['dup'], b['gate'], b['up']
    _tile_nofx(b, a, dg)
    for m, n in _tile_cells(a):
        dh, g, u = dg.a[m * dg.cols + n], gate.a[m * gate.cols + n], up.a[m * up.cols + n]
        dg.a[m * dg.cols + n] = dh * u * _silu_d(g)
        du.a[m * du.cols + n] = dh * _silu(g)
