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
F_TA, F_TB, F_ACC, F_GELU = 1, 2, 4, 8


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
    return 2.0 * max(0, min(MM_TILE, a['M'] - a['m0'])) * max(0, min(MM_TILE, a['N'] - a['n0'])) * a['K']


@impl('matmul_tile', _mm_flops)
def _matmul_tile(b, a):
    M, N, K, fl = a['M'], a['N'], a['K'], a['flags']
    A, B, out = b['a'], b['b'], b['out']
    m1, n1 = min(a['m0'] + MM_TILE, M), min(a['n0'] + MM_TILE, N)
    bcols = [[(B.a[n * B.cols + k] if fl & F_TB else B.a[k * B.cols + n]) for k in range(K)] for n in range(a['n0'], n1)]
    for m in range(a['m0'], m1):
        arow = [(A.a[k * A.cols + m] if fl & F_TA else A.a[m * A.cols + k]) for k in range(K)]
        vals = []
        for j, n in enumerate(range(a['n0'], n1)):
            v = a['alpha'] * sum(x * y for x, y in zip(arow, bcols[j]))
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
def _attn_fl(a):
    return 4.0 * ATTN_Q * (a.get('q0', a.get('k0', 0)) + ATTN_Q) * a['hd']


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
    for t in range(a['q0'], min(a['q0'] + ATTN_Q, T)):
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
    for t in range(a['q0'], min(a['q0'] + ATTN_Q, T)):
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
    k1 = min(a['k0'] + ATTN_Q, T)
    dk = {j: [0.0] * hd for j in range(a['k0'], k1)}
    dv = {j: [0.0] * hd for j in range(a['k0'], k1)}
    for h in range(kvh * G, (kvh + 1) * G):
        for t in range(a['k0'], T):
            q, do, terms = _ds(b, a, bb, h, kvh, t, range(a['k0'], min(k1, t + 1)))
            for j, p, ds in terms:
                dv[j] = [x + p * y for x, y in zip(dv[j], do)]
                dk[j] = [x + a['scale'] * ds * y for x, y in zip(dk[j], q)]
    for j in range(a['k0'], k1):
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
