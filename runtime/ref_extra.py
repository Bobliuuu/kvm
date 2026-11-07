"""CPU reference (compiler golden, float32 results) of the extra operator pack (include/kvm_ops_extra.def, ids 96-121).

Same model as ref_gpt.py: inputs/outputs are float32, intermediates in double, every stored result rounded to float32. The
semantics are the comments of kvm_ops_extra.def; the frontend's float64 oracle (frontend/kvmfe/opset.py) is an independent
second implementation of the same text.
"""
import math
import struct

from ref_gpt import GPT_OPS, _R, _flat_ext, _need, _rng, _matmul_tile
from ref_types import RefOp, f32

M32 = 0xFFFFFFFF


# ---- scalar helpers ----------------------------------------------------------------------------------------------------
def _sig(x):
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def _silu(x):
    return x * _sig(x)


def _silu_d(x):
    s = _sig(x)
    return s * (1.0 + x * (1.0 - s))


def _gelu_d(x):
    t = math.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x))
    return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * 0.7978845608028654 * (1 + 3 * 0.044715 * x * x)


_ACT = {0: (lambda x: x if x > 0 else 0.0, lambda x: 1.0 if x > 0 else 0.0),
        1: (_silu, _silu_d),
        2: (math.tanh, lambda x: 1.0 - math.tanh(x) ** 2),
        3: (_sig, lambda x: _sig(x) * (1.0 - _sig(x)))}


def bf16_round(x):
    """float32 value rounded to the nearest bfloat16 (ties to even); inf/nan unchanged."""
    if x != x or x in (math.inf, -math.inf):
        return x
    bits = struct.unpack('<I', struct.pack('<f', x))[0]
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack('<f', struct.pack('<I', bits))[0]


def _kind(k, what):
    _need(k in _ACT, f'{what}: kind {k} not in 0..3')
    return _ACT[k]


# ---- elementwise -------------------------------------------------------------------------------------------------------
def _act_fwd(b, a):
    f = _kind(a['kind'], 'act_fwd')[0]
    for i in range(a['e0'], a['e1']):
        b['out'][i] = f32(f(b['x'][i]))


def _act_bwd(b, a):
    d = _kind(a['kind'], 'act_bwd')[1]
    for i in range(a['e0'], a['e1']):
        v = f32(b['dy'][i] * d(b['x'][i]))
        b['dx'][i] = f32(b['dx'][i] + v) if a['acc'] else v


def _silu_mul_fwd(b, a):
    for i in range(a['e0'], a['e1']):
        b['out'][i] = f32(_silu(b['gate'][i]) * b['up'][i])


def _silu_mul_bwd(b, a):
    g, u, dy, dg, du = (b[n] for n in ('gate', 'up', 'dy', 'dgate', 'dup'))
    for i in range(a['e0'], a['e1']):
        vg, vu = f32(dy[i] * f32(u[i] * _silu_d(g[i]))), f32(dy[i] * _silu(g[i]))
        dg[i] = f32(dg[i] + vg) if a['acc'] else vg
        du[i] = f32(du[i] + vu) if a['acc'] else vu


def _softmax_rows_fwd(b, a):
    x, out, d, sc = b['x'], b['out'], a['d'], a['scale']
    for r in range(a['r0'], a['r1']):
        s = [x.at(r, c) * sc for c in range(d)]
        m = max(s)
        e = [math.exp(v - m) for v in s]
        z = sum(e)
        for c in range(d):
            out[r * out.cols + c] = f32(e[c] / z)


def _softmax_rows_bwd(b, a):
    p, dp, dx, d, sc = b['p'], b['dp'], b['dx'], a['d'], a['scale']
    for r in range(a['r0'], a['r1']):
        dot = sum(p.at(r, c) * dp.at(r, c) for c in range(d))
        for c in range(d):
            v = f32(sc * p.at(r, c) * (dp.at(r, c) - dot))
            o = r * dx.cols + c
            dx[o] = f32(dx[o] + v) if a['acc'] else v


def _flat_rows_ext(names, r1, d):
    return {n: (r1, d) for n in names}


# ---- gather / scatter ----------------------------------------------------------------------------------------------------
def _gather_rows_scaled(b, a):
    src, idx, sc, dst, d = b['src'], b['idx'], b['scale'], b['dst'], a['d']
    for r in range(a['r0'], a['r1']):
        j = idx[a['idx0'] + r]
        s = sc[a['idx0'] + r] if sc is not None else 1.0
        for c in range(d):
            v = f32(s * src.at(j, c)) if 0 <= j < src.rows else 0.0
            o = r * dst.cols + c
            dst[o] = f32(dst[o] + v) if a['acc'] else v


def _scatter_add_rows(b, a):
    src, idx, sc, dst, d = b['src'], b['idx'], b['scale'], b['dst'], a['d']
    for v in range(a['v0'], a['v1']):
        acc = [0.0] * d
        for r in range(a['N']):
            if idx[a['idx0'] + r] == v:
                s = sc[a['idx0'] + r] if sc is not None else 1.0
                for c in range(d):
                    acc[c] = f32(acc[c] + f32(s * src.at(r, c)))
        for c in range(d):
            o = v * dst.cols + c
            dst[o] = f32(dst[o] + acc[c]) if a['acc'] else acc[c]


def _row_dot(b, a):
    A, B, idx, out, d = b['a'], b['b'], b['idx'], b['out'], a['d']
    for r in range(a['r0'], a['r1']):
        j = idx[a['idx0'] + r] if idx is not None else r
        v = f32(sum(A.at(j, c) * B.at(r, c) for c in range(d))) if 0 <= j < A.rows else 0.0
        o = a['idx0'] + r
        out[o] = f32(out[o] + v) if a['acc'] else v


# ---- MoE routing -----------------------------------------------------------------------------------------------------
def _moe_geom(a, C=None):
    _need(a['N'] > 0 and a['E'] > 0 and 0 < a['k'] <= a['E'], 'moe: need N > 0 and 0 < k <= E')
    _need(C is None or C > 0, 'moe: capacity must be > 0')


def _moe_route(b, a):
    N, E, k, C = a['N'], a['E'], a['k'], a['C']
    P, sel, slot, eidx = b['probs'], b['sel'], b['slot'], b['eidx']
    for s in range(E * C):
        eidx[s] = -1
    cnt = [0] * E
    for t in range(N):
        order = sorted(range(E), key=lambda e: (-P.at(t, e), e))[:k]
        for j, e in enumerate(order):
            sel[t * k + j] = e
            if cnt[e] < C:
                slot[t * k + j] = e * C + cnt[e]
                eidx[e * C + cnt[e]] = t
                cnt[e] += 1
            else:
                slot[t * k + j] = -1


def _moe_gates_fwd(b, a):
    N, E, k = a['N'], a['E'], a['k']
    P, sel, slot, gs = b['probs'], b['sel'], b['slot'], b['gs']
    for s in range(len(gs)):
        gs[s] = 0.0
    for t in range(N):
        p = [P.at(t, sel[t * k + j]) for j in range(k)]
        z = sum(p) if a['renorm'] else 1.0
        for j in range(k):
            if slot[t * k + j] >= 0:
                gs[slot[t * k + j]] = f32(p[j] / z)


def _moe_gates_bwd(b, a):
    N, E, k = a['N'], a['E'], a['k']
    P, sel, slot, dgs, dP = b['probs'], b['sel'], b['slot'], b['dgs'], b['dprobs']
    if not a['acc']:
        for i in range(N * E):
            dP[i] = 0.0
    for t in range(N):
        p = [P.at(t, sel[t * k + j]) for j in range(k)]
        dg = [dgs[slot[t * k + j]] if slot[t * k + j] >= 0 else 0.0 for j in range(k)]
        if a['renorm']:
            z = sum(p)
            w = sum(dg[j] * p[j] for j in range(k)) / (z * z)
            dp = [dg[j] / z - w for j in range(k)]
        else:
            dp = dg
        for j in range(k):
            o = t * E + sel[t * k + j]
            dP[o] = f32(dP[o] + dp[j])


def _balance_f(b, a):
    N, E, k, sel = a['N'], a['E'], a['k'], b['sel']
    cnt = [0] * E
    for i in range(N * k):
        cnt[sel[i]] += 1
    return [c / (N * k) for c in cnt]


def _moe_balance_fwd(b, a):
    N, E = a['N'], a['E']
    f = _balance_f(b, a)
    pm = [sum(b['probs'].at(t, e) for t in range(N)) / N for e in range(E)]
    b['loss'][0] = f32(a['coef'] * E * sum(f[e] * pm[e] for e in range(E)))


def _moe_balance_bwd(b, a):
    N, E = a['N'], a['E']
    f = _balance_f(b, a)
    dP = b['dprobs']
    for t in range(N):
        for e in range(E):
            v = f32(a['coef'] * E * f[e] / N)
            o = t * E + e
            dP[o] = f32(dP[o] + v) if a['acc'] else v


# ---- bidirectional attention -------------------------------------------------------------------------------------------
def _full_fwd(b, a):
    q, k, v, o, lse = b['q'], b['k'], b['v'], b['o'], b['lse']
    bb, h, T, H, Hkv, hd, sc = a['b'], a['h'], a['T'], a['H'], a['Hkv'], a['hd'], a['scale']
    kvh = h // (H // Hkv)
    for gq in range(a['q0'], min(a['q0'] + 32, T)):
        s = [sum(q.at(bb * T + gq, h * hd + c) * k.at(bb * T + j, kvh * hd + c) for c in range(hd)) * sc for j in range(T)]
        m = max(s)
        p = [math.exp(x - m) for x in s]
        l = sum(p)
        for c in range(hd):
            o[(bb * T + gq) * o.cols + h * hd + c] = f32(sum(p[j] * v.at(bb * T + j, kvh * hd + c) for j in range(T)) / l)
        lse[(bb * H + h) * T + gq] = f32(m + math.log(l))


def _full_dq(b, a):
    q, k, v, do, lse, delta, dq = (b[n] for n in ('q', 'k', 'v', 'do', 'lse', 'delta', 'dq'))
    bb, h, T, H, Hkv, hd, sc = a['b'], a['h'], a['T'], a['H'], a['Hkv'], a['hd'], a['scale']
    kvh = h // (H // Hkv)
    for gq in range(a['q0'], min(a['q0'] + 32, T)):
        li, di = lse[(bb * H + h) * T + gq], delta[(bb * H + h) * T + gq]
        acc = [0.0] * hd
        for j in range(T):
            s = sum(q.at(bb * T + gq, h * hd + c) * k.at(bb * T + j, kvh * hd + c) for c in range(hd))
            dp = sum(do.at(bb * T + gq, h * hd + c) * v.at(bb * T + j, kvh * hd + c) for c in range(hd))
            ds = math.exp(s * sc - li) * (dp - di)
            for c in range(hd):
                acc[c] += ds * k.at(bb * T + j, kvh * hd + c)
        for c in range(hd):
            dq[(bb * T + gq) * dq.cols + h * hd + c] = f32(acc[c] * sc)


def _full_dkv(b, a):
    q, k, v, do, lse, delta, dk, dv = (b[n] for n in ('q', 'k', 'v', 'do', 'lse', 'delta', 'dk', 'dv'))
    bb, kvh, T, H, hd, sc = a['b'], a['kvh'], a['T'], a['H'], a['hd'], a['scale']
    G = H // (k.cols // hd)
    for gk in range(a['k0'], min(a['k0'] + 32, T)):
        sk, sv = [0.0] * hd, [0.0] * hd
        for h in range(kvh * G, (kvh + 1) * G):
            for gq in range(T):
                li, di = lse[(bb * H + h) * T + gq], delta[(bb * H + h) * T + gq]
                s = sum(q.at(bb * T + gq, h * hd + c) * k.at(bb * T + gk, kvh * hd + c) for c in range(hd))
                dp = sum(do.at(bb * T + gq, h * hd + c) * v.at(bb * T + gk, kvh * hd + c) for c in range(hd))
                p = math.exp(s * sc - li)
                for c in range(hd):
                    sv[c] += p * do.at(bb * T + gq, h * hd + c)
                    sk[c] += p * (dp - di) * q.at(bb * T + gq, h * hd + c)
        for c in range(hd):
            dk[(bb * T + gk) * dk.cols + kvh * hd + c] = f32(sk[c] * sc)
            dv[(bb * T + gk) * dv.cols + kvh * hd + c] = f32(sv[c])


def _geom(a, kv=False):
    _need(a['hd'] in (16, 32, 64, 128), f"attention: hd {a['hd']} not in 16/32/64/128")
    _need(a['T'] > 0 and a['H'] > 0 and a['b'] >= 0 and 0 <= a['q0' if 'q0' in a else 'k0'] < a['T'], 'attn_full: bad T/H/b/q0')
    if 'Hkv' in a:
        _need(a['Hkv'] > 0 and a['H'] % a['Hkv'] == 0, 'attn_full: bad Hkv')


def _full_fwd_ext(a):
    _geom(a)
    R = (a['b'] + 1) * a['T']
    return {'q': (R, a['H'] * a['hd']), 'k': (R, a['Hkv'] * a['hd']), 'v': (R, a['Hkv'] * a['hd']),
            'o': (R, a['H'] * a['hd']), 'lse': R * a['H']}


def _full_dq_ext(a):
    _geom(a)
    R, hd = (a['b'] + 1) * a['T'], a['hd']
    return {'q': (R, a['H'] * hd), 'k': (R, a['Hkv'] * hd), 'v': (R, a['Hkv'] * hd), 'do': (R, a['H'] * hd),
            'lse': R * a['H'], 'delta': R * a['H'], 'dq': (R, a['H'] * hd)}


def _full_dkv_ext(a):
    _geom(a)
    _need(a['kvh'] >= 0, 'attn_full_bwd_dkv: bad kvh')
    R, hd = (a['b'] + 1) * a['T'], a['hd']
    return {'q': (R, a['H'] * hd), 'k': (R, (a['kvh'] + 1) * hd), 'v': (R, (a['kvh'] + 1) * hd), 'do': (R, a['H'] * hd),
            'lse': R * a['H'], 'delta': R * a['H'], 'dk': (R, (a['kvh'] + 1) * hd), 'dv': (R, (a['kvh'] + 1) * hd)}


# ---- mixed precision, schedules, epilogues ------------------------------------------------------------------------------
def _round_bf16(b, a):
    for i in range(a['e0'], a['e1']):
        b['out'][i] = f32(bf16_round(b['x'][i]))


def _matmul_tile_bf16(b, a):
    rb = dict(b)
    rb['a'] = type(b['a'])([bf16_round(x) for x in b['a']], b['a'].rows, b['a'].cols)
    rb['b'] = type(b['b'])([bf16_round(x) for x in b['b']], b['b'].rows, b['b'].cols)
    _matmul_tile(rb, a)
    # bf16-STORAGE outputs (dtype 'bf16'): one rounding per stored value, after the fp32 epilogue (pre before gelu, as on the device)
    for name in ('out', 'pre'):
        t = b.get(name)
        if t is not None and t.dtype == 'bf16':
            for m in range(a['m0'], min(a['m0'] + 64, a['M'])):
                for n in range(a['n0'], min(a['n0'] + 64, a['N'])):
                    t[m * t.cols + n] = bf16_round(t[m * t.cols + n])


def _matmul_mma_bf16(b, a):
    """The 128 x 256 tile as the eight 64 x 64 matmul_tile_bf16 tiles it equals (same operand rounding, same epilogue order)."""
    for dm in (0, 64):
        for dn in (0, 64, 128, 192):
            if a['m0'] + dm < a['M'] and a['n0'] + dn < a['N']:
                _matmul_tile_bf16(b, {**a, 'm0': a['m0'] + dm, 'n0': a['n0'] + dn})


def _adamw_lr(b, a):
    from ref_gpt import _adamw
    _adamw(b, {**a, 'lr': b['lr'][0]})


def _lr_schedule(b, a):
    t = b['step'][0]
    warm = t / a['warmup'] if a['warmup'] > 0 and t <= a['warmup'] else 1.0
    prog = min(max((t - a['warmup']) / max(a['total'] - a['warmup'], 1), 0.0), 1.0)
    mr = a['min_ratio']
    f = {0: 1.0, 1: mr + (1 - mr) * 0.5 * (1 + math.cos(math.pi * prog)), 2: 1.0 - (1 - mr) * prog}[a['kind']]
    b['lr'][0] = f32(a['base'] * warm * f)


def _scale_flat(b, a):
    for i in range(a['e0'], a['e1']):
        b['out'][i] = f32(a['alpha'] * b['x'][i])


def _add_rowtile(b, a):
    x, pos, out, d = b['x'], b['pos'], b['out'], a['d']
    for r in range(a['r0'], a['r1']):
        for c in range(d):
            out[r * out.cols + c] = f32(x.at(r, c) + pos.at(r % a['T'], c))


def _tile_mm(b, a, out, extra=None):
    """matmul_tile on the 64x64 tile with no epilogue, written into `out`."""
    mb = {'a': b['a'], 'b': b['b'], 'out': out, 'bias': None, 'res': None, 'pre': None}
    _matmul_tile(mb, {**a, 'flags': a['flags'] & 3})


def _tile_range(a):
    return range(a['m0'], min(a['m0'] + 64, a['M'])), range(a['n0'], min(a['n0'] + 64, a['N']))


def _mm_dgelu(b, a):
    _tile_mm(b, a, b['out'])
    pre, out = b['pre'], b['out']
    ms, ns = _tile_range(a)
    for m in ms:
        for n in ns:
            out[m * out.cols + n] = f32(out[m * out.cols + n] * _gelu_d(pre.at(m, n)))


def _mm_silu_mul(b, a):
    _tile_mm(b, a, b['up'])
    gate, up, out = b['gate'], b['up'], b['out']
    ms, ns = _tile_range(a)
    for m in ms:
        for n in ns:
            out[m * out.cols + n] = f32(_silu(gate.at(m, n)) * up.at(m, n))


def _mm_silu_mul_bwd(b, a):
    _tile_mm(b, a, b['dgate'])
    gate, up, dg, du = b['gate'], b['up'], b['dgate'], b['dup']
    ms, ns = _tile_range(a)
    for m in ms:
        for n in ns:
            dh, o = dg.at(m, n), m * dg.cols + n
            dg[o] = f32(dh * f32(up.at(m, n) * _silu_d(gate.at(m, n))))
            du[m * du.cols + n] = f32(dh * _silu(gate.at(m, n)))


def _mm_ext(extra):
    def ext(a):
        M, N, K, fl = a['M'], a['N'], a['K'], a['flags']
        _need(M > 0 and N > 0 and K > 0 and 0 <= a['m0'] < M and 0 <= a['n0'] < N, 'matmul_tile*: bad M/N/K/m0/n0')
        _need(fl & ~19 == 0, f'matmul_tile*: unsupported flags {fl}')
        return {'a': (K, M) if fl & 1 else (M, K), 'b': (N, K) if fl & 2 else (K, N), **{n: (M, N) for n in extra}}
    return ext


def _rows_ext(a, names, mat=True):
    _rng(a['r0'], a['r1'])
    return {n: (a['r1'], a['d']) for n in names}


def _bad_if(cond, msg):
    _need(not cond, msg)


EXTRA_OPS = {
    'act_fwd': _R(_act_fwd, lambda a: (_kind(a['kind'], 'act_fwd'), _flat_ext(bufs=('x', 'out'))(a))[1]),
    'act_bwd': _R(_act_bwd, lambda a: (_kind(a['kind'], 'act_bwd'), _flat_ext(bufs=('x', 'dy', 'dx'))(a))[1]),
    'silu_mul_fwd': _R(_silu_mul_fwd, _flat_ext(bufs=('gate', 'up', 'out'))),
    'silu_mul_bwd': _R(_silu_mul_bwd, _flat_ext(bufs=('gate', 'up', 'dy', 'dgate', 'dup'))),
    'softmax_rows_fwd': _R(_softmax_rows_fwd, lambda a: (_need(a['d'] > 0, 'softmax_rows: d > 0'), _rows_ext(a, ('x', 'out')))[1]),
    'softmax_rows_bwd': _R(_softmax_rows_bwd, lambda a: (_need(a['d'] > 0, 'softmax_rows: d > 0'), _rows_ext(a, ('p', 'dp', 'dx')))[1]),
    'gather_rows_scaled': _R(_gather_rows_scaled, lambda a: (_rng(a['r0'], a['r1']), _need(a['d'] > 0, 'gather_rows_scaled: d > 0'),
                                                              {'src': (0, a['d']), 'idx': a['idx0'] + a['r1'], 'scale': a['idx0'] + a['r1'], 'dst': (a['r1'], a['d'])})[2],
                             ints=('idx',)),
    'scatter_add_rows': _R(_scatter_add_rows, lambda a: (_need(a['N'] >= 0 and 0 <= a['v0'] <= a['v1'] and a['d'] > 0, 'scatter_add_rows: bad N/v0/v1/d'),
                                                          {'src': (a['N'], a['d']), 'idx': a['idx0'] + a['N'], 'scale': a['idx0'] + a['N'], 'dst': (a['v1'], a['d'])})[1],
                           ints=('idx',)),
    'row_dot': _R(_row_dot, lambda a: (_rng(a['r0'], a['r1']), _need(a['d'] > 0, 'row_dot: d > 0'),
                                       {'a': (0, a['d']), 'b': (a['r1'], a['d']), 'idx': a['idx0'] + a['r1'], 'out': a['idx0'] + a['r1']})[2], ints=('idx',)),
    'moe_route': _R(_moe_route, lambda a: (_moe_geom(a, a['C']), {'probs': (a['N'], a['E']), 'sel': a['N'] * a['k'], 'slot': a['N'] * a['k'],
                                                                   'eidx': a['E'] * a['C']})[1], ints=('sel', 'slot', 'eidx')),
    'moe_gates_fwd': _R(_moe_gates_fwd, lambda a: (_moe_geom(a), {'probs': (a['N'], a['E']), 'sel': a['N'] * a['k'], 'slot': a['N'] * a['k'],
                                                                   'gs': 1})[1], ints=('sel', 'slot')),
    'moe_gates_bwd': _R(_moe_gates_bwd, lambda a: (_moe_geom(a), {'probs': (a['N'], a['E']), 'sel': a['N'] * a['k'], 'slot': a['N'] * a['k'],
                                                                   'dgs': 1, 'dprobs': (a['N'], a['E'])})[1], ints=('sel', 'slot')),
    'moe_balance_fwd': _R(_moe_balance_fwd, lambda a: (_moe_geom(a), {'probs': (a['N'], a['E']), 'sel': a['N'] * a['k'], 'loss': 1})[1],
                          ints=('sel',)),
    'moe_balance_bwd': _R(_moe_balance_bwd, lambda a: (_moe_geom(a), {'probs': (a['N'], a['E']), 'sel': a['N'] * a['k'],
                                                                       'dprobs': (a['N'], a['E'])})[1], ints=('sel',)),
    'attn_full_fwd': _R(_full_fwd, _full_fwd_ext),
    'attn_full_bwd_dq': _R(_full_dq, _full_dq_ext),
    'attn_full_bwd_dkv': _R(_full_dkv, _full_dkv_ext),
    'round_bf16': _R(_round_bf16, _flat_ext(bufs=('x', 'out')), anydt=('x', 'out')),      # storage conversion: each side f32 or bf16
    'matmul_tile_bf16': _R(_matmul_tile_bf16, GPT_OPS['matmul_tile'].extents, anydt=('a', 'b', 'out', 'res', 'pre')),
    'matmul_mma_bf16': _R(_matmul_mma_bf16, GPT_OPS['matmul_tile'].extents, anydt=('a', 'b', 'out', 'res', 'pre')),
    'adamw_lr': _R(_adamw_lr, lambda a: (_rng(a['e0'], a['e1']), {'p': a['e1'], 'g': a['e1'], 'm': a['e1'], 'v': a['e1'], 'gscale': 1, 'step': 1,
                                                                  'lr': 1})[1], ints=('step',)),
    'lr_schedule': _R(_lr_schedule, lambda a: (_need(a['kind'] in (0, 1, 2), 'lr_schedule: kind not in 0..2'), {'step': 1, 'lr': 1})[1],
                      ints=('step',)),
    'scale_flat': _R(_scale_flat, _flat_ext(bufs=('x', 'out'))),
    'add_rowtile': _R(_add_rowtile, lambda a: (_rng(a['r0'], a['r1']), _need(a['T'] > 0 and a['d'] > 0, 'add_rowtile: bad T/d'),
                                               {'x': (a['r1'], a['d']), 'pos': (a['T'], a['d']), 'out': (a['r1'], a['d'])})[2]),
    'matmul_tile_dgelu': _R(_mm_dgelu, _mm_ext(('out', 'pre'))),
    'matmul_tile_silu_mul': _R(_mm_silu_mul, _mm_ext(('out', 'gate', 'up'))),
    'matmul_tile_silu_mul_bwd': _R(_mm_silu_mul_bwd, _mm_ext(('gate', 'up', 'dgate', 'dup'))),
}


# ---- bf16 STORAGE variants of GPT-pack ops --------------------------------------------------------------------------------------
# These device ops take fp32 or bf16-storage buffers (ld_s / st_s) on the arguments below; the arithmetic is the fp32 op's, and every value stored
# to a bf16 buffer is rounded once (RNE). The reference runs the fp32 model, then rounds the whole bf16 output buffer: rounding is idempotent,
# so values the task did not write (already bf16) are unchanged. Cost is O(buffer) per task: this is the small-model golden, not a 162M oracle.
_BF16_STORAGE = {   # op: (buffers that may be bf16 storage, of which written)
    'embedding_fwd': (('out',), ('out',)),
    'layernorm_fwd': (('x', 'out'), ('out',)),
    'rmsnorm_fwd': (('x', 'out'), ('out',)),
    'layernorm_bwd_dx': (('x', 'dy', 'dx'), ('dx',)),
    'rmsnorm_bwd_dx': (('x', 'dy', 'dx'), ('dx',)),
    'norm_bwd_params': (('x', 'dy'), ()),
    'colsum': (('x',), ()),
    'gelu_bwd': (('x', 'dy', 'dx'), ('dx',)),
    'residual_add': (('a', 'b', 'out'), ('out',)),
    'rope_fwd': (('x', 'out'), ('out',)),
    'rope_bwd': (('x', 'out'), ('out',)),
    'softmax_ce_fwd': (('logits',), ()),
    'softmax_ce_bwd': (('logits', 'dlogits'), ('dlogits',)),
}


def _bf16_storage(base, written):
    def run(b, a):
        base.run(b, a)
        for n in written:
            t = b.get(n)
            if t is not None and t.dtype == 'bf16':
                t[:] = [bf16_round(x) for x in t]
    return run


for _name, (_any, _written) in _BF16_STORAGE.items():
    _base = GPT_OPS[_name]
    EXTRA_OPS[_name] = RefOp(_bf16_storage(_base, _written), _base.extents, _base.mins, _base.ints, _base.anydt | frozenset(_any), _base.opaque)
