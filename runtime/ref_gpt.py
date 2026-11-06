"""CPU reference for the GPT operator library (include/kvm_ops_gpt.def, device code in kvm_ops_gpt.cuh).

Numerics: inputs and outputs are float32 values; inside an operator, reductions and transcendentals run in double and
every stored result is rounded to float32 (AdamW follows the device's float32 step order). This matches the device to its
documented tolerance (1e-5 relative for norms, activations, losses, optimizer; 2e-5 attention) but is not bitwise: the
device sums in float32 trees and uses TF32 tensor cores for matmul_tile. `sample` and `argmax` are exact (same Philox).
"""
import math

from ref_types import Buf, RefOp, f32

M32 = 0xFFFFFFFF
_I = frozenset


def _rng(lo, hi, what='range'):
    if not 0 <= lo <= hi:
        raise ValueError(f'{what} [{lo},{hi}) invalid')


def _need(cond, msg):
    if not cond:
        raise ValueError(msg)


def _gelu(x):
    return 0.5 * x * (1.0 + math.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))


def _gelu_grad(x):
    t = math.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x))
    return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * 0.7978845608028654 * (1 + 3 * 0.044715 * x * x)


def _token(i, n, what):
    if not 0 <= i < n:
        raise ValueError(f'{what} id {i} outside [0,{n})')
    return i


# ---- embeddings ---------------------------------------------------------------------------------------------------------
def _embedding_fwd(b, a):
    ids, wte, wpe, out = b['ids'], b['wte'], b['wpe'], b['out']
    for r in range(a['r0'], a['r1']):
        i = _token(ids[r], wte.rows, 'token')
        for c in range(a['d']):
            v = wte.at(i, c)
            if wpe is not None:
                v = f32(v + wpe.at(r % a['T'], c))
            out[r * out.cols + c] = v


def _embedding_bwd(b, a):
    ids, dout, dt, d, N, mod = b['ids'], b['dout'], b['dtab'], a['d'], a['N'], a['mod']
    _need(ids is not None or mod > 0, 'embedding_bwd: ids absent needs mod > 0')
    for key in range(a['v0'], a['v1']):
        base = key * dt.cols
        if not a['acc']:
            for c in range(d):
                dt[base + c] = 0.0
        rows = [r for r in range(N) if ids[r] == key] if ids is not None else range(key, N, mod)
        for r in rows:
            for c in range(d):
                dt[base + c] = f32(dt[base + c] + dout.at(r, c))


# ---- matmul -------------------------------------------------------------------------------------------------------------
def _matmul_tile(b, a):
    A, B, out, bias, res, pre = b['a'], b['b'], b['out'], b['bias'], b['res'], b['pre']
    M, N, K, fl, alpha = a['M'], a['N'], a['K'], a['flags'], a['alpha']
    ta, tb, accum, gelu = fl & 1, fl & 2, fl & 4, fl & 8
    tm, tn = (256, 128) if fl & 64 else (128, 128) if fl & 32 else (64, 64)
    for m in range(a['m0'], min(a['m0'] + tm, M)):
        for n in range(a['n0'], min(a['n0'] + tn, N)):
            acc = 0.0
            for k in range(K):
                x = A.at(k, m) if ta else A.at(m, k)
                y = B.at(n, k) if tb else B.at(k, n)
                acc = f32(acc + f32(x * y))
            v = f32(acc * alpha)
            if bias is not None:
                v = f32(v + bias[n])
            if pre is not None:
                pre[m * pre.cols + n] = v
            if gelu:
                v = f32(_gelu(v))
            if res is not None:
                v = f32(v + res.at(m, n))
            o = m * out.cols + n
            out[o] = f32(out[o] + v) if accum else v


def _matmul_extents(a):
    M, N, K, fl = a['M'], a['N'], a['K'], a['flags']
    _need(M > 0 and N > 0 and K > 0 and 0 <= a['m0'] < M and 0 <= a['n0'] < N, 'matmul_tile: bad M/N/K/m0/n0')
    _need(fl & ~(127 | 256) == 0 and not (fl & 32 and fl & 64), f'matmul_tile: unsupported flags {fl}')   # 256: truncated operands, same exact-product golden
    return {'a': (K, M) if fl & 1 else (M, K), 'b': (N, K) if fl & 2 else (K, N), 'out': (M, N), 'bias': N,
            'res': (M, N), 'pre': (M, N)}


# ---- normalization ------------------------------------------------------------------------------------------------------
def _norm_fwd(rms):
    def run(b, a):
        x, g, be, out, mean, rstd, d = b['x'], b['gamma'], b.get('beta'), b['out'], b.get('mean'), b['rstd'], a['d']
        for r in range(a['r0'], a['r1']):
            row = [x.at(r, c) for c in range(d)]
            mu = 0.0 if rms else sum(row) / d
            rs = f32(1.0 / math.sqrt(sum((v - mu) ** 2 for v in row) / d + a['eps']))
            for c in range(d):
                out[r * out.cols + c] = f32((row[c] - mu) * rs * g[c] + (0.0 if rms else be[c]))
            rstd[r] = rs
            if not rms:
                mean[r] = f32(mu)
    return run


def _norm_fwd_ext(rms):
    def ext(a):
        _rng(a['r0'], a['r1'])
        r1, d = a['r1'], a['d']
        e = {'x': (r1, d), 'gamma': d, 'out': (r1, d), 'rstd': r1}
        return e if rms else {**e, 'beta': d, 'mean': r1}
    return ext


def _norm_bwd_dx(rms):
    def run(b, a):
        x, dy, g, mean, rstd, dx, d = b['x'], b['dy'], b['gamma'], b.get('mean'), b['rstd'], b['dx'], a['d']
        for r in range(a['r0'], a['r1']):
            mu = 0.0 if rms else mean[r]
            rs = rstd[r]
            gs = [dy.at(r, c) * g[c] for c in range(d)]
            xh = [(x.at(r, c) - mu) * rs for c in range(d)]
            s1 = 0.0 if rms else sum(gs) / d
            s2 = sum(p * q for p, q in zip(gs, xh)) / d
            for c in range(d):
                v = f32(rs * (gs[c] - s1 - xh[c] * s2))
                o = r * dx.cols + c
                dx[o] = f32(dx[o] + v) if a['acc'] else v
    return run


def _norm_bwd_dx_ext(rms):
    def ext(a):
        _rng(a['r0'], a['r1'])
        r1, d = a['r1'], a['d']
        e = {'x': (r1, d), 'dy': (r1, d), 'gamma': d, 'rstd': r1, 'dx': (r1, d)}
        return e if rms else {**e, 'mean': r1}
    return ext


def _norm_bwd_params(b, a):
    x, dy, mean, rstd, dg, db = b['x'], b['dy'], b['mean'], b['rstd'], b['dgamma'], b['dbeta']
    for c in range(a['c0'], a['c1']):
        sg = sb = 0.0
        for r in range(a['R']):
            g = dy.at(r, c)
            sg += g * (x.at(r, c) - (mean[r] if mean is not None else 0.0)) * rstd[r]
            sb += g
        dg[c] = f32(dg[c] + sg) if a['acc'] else f32(sg)
        if db is not None:
            db[c] = f32(db[c] + sb) if a['acc'] else f32(sb)


def _norm_bwd_params_ext(a):
    _need(a['R'] >= 0 and 0 <= a['c0'] <= a['c1'] <= a['d'], 'norm_bwd_params: bad R/c0/c1/d')
    R, d = a['R'], a['d']
    return {'x': (R, d), 'dy': (R, d), 'mean': R, 'rstd': R, 'dgamma': d, 'dbeta': d}


def _colsum(b, a):
    x, out = b['x'], b['out']
    for c in range(a['c0'], a['c1']):
        s = 0.0
        for r in range(a['R']):
            s = f32(s + x.at(r, c))
        out[c] = f32(out[c] + s) if a['acc'] else s


def _colsum_ext(a):
    _need(a['R'] >= 0 and 0 <= a['c0'] <= a['c1'], 'colsum: bad R/c0/c1')
    return {'x': (a['R'], a['c1']), 'out': a['c1']}


# ---- elementwise --------------------------------------------------------------------------------------------------------
def _flat_ext(**names):
    """extents for flat [e0,e1) ops: every named buf arg needs e1 elements."""
    def ext(a):
        e0, e1 = a[names.get('lo', 'e0')], a[names.get('hi', 'e1')]
        _rng(e0, e1)
        return {k: e1 for k in names['bufs']}
    return ext


def _gelu_fwd(b, a):
    for i in range(a['e0'], a['e1']):
        b['out'][i] = f32(_gelu(b['x'][i]))


def _gelu_bwd(b, a):
    x, dy, dx = b['x'], b['dy'], b['dx']
    for i in range(a['e0'], a['e1']):
        v = f32(dy[i] * _gelu_grad(x[i]))
        dx[i] = f32(dx[i] + v) if a['acc'] else v


def _residual_add(b, a):
    for i in range(a['e0'], a['e1']):
        b['out'][i] = f32(b['a'][i] + b['b'][i])


def _zero_buf(b, a):
    x = b['x']
    zero = 0 if x.dtype == 'i32' else 0.0
    for i in range(a['e0'], a['e1']):
        x[i] = zero


def _counter_inc(b, a):
    x = b['x']
    for i in range(a['i0'], a['i1']):
        x[i] = (x[i] + a['delta'] + 2**31) % 2**32 - 2**31


def _rope(sign):
    def run(b, a):
        x, out, pos, H, hd, T = b['x'], b['out'], b['pos'], a['H'], a['hd'], a['T']
        p0, half = (pos[0] if pos is not None else 0), a['hd'] // 2
        for r in range(a['r0'], a['r1']):
            for h in range(H):
                for i in range(half):
                    ang = f32(f32((r % T) + p0) * f32(math.pow(a['theta'], -f32(2 * i) / hd)))
                    s, c = sign * f32(math.sin(ang)), f32(math.cos(ang))
                    x1, x2 = x.at(r, h * hd + i), x.at(r, h * hd + i + half)
                    out[r * out.cols + h * hd + i] = f32(x1 * c - x2 * s)
                    out[r * out.cols + h * hd + i + half] = f32(x1 * s + x2 * c)
    return run


def _rope_ext(a):
    _rng(a['r0'], a['r1'])
    _need(a['T'] > 0 and a['H'] > 0 and a['hd'] > 0 and a['hd'] % 2 == 0, 'rope: bad T/H/hd')
    return {'x': (a['r1'], a['H'] * a['hd']), 'out': (a['r1'], a['H'] * a['hd']), 'pos': 1}


# ---- attention ----------------------------------------------------------------------------------------------------------
def _geom(a, kv=False):
    _need(a['hd'] in (16, 32, 64, 128), f"attention: hd {a['hd']} not in 16/32/64/128")
    _need(a['T'] > 0 and a['H'] > 0 and a['Hkv'] > 0 and a['H'] % a['Hkv'] == 0 and a['b'] >= 0 and a['q0'] >= 0 and (a['q0'] & 0xFFFF) < a['T'],
          'attention: bad T/H/Hkv/b/q0')
    _need(0 <= a['h'] < (a['Hkv'] if kv else a['H']), 'attention: head index out of range')


def _attn_fwd(b, a):
    q, k, v, o, lse = b['q'], b['k'], b['v'], b['o'], b['lse']
    bb, h, T, H, Hkv, hd, sc = a['b'], a['h'], a['T'], a['H'], a['Hkv'], a['hd'], a['scale']
    kvh = h // (H // Hkv)
    for gq in range(a['q0'] & 0xFFFF, min((a['q0'] & 0xFFFF) + 32 * ((a['q0'] >> 16) + 1), T)):
        qr = [q.at(bb * T + gq, h * hd + c) for c in range(hd)]
        s = [sum(qr[c] * k.at(bb * T + j, kvh * hd + c) for c in range(hd)) * sc for j in range(gq + 1)]
        m = max(s)
        p = [math.exp(x - m) for x in s]
        l = sum(p)
        for c in range(hd):
            o[(bb * T + gq) * o.cols + h * hd + c] = f32(sum(p[j] * v.at(bb * T + j, kvh * hd + c) for j in range(gq + 1)) / l)
        lse[(bb * H + h) * T + gq] = f32(m + math.log(l))


def _attn_fwd_ext(a):
    _geom(a)
    R = (a['b'] + 1) * a['T']
    return {'q': (R, a['H'] * a['hd']), 'k': (R, a['Hkv'] * a['hd']), 'v': (R, a['Hkv'] * a['hd']),
            'o': (R, a['H'] * a['hd']), 'lse': R * a['H']}


def _attn_bwd_delta(b, a):
    o, do, delta, T, H, hd = b['o'], b['do'], b['delta'], a['T'], a['H'], a['hd']
    for r in range(a['r0'], a['r1']):
        for h in range(H):
            delta[((r // T) * H + h) * T + r % T] = f32(sum(o.at(r, h * hd + c) * do.at(r, h * hd + c) for c in range(hd)))


def _attn_bwd_delta_ext(a):
    _rng(a['r0'], a['r1'])
    _need(a['T'] > 0 and a['H'] > 0 and a['hd'] > 0, 'attn_bwd_delta: bad T/H/hd')
    r1 = a['r1']
    return {'o': (r1, a['H'] * a['hd']), 'do': (r1, a['H'] * a['hd']), 'delta': -(-r1 // a['T']) * a['H'] * a['T']}


def _attn_bwd_dq(b, a):
    q, k, v, do, lse, delta, dq = b['q'], b['k'], b['v'], b['do'], b['lse'], b['delta'], b['dq']
    bb, h, T, H, Hkv, hd, sc = a['b'], a['h'], a['T'], a['H'], a['Hkv'], a['hd'], a['scale']
    kvh = h // (H // Hkv)
    for gq in range(a['q0'] & 0xFFFF, min((a['q0'] & 0xFFFF) + 32 * ((a['q0'] >> 16) + 1), T)):
        li, di = lse[(bb * H + h) * T + gq], delta[(bb * H + h) * T + gq]
        acc = [0.0] * hd
        for j in range(gq + 1):
            s = sum(q.at(bb * T + gq, h * hd + c) * k.at(bb * T + j, kvh * hd + c) for c in range(hd))
            dp = sum(do.at(bb * T + gq, h * hd + c) * v.at(bb * T + j, kvh * hd + c) for c in range(hd))
            ds = math.exp(s * sc - li) * (dp - di)
            for c in range(hd):
                acc[c] += ds * k.at(bb * T + j, kvh * hd + c)
        for c in range(hd):
            dq[(bb * T + gq) * dq.cols + h * hd + c] = f32(acc[c] * sc)


def _attn_bwd_dq_ext(a):
    _geom(a)
    R, hd = (a['b'] + 1) * a['T'], a['hd']
    return {'q': (R, a['H'] * hd), 'k': (R, a['Hkv'] * hd), 'v': (R, a['Hkv'] * hd), 'do': (R, a['H'] * hd),
            'lse': R * a['H'], 'delta': R * a['H'], 'dq': (R, a['H'] * hd)}


def _attn_bwd_dkv(b, a):
    q, k, v, do, lse, delta, dk, dv = (b[n] for n in ('q', 'k', 'v', 'do', 'lse', 'delta', 'dk', 'dv'))
    bb, kvh, T, H, hd, sc = a['b'], a['kvh'], a['T'], a['H'], a['hd'], a['scale']
    G = H // (k.cols // hd)
    for gk in range(a['k0'] & 0xFFFF, min((a['k0'] & 0xFFFF) + 32 * ((a['k0'] >> 16) + 1), T)):
        sk, sv = [0.0] * hd, [0.0] * hd
        for h in range(kvh * G, (kvh + 1) * G):
            for gq in range(gk, T):
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


def _attn_bwd_dkv_ext(a):
    _need(a['hd'] in (16, 32, 64, 128), f"attention: hd {a['hd']} not in 16/32/64/128")
    _need(a['T'] > 0 and a['H'] > 0 and a['b'] >= 0 and 0 <= (a['k0'] & 0xFFFF) < a['T'] and a['k0'] >= 0 and a['kvh'] >= 0, 'attn_bwd_dkv: bad T/H/b/k0/kvh')
    R, hd = (a['b'] + 1) * a['T'], a['hd']
    return {'q': (R, a['H'] * hd), 'k': (R, (a['kvh'] + 1) * hd), 'v': (R, (a['kvh'] + 1) * hd), 'do': (R, a['H'] * hd),
            'lse': R * a['H'], 'delta': R * a['H'], 'dk': (R, (a['kvh'] + 1) * hd), 'dv': (R, (a['kvh'] + 1) * hd)}


def _kv_append(b, a):
    kn, vn, kc, vc, lens = b['knew'], b['vnew'], b['kc'], b['vc'], b['lens']
    nn, Hkv, hd, Tmax = a['n_new'], a['Hkv'], a['hd'], a['Tmax']
    for bb in range(a['b0'], a['b1']):
        if lens[bb] < 0 or lens[bb] + nn > Tmax:
            raise ValueError(f'kv_append: sequence {bb} length {lens[bb]} + {nn} exceeds Tmax {Tmax}')
    for bb in range(a['b0'], a['b1']):
        for t in range(nn):
            for h in range(Hkv):
                row = (bb * Hkv + h) * Tmax + lens[bb] + t
                for c in range(hd):
                    kc[row * kc.cols + c] = kn.at(bb * nn + t, h * hd + c)
                    vc[row * vc.cols + c] = vn.at(bb * nn + t, h * hd + c)


def _kv_append_ext(a):
    _rng(a['b0'], a['b1'])
    _need(a['n_new'] > 0 and a['Hkv'] > 0 and a['hd'] > 0 and a['Tmax'] > 0, 'kv_append: bad n_new/Hkv/hd/Tmax')
    b1, nn, Hkv, hd = a['b1'], a['n_new'], a['Hkv'], a['hd']
    return {'knew': (b1 * nn, Hkv * hd), 'vnew': (b1 * nn, Hkv * hd), 'kc': (b1 * Hkv * a['Tmax'], hd),
            'vc': (b1 * Hkv * a['Tmax'], hd), 'lens': b1}


def _attn_decode(b, a):
    q, kc, vc, lens, o = b['q'], b['kc'], b['vc'], b['lens'], b['o']
    bb, h, H, Hkv, hd, Tmax = a['b'], a['h'], a['H'], a['Hkv'], a['hd'], a['Tmax']
    n = lens[bb] + 1
    if lens[bb] < 0 or n > Tmax:
        raise ValueError(f'attn_decode: sequence {bb} length {lens[bb]} exceeds Tmax {Tmax}')
    base = (bb * Hkv + h // (H // Hkv)) * Tmax
    s = [sum(q.at(bb, h * hd + c) * kc.at(base + j, c) for c in range(hd)) * a['scale'] for j in range(n)]
    m = max(s)
    p = [math.exp(x - m) for x in s]
    l = sum(p)
    for c in range(hd):
        o[bb * o.cols + h * hd + c] = f32(sum(p[j] * vc.at(base + j, c) for j in range(n)) / l)


def _attn_decode_ext(a):
    _need(a['b'] >= 0 and 0 <= a['h'] < a['H'] and a['Hkv'] > 0 and a['H'] % a['Hkv'] == 0 and a['Tmax'] > 0, 'attn_decode: bad b/h/H/Hkv/Tmax')
    _need(0 < a['hd'] <= 128, 'attn_decode: hd must be in 1..128')
    n, hd = a['b'] + 1, a['hd']
    return {'q': (n, a['H'] * hd), 'kc': (n * a['Hkv'] * a['Tmax'], hd), 'vc': (n * a['Hkv'] * a['Tmax'], hd), 'lens': n,
            'o': (n, a['H'] * hd)}


# ---- loss ---------------------------------------------------------------------------------------------------------------
def _ce_fwd(b, a):
    lg, tg, rl, ls, V = b['logits'], b['tgt'], b['rowloss'], b['lse'], a['V']
    for r in range(a['r0'], a['r1']):
        t = tg[r]
        if t == a['ignore']:
            rl[r] = ls[r] = 0.0
            continue
        _token(t, V, 'target')
        row = [lg.at(r, c) for c in range(V)]
        m = max(row)
        lse = m + math.log(sum(math.exp(x - m) for x in row))
        ls[r] = f32(lse)
        rl[r] = f32(lse - row[t])


def _ce_reduce(b, a):
    tg, rl = b['tgt'], b['rowloss']
    s, c = 0.0, 0
    for r in range(a['N']):
        if tg[r] != a['ignore']:
            s += rl[r]
            c += 1
    b['loss'][0] = f32(s / c) if c else 0.0
    b['loss'][1] = f32(1.0 / c) if c else 0.0


def _ce_bwd(b, a):
    lg, tg, ls, loss, dl, V = b['logits'], b['tgt'], b['lse'], b['loss'], b['dlogits'], a['V']
    f = f32(loss[1] * a['scale'])
    for r in range(a['r0'], a['r1']):
        t = tg[r]
        if t != a['ignore']:
            _token(t, V, 'target')
        for c in range(V):
            dl[r * dl.cols + c] = 0.0 if t == a['ignore'] else f32((math.exp(lg.at(r, c) - ls[r]) - (1.0 if c == t else 0.0)) * f)


# ---- optimizer ----------------------------------------------------------------------------------------------------------
def _adamw(b, a):
    p, g, m, v, gs, st = b['p'], b['g'], b['m'], b['v'], b['gscale'], b['step']
    t = st[0]
    _need(t >= 1, f'adamw: step counter {t} < 1')
    b1, b2, lr, eps, wd = a['b1'], a['b2'], a['lr'], a['eps'], a['wd']
    gsc = gs[0] if gs is not None else 1.0
    bc1, bc2 = f32(1.0 - f32(math.pow(b1, t))), f32(1.0 - f32(math.pow(b2, t)))
    for i in range(a['e0'], a['e1']):
        gi = f32(g[i] * gsc)
        mi = f32(f32(b1 * m[i]) + f32(f32(1.0 - b1) * gi))
        vi = f32(f32(b2 * v[i]) + f32(f32(f32(1.0 - b2) * gi) * gi))
        m[i], v[i] = mi, vi
        decay = f32(f32(lr * wd) * p[i])
        step = f32(f32(lr * f32(mi / bc1)) / f32(f32(math.sqrt(f32(vi / bc2))) + eps))
        p[i] = f32(f32(p[i] - decay) - step)


def _gradnorm_partial(b, a):
    b['partial'][a['slot']] = f32(sum(b['g'][i] * b['g'][i] for i in range(a['e0'], a['e1'])))


def _gradnorm_finish(b, a):
    n = f32(math.sqrt(sum(b['partial'][i] for i in range(a['nslots']))))
    mx = a['max_norm']
    b['out'][0] = f32(min(1.0, f32(mx / f32(n + f32(1e-6))))) if mx > 0 else 1.0
    b['out'][1] = n


# ---- sampling (Philox-4x32-10, identical to the device) ------------------------------------------------------------------
def philox4x32_10(c, key):
    """Philox-4x32-10 (Salmon et al. 2011): counter c and key are lists of 32-bit words."""
    c, key = list(c), list(key)
    for _ in range(10):
        p0, p1 = 0xD2511F53 * c[0], 0xCD9E8D57 * c[2]
        c = [(p1 >> 32) ^ c[1] ^ key[0], p1 & M32, (p0 >> 32) ^ c[3] ^ key[1], p0 & M32]
        key = [(key[0] + 0x9E3779B9) & M32, (key[1] + 0xBB67AE85) & M32]
    return c


def philox_uniform(seed, ctr, row):
    """The device's sampling uniform: counter (row, ctr, 0, 0), key (seed, 0x9E3779B9), top 24 bits of word 0."""
    return (philox4x32_10([row & M32, ctr & M32, 0, 0], [seed & M32, 0x9E3779B9])[0] >> 8) * (1.0 / 16777216.0)


def _order_key(x):
    import struct
    bits = struct.unpack('<I', struct.pack('<f', x))[0]
    return bits ^ (M32 if bits >> 31 else 0x80000000)


def _argmax_row(row):
    bv, bi = -math.inf, 0x7FFFFFFF
    for c, x in enumerate(row):
        if x > bv or (x == bv and c < bi):
            bv, bi = x, c
    return 0 if bi == 0x7FFFFFFF else bi


def _sample(b, a):
    lg, out, ctr, V = b['logits'], b['out'], b['ctr'], a['V']
    cv = ctr[0] if ctr is not None else 0
    for r in range(a['r0'], a['r1']):
        row = [lg.at(r, c) for c in range(V)]
        if a['temp'] <= 0:
            out[r] = _argmax_row(row)
            continue
        mx = max([x for x in row if x == x], default=-math.inf)
        thr = 0
        if 0 < a['topk'] < V:
            thr = sorted((_order_key(x) for x in row), reverse=True)[a['topk'] - 1]
        w = [math.exp(f32(x - mx) / a['temp']) if x == x and _order_key(x) >= thr else None for x in row]
        total = 0.0
        for x in w:
            total += x or 0.0
        target = philox_uniform(a['seed'], cv, r) * total
        run, pick, last = 0.0, -1, -1
        for j, x in enumerate(w):
            if x is not None:
                run += x
                last = j
                if run > target:
                    pick = j
                    break
        out[r] = pick if pick >= 0 else max(last, 0)


def _argmax(b, a):
    for r in range(a['r0'], a['r1']):
        b['out'][r] = _argmax_row([b['logits'].at(r, c) for c in range(a['V'])])


# ---- batch plumbing -----------------------------------------------------------------------------------------------------
def _select_window(b, a):
    src, step, dst = b['src'], b['step'], b['dst']
    base = step[0] * a['stride'] + a['off']
    _need(base >= 0 and base + a['i1'] <= len(src), f'select_window: window [{base},{base + a["i1"]}) outside src ({len(src)})')
    for i in range(a['i0'], a['i1']):
        dst[i] = src[base + i]


def _gather_rows(b, a):
    src, idx, dst, d = b['src'], b['idx'], b['dst'], a['d']
    for r in range(a['r0'], a['r1']):
        i = _token(idx[r], src.rows, 'row')
        for c in range(d):
            dst[r * dst.cols + c] = src.at(i, c)


def _R(run, ext, mins=None, ints=(), anydt=()):
    return RefOp(run, ext, mins or {}, _I(ints), _I(anydt))


def _rows_ext(bufs):
    """extents for ops over rows [r0,r1) of [r1 x d] matrices plus flat vectors of r1."""
    def ext(a):
        _rng(a['r0'], a['r1'])
        return {n: (a['r1'], a['d']) if m else a['r1'] for n, m in bufs.items()}
    return ext


GPT_OPS = {
    'embedding_fwd': _R(_embedding_fwd, lambda a: (_rng(a['r0'], a['r1']), _need(a['T'] > 0 and a['d'] > 0, 'embedding_fwd: bad T/d'),
                                                   {'ids': a['r1'], 'wte': (0, a['d']), 'wpe': (a['T'], a['d']), 'out': (a['r1'], a['d'])})[2],
                        ints=('ids',)),
    'embedding_bwd': _R(_embedding_bwd, lambda a: (_need(a['N'] >= 0 and 0 <= a['v0'] <= a['v1'] and a['d'] > 0, 'embedding_bwd: bad N/v0/v1/d'),
                                                   {'ids': a['N'], 'dout': (a['N'], a['d']), 'dtab': (a['v1'], a['d'])})[1], ints=('ids',)),
    'matmul_tile': _R(_matmul_tile, _matmul_extents),
    'layernorm_fwd': _R(_norm_fwd(False), _norm_fwd_ext(False)),
    'rmsnorm_fwd': _R(_norm_fwd(True), _norm_fwd_ext(True)),
    'layernorm_bwd_dx': _R(_norm_bwd_dx(False), _norm_bwd_dx_ext(False)),
    'rmsnorm_bwd_dx': _R(_norm_bwd_dx(True), _norm_bwd_dx_ext(True)),
    'norm_bwd_params': _R(_norm_bwd_params, _norm_bwd_params_ext),
    'gelu_fwd': _R(_gelu_fwd, _flat_ext(bufs=('x', 'out'))),
    'gelu_bwd': _R(_gelu_bwd, _flat_ext(bufs=('x', 'dy', 'dx'))),
    'residual_add': _R(_residual_add, _flat_ext(bufs=('a', 'b', 'out'))),
    'rope_fwd': _R(_rope(1), _rope_ext, ints=('pos',)),
    'rope_bwd': _R(_rope(-1), _rope_ext, ints=('pos',)),
    'attn_fwd': _R(_attn_fwd, _attn_fwd_ext),
    'attn_bwd_delta': _R(_attn_bwd_delta, _attn_bwd_delta_ext),
    'attn_bwd_dq': _R(_attn_bwd_dq, _attn_bwd_dq_ext),
    'attn_bwd_dkv': _R(_attn_bwd_dkv, _attn_bwd_dkv_ext),
    'kv_append': _R(_kv_append, _kv_append_ext, ints=('lens',)),
    'attn_decode': _R(_attn_decode, _attn_decode_ext, ints=('lens',)),
    'softmax_ce_fwd': _R(_ce_fwd, lambda a: (_rng(a['r0'], a['r1']), _need(a['V'] > 0, 'softmax_ce_fwd: V must be > 0'),
                                             {'logits': (a['r1'], a['V']), 'tgt': a['r1'], 'rowloss': a['r1'], 'lse': a['r1']})[2], ints=('tgt',)),
    'softmax_ce_reduce': _R(_ce_reduce, lambda a: (_need(a['N'] >= 0, 'softmax_ce_reduce: N < 0'),
                                                   {'rowloss': a['N'], 'tgt': a['N'], 'loss': 2})[1], ints=('tgt',)),
    'softmax_ce_bwd': _R(_ce_bwd, lambda a: (_rng(a['r0'], a['r1']), _need(a['V'] > 0, 'softmax_ce_bwd: V must be > 0'),
                                             {'logits': (a['r1'], a['V']), 'tgt': a['r1'], 'lse': a['r1'], 'loss': 2,
                                              'dlogits': (a['r1'], a['V'])})[2], ints=('tgt',)),
    'adamw': _R(_adamw, lambda a: (_rng(a['e0'], a['e1']), {'p': a['e1'], 'g': a['e1'], 'm': a['e1'], 'v': a['e1'], 'gscale': 1, 'step': 1})[1],
                ints=('step',)),
    'gradnorm_partial': _R(_gradnorm_partial, lambda a: (_rng(a['e0'], a['e1']), _need(a['slot'] >= 0, 'gradnorm_partial: slot < 0'),
                                                         {'g': a['e1'], 'partial': a['slot'] + 1})[2]),
    'gradnorm_finish': _R(_gradnorm_finish, lambda a: (_need(a['nslots'] >= 0, 'gradnorm_finish: nslots < 0'),
                                                       {'partial': a['nslots'], 'out': 2})[1]),
    'sample': _R(_sample, lambda a: (_rng(a['r0'], a['r1']), _need(a['V'] > 0, 'sample: V must be > 0'),
                                     {'logits': (a['r1'], a['V']), 'out': a['r1'], 'ctr': 1})[2], ints=('out', 'ctr')),
    'argmax': _R(_argmax, lambda a: (_rng(a['r0'], a['r1']), _need(a['V'] > 0, 'argmax: V must be > 0'),
                                     {'logits': (a['r1'], a['V']), 'out': a['r1']})[2], ints=('out',)),
    'zero_buf': _R(_zero_buf, _flat_ext(bufs=('x',)), anydt=('x',)),
    'counter_inc': _R(_counter_inc, _flat_ext(lo='i0', hi='i1', bufs=('x',)), ints=('x',)),
    'colsum': _R(_colsum, _colsum_ext),
    'select_window': _R(_select_window, lambda a: (_rng(a['i0'], a['i1']), {'step': 1, 'dst': a['i1']})[1], ints=('step',), anydt=('src', 'dst')),
    'gather_rows': _R(_gather_rows, lambda a: (_rng(a['r0'], a['r1']), _need(a['d'] > 0, 'gather_rows: d must be > 0'),
                                               {'src': (0, a['d']), 'idx': a['r1'], 'dst': (a['r1'], a['d'])})[2], ints=('idx',)),
}
