"""Independent float64 reference of the frontend's GPT: plain Python lists, hand-written forward AND backward, AdamW.

Shares nothing with kvmfe except the deterministic initial values and batches (so both start from the same point). It does not
use tiles, buffers, device ops or autodiff: it is the oracle the emitted workload is checked against.
"""
import math

import _path  # noqa: F401
from kvmfe.emit import init_values
from kvmfe.nn import GPT
from kvmfe.train import batch_tokens


def mm(a, b):                       # [m,k] @ [k,n]
    bt = list(zip(*b))
    return [[sum(x * y for x, y in zip(r, c)) for c in bt] for r in a]


def tr(a):
    return [list(c) for c in zip(*a)]


def init_params(cfg):
    p = {}
    for n, s in GPT(cfg).specs.items():
        vals = init_values(('param', s.init, f'{cfg.seed}/{n}'), s.rows, s.cols)
        p[n] = [vals[r * s.cols:(r + 1) * s.cols] for r in range(s.rows)]
    return p


def gelu(x):
    return 0.5 * x * (1 + math.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))


def gelu_d(x):
    t = math.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3))
    return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * 0.7978845608028654 * (1 + 3 * 0.044715 * x * x)


def norm_f(x, g, b, rms, eps=1e-5):
    out, cache = [], []
    for r in x:
        mu = 0.0 if rms else sum(r) / len(r)
        rs = 1 / math.sqrt(sum((v - mu) ** 2 for v in r) / len(r) + eps)
        xh = [(v - mu) * rs for v in r]
        out.append([h * gg + (0 if rms else bb) for h, gg, bb in zip(xh, g, b or g)])
        cache.append((xh, rs))
    return out, cache


def norm_b(dy, cache, g, rms):
    dx, dg, db = [], [0.0] * len(g), [0.0] * len(g)
    for d, (xh, rs) in zip(dy, cache):
        gc = [a * gg for a, gg in zip(d, g)]
        s1 = 0.0 if rms else sum(gc) / len(gc)
        s2 = sum(a * h for a, h in zip(gc, xh)) / len(gc)
        dx.append([rs * (a - s1 - h * s2) for a, h in zip(gc, xh)])
        for j, (a, h) in enumerate(zip(d, xh)):
            dg[j] += a * h
            db[j] += a
    return dx, dg, db


def rope(x, cfg, sign=1.0, pos0=0):
    H = cfg.heads
    hd = cfg.d_model // H
    out = []
    for r, row in enumerate(x):
        o = list(row)
        for h in range(H):
            for i in range(hd // 2):
                ang = (r % cfg.seq + pos0) * cfg.rope_theta ** (-2.0 * i / hd)
                c, s = math.cos(ang), sign * math.sin(ang)
                a, b = row[h * hd + i], row[h * hd + i + hd // 2]
                o[h * hd + i], o[h * hd + i + hd // 2] = a * c - b * s, a * s + b * c
        out.append(o)
    return out


def attn(q, k, v, cfg, do=None):
    """Causal attention per sequence/head. Returns o (and dq,dk,dv when do is given)."""
    T, H = cfg.seq, cfg.heads
    hd = cfg.d_model // H
    sc = hd ** -0.5
    o = [[0.0] * cfg.d_model for _ in q]
    dq = [[0.0] * cfg.d_model for _ in q]
    dk = [[0.0] * cfg.d_model for _ in q]
    dv = [[0.0] * cfg.d_model for _ in q]
    for b in range(len(q) // T):
        for h in range(H):
            sl = slice(h * hd, (h + 1) * hd)
            for t in range(T):
                qi = q[b * T + t][sl]
                s = [sc * sum(x * y for x, y in zip(qi, k[b * T + j][sl])) for j in range(t + 1)]
                m = max(s)
                e = [math.exp(x - m) for x in s]
                z = sum(e)
                p = [x / z for x in e]
                for j, pj in enumerate(p):
                    for d_ in range(hd):
                        o[b * T + t][h * hd + d_] += pj * v[b * T + j][h * hd + d_]
                if do is None:
                    continue
                dot = do[b * T + t][sl]
                dp = [sum(x * y for x, y in zip(dot, v[b * T + j][sl])) for j in range(t + 1)]
                dl = sum(pj * dj for pj, dj in zip(p, dp))
                for j in range(t + 1):
                    ds = p[j] * (dp[j] - dl)
                    for d_ in range(hd):
                        dv[b * T + j][h * hd + d_] += p[j] * dot[d_]
                        dq[b * T + t][h * hd + d_] += sc * ds * k[b * T + j][h * hd + d_]
                        dk[b * T + j][h * hd + d_] += sc * ds * qi[d_]
    return (o, dq, dk, dv) if do is not None else o


def lin(x, w, b):
    y = mm(x, w)
    return [[v + bb for v, bb in zip(r, b[0])] for r in y] if b else y


def lin_b(x, w, dy, has_b):
    return mm(dy, tr(w)), mm(tr(x), dy), [sum(c) for c in zip(*dy)] if has_b else None


def loss_and_grads(cfg, p, ids, tgt):
    rms = cfg.norm == 'rmsnorm'
    P = lambda n: p[n]
    c = {}
    x = [list(P('gpt.wte')[int(i)]) for i in ids]
    blocks = []
    for l in range(cfg.layers):
        n = f'gpt.h{l}'
        bk = {'x0': x}
        h, bk['n1'] = norm_f(x, P(f'{n}.n1.g')[0], None if rms else P(f'{n}.n1.b')[0], rms)
        bk['h'] = h
        q0 = lin(h, P(f'{n}.attn.q.w'), P(f'{n}.attn.q.b') if cfg.bias else None)
        k0 = lin(h, P(f'{n}.attn.k.w'), P(f'{n}.attn.k.b') if cfg.bias else None)
        v = lin(h, P(f'{n}.attn.v.w'), P(f'{n}.attn.v.b') if cfg.bias else None)
        q, k = rope(q0, cfg), rope(k0, cfg)
        o = attn(q, k, v, cfg)
        a = lin(o, P(f'{n}.attn.proj.w'), P(f'{n}.attn.proj.b') if cfg.bias else None)
        x2 = [[u + w for u, w in zip(r1, r2)] for r1, r2 in zip(x, a)]
        h2, bk['n2'] = norm_f(x2, P(f'{n}.n2.g')[0], None if rms else P(f'{n}.n2.b')[0], rms)
        pre = lin(h2, P(f'{n}.mlp.fc1.w'), P(f'{n}.mlp.fc1.b') if cfg.bias else None)
        g = [[gelu(v_) for v_ in r] for r in pre]
        m = lin(g, P(f'{n}.mlp.fc2.w'), P(f'{n}.mlp.fc2.b') if cfg.bias else None)
        x3 = [[u + w for u, w in zip(r1, r2)] for r1, r2 in zip(x2, m)]
        bk.update(h2=h2, q=q, k=k, v=v, o=o, x2=x2, pre=pre, g=g)
        blocks.append(bk)
        x = x3
    hf, nfc = norm_f(x, P('gpt.nf.g')[0], None if rms else P('gpt.nf.b')[0], rms)
    logits = mm(hf, P('gpt.head.w'))
    loss, dl = 0.0, []
    for r, t in zip(logits, tgt):
        mx = max(r)
        e = [math.exp(v - mx) for v in r]
        z = sum(e)
        loss += math.log(z) + mx - r[int(t)]
        dl.append([(ev / z - (j == int(t))) / len(ids) for j, ev in enumerate(e)])
    loss /= len(ids)
    G = {}
    dhf, G['gpt.head.w'], _ = lin_b(hf, P('gpt.head.w'), dl, False)
    dx, dg, db = norm_b(dhf, nfc, P('gpt.nf.g')[0], rms)
    G['gpt.nf.g'] = [dg]
    if not rms:
        G['gpt.nf.b'] = [db]
    for l in reversed(range(cfg.layers)):
        n, bk = f'gpt.h{l}', blocks[l]
        dg_, G[f'{n}.mlp.fc2.w'], db2 = lin_b(bk['g'], P(f'{n}.mlp.fc2.w'), dx, cfg.bias)
        dpre = [[gelu_d(a) * b for a, b in zip(r1, r2)] for r1, r2 in zip(bk['pre'], dg_)]
        dh2, G[f'{n}.mlp.fc1.w'], db1 = lin_b(bk['h2'], P(f'{n}.mlp.fc1.w'), dpre, cfg.bias)
        dxn, dg2, dbt2 = norm_b(dh2, bk['n2'], P(f'{n}.n2.g')[0], rms)
        dx2 = [[a + b for a, b in zip(r1, r2)] for r1, r2 in zip(dx, dxn)]
        do_, G[f'{n}.attn.proj.w'], dbp = lin_b(bk['o'], P(f'{n}.attn.proj.w'), dx2, cfg.bias)
        _, dq, dk, dv = attn(bk['q'], bk['k'], bk['v'], cfg, do_)
        dq0, dk0 = rope(dq, cfg, -1.0), rope(dk, cfg, -1.0)
        dh = [[0.0] * cfg.d_model for _ in dq0]
        for nm, d_ in (('q', dq0), ('k', dk0), ('v', dv)):
            dhi, G[f'{n}.attn.{nm}.w'], dbi = lin_b(bk['h'], P(f'{n}.attn.{nm}.w'), d_, cfg.bias)
            dh = [[a + b for a, b in zip(r1, r2)] for r1, r2 in zip(dh, dhi)]
            if cfg.bias:
                G[f'{n}.attn.{nm}.b'] = [dbi]
        dxn1, dg1, dbt1 = norm_b(dh, bk['n1'], P(f'{n}.n1.g')[0], rms)
        dx = [[a + b for a, b in zip(r1, r2)] for r1, r2 in zip(dx2, dxn1)]
        if cfg.bias:
            G[f'{n}.mlp.fc2.b'], G[f'{n}.mlp.fc1.b'], G[f'{n}.attn.proj.b'] = [db2], [db1], [dbp]
        G[f'{n}.n2.g'], G[f'{n}.n1.g'] = [dg2], [dg1]
        if not rms:
            G[f'{n}.n2.b'], G[f'{n}.n1.b'] = [dbt2], [dbt1]
    gw = [[0.0] * cfg.d_model for _ in range(cfg.vocab)]
    for i, r in zip(ids, dx):
        gw[int(i)] = [a + b for a, b in zip(gw[int(i)], r)]
    G['gpt.wte'] = gw
    return loss, G


def train_reference(cfg, steps, opt):
    p, specs = init_params(cfg), GPT(cfg).specs
    m = {n: [[0.0] * len(r) for r in v] for n, v in p.items()}
    v2 = {n: [[0.0] * len(r) for r in v] for n, v in p.items()}
    losses, history = [], [{n: [r[:] for r in v] for n, v in p.items()}]
    for s in range(steps):
        ids, tgt = batch_tokens(cfg, s)
        loss, G = loss_and_grads(cfg, p, ids, tgt)
        losses.append(loss)
        for n in p:
            wd = opt.weight_decay if specs[n].decay else 0.0
            for i, row in enumerate(p[n]):
                for j, w in enumerate(row):
                    g = G[n][i][j]
                    m[n][i][j] = opt.b1 * m[n][i][j] + (1 - opt.b1) * g
                    v2[n][i][j] = opt.b2 * v2[n][i][j] + (1 - opt.b2) * g * g
                    mh, vh = m[n][i][j] / (1 - opt.b1 ** (s + 1)), v2[n][i][j] / (1 - opt.b2 ** (s + 1))
                    row[j] = w - opt.lr * wd * w - opt.lr * mh / (math.sqrt(vh) + opt.eps)
        history.append({n: [r[:] for r in v] for n, v in p.items()})
    return losses, history


def greedy(cfg, p, prompt, n_new):
    """Greedy generation by re-running the full forward each token (no cache): the oracle for the KV-cache serving path."""
    toks, out = list(prompt), []
    for _ in range(n_new):
        logits = _logits(_Seq(cfg, len(toks)), p, toks)
        out.append(max(range(len(logits[-1])), key=lambda j: logits[-1][j]))
        toks.append(out[-1])
    return out


class _Seq:
    """cfg view with a different sequence length (the oracle has no 32-multiple restriction)."""
    def __init__(self, cfg, T):
        self.__dict__.update(cfg.__dict__)
        self.seq, self.batch = T, 1


def _logits(cfg, p, ids):
    rms = cfg.norm == 'rmsnorm'
    x = [list(p['gpt.wte'][int(i)]) for i in ids]
    for l in range(cfg.layers):
        n = f'gpt.h{l}'
        h, _ = norm_f(x, p[f'{n}.n1.g'][0], None if rms else p[f'{n}.n1.b'][0], rms)
        b = lambda s: p[f'{n}.attn.{s}.b'] if cfg.bias else None
        q = rope(lin(h, p[f'{n}.attn.q.w'], b('q')), cfg)
        k = rope(lin(h, p[f'{n}.attn.k.w'], b('k')), cfg)
        v = lin(h, p[f'{n}.attn.v.w'], b('v'))
        a = lin(attn(q, k, v, cfg), p[f'{n}.attn.proj.w'], b('proj'))
        x = [[u + w for u, w in zip(r1, r2)] for r1, r2 in zip(x, a)]
        h2, _ = norm_f(x, p[f'{n}.n2.g'][0], None if rms else p[f'{n}.n2.b'][0], rms)
        g = [[gelu(v_) for v_ in r] for r in lin(h2, p[f'{n}.mlp.fc1.w'], p[f'{n}.mlp.fc1.b'] if cfg.bias else None)]
        m = lin(g, p[f'{n}.mlp.fc2.w'], p[f'{n}.mlp.fc2.b'] if cfg.bias else None)
        x = [[u + w for u, w in zip(r1, r2)] for r1, r2 in zip(x, m)]
    hf, _ = norm_f(x, p['gpt.nf.g'][0], None if rms else p['gpt.nf.b'][0], rms)
    return mm(hf, p['gpt.head.w'])
