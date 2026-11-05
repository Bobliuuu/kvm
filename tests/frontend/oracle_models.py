"""Independent float64 forward oracles (numpy) of the model zoo: Llama/GQA/SwiGLU/tied embeddings/MoE, MLP classifier, ViT.

Shares nothing with kvmfe except the deterministic initial values and data streams (so both start from the same point). No tiles,
buffers, device ops or autodiff: gradients are checked by central finite differences of these functions (fd_check), which makes
the frontend's autodiff the thing under test. mixed precision: `lp` rounds matmul operands and stored activations to bfloat16.
"""
import math

import numpy as np

import _path  # noqa: F401
from kvmfe.emit import init_values
from kvmfe.train import param_init


def params(model):
    out = {}
    for n, s in model.specs.items():
        out[n] = np.array(init_values(param_init(s, model.cfg.seed), s.rows, s.cols), dtype=np.float64).reshape(s.rows, s.cols)
    return out


def streams(model):
    """The first microbatch of every stream as float64 / int arrays (f32-rounded exactly as the emitted buffers are)."""
    out = {}
    for st in model.batch_streams(1):
        a = np.array(st.values[:st.rows * st.cols], dtype=np.float32 if st.dtype == 'f32' else np.int64).reshape(st.rows, st.cols)
        out[st.name] = a.astype(np.float64) if st.dtype == 'f32' else a[:, 0]
    return out


def bf16(x):
    """Round float64 values through float32 to the nearest bfloat16 (ties to even)."""
    f = np.asarray(x, dtype=np.float32)
    u = f.view(np.uint32).astype(np.uint64)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000).astype(np.uint32)
    return r.view(np.float32).astype(np.float64)


def mm(a, b, lp=False):
    return bf16(a) @ bf16(b) if lp else a @ b


def lin(x, w, b=None, lp=False, gelu=False, res=None):
    y = mm(x, w, lp)
    if b is not None:
        y = y + b
    if gelu:
        y = 0.5 * y * (1 + np.tanh(0.7978845608028654 * (y + 0.044715 * y ** 3)))
    if res is not None:
        y = y + res
    return bf16(y) if lp else y


def rmsnorm(x, g, eps=1e-5):
    return x / np.sqrt((x * x).mean(1, keepdims=True) + eps) * g


def layernorm(x, g, b, eps=1e-5):
    mu = x.mean(1, keepdims=True)
    return (x - mu) / np.sqrt(((x - mu) ** 2).mean(1, keepdims=True) + eps) * g + b


def rope(x, H, T, theta):
    R, hd = x.shape[0], x.shape[1] // H
    half = hd // 2
    ang = (np.arange(R) % T)[:, None] * theta ** (-2.0 * np.arange(half) / hd)[None, :]
    c, s = np.cos(ang), np.sin(ang)
    out = x.copy().reshape(R, H, hd)
    a, b = out[:, :, :half].copy(), out[:, :, half:].copy()
    out[:, :, :half] = a * c[:, None, :] - b * s[:, None, :]
    out[:, :, half:] = a * s[:, None, :] + b * c[:, None, :]
    return out.reshape(R, H * hd)


def attention(q, k, v, H, Hkv, T, causal):
    R, hd = q.shape[0], q.shape[1] // H
    G = H // Hkv
    out = np.zeros_like(q)
    for b in range(R // T):
        sl = slice(b * T, (b + 1) * T)
        for h in range(H):
            qh = q[sl, h * hd:(h + 1) * hd]
            kh, vh = k[sl, h // G * hd:(h // G + 1) * hd], v[sl, h // G * hd:(h // G + 1) * hd]
            s = qh @ kh.T * hd ** -0.5
            if causal:
                s = np.where(np.tril(np.ones((T, T), bool)), s, -np.inf)
            p = np.exp(s - s.max(1, keepdims=True))
            out[sl, h * hd:(h + 1) * hd] = (p / p.sum(1, keepdims=True)) @ vh
    return out


def ce(logits, tgt):
    m = logits.max(1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(logits - m).sum(1))
    return float((lse - logits[np.arange(len(tgt)), tgt]).mean())


def silu(x):
    return x / (1 + np.exp(-x))


def swiglu(h, p, n, lp, res=None):
    return lin(silu(lin(h, p[f'{n}.gate.w'], lp=lp)) * lin(h, p[f'{n}.up.w'], lp=lp), p[f'{n}.down.w'], lp=lp, res=res)


def moe_route(probs, k, C):
    """Independent routing: top-k per token (ties -> lowest expert), capacity by token order. Returns sel, slot, eidx."""
    N, E = probs.shape
    sel, slot, eidx, cnt = np.zeros((N, k), int), -np.ones((N, k), int), -np.ones(E * C, int), [0] * E
    for t in range(N):
        order = sorted(range(E), key=lambda e: (-probs[t, e], e))[:k]
        for j, e in enumerate(order):
            sel[t, j] = e
            if cnt[e] < C:
                slot[t, j] = e * C + cnt[e]
                eidx[e * C + cnt[e]] = t
                cnt[e] += 1
    return sel, slot, eidx


def moe_ffn(h, p, n, cfg, res, lp, aux):
    N, E, k, C = h.shape[0], cfg.experts, cfg.top_k, cfg.capacity
    logits = lin(h, p[f'{n}.router.w'])        # the router stays fp32
    pr = np.exp(logits - logits.max(1, keepdims=True))
    pr /= pr.sum(1, keepdims=True)
    sel, slot, eidx = moe_route(pr, k, C)
    f = np.bincount(sel.ravel(), minlength=E) / (N * k)
    aux.append(cfg.balance_coef * E * float((f * pr.mean(0)).sum()))
    gate = np.zeros(E * C)
    for t in range(N):
        z = pr[t, sel[t]].sum()
        for j in range(k):
            if slot[t, j] >= 0:
                gate[slot[t, j]] = pr[t, sel[t, j]] / z
    out = res.copy()
    for e in range(E):
        for r in range(C):
            t = eidx[e * C + r]
            if t >= 0:
                ye = swiglu(h[t:t + 1], p, f'{n}.e{e}', lp)
                out[t] += gate[e * C + r] * ye[0]
    return out


def llama_logits(c, lp, p, ids, aux):
    x = p['llama.wte'][ids]
    x = bf16(x) if lp else x
    for l in range(c.layers):
        n = f'llama.h{l}'
        h = rmsnorm(x, p[f'{n}.n1.g'])
        q = rope(lin(h, p[f'{n}.attn.q.w'], lp=lp), c.heads, c.seq, c.rope_theta)
        k = rope(lin(h, p[f'{n}.attn.k.w'], lp=lp), c.kv_heads, c.seq, c.rope_theta)
        v = lin(h, p[f'{n}.attn.v.w'], lp=lp)
        x = lin(attention(q, k, v, c.heads, c.kv_heads, c.seq, True), p[f'{n}.attn.o.w'], lp=lp, res=x)
        h = rmsnorm(x, p[f'{n}.n2.g'])
        x = moe_ffn(h, p, f'{n}.moe', c, x, lp, aux) if c.experts else swiglu(h, p, f'{n}.mlp', lp, res=x)
    h = rmsnorm(x, p['llama.nf.g'])
    return lin(h, p['llama.wte'].T if c.tie_embeddings else p['llama.head.w'], lp=lp)


def llama(model, p, data):
    """Returns {'loss': ce, 'aux0': ...}."""
    aux = []
    logits = llama_logits(model.cfg, model.lp, p, data['ids'].astype(int), aux)
    return {'loss': ce(logits, data['tgt'].astype(int)), **{f'aux{i}': a for i, a in enumerate(aux)}}


def greedy(model, p, prompt, n_new):
    """Greedy decode by re-running the full forward on the growing sequence (no KV cache): the oracle of the cached decode."""
    import dataclasses
    toks = list(prompt)
    for _ in range(n_new):
        c = dataclasses.replace(model.cfg, seq=len(toks), batch=1)
        toks.append(int(np.argmax(llama_logits(c, model.lp, p, np.array(toks), [])[-1])))
    return toks[len(prompt):]


def mlp(model, p, data):
    c, x = model.cfg, data['x']
    f = {'relu': lambda v: np.maximum(v, 0), 'silu': silu, 'tanh': np.tanh, 'sigmoid': lambda v: 1 / (1 + np.exp(-v))}[c.act]
    n = len(c.hidden) + 1
    for i in range(n):
        x = lin(x, p[f'mlp.fc{i}.w'], p[f'mlp.fc{i}.b'], lp=model.lp)
        if i < n - 1:
            x = f(x)
    return {'loss': ce(x, data['y'].astype(int))}


def vit(model, p, data):
    c, lp = model.cfg, model.lp
    x = lin(data['patches'], p['vit.embed.w'], p['vit.embed.b'], lp=lp) + np.tile(p['vit.pos'], (c.batch, 1))
    for l in range(c.layers):
        n = f'vit.h{l}'
        h = layernorm(x, p[f'{n}.n1.g'], p[f'{n}.n1.b'])
        q, k, v = (lin(h, p[f'{n}.attn.{m}.w'], p[f'{n}.attn.{m}.b'], lp=lp) for m in 'qkv')
        x = lin(attention(q, k, v, c.heads, c.heads, c.patches, False), p[f'{n}.attn.o.w'], p[f'{n}.attn.o.b'], lp=lp, res=x)
        h = lin(layernorm(x, p[f'{n}.n2.g'], p[f'{n}.n2.b']), p[f'{n}.mlp.fc1.w'], p[f'{n}.mlp.fc1.b'], lp=lp, gelu=True)
        x = lin(h, p[f'{n}.mlp.fc2.w'], p[f'{n}.mlp.fc2.b'], lp=lp, res=x)
    pooled = layernorm(x, p['vit.nf.g'], p['vit.nf.b']).reshape(c.batch, c.patches, -1).mean(1)
    return {'loss': ce(lin(pooled, p['vit.head.w'], p['vit.head.b'], lp=lp), data['y'].astype(int))}


ORACLES = {'Llama': llama, 'MLPClassifier': mlp, 'ViT': vit}


def forward(model, p, data):
    return ORACLES[type(model).__name__](model, p, data)


def total_loss(model, p, data):
    return sum(forward(model, p, data).values())


def fd_grad(model, p, data, name, idx, h=1e-5):
    """Central finite difference of the summed loss (ce + auxiliary) w.r.t. one parameter element."""
    old = p[name][idx]
    p[name][idx] = old + h
    lp_ = total_loss(model, p, data)
    p[name][idx] = old - h
    lm = total_loss(model, p, data)
    p[name][idx] = old
    return (lp_ - lm) / (2 * h)
