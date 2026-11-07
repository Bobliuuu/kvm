"""PyTorch (autograd, float64) references of the model zoo: GPT, Llama / GQA / tied head / MoE, MLP classifier, ViT, optionally with
the bf16 emulation of the frontend's `precision='bf16'`.

This is the independent reference of goal 2 (not the repo's CPU golden): plain torch ops on the model's initial values and first data
batch, loss and gradients by torch autograd. Nothing here uses kvmfe's tiles, buffers, ops or autodiff; it shares only the
deterministic initial values and data streams (kvmfe.emit.init_values, tests/frontend/oracle_models.py) so both sides start at the same point.

  reference(model) -> (loss dict, {param name: grad})   the gradient of sum(loss dict) (cross entropy + MoE balance loss are both backward roots)

bf16 (`precision='bf16'`, lp): matmul operands and stored activations are rounded to bfloat16 (round to nearest even), accumulation and
parameters stay wide; the rounding is treated as the identity in the backward pass, as the frontend's cast op does.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'frontend'))
sys.path.insert(0, str(ROOT / 'tests' / 'frontend'))
import oracle_models as om  # noqa: E402

DT = torch.float64


def _bf16(x):
    return x.to(torch.bfloat16).to(DT)


class _CastAct(torch.autograd.Function):
    """Activation cast: rounded forward, and the gradient is rounded the same way (autodiff.py `cast` vjp: straight-through, rounded)."""
    @staticmethod
    def forward(ctx, x):
        return _bf16(x)

    @staticmethod
    def backward(ctx, g):
        return _bf16(g)


def rb(x, lp, param=False):
    """bf16 rounding of an activation (rounded gradient) or of a parameter used as a matmul operand (fp32 master: identity gradient)."""
    if not lp:
        return x
    return x + (_bf16(x) - x).detach() if param else _CastAct.apply(x)


def lin(x, w, b=None, lp=False, gelu=False, res=None):
    y = rb(x, lp) @ rb(w, lp, param=True)
    if b is not None:
        y = y + b
    if gelu:
        y = F.gelu(y, approximate='tanh')
    if res is not None:
        y = y + res
    return rb(y, lp)


def rmsnorm(x, g, eps=1e-5):
    return x * torch.rsqrt((x * x).mean(1, keepdim=True) + eps) * g


def layernorm(x, g, b, eps=1e-5):
    return F.layer_norm(x, (x.shape[1],), g.reshape(-1), b.reshape(-1), eps)


def rope(x, H, T, theta):
    R, hd = x.shape
    hd //= H
    half = hd // 2
    ang = (torch.arange(R, dtype=DT) % T)[:, None] * theta ** (-2.0 * torch.arange(half, dtype=DT) / hd)[None, :]
    c, s = ang.cos()[:, None, :], ang.sin()[:, None, :]
    x = x.reshape(R, H, hd)
    a, b = x[..., :half], x[..., half:]
    return torch.cat([a * c - b * s, a * s + b * c], -1).reshape(R, H * hd)


def attention(q, k, v, H, Hkv, T, causal):
    """q [B*T, H*hd], k/v [B*T, Hkv*hd]; grouped-query: head h reads kv head h // (H // Hkv)."""
    R, hd = q.shape[0], q.shape[1] // H
    B = R // T
    qh = q.reshape(B, T, H, hd).transpose(1, 2)
    kh = k.reshape(B, T, Hkv, hd).transpose(1, 2).repeat_interleave(H // Hkv, dim=1)
    vh = v.reshape(B, T, Hkv, hd).transpose(1, 2).repeat_interleave(H // Hkv, dim=1)
    o = F.scaled_dot_product_attention(qh, kh, vh, is_causal=causal)
    return o.transpose(1, 2).reshape(R, H * hd)


def swiglu(h, p, n, lp, res=None):
    return lin(F.silu(lin(h, p[f'{n}.gate.w'], lp=lp)) * lin(h, p[f'{n}.up.w'], lp=lp), p[f'{n}.down.w'], lp=lp, res=res)


def moe_ffn(h, p, n, cfg, res, lp, aux):
    N, E, k, C = h.shape[0], cfg.experts, cfg.top_k, cfg.capacity
    pr = torch.softmax(lin(h, p[f'{n}.router.w']), 1)               # the router stays fp32
    sel, slot, eidx = om.moe_route(pr.detach().numpy(), k, C)       # discrete routing (top-k, capacity by token order): not differentiable
    f = torch.tensor(np.bincount(sel.ravel(), minlength=E) / (N * k), dtype=DT)
    aux.append(cfg.balance_coef * E * (f * pr.mean(0)).sum())
    out = res.clone()
    for e in range(E):
        rows = [(r, int(eidx[e * C + r])) for r in range(C) if eidx[e * C + r] >= 0]
        if not rows:
            continue
        ts = torch.tensor([t for _, t in rows])
        gate = torch.stack([pr[t, torch.tensor(sel[t])].sum() for _, t in rows])
        ge = torch.stack([pr[t, e] for _, t in rows]) / gate
        out = out.index_add(0, ts, ge[:, None] * swiglu(h[ts], p, f'{n}.e{e}', lp))
    return out


def llama(model, p, data):
    c, lp = model.cfg, model.lp
    ids, tgt = data['ids'].long(), data['tgt'].long()
    aux = []
    x = rb(p['llama.wte'][ids], lp)
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
    logits = lin(h, p['llama.wte'].T if c.tie_embeddings else p['llama.head.w'], lp=lp)
    return {'loss': F.cross_entropy(logits, tgt), **{f'aux{i}': a for i, a in enumerate(aux)}}


def gpt(model, p, data):
    c = model.cfg
    ids, tgt = data['ids'].long(), data['tgt'].long()
    x = p['gpt.wte'][ids]
    for l in range(c.layers):
        n = f'gpt.h{l}'
        h = layernorm(x, p[f'{n}.n1.g'], p[f'{n}.n1.b'])
        q, k, v = (lin(h, p[f'{n}.attn.{m}.w'], p[f'{n}.attn.{m}.b']) for m in 'qkv')
        q, k = rope(q, c.heads, c.seq, c.rope_theta), rope(k, c.heads, c.seq, c.rope_theta)
        x = lin(attention(q, k, v, c.heads, c.heads, c.seq, True), p[f'{n}.attn.proj.w'], p[f'{n}.attn.proj.b'], res=x)
        h = lin(layernorm(x, p[f'{n}.n2.g'], p[f'{n}.n2.b']), p[f'{n}.mlp.fc1.w'], p[f'{n}.mlp.fc1.b'], gelu=True)
        x = lin(h, p[f'{n}.mlp.fc2.w'], p[f'{n}.mlp.fc2.b'], res=x)
    h = layernorm(x, p['gpt.nf.g'], p['gpt.nf.b'])
    return {'loss': F.cross_entropy(h @ p['gpt.head.w'], tgt)}


def mlp(model, p, data):
    c, lp = model.cfg, model.lp
    act = {'relu': torch.relu, 'silu': F.silu, 'tanh': torch.tanh, 'sigmoid': torch.sigmoid}[c.act]
    x, n = data['x'], len(c.hidden) + 1
    for i in range(n):
        x = lin(x, p[f'mlp.fc{i}.w'], p[f'mlp.fc{i}.b'], lp=lp)
        if i < n - 1:
            x = act(x)
    return {'loss': F.cross_entropy(x, data['y'].long())}


def vit(model, p, data):
    c, lp = model.cfg, model.lp
    x = lin(data['patches'], p['vit.embed.w'], p['vit.embed.b'], lp=lp) + p['vit.pos'].repeat(c.batch, 1)
    for l in range(c.layers):
        n = f'vit.h{l}'
        h = layernorm(x, p[f'{n}.n1.g'], p[f'{n}.n1.b'])
        q, k, v = (lin(h, p[f'{n}.attn.{m}.w'], p[f'{n}.attn.{m}.b'], lp=lp) for m in 'qkv')
        x = lin(attention(q, k, v, c.heads, c.heads, c.patches, False), p[f'{n}.attn.o.w'], p[f'{n}.attn.o.b'], lp=lp, res=x)
        h = lin(layernorm(x, p[f'{n}.n2.g'], p[f'{n}.n2.b']), p[f'{n}.mlp.fc1.w'], p[f'{n}.mlp.fc1.b'], lp=lp, gelu=True)
        x = lin(h, p[f'{n}.mlp.fc2.w'], p[f'{n}.mlp.fc2.b'], lp=lp, res=x)
    pooled = layernorm(x, p['vit.nf.g'], p['vit.nf.b']).reshape(c.batch, c.patches, -1).mean(1)
    return {'loss': F.cross_entropy(lin(pooled, p['vit.head.w'], p['vit.head.b'], lp=lp), data['y'].long())}


FORWARDS = {'GPT': gpt, 'Llama': llama, 'MLPClassifier': mlp, 'ViT': vit}


def reference(model):
    """(losses, grads) of the model's first training batch from its initial values, in float64."""
    p = {n: torch.from_numpy(v).to(DT).requires_grad_(True) for n, v in om.params(model).items()}
    data = {n: torch.from_numpy(np.asarray(v)).to(DT if v.dtype.kind == 'f' else torch.int64) for n, v in om.streams(model).items()}
    if type(model).__name__ == 'GPT':          # token streams of the GPT config are named stream_ids / stream_tgt in the workload; ids / tgt here
        data = {k.replace('stream_', ''): v for k, v in data.items()}
    losses = FORWARDS[type(model).__name__](model, p, data)
    sum(losses.values()).backward()
    return {k: float(v.detach()) for k, v in losses.items()}, {n: q.grad.detach().numpy() for n, q in p.items()}
