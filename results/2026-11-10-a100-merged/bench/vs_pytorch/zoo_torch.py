"""The PyTorch twins of the hold-out's zoo rows (Llama / grouped-query, ViT): the forward passes of frontend/kvmfe/models.py in float32 on the GPU, on the executor's own bytes.

torch_bench.Trainer reads the initial weights, data streams and optimizer from the workload (one code path with the GPT row) and calls `loss(p, ids, tgt)` here. The math is the same as
the frontend's float64 reference (examples/models/torch_models.py): RMSNorm or LayerNorm (eps 1e-5), RoPE (rotate-half), scaled dot product attention (scale hd^-0.5), SwiGLU or tanh-GELU,
mean cross-entropy. For a ViT `ids` is the step's patches, flattened by the stream window: its loss restores [B*P, patch_dim]. The MoE family is left out on purpose: its discrete routing
(top-k, capacity by token order) is data-dependent control flow that a CUDA graph capture cannot hold, so the conventional side would be eager only and the bar would mean little.
"""
import torch
import torch.nn.functional as F

EPS = 1e-5


def rope_tables(theta, seq, hd, dev):
    inv = theta ** (-2.0 * torch.arange(hd // 2, dtype=torch.float64) / hd)
    ang = torch.arange(seq, dtype=torch.float64)[:, None] * inv[None]
    return ang.cos().float().to(dev), ang.sin().float().to(dev)                  # [T, hd/2]


def rope(x, cos, sin):                                                            # x [B, H, T, hd]
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)


def rmsnorm(x, g):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * g


def linear(x, w, b=None):
    return F.linear(x, w.t(), b)


class TorchLlama:
    """RMSNorm, RoPE, grouped-query causal attention, SwiGLU, optionally a tied head (models.py Llama, experts == 0)."""

    def __init__(self, cfg, dev):
        if cfg.experts:
            raise SystemExit('zoo_torch: the MoE family has no conventional twin (see the module note)')
        if cfg.precision != 'f32':
            raise SystemExit(f'zoo_torch: precision {cfg.precision} is not a panel row; f32 only')
        self.cfg = cfg
        self.cos, self.sin = rope_tables(cfg.rope_theta, cfg.seq, cfg.d_model // cfg.heads, dev)

    def loss(self, p, ids, tgt):
        c = self.cfg
        B, T, H, Hkv, d = c.batch, c.seq, c.heads, c.kv_heads, c.d_model
        hd = d // H
        x = F.embedding(ids, p['llama.wte'])                                      # [B*T, d]
        for i in range(c.layers):
            n = f'llama.h{i}'
            h = rmsnorm(x, p[f'{n}.n1.g'])
            q = rope(linear(h, p[f'{n}.attn.q.w']).view(B, T, H, hd).transpose(1, 2), self.cos, self.sin)
            k = rope(linear(h, p[f'{n}.attn.k.w']).view(B, T, Hkv, hd).transpose(1, 2), self.cos, self.sin)
            v = linear(h, p[f'{n}.attn.v.w']).view(B, T, Hkv, hd).transpose(1, 2)
            if H != Hkv:                                                          # grouped-query: head h reads kv head h // (H // Hkv)
                k, v = k.repeat_interleave(H // Hkv, dim=1), v.repeat_interleave(H // Hkv, dim=1)
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B * T, d)
            x = x + linear(a, p[f'{n}.attn.o.w'])
            h = rmsnorm(x, p[f'{n}.n2.g'])
            x = x + linear(F.silu(linear(h, p[f'{n}.mlp.gate.w'])) * linear(h, p[f'{n}.mlp.up.w']), p[f'{n}.mlp.down.w'])
        h = rmsnorm(x, p['llama.nf.g'])
        logits = h @ p['llama.wte'].t() if c.tie_embeddings else h @ p['llama.head.w']
        return F.cross_entropy(logits, tgt)


class TorchViT:
    """Patch embedding as a matmul, a learned position table, bidirectional attention, a mean over patches, a linear head (models.py ViT)."""

    def __init__(self, cfg, dev):
        if cfg.precision != 'f32':
            raise SystemExit(f'zoo_torch: precision {cfg.precision} is not a panel row; f32 only')
        self.cfg = cfg

    def loss(self, p, ids, tgt):
        c = self.cfg
        B, P, H, d = c.batch, c.patches, c.heads, c.d_model
        hd = d // H
        x = linear(ids.view(B * P, c.patch_dim), p['vit.embed.w'], p['vit.embed.b']) + p['vit.pos'].repeat(B, 1)
        for i in range(c.layers):
            n = f'vit.h{i}'
            h = F.layer_norm(x, (d,), p[f'{n}.n1.g'], p[f'{n}.n1.b'], EPS)
            q, k, v = (linear(h, p[f'{n}.attn.{m}.w'], p[f'{n}.attn.{m}.b']).view(B, P, H, hd).transpose(1, 2) for m in 'qkv')
            a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B * P, d)
            x = x + linear(a, p[f'{n}.attn.o.w'], p[f'{n}.attn.o.b'])
            h = F.layer_norm(x, (d,), p[f'{n}.n2.g'], p[f'{n}.n2.b'], EPS)
            h = F.gelu(linear(h, p[f'{n}.mlp.fc1.w'], p[f'{n}.mlp.fc1.b']), approximate='tanh')
            x = x + linear(h, p[f'{n}.mlp.fc2.w'], p[f'{n}.mlp.fc2.b'])
        pooled = F.layer_norm(x, (d,), p['vit.nf.g'], p['vit.nf.b'], EPS).reshape(B, P, d).mean(1)
        return F.cross_entropy(linear(pooled, p['vit.head.w'], p['vit.head.b']), tgt.view(-1))


TWINS = {'llama': TorchLlama, 'vit': TorchViT}
STREAMS = {'llama': ('stream_ids', 'stream_tgt'), 'vit': ('stream_patches', 'stream_y')}


def twin(family, model_cfg, dev):
    """(config, parameter specs, twin, (data stream, target stream)) of a zoo row's info.json."""
    from kvmfe import models
    if family not in TWINS:
        raise SystemExit(f'zoo_torch: no twin for family {family!r} (have {sorted(TWINS)})')
    model = models.build({'family': family, **model_cfg})
    return model.cfg, model.specs, TWINS[family](model.cfg, dev), STREAMS[family]
