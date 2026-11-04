"""Model zoo built only from the frontend's generic builder ops (no model-specific kernels): a Llama-style decoder (RMSNorm,
RoPE, GQA, SwiGLU, optional tied embeddings, optional MoE feed-forward), an MLP classifier, and a conv-free ViT. Each model is a
config dataclass plus a Module; all of them implement the training protocol of train.build_training:

    cfg.name, cfg.seed;  model.specs;  model.batch_streams(micros);  model.losses(b, batch, w, grad_scale)

so autodiff, microbatch accumulation, LR schedules, loss scaling, rematerialization and fusion apply to all of them unchanged.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, fields

from .ir import FrontendError
from .nn import Linear, Module, Norm, Stream, token_streams
from .opset import ATTN_Q  # noqa: F401  (kept for config validation messages)


def _load(cls, spec, presets):
    if isinstance(spec, cls):
        return spec
    if isinstance(spec, str):
        if spec in presets:
            return presets[spec]
        spec = json.loads(spec) if spec.lstrip().startswith('{') else json.loads(open(spec).read())
    try:
        return cls(**spec)
    except TypeError as e:
        raise FrontendError(f'bad {cls.__name__}: {e}; fields: {[f.name for f in fields(cls)]}') from None


def _check_hd(d_model, heads):
    if d_model % heads or (d_model // heads) not in (16, 32, 64, 128):
        raise FrontendError(f'head dim d_model/heads = {d_model}/{heads} must be 16, 32, 64 or 128 (attention ops)')


# ====================================================================================================== Llama / MoE decoder
@dataclass(frozen=True)
class LlamaConfig:
    """Llama-style decoder: RMSNorm, RoPE, grouped-query attention, SwiGLU (or a top-k MoE of SwiGLU experts when experts > 0).
    precision 'bf16': every matmul runs on bf16 operands (fp32 accumulate) and activations are stored bf16-rounded, while the
    parameters stay fp32 masters. remat: recompute each block's forward inside its backward (activation checkpointing)."""
    name: str = 'llama_tiny'
    vocab: int = 64
    d_model: int = 64
    layers: int = 2
    heads: int = 4
    kv_heads: int = 2
    ffn: int = 96
    seq: int = 32
    batch: int = 1
    tie_embeddings: bool = False
    rope_theta: float = 10000.0
    init_std: float = 0.02
    seed: int = 0
    precision: str = 'f32'
    remat: bool = False
    experts: int = 0
    top_k: int = 2
    capacity_factor: float = 1.25
    balance_coef: float = 0.01

    def __post_init__(self):
        _check_hd(self.d_model, self.heads)
        if self.heads % self.kv_heads:
            raise FrontendError(f'heads {self.heads} must be a multiple of kv_heads {self.kv_heads}')
        if self.precision not in ('f32', 'bf16'):
            raise FrontendError(f"precision must be 'f32' or 'bf16', got {self.precision!r}")
        if self.experts and not 0 < self.top_k <= self.experts:
            raise FrontendError(f'need 0 < top_k ({self.top_k}) <= experts ({self.experts})')

    @property
    def tokens(self):
        return self.batch * self.seq

    @property
    def capacity(self):
        """Per-expert token capacity of one microbatch (earlier tokens win; overflow is dropped)."""
        return max(1, math.ceil(self.capacity_factor * self.tokens * self.top_k / self.experts)) if self.experts else 0

    @classmethod
    def load(cls, spec):
        return _load(cls, spec, LLAMA_PRESETS)


LLAMA_PRESETS = {
    'llama_tiny': LlamaConfig(),
    'llama_gqa_tied': LlamaConfig('llama_gqa_tied', tie_embeddings=True),
    'moe_tiny': LlamaConfig('moe_tiny', layers=1, heads=4, kv_heads=4, ffn=64, batch=2, experts=4, top_k=2),
}


class SwiGLU(Module):
    def __init__(self, name, cfg):
        super().__init__(name)
        std = cfg.init_std
        self.gate, self.up = (Linear(f'{name}.{n}', cfg.d_model, cfg.ffn, False, std) for n in ('gate', 'up'))
        self.down = Linear(f'{name}.down', cfg.ffn, cfg.d_model, False, std / (2 * cfg.layers) ** 0.5)
        self.absorb(self.gate, self.up, self.down)
        self.lp = cfg.precision == 'bf16'

    def __call__(self, b, h, w, res, aux=None):
        gate, up = self.gate(b, h, w, bf16=self.lp), self.up(b, h, w, bf16=self.lp)
        return self.down(b, b.silu_mul(gate, up), w, res=res, bf16=self.lp)


class MoEFFN(Module):
    """Top-k routed mixture of SwiGLU experts with per-expert capacity. Dispatch = gather_rows_scaled, combine = scatter_add_rows
    with the router gate as scale, router probabilities get a load-balancing auxiliary loss; autodiff flows through all of it."""
    def __init__(self, name, cfg):
        super().__init__(name)
        self.cfg, self.lp = cfg, cfg.precision == 'bf16'
        self.router = Linear(f'{name}.router', cfg.d_model, cfg.experts, False, cfg.init_std)
        self.experts = [SwiGLU(f'{name}.e{e}', cfg) for e in range(cfg.experts)]
        self.absorb(self.router, *self.experts)

    def __call__(self, b, h, w, res, aux):
        c = self.cfg
        N, E, C = h.rows, c.experts, c.capacity
        probs = b.softmax(self.router(b, h, w))
        sel, slot, eidx = b.moe_route(probs, c.top_k, C)
        gates = b.moe_gates(probs, sel, slot, E * C)
        aux.append(b.moe_balance(probs, sel, c.balance_coef))
        combined = None
        for e, ex in enumerate(self.experts):
            xe = b.gather(h, eidx, C, idx0=e * C)
            ye = ex.down(b, b.silu_mul(ex.gate(b, xe, w, bf16=self.lp), ex.up(b, xe, w, bf16=self.lp)), w, bf16=self.lp)
            combined = b.scatter_add(ye, eidx, N, idx0=e * C, scale=gates, into=combined)
        return b.add(combined, res)


class LlamaBlock(Module):
    def __init__(self, name, cfg):
        super().__init__(name)
        self.cfg, self.lp = cfg, cfg.precision == 'bf16'
        d, hd, std = cfg.d_model, cfg.d_model // cfg.heads, cfg.init_std
        self.n1, self.n2 = Norm(f'{name}.n1', d, 'rmsnorm'), Norm(f'{name}.n2', d, 'rmsnorm')
        self.q = Linear(f'{name}.attn.q', d, cfg.heads * hd, False, std)
        self.k = Linear(f'{name}.attn.k', d, cfg.kv_heads * hd, False, std)
        self.v = Linear(f'{name}.attn.v', d, cfg.kv_heads * hd, False, std)
        self.o = Linear(f'{name}.attn.o', cfg.heads * hd, d, False, std / (2 * cfg.layers) ** 0.5)
        self.ffn = MoEFFN(f'{name}.moe', cfg) if cfg.experts else SwiGLU(f'{name}.mlp', cfg)
        self.absorb(self.n1, self.n2, self.q, self.k, self.v, self.o, self.ffn)

    def qkv(self, b, h, w, seq, pos=None):
        c = self.cfg
        return (b.rope(self.q(b, h, w, bf16=self.lp), c.heads, seq, c.rope_theta, pos),
                b.rope(self.k(b, h, w, bf16=self.lp), c.kv_heads, seq, c.rope_theta, pos), self.v(b, h, w, bf16=self.lp))

    # decoder-block interface shared with nn.TransformerBlock (used by serve.py)
    def attn_inputs(self, b, x, w, seq, pos=None):
        return self.qkv(b, self.n1(b, x, w), w, seq, pos)

    def attn_out(self, b, a, w, res):
        return self.o(b, a, w, res=res, bf16=self.lp)

    def ffn_block(self, b, x, w):
        return self.ffn(b, self.n2(b, x, w), w, x, [])

    def __call__(self, b, x, w, aux):
        c = self.cfg
        with b.scope(self.name), b.remat(c.remat):
            q, k, v = self.qkv(b, self.n1(b, x, w), w, c.seq)
            x = self.attn_out(b, b.attention(q, k, v, c.heads, c.seq, c.kv_heads), w, x)
            return self.ffn(b, self.n2(b, x, w), w, x, aux)


class Llama(Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__('llama')
        self.cfg, self.lp = cfg, cfg.precision == 'bf16'
        self.wte = self.add('wte', cfg.vocab, cfg.d_model, ('normal', cfg.init_std))
        self.blocks = [LlamaBlock(f'llama.h{i}', cfg) for i in range(cfg.layers)]
        self.nf = Norm('llama.nf', cfg.d_model, 'rmsnorm')
        self.head = None if cfg.tie_embeddings else Linear('llama.head', cfg.d_model, cfg.vocab, False, cfg.init_std)
        self.absorb(*self.blocks, self.nf, *([self.head] if self.head else []))

    def logits(self, b, x, w):
        h = self.nf(b, x, w)
        if self.head:
            return self.head(b, h, w, bf16=self.lp)
        out = b.linear(h, w[self.wte], wt=True, bf16=self.lp)              # tied: logits = h @ wte^T
        return b.cast(out) if self.lp else out

    @property
    def kv_heads(self):
        return self.cfg.kv_heads

    def embed(self, b, ids, w, seq):
        x = b.embedding(ids, w[self.wte], seq)
        return b.cast(x) if self.lp else x

    def __call__(self, b, ids, w, aux=None):
        x = self.embed(b, ids, w, self.cfg.seq)
        aux = [] if aux is None else aux
        for blk in self.blocks:
            x = blk(b, x, w, aux)
        return self.logits(b, x, w)

    def batch_streams(self, micros):
        return token_streams(self.cfg, micros)

    def losses(self, b, batch, w, grad_scale=1.0):
        aux = []
        ce = b.ce_loss(self(b, batch['ids'], w, aux), batch['tgt'], grad_scale)
        return [('loss', ce)] + [(f'aux{i}', t) for i, t in enumerate(aux)]


# ====================================================================================================== MLP classifier
@dataclass(frozen=True)
class MLPConfig:
    """Plain MLP classifier on synthetic Gaussian clusters (class means drawn from the seed)."""
    name: str = 'mlp_tiny'
    in_dim: int = 16
    hidden: tuple = (32, 32)
    classes: int = 4
    batch: int = 32
    act: str = 'relu'
    noise: float = 0.6
    seed: int = 0
    precision: str = 'f32'

    def __post_init__(self):
        object.__setattr__(self, 'hidden', tuple(self.hidden))
        if self.act not in ('relu', 'silu', 'tanh', 'sigmoid'):
            raise FrontendError(f'unknown activation {self.act!r}')

    @classmethod
    def load(cls, spec):
        return _load(cls, spec, MLP_PRESETS)


MLP_PRESETS = {'mlp_tiny': MLPConfig()}


class MLPClassifier(Module):
    def __init__(self, cfg: MLPConfig):
        super().__init__('mlp')
        self.cfg, self.lp = cfg, cfg.precision == 'bf16'
        dims = [cfg.in_dim, *cfg.hidden, cfg.classes]
        self.layers = [Linear(f'mlp.fc{i}', a, c, True, math.sqrt(2.0 / a)) for i, (a, c) in enumerate(zip(dims, dims[1:]))]
        self.absorb(*self.layers)

    def __call__(self, b, x, w):
        for i, fc in enumerate(self.layers):
            x = fc(b, x, w, bf16=self.lp)
            if i < len(self.layers) - 1:
                x = b.act(x, self.cfg.act)
        return x

    def batch_streams(self, micros):
        c = self.cfg
        rng = random.Random(c.seed * 7919 + 17)
        means = [[rng.gauss(0, 1.5) for _ in range(c.in_dim)] for _ in range(c.classes)]
        xs, ys = [], []
        for _ in range(micros * c.batch):
            y = rng.randrange(c.classes)
            xs += [m + c.noise * rng.gauss(0, 1) for m in means[y]]
            ys.append(y)
        return [Stream('x', c.batch, c.in_dim, 'f32', xs), Stream('y', c.batch, 1, 'i32', ys)]

    def losses(self, b, batch, w, grad_scale=1.0):
        return [('loss', b.ce_loss(self(b, batch['x'], w), batch['y'], grad_scale))]


# ====================================================================================================== ViT (patch embedding as a matmul)
@dataclass(frozen=True)
class ViTConfig:
    """Conv-free ViT: images arrive pre-cut into patches (rows = B*P, cols = patch*patch*channels); patch embedding is a matmul,
    positions are a learned table, attention is bidirectional, pooling is a mean over patches (a scatter-add), head is linear."""
    name: str = 'vit_tiny'
    image: int = 16
    patch: int = 4
    channels: int = 1
    d_model: int = 32
    layers: int = 1
    heads: int = 2
    ffn_mult: int = 4
    classes: int = 4
    batch: int = 8
    init_std: float = 0.05
    noise: float = 0.5
    seed: int = 0
    precision: str = 'f32'

    def __post_init__(self):
        _check_hd(self.d_model, self.heads)
        if self.image % self.patch:
            raise FrontendError(f'image {self.image} must be a multiple of patch {self.patch}')

    @property
    def patches(self):
        return (self.image // self.patch) ** 2

    @property
    def patch_dim(self):
        return self.patch * self.patch * self.channels

    @classmethod
    def load(cls, spec):
        return _load(cls, spec, VIT_PRESETS)


VIT_PRESETS = {'vit_tiny': ViTConfig()}


class ViTBlock(Module):
    def __init__(self, name, cfg):
        super().__init__(name)
        d, std, self.cfg, self.lp = cfg.d_model, cfg.init_std, cfg, cfg.precision == 'bf16'
        self.n1, self.n2 = Norm(f'{name}.n1', d, 'layernorm'), Norm(f'{name}.n2', d, 'layernorm')
        self.q, self.k, self.v = (Linear(f'{name}.attn.{n}', d, d, True, std) for n in 'qkv')
        self.o = Linear(f'{name}.attn.o', d, d, True, std / (2 * cfg.layers) ** 0.5)
        self.fc1 = Linear(f'{name}.mlp.fc1', d, cfg.ffn_mult * d, True, std)
        self.fc2 = Linear(f'{name}.mlp.fc2', cfg.ffn_mult * d, d, True, std / (2 * cfg.layers) ** 0.5)
        self.absorb(self.n1, self.n2, self.q, self.k, self.v, self.o, self.fc1, self.fc2)

    def __call__(self, b, x, w):
        c, lp = self.cfg, self.lp
        with b.scope(self.name):
            h = self.n1(b, x, w)
            a = b.attention(self.q(b, h, w, bf16=lp), self.k(b, h, w, bf16=lp), self.v(b, h, w, bf16=lp), c.heads, c.patches,
                            causal=False)
            x = self.o(b, a, w, res=x, bf16=lp)
            return self.fc2(b, self.fc1(b, self.n2(b, x, w), w, gelu=True, bf16=lp), w, res=x, bf16=lp)


class ViT(Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__('vit')
        self.cfg, self.lp = cfg, cfg.precision == 'bf16'
        d = cfg.d_model
        self.embed = Linear('vit.embed', cfg.patch_dim, d, True, cfg.init_std)
        self.pos = self.add('pos', cfg.patches, d, ('normal', cfg.init_std))
        self.blocks = [ViTBlock(f'vit.h{i}', cfg) for i in range(cfg.layers)]
        self.nf = Norm('vit.nf', d, 'layernorm')
        self.head = Linear('vit.head', d, cfg.classes, True, cfg.init_std)
        self.absorb(self.embed, *self.blocks, self.nf, self.head)

    def __call__(self, b, patches, w):
        c = self.cfg
        n = c.batch * c.patches
        bidx = b.tensor('pool_idx', n, 1, 'input', buf='pool_idx', dtype='i32', init=('values', [r // c.patches for r in range(n)]))
        inv = b.tensor('pool_scale', n, 1, 'input', buf='pool_scale', init=('values', [1.0 / c.patches] * n))
        x = b.add_rowtile(self.embed(b, patches, w, bf16=self.lp), w[self.pos], c.patches)
        for blk in self.blocks:
            x = blk(b, x, w)
        pooled = b.scatter_add(self.nf(b, x, w), bidx, c.batch, scale=inv)          # mean over each image's patches
        return self.head(b, pooled, w, bf16=self.lp)

    def batch_streams(self, micros):
        c = self.cfg
        rng = random.Random(c.seed * 104729 + 3)
        side = c.image
        proto = []
        for k in range(c.classes):                       # class k: a smooth oriented wave pattern
            fx, fy, ph = 1 + k % 2, 1 + k // 2, rng.random() * 6.28
            proto.append([[math.sin(2 * math.pi * (fx * x + fy * y) / side + ph) for x in range(side)] for y in range(side)])
        px, ys = [], []
        for _ in range(micros * c.batch):
            k = rng.randrange(c.classes)
            img = [[proto[k][y][x] + c.noise * rng.gauss(0, 1) for x in range(side)] for y in range(side)]
            for py in range(0, side, c.patch):
                for pxx in range(0, side, c.patch):
                    px += [img[py + i][pxx + j] for i in range(c.patch) for j in range(c.patch)]
            ys.append(k)
        return [Stream('patches', c.batch * c.patches, c.patch_dim, 'f32', px), Stream('y', c.batch, 1, 'i32', ys)]

    def losses(self, b, batch, w, grad_scale=1.0):
        return [('loss', b.ce_loss(self(b, batch['patches'], w), batch['y'], grad_scale))]


_REGISTRY = {LlamaConfig: Llama, MLPConfig: MLPClassifier, ViTConfig: ViT}
PRESETS = {**LLAMA_PRESETS, **MLP_PRESETS, **VIT_PRESETS}


FAMILIES = {'llama': LlamaConfig, 'mlp': MLPConfig, 'vit': ViTConfig}


def build(cfg):
    """Config object, preset name or JSON -> model object. A JSON/dict config may carry "family": llama | mlp | vit to pick the
    class; without it the first config class that accepts every field wins."""
    if isinstance(cfg, str) and cfg.lstrip().startswith('{'):
        cfg = json.loads(cfg)
    if isinstance(cfg, dict) and 'family' in cfg:
        cfg = dict(cfg)
        fam = cfg.pop('family')
        if fam not in FAMILIES:
            raise FrontendError(f'unknown model family {fam!r}; known: {sorted(FAMILIES)}')
        return _REGISTRY[FAMILIES[fam]](FAMILIES[fam](**cfg))
    if isinstance(cfg, (str, dict)):
        for cls in _REGISTRY:
            try:
                return _REGISTRY[cls](cls.load(cfg))
            except (FrontendError, TypeError, KeyError):
                continue
        raise FrontendError(f'no model config matches {cfg!r}; presets: {sorted(PRESETS)}')
    if type(cfg) not in _REGISTRY:
        raise FrontendError(f'unknown model config type {type(cfg).__name__}; known: {[c.__name__ for c in _REGISTRY]}')
    return _REGISTRY[type(cfg)](cfg)
