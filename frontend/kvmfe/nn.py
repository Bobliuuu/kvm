"""Module-style builders. Modules own parameter *specs* and are called with a `w` dict of weight tensors, so one definition serves
training (every step) and serving (any weights version)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from .ir import FrontendError, Tensor
from .opset import ATTN_Q


@dataclass(frozen=True)
class ModelConfig:
    """Declarative GPT config; the graph, tiles and memory plan all derive from it and a Tiling."""
    name: str = 'gpt_tiny'
    vocab: int = 64
    d_model: int = 32
    layers: int = 1
    heads: int = 2
    seq: int = 32
    batch: int = 1
    ffn_mult: int = 4
    norm: str = 'layernorm'          # layernorm | rmsnorm
    bias: bool = True
    init_std: float = 0.02
    rope_theta: float = 10000.0
    seed: int = 0
    repeat_batch: bool = False       # every step trains on step 0's batch (an overfitting check: the loss must fall far below ln(vocab))

    def __post_init__(self):
        if self.d_model % self.heads or (self.d_model // self.heads) not in (16, 32, 64, 128):
            raise FrontendError(f'head dim d_model/heads = {self.d_model}/{self.heads} must be 16, 32, 64 or 128 (attention ops)')
        if self.seq % ATTN_Q:
            raise FrontendError(f'seq {self.seq} must be a multiple of {ATTN_Q} (attention block size)')
        if self.norm not in ('layernorm', 'rmsnorm'):
            raise FrontendError(f'norm must be layernorm or rmsnorm, got {self.norm!r}')

    @classmethod
    def load(cls, spec):
        """spec: a preset name, a path to a JSON file, a JSON string, or a dict."""
        if isinstance(spec, cls):
            return spec
        if isinstance(spec, str):
            if spec in PRESETS:
                return PRESETS[spec]
            spec = json.loads(spec) if spec.lstrip().startswith('{') else json.loads(open(spec).read())
        try:
            return cls(**spec)
        except TypeError as e:
            raise FrontendError(f'bad model config: {e}; fields: {sorted(asdict(cls()))}') from None

    @property
    def tokens(self):
        return self.batch * self.seq

    @property
    def params(self):
        return sum(p.rows * p.cols for p in GPT(self).specs.values())


PRESETS = {
    'gpt_tiny': ModelConfig(),
    'gpt_small': ModelConfig('gpt_small', vocab=128, d_model=64, layers=2, heads=4, seq=64, batch=2),
    # 12 layers, d768, 12 heads, seq 1024, vocab 50304, untied head, RoPE: ~162.3M parameters
    'gpt_162m': ModelConfig('gpt_162m', vocab=50304, d_model=768, layers=12, heads=12, seq=1024, batch=4),
}


@dataclass(frozen=True)
class ParamSpec:
    name: str
    rows: int
    cols: int
    init: tuple          # ('normal', std) | ('fill', v)
    decay: bool = False  # AdamW weight decay on matrices only
    invariant: bool = False  # the loss is exactly invariant to this parameter (true gradient identically 0); see Linear(b_invariant)


class Module:
    def __init__(self, name):
        self.name, self.specs = name, {}

    def add(self, name, rows, cols, init, decay=False, invariant=False):
        full = f'{self.name}.{name}'
        self.specs[full] = ParamSpec(full, rows, cols, init, decay, invariant)
        return full

    def absorb(self, *mods):
        for m in mods:
            self.specs.update(m.specs)


class Linear(Module):
    def __init__(self, name, din, dout, bias, std, b_invariant=False):
        """b_invariant: the bias is the key projection of attention without positions (ViT): softmax over keys is unchanged when every key
        moves by the same vector (q . (k + c) = q . k + q . c, constant per query), so d loss / d b is zero for the exact computation.
        Whatever rounding noise the executor or the golden holds there is then driven through AdamW's normalised step."""
        super().__init__(name)
        self.w = self.add('w', din, dout, ('normal', std), decay=True)
        self.b = self.add('b', 1, dout, ('fill', 0.0), invariant=b_invariant) if bias else None

    def __call__(self, b, x, w, res=None, gelu=False, bf16=False):
        out = b.linear(x, w[self.w], w[self.b] if self.b else None, res, gelu, bf16=bf16)
        return b.cast(out) if bf16 else out          # bf16 mode: activations are stored rounded as well


class Norm(Module):
    def __init__(self, name, d, kind):
        super().__init__(name)
        self.kind = kind
        self.g = self.add('g', 1, d, ('fill', 1.0))
        self.b = self.add('b', 1, d, ('fill', 0.0)) if kind == 'layernorm' else None

    def __call__(self, b, x, w):
        return b.norm(x, w[self.g], w[self.b] if self.b else None, self.kind)


class CausalSelfAttention(Module):
    def __init__(self, name, cfg):
        super().__init__(name)
        self.cfg = cfg
        std = cfg.init_std
        self.q, self.k, self.v = (Linear(f'{name}.{n}', cfg.d_model, cfg.d_model, cfg.bias, std) for n in 'qkv')
        self.proj = Linear(f'{name}.proj', cfg.d_model, cfg.d_model, cfg.bias, std / (2 * cfg.layers) ** 0.5)
        self.absorb(self.q, self.k, self.v, self.proj)

    def qkv(self, b, h, w, seq, pos=None):
        c = self.cfg
        return (b.rope(self.q(b, h, w), c.heads, seq, c.rope_theta, pos), b.rope(self.k(b, h, w), c.heads, seq, c.rope_theta, pos),
                self.v(b, h, w))

    def __call__(self, b, h, w, res):
        c = self.cfg
        q, k, v = self.qkv(b, h, w, c.seq)
        return self.proj(b, b.attention(q, k, v, c.heads, c.seq), w, res=res)


class MLP(Module):
    def __init__(self, name, cfg):
        super().__init__(name)
        self.fc1 = Linear(f'{name}.fc1', cfg.d_model, cfg.ffn_mult * cfg.d_model, cfg.bias, cfg.init_std)
        self.fc2 = Linear(f'{name}.fc2', cfg.ffn_mult * cfg.d_model, cfg.d_model, cfg.bias, cfg.init_std / (2 * cfg.layers) ** 0.5)
        self.absorb(self.fc1, self.fc2)

    def __call__(self, b, h, w, res):
        return self.fc2(b, self.fc1(b, h, w, gelu=True), w, res=res)


class TransformerBlock(Module):
    def __init__(self, name, cfg):
        super().__init__(name)
        self.n1, self.attn = Norm(f'{name}.n1', cfg.d_model, cfg.norm), CausalSelfAttention(f'{name}.attn', cfg)
        self.n2, self.mlp = Norm(f'{name}.n2', cfg.d_model, cfg.norm), MLP(f'{name}.mlp', cfg)
        self.absorb(self.n1, self.attn, self.n2, self.mlp)

    # decoder-block interface shared with models.LlamaBlock (used by serve.py): attention inputs, attention output, feed-forward
    def attn_inputs(self, b, x, w, seq, pos=None):
        return self.attn.qkv(b, self.n1(b, x, w), w, seq, pos)

    def attn_out(self, b, a, w, res):
        return self.attn.proj(b, a, w, res=res)

    def ffn_block(self, b, x, w):
        return self.mlp(b, self.n2(b, x, w), w, res=x)

    def __call__(self, b, x, w):
        with b.scope(self.name):
            x = self.attn(b, self.n1(b, x, w), w, res=x)
            return self.ffn_block(b, x, w)


class GPT(Module):
    """Pre-norm GPT: token embedding, RoPE, GELU MLP, untied LM head (no learned position table)."""
    def __init__(self, cfg: ModelConfig):
        super().__init__('gpt')
        self.cfg = cfg
        self.wte = self.add('wte', cfg.vocab, cfg.d_model, ('normal', cfg.init_std))
        self.blocks = [TransformerBlock(f'gpt.h{i}', cfg) for i in range(cfg.layers)]
        self.nf = Norm('gpt.nf', cfg.d_model, cfg.norm)
        self.head = Linear('gpt.head', cfg.d_model, cfg.vocab, False, cfg.init_std)
        self.absorb(*self.blocks, self.nf, self.head)

    kv_heads = property(lambda self: self.cfg.heads)

    def embed(self, b, ids, w, seq):
        return b.embedding(ids, w[self.wte], seq)

    def logits(self, b, x, w):
        return self.head(b, self.nf(b, x, w), w)

    def __call__(self, b, ids, w):
        x = self.embed(b, ids, w, self.cfg.seq)
        for blk in self.blocks:
            x = blk(b, x, w)
        return self.logits(b, x, w)


    # ---- the training-program protocol (train.build_training): data streams and loss roots
    def batch_streams(self, micros):
        return token_streams(self.cfg, micros)

    def losses(self, b, batch, w, grad_scale=1.0):
        return [('loss', b.ce_loss(self(b, batch['ids'], w), batch['tgt'], grad_scale))]


@dataclass
class Stream:
    """A host-provided input stream consumed one window per microbatch: micro m is elements [m*rows*cols, (m+1)*rows*cols)."""
    name: str
    rows: int            # rows per microbatch
    cols: int
    dtype: str           # f32 | i32
    values: list         # micros * rows * cols numbers


def token_streams(cfg, micros):
    """Next-token bigram-chain batches (see train.batch_tokens) for `micros` microbatches: streams ids and tgt."""
    from .train import batch_stream
    ids, tgt = batch_stream(cfg, micros)
    return [Stream('ids', cfg.tokens, 1, 'i32', ids), Stream('tgt', cfg.tokens, 1, 'i32', tgt)]
