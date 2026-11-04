"""Serving graph builder: prefill + greedy decode with a KV cache, one graph per request, parameterized by the prompt.

A request reads the trainer's weight buffers under a version policy of the `weights` variable (see train.py):
  exact    -> the request sees exactly the weights stable at `version`; the compiler orders it before the next overwrite
  at_least -> any version >= `version`; reads whatever the buffers hold (may mix two versions if a step lands mid-request)
Requests are ordinary graphs on the same scheduler (own priority and deadline); nothing here knows about training.

Structure: the first P-1 prompt tokens are prefilled in one batch (rows tiled like training, K/V appended to the cache), then every
step (the last prompt token, then each generated token) is a one-row decode: embed -> layers -> head -> argmax. All decode buffers
have one row, so no op needs row offsets into a larger buffer.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import log
from .emit import GraphPolicy, lower_program, to_workload
from .ir import Builder, FrontendError, Tensor
from .lower import Tiling
from .nn import GPT


@dataclass(frozen=True)
class ServeRequest:
    name: str
    prompt: tuple
    max_new: int = 4
    version: int = 0                 # `weights` variable version (train.weights_version(k) = stable after k updates)
    policy: str = 'exact'            # exact | at_least
    priority: int = 10
    deadline_us: int = 2000


def build_serving(model: GPT, req: ServeRequest) -> Builder:
    cfg = model.cfg
    P, N = len(req.prompt), req.max_new
    if P < 1 or N < 1:
        raise FrontendError(f'{req.name}: need a nonempty prompt and max_new >= 1')
    if any(not 0 <= t < cfg.vocab for t in req.prompt):
        raise FrontendError(f'{req.name}: prompt token outside [0, {cfg.vocab})')
    if req.policy not in ('exact', 'at_least'):
        raise FrontendError(f'policy must be exact or at_least, got {req.policy!r}')
    b = Builder(req.name)
    b.phase = 'serve'
    H, d, Tmax = cfg.heads, cfg.d_model, P - 1 + N
    w = {n: Tensor(s.name, s.rows, s.cols, 'param', buf=f'w.{s.name}') for n, s in model.specs.items()}
    pin = lambda n, r, c, init=None, dtype='f32': Tensor(f'{req.name}.{n}', r, c, 'state', buf=f'{req.name}.{n}', init=init, dtype=dtype)
    lens = pin('lens', 1, 1, ('fill', 0.0), 'i32')
    kc = [pin(f'kc{i}', H * Tmax, d // H) for i in range(cfg.layers)]
    vc = [pin(f'vc{i}', H * Tmax, d // H) for i in range(cfg.layers)]
    b.tensors += [lens, *kc, *vc]

    def qkv(blk, h, seq, pos):
        a = blk.attn
        return (b.rope(a.q(b, h, w), H, seq, cfg.rope_theta, pos), b.rope(a.k(b, h, w), H, seq, cfg.rope_theta, pos), a.v(b, h, w))

    def head(x):
        return model.head(b, model.nf(b, x, w), w)

    if P > 1:                                                    # prefill the first P-1 tokens
        n = P - 1
        ids = pin('prompt', n, 1, ('values', [float(t) for t in req.prompt[:-1]]), 'i32')
        x = b.embedding(ids, w[model.wte], n)
        for l, blk in enumerate(model.blocks):
            with b.scope(f'prefill.h{l}'):
                q, k, v = qkv(blk, blk.n1(b, x, w), n, None)
                b.kv_append(k, v, kc[l], vc[l], lens, n, H, Tmax)
                x = blk.attn.proj(b, b.attention(q, k, v, H, n), w, res=x)
                x = blk.mlp(b, blk.n2(b, x, w), w, res=x)
        b.counter_inc(lens, n)
        b.tensors.append(ids)

    cur = pin('cur0', 1, 1, ('values', [float(req.prompt[-1])]), 'i32')
    for i in range(N):                                           # one-row decode steps
        with b.scope(f'dec{i}'):
            x = b.embedding(cur, w[model.wte], 1)
            for l, blk in enumerate(model.blocks):
                q, k, v = qkv(blk, blk.n1(b, x, w), 1, lens)
                b.kv_append(k, v, kc[l], vc[l], lens, 1, H, Tmax)
                x = blk.attn.proj(b, b.attn_decode(q, kc[l], vc[l], lens, H, Tmax), w, res=x)
                x = blk.mlp(b, blk.n2(b, x, w), w, res=x)
            b.counter_inc(lens, 1)
            nxt = pin(f'cur{i + 1}', 1, 1, None, 'i32')
            b.tensors.append(nxt)
            b.argmax(head(x), nxt)
            cur = nxt
    log.info('serve %s: prompt %d tokens, %d new, weights %s version %d', req.name, P, N, req.policy, req.version)
    return b


def compile_serving(model: GPT, req: ServeRequest, tiling: Tiling, reuse='shape'):
    b = build_serving(model, req)
    low = lower_program(b.ops, tiling, reuse, 0, prefix=f'{req.name}.act')
    spec = to_workload(low, {req.name: GraphPolicy(req.priority, lambda t: req.deadline_us, (req.policy, req.version))})
    return low, spec
