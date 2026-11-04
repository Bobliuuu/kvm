"""Reverse-mode autodiff over logical ops. A VJP rule is a composition of registered logical ops, so autodiff is generic:
adding a differentiable op means registering its lowering and one `@vjp` rule here.

Gradients are accumulated in place (the device ops take an `acc` flag): the first contribution to a tensor allocates its gradient,
later ones pass it back as `into=ctx.into(t)`. A residual connection therefore carries ONE gradient buffer through the whole
network (the fused `res` epilogue forwards its output gradient unchanged to the residual input). Dataflow turns the resulting
in-place hazards into tile-level edges.
"""
from __future__ import annotations

from typing import Callable

from . import log
from .ir import Builder, FrontendError, LogicalOp, Tensor

VJP: dict[str, Callable] = {}


def vjp(kind):
    """Register fn(b, op, g, ctx) -> {input role: gradient Tensor}; g maps output role -> gradient Tensor (None if unused)."""
    def deco(fn):
        if kind in VJP:
            raise ValueError(f'duplicate VJP for {kind}')
        VJP[kind] = fn
        return fn
    return deco


class Ctx:
    def __init__(self, seeds):
        self.grads: dict[int, Tensor] = dict(seeds)

    def into(self, t: Tensor):
        """The existing gradient of t to accumulate into, or None for the first contribution."""
        return self.grads.get(t.uid) if t.rg else None


@vjp('linear')
def _(b, op, g, ctx):
    x, w, dout = op.ins['x'], op.ins['w'], g['out']
    r = {}
    if 'res' in op.ins:
        r['res'] = dout                                         # out = f(x) + res  =>  d res = d out (aliased, see module doc)
    dz = b.gelu_bwd(op.outs['pre'], dout) if op.attrs['gelu'] else dout
    if 'bias' in op.ins:
        r['bias'] = b.colsum(dz, into=ctx.into(op.ins['bias']))
    if w.rg:
        r['w'] = b.mm(x, dz, ta=True, into=ctx.into(w))
    if x.rg:
        r['x'] = b.mm(dz, w, tb=True, into=ctx.into(x))
    return r


def _norm_vjp(b, op, g, ctx):
    x = op.ins['x']
    pg = (ctx.into(op.ins['g']), ctx.into(op.ins['beta'])) if 'beta' in op.ins else (ctx.into(op.ins['g']), None)
    both = all(p is not None for p in pg[:(2 if 'beta' in op.ins else 1)])
    dx, dg, db = b.norm_bwd(op, g['out'], into=ctx.into(x), into_params=pg if both else None)
    return {'x': dx, 'g': dg, **({'beta': db} if db is not None else {})}


vjp('layernorm')(_norm_vjp)
vjp('rmsnorm')(_norm_vjp)


@vjp('rope')
def _(b, op, g, ctx):
    return {'x': b.rope_bwd(op, g['out'])}


@vjp('attention')
def _(b, op, g, ctx):
    dq, dk, dv = b.attention_bwd(op, g['out'])
    return {'q': dq, 'k': dk, 'v': dv}


@vjp('embedding')
def _(b, op, g, ctx):
    return {'wte': b.embedding_bwd(op, g['out'], into=ctx.into(op.ins['wte']))}


@vjp('ce_loss')
def _(b, op, g, ctx):
    return {'logits': b.ce_bwd(op)}      # d(mean loss)/d logits = (softmax - onehot) / count, scale 1 (loss is the root)


def backward(b: Builder, loss: Tensor, fwd_ops: list[LogicalOp], wrt: list[Tensor]) -> dict[Tensor, Tensor]:
    """Append backward ops for `fwd_ops` to b; returns {param: gradient tensor}. d(loss)/d(loss) = 1."""
    prev, b.phase = b.phase, 'bwd'
    ctx = Ctx({loss.uid: loss})
    for op in reversed(fwd_ops):
        g = {r: ctx.grads.get(t.uid) for r, t in op.outs.items()}
        if all(v is None for v in g.values()):
            continue
        if op.kind not in VJP:
            raise FrontendError(f'op {op.kind!r} ({op.name}) has no VJP rule; register one with @autodiff.vjp({op.kind!r})')
        with b.scope('grad'):
            contrib = VJP[op.kind](b, op, g, ctx)
        for role, gt in contrib.items():
            t = op.ins[role]
            if not t.rg:
                continue
            old = ctx.grads.get(t.uid)
            if old is None:
                ctx.grads[t.uid] = gt
            elif old is not gt:
                with b.scope('grad'):
                    b.add_into(old, gt)
    b.phase = prev
    missing = [t.name for t in wrt if t.uid not in ctx.grads]
    if missing:
        raise FrontendError(f'no gradient reached parameter(s) {missing[:5]}; are they used by the loss?')
    log.debug('autodiff: %d forward ops -> %d total ops', len(fwd_ops), len(b.ops))
    return {t: ctx.grads[t.uid] for t in wrt}
