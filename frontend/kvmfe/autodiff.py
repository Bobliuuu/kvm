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
    x, w, dout, wt, lp = op.ins['x'], op.ins['w'], g['out'], op.attrs['wt'], op.attrs['bf16']
    r = {}
    if 'res' in op.ins:
        r['res'] = dout                                         # out = f(x) + res  =>  d res = d out (aliased, see module doc)
    dz = b.gelu_bwd(op.outs['pre'], dout) if op.attrs['gelu'] else dout
    if 'bias' in op.ins:
        r['bias'] = b.colsum(dz, into=ctx.into(op.ins['bias']))
    if w.rg:                                                    # w is [K,N] (or [N,K] when wt): dW = x^T dz  (dz^T x when wt)
        r['w'] = b.mm(dz, x, ta=True, into=ctx.into(w), bf16=lp) if wt else b.mm(x, dz, ta=True, into=ctx.into(w), bf16=lp)
    if x.rg:
        r['x'] = b.mm(dz, w, into=ctx.into(x), bf16=lp) if wt else b.mm(dz, w, tb=True, into=ctx.into(x), bf16=lp)
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


@vjp('add')
def _(b, op, g, ctx):
    return {'a': g['out'], 'b': g['out']}      # both operands share the output gradient buffer (see module doc)


@vjp('act')
def _(b, op, g, ctx):
    x = op.ins['x']
    return {'x': b.act_bwd(x, g['out'], op.attrs['fn'], into=ctx.into(x))}


@vjp('silu_mul')
def _(b, op, g, ctx):
    dgate, dup = b.silu_mul_bwd(op.ins['gate'], op.ins['up'], g['out'])
    return {'gate': dgate, 'up': dup}


@vjp('softmax')
def _(b, op, g, ctx):
    return {'x': b.softmax_bwd(op, g['out'], into=ctx.into(op.ins['x']))}


@vjp('gather')
def _(b, op, g, ctx):
    src, idx, sc, o, n = op.ins['src'], op.ins['idx'], op.ins.get('scale'), op.attrs['idx0'], op.attrs['n']
    r = {}
    if src.rg:
        r['src'] = b.scatter_add(g['out'], idx, src.rows, o, sc, into=ctx.into(src))
    if sc is not None and sc.rg:
        r['scale'] = b.row_dot(src, g['out'], idx, o, ctx.into(sc) or b._out('dscale', *sc.shape))
    return r


@vjp('scatter_add')
def _(b, op, g, ctx):
    src, idx, sc, o = op.ins['src'], op.ins['idx'], op.ins.get('scale'), op.attrs['idx0']
    r = {}
    if src.rg:
        r['src'] = b.gather(g['out'], idx, src.rows, o, sc, into=ctx.into(src))
    if sc is not None and sc.rg:
        r['scale'] = b.row_dot(g['out'], src, idx, o, ctx.into(sc) or b._out('dscale', *sc.shape))
    return r


@vjp('moe_gates')
def _(b, op, g, ctx):
    probs = op.ins['probs']
    return {'probs': b.moe_gates_bwd(op, g['out'], into=ctx.into(probs))}


@vjp('moe_balance')
def _(b, op, g, ctx):
    probs = op.ins['probs']
    return {'probs': b.moe_balance_bwd(op, into=ctx.into(probs))}      # the loss is a root: seeded with 1


@vjp('add_rowtile')
def _(b, op, g, ctx):
    r = {'x': g['out']}
    if op.ins['pos'].rg:
        r['pos'] = b.pos_bwd(op, g['out'], into=ctx.into(op.ins['pos']))
    return r


@vjp('cast')
def _(b, op, g, ctx):
    return {'x': b.cast(g['out'], op.attrs['dtype'])}                    # straight-through: the gradient is rounded the same way


def _recompute(b: Builder, group: list[LogicalOp], ctx: Ctx) -> list[LogicalOp]:
    """Rematerialization: re-emit `group` (forward ops of one checkpoint region) with fresh output tensors right before its
    backward, and hand the gradients of the region's original outputs to the clones. The originals then die at their last
    forward consumer, so the planner can reuse their memory."""
    clone: dict[int, Tensor] = {}
    out = []
    with b.scope('remat'):
        for op in group:
            ins = {r: clone.get(t.uid, t) for r, t in op.ins.items()}
            outs = {}
            for r, t in op.outs.items():          # an in-place output (scatter_add into=..) keeps the identity of its clone
                c = clone.get(t.uid) or b.tensor(t.name.rsplit('.', 1)[-1] + '.re', t.rows, t.cols, dtype=t.dtype, rg=t.rg)
                clone[t.uid], outs[r] = c, c
            out.append(b.emit(op.kind, ins, outs, **op.attrs))
    for uid, c in clone.items():
        if uid in ctx.grads:
            ctx.grads[c.uid] = ctx.grads.pop(uid)
    return out


def backward(b: Builder, loss, fwd_ops: list[LogicalOp], wrt: list[Tensor], accumulate: dict | None = None) -> dict[Tensor, Tensor]:
    """Append backward ops for `fwd_ops` to b; returns {param: gradient tensor}. d(loss)/d(loss) = 1 for every root in `loss`
    (a tensor or a list: cross-entropy plus auxiliary losses). `accumulate` maps parameters to gradient tensors of an earlier
    microbatch: this backward adds into them instead of allocating (gradient accumulation). Ops inside `Builder.remat()`
    regions are recomputed first (see _recompute)."""
    prev, b.phase = b.phase, 'bwd'
    roots = [loss] if isinstance(loss, Tensor) else list(loss)
    ctx = Ctx({t.uid: t for t in roots})
    ctx.grads.update({w.uid: g for w, g in (accumulate or {}).items()})

    def step(op):
        g = {r: ctx.grads.get(t.uid) for r, t in op.outs.items()}
        if all(v is None for v in g.values()):
            return
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

    i = len(fwd_ops) - 1
    while i >= 0:
        op = fwd_ops[i]
        if op.remat is None:
            step(op)
            i -= 1
            continue
        j = i
        while j > 0 and fwd_ops[j - 1].remat == op.remat:
            j -= 1
        for cop in reversed(_recompute(b, fwd_ops[j:i + 1], ctx)):
            step(cop)
        i = j - 1
    b.phase = prev
    missing = [t.name for t in wrt if t.uid not in ctx.grads]
    if missing:
        raise FrontendError(f'no gradient reached parameter(s) {missing[:5]}; are they used by the loss?')
    log.debug('autodiff: %d forward ops -> %d total ops', len(fwd_ops), len(b.ops))
    return {t: ctx.grads[t.uid] for t in wrt}
