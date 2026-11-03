"""Logical program IR: tensors (dense 2-D), regions, logical ops and the tracing Builder.

A program is the *sequential* semantics: logical ops in program order. Everything parallel (tiles, edges, memory reuse) is derived
from it by lower.py / dataflow.py / plan.py. Builder methods only infer shapes and record the op.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass


class FrontendError(ValueError):
    """The user's model/program is invalid; the message names the op and tensors involved."""


class Tensor:
    """A dense rows x cols array. kind: param | state | input | act. dtype: f32 | i32 (index buffers).
    `buf` pins a physical buffer name (weights, optimizer state, request buffers); None lets the memory planner choose."""
    __slots__ = ('uid', 'name', 'rows', 'cols', 'kind', 'buf', 'init', 'rg', 'dtype')
    _uid = itertools.count()

    def __init__(self, name, rows, cols, kind='act', buf=None, init=None, rg=False, dtype='f32'):
        if rows < 1 or cols < 1:
            raise FrontendError(f'tensor {name!r}: bad shape {rows}x{cols}')
        self.uid, self.name, self.rows, self.cols, self.kind = next(Tensor._uid), name, rows, cols, kind
        self.buf, self.init, self.rg, self.dtype = buf, init, rg, dtype

    shape = property(lambda s: (s.rows, s.cols))
    numel = property(lambda s: s.rows * s.cols)

    def region(self, r0=0, r1=None, c0=0, c1=None):
        r1, c1 = self.rows if r1 is None else min(r1, self.rows), self.cols if c1 is None else min(c1, self.cols)
        if not (0 <= r0 < r1 and 0 <= c0 < c1):
            raise FrontendError(f'empty/invalid region rows[{r0}:{r1}] cols[{c0}:{c1}] of {self.name} {self.rows}x{self.cols}')
        return Region(self, r0, r1, c0, c1)

    def __repr__(self):
        return f'<{self.name} {self.rows}x{self.cols}>'


@dataclass(frozen=True)
class Region:
    """Half-open rectangle of a tensor."""
    t: Tensor
    r0: int
    r1: int
    c0: int
    c1: int

    shape = property(lambda s: (s.r1 - s.r0, s.c1 - s.c0))

    def __hash__(self):
        return hash((self.t.uid, self.r0, self.r1, self.c0, self.c1))


@dataclass
class LogicalOp:
    kind: str
    ins: dict
    outs: dict
    attrs: dict
    name: str
    step: int = 0
    phase: str = 'fwd'              # data | fwd | bwd | opt | serve | game
    graph: str = 'main'

    def tensors(self):
        return list(self.ins.values()) + list(self.outs.values())


class Builder:
    """Records logical ops in program order. All shape inference lives here; lowering never guesses shapes."""

    def __init__(self, graph='main'):
        self.graph, self.ops, self.tensors = graph, [], []
        self.step, self.phase, self._scope, self._n = 0, 'fwd', [], itertools.count()

    def scope(self, name):
        b = self

        class _S:
            def __enter__(s):
                b._scope.append(name)

            def __exit__(s, *a):
                b._scope.pop()
        return _S()

    def tensor(self, name, rows, cols, kind='act', **kw):
        t = Tensor('.'.join(self._scope + [name]), rows, cols, kind, **kw)
        self.tensors.append(t)
        return t

    def _out(self, name, rows, cols, **kw):
        return self.tensor(f'{name}{next(self._n)}', rows, cols, **kw)

    def emit(self, kind, ins, outs, **attrs):
        op = LogicalOp(kind, ins, outs, attrs, '.'.join(self._scope + [kind]), self.step, self.phase, self.graph)
        if self.phase == 'fwd':
            for t in outs.values():
                t.rg = t.rg or any(i.rg for i in ins.values())
        self.ops.append(op)
        return op

    @staticmethod
    def _check(cond, op, msg, *ts):
        if not cond:
            raise FrontendError(f"{op}: {msg}; operands {[(t.name, t.shape) for t in ts]}")

    # ---- forward ops (each has a lowering in lower.py and, if differentiable, a VJP in autodiff.py)
    def embedding(self, ids, wte, seq):
        self._check(ids.cols == 1 and ids.dtype == 'i32', 'embedding', 'ids must be an i32 column', ids)
        out = self._out('emb', ids.rows, wte.cols)
        self.emit('embedding', {'ids': ids, 'wte': wte}, {'out': out}, seq=seq)
        return out

    def linear(self, x, w, bias=None, res=None, gelu=False):
        """out = [gelu](x @ w + bias) [+ res]; one fused matmul_tile epilogue. With gelu the pre-activation is also kept."""
        self._check(x.cols == w.rows, 'linear', f'inner dims {x.cols} != {w.rows}', x, w)
        self._check(bias is None or bias.shape == (1, w.cols), 'linear', 'bias must be 1 x N', w)
        self._check(res is None or res.shape == (x.rows, w.cols), 'linear', 'res must match the output', x, w)
        ins = {'x': x, 'w': w, **({'bias': bias} if bias else {}), **({'res': res} if res else {})}
        outs = {'out': self._out('lin', x.rows, w.cols)}
        if gelu:
            outs['pre'] = self._out('pre', x.rows, w.cols)
        self.emit('linear', ins, outs, gelu=gelu)
        return outs['out']

    def norm(self, x, g, beta=None, kind='layernorm', eps=1e-5):
        self._check(g.shape == (1, x.cols), kind, 'gamma must be 1 x d', x, g)
        ins, outs = {'x': x, 'g': g}, {'out': self._out('norm', x.rows, x.cols), 'rstd': self._out('rstd', x.rows, 1)}
        if kind == 'layernorm':
            self._check(beta is not None and beta.shape == (1, x.cols), kind, 'needs beta 1 x d', x)
            ins['beta'], outs['mean'] = beta, self._out('mean', x.rows, 1)
        elif kind != 'rmsnorm':
            raise FrontendError(f'unknown norm kind {kind!r}')
        self.emit(kind, ins, outs, eps=eps)
        return outs['out']

    def rope(self, x, heads, seq, theta=10000.0, pos=None):
        """Rotary embedding; position of row r = r % seq (+ pos[0] when `pos` is an i32 scalar buffer, used when serving)."""
        self._check(x.cols % heads == 0 and (x.cols // heads) % 2 == 0, 'rope', 'head dim must be even', x)
        out = self._out('rope', x.rows, x.cols)
        self.emit('rope', {'x': x, **({'pos': pos} if pos else {})}, {'out': out}, heads=heads, seq=seq, theta=theta)
        return out

    def kv_append(self, knew, vnew, kc, vc, lens, n_new, heads, tmax):
        """Write n_new rows of knew/vnew (post-rope) into the per-head KV cache at positions lens[0].. (one sequence)."""
        self.emit('kv_append', {'knew': knew, 'vnew': vnew, 'kc': kc, 'vc': vc, 'lens': lens}, {'kc': kc, 'vc': vc},
                  n_new=n_new, heads=heads, tmax=tmax)

    def attn_decode(self, q, kc, vc, lens, heads, tmax):
        """One query row attends to cache positions [0, lens[0]] inclusive."""
        out = self._out('dec', 1, q.cols)
        self.emit('attn_decode', {'q': q, 'kc': kc, 'vc': vc, 'lens': lens}, {'out': out}, heads=heads, tmax=tmax)
        return out

    def attention(self, q, k, v, heads, seq):
        self._check(q.shape == k.shape == v.shape and q.cols % heads == 0 and q.rows % seq == 0, 'attention',
                    'q/k/v must share a shape, cols % heads == 0, rows % seq == 0', q, k, v)
        out, lse = self._out('attn', q.rows, q.cols), self._out('lse', q.rows // seq * heads, seq)
        self.emit('attention', {'q': q, 'k': k, 'v': v}, {'out': out, 'lse': lse}, heads=heads, seq=seq)
        return out

    def ce_loss(self, logits, tgt):
        """Mean softmax cross-entropy over rows (loss[0,0]); loss[0,1] holds 1/count for the backward."""
        self._check(tgt.shape == (logits.rows, 1) and tgt.dtype == 'i32', 'ce_loss', 'tgt must be an i32 column', logits, tgt)
        outs = {'loss': self._out('loss', 1, 2), 'rowloss': self._out('rowloss', logits.rows, 1),
                'lse': self._out('lse', logits.rows, 1)}
        self.emit('ce_loss', {'logits': logits, 'tgt': tgt}, outs)
        return outs['loss']

    def argmax(self, logits, out=None):
        out = out or self._out('tok', logits.rows, 1, dtype='i32')
        self.emit('argmax', {'logits': logits}, {'out': out})
        return out

    # ---- backward-capable building blocks (emitted by VJP rules; `into` accumulates in place into an existing gradient)
    def mm(self, a, b, ta=False, tb=False, into=None):
        """out = op(a) @ op(b) (device matmul_tile without epilogue)."""
        M, K = (a.cols, a.rows) if ta else (a.rows, a.cols)
        K2, N = (b.cols, b.rows) if tb else (b.rows, b.cols)
        self._check(K == K2, 'mm', f'inner dims {K} != {K2} (ta={ta}, tb={tb})', a, b)
        out = into or self._out('mm', M, N)
        self._check(out.shape == (M, N), 'mm', 'into has the wrong shape', out)
        self.emit('mm', {'a': a, 'b': b, **({'into': into} if into else {})}, {'out': out}, ta=ta, tb=tb)
        return out

    def colsum(self, x, into=None):
        out = into or self._out('colsum', 1, x.cols)
        self.emit('colsum', {'x': x, **({'into': into} if into else {})}, {'out': out})
        return out

    def gelu_bwd(self, pre, dy):
        out = self._out('dgelu', pre.rows, pre.cols)
        self.emit('gelu_bwd', {'pre': pre, 'dy': dy}, {'out': out})
        return out

    def add_into(self, dst, src):
        self._check(dst.shape == src.shape, 'add_into', 'shape mismatch', dst, src)
        self.emit('add_into', {'dst': dst, 'src': src}, {'dst': dst})

    def norm_bwd(self, fwd, dy, into=None, into_params=None):
        """Returns (dx, dgamma, dbeta|None). `into` accumulates into dx; `into_params` = (dgamma, dbeta) existing grads."""
        ins = {k: fwd.ins[k] for k in ('x', 'g')}
        ins.update(dy=dy, rstd=fwd.outs['rstd'], **({'mean': fwd.outs['mean']} if 'mean' in fwd.outs else {}))
        outs = {'dx': into or self._out('dnorm', *dy.shape)}
        pg = into_params or (None, None)
        outs['dg'] = pg[0] or self._out('dg', 1, dy.cols)
        if 'mean' in fwd.outs:
            outs['db'] = pg[1] or self._out('db', 1, dy.cols)
        if into:
            ins['into'] = into
        self.emit(fwd.kind + '_bwd', ins, outs, eps=fwd.attrs['eps'], acc_x=into is not None, acc_p=into_params is not None)
        return outs['dx'], outs['dg'], outs.get('db')

    def attention_bwd(self, fwd, dout):
        q, k, v = (fwd.ins[n] for n in 'qkv')
        outs = {n: self._out('d' + n, *q.shape) for n in ('dq', 'dk', 'dv')}
        outs['delta'] = self._out('delta', *fwd.outs['lse'].shape)
        self.emit('attention_bwd', {'q': q, 'k': k, 'v': v, 'o': fwd.outs['out'], 'do': dout, 'lse': fwd.outs['lse']}, outs, **fwd.attrs)
        return outs['dq'], outs['dk'], outs['dv']

    def rope_bwd(self, fwd, dy):
        out = self._out('drope', *dy.shape)
        self.emit('rope_bwd', {'dy': dy}, {'out': out}, **fwd.attrs)
        return out

    def embedding_bwd(self, fwd, dout, into=None):
        out = into or self._out('dwte', *fwd.ins['wte'].shape)
        self.emit('embedding_bwd', {'ids': fwd.ins['ids'], 'dout': dout, **({'into': into} if into else {})}, {'dtab': out}, **fwd.attrs)
        return out

    def ce_bwd(self, fwd):
        out = self._out('dlogits', *fwd.ins['logits'].shape)
        self.emit('ce_bwd', {'logits': fwd.ins['logits'], 'tgt': fwd.ins['tgt'], 'lse': fwd.outs['lse'], 'loss': fwd.outs['loss']},
                  {'out': out})
        return out

    # ---- optimizer / bookkeeping
    def adamw(self, w, g, m, v, step, tok=None, **hp):
        """In-place AdamW on w (device op adamw). `tok` is the version token the update must wait for (see train.py)."""
        self.emit('adamw', {'w': w, 'g': g, 'm': m, 'v': v, 'step': step, **({'tok': tok} if tok else {})},
                  {'w': w, 'm': m, 'v': v}, **hp)

    def sgd(self, w, g, lr, tok=None):
        self.emit('sgd', {'w': w, 'g': g, **({'tok': tok} if tok else {})}, {'w': w}, lr=lr)

    def counter_inc(self, x, delta=1):
        self.emit('counter_inc', {'x': x}, {'x': x}, delta=delta)

    def barrier(self, reads, writes, publishes=None):
        """No-op task ordered after every region read and before every region written; may publish a version."""
        self.emit('barrier', {f'r{i}': t for i, t in enumerate(reads)}, {f'w{i}': t for i, t in enumerate(writes)}, publishes=publishes)
