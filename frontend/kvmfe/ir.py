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
    """A dense rows x cols array. kind: param | state | input | act. dtype: f32 | i32 (index buffers) | bf16 (real 2-byte storage;
    accepted by cast/linear/mm only).
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
    remat: int | None = None        # rematerialization group (ops recomputed in the backward pass), see Builder.remat
    pred: tuple | None = None       # (i32 tensor, flat index, negate): run only when the predicate holds (Builder.when)

    def tensors(self):
        return list(self.ins.values()) + list(self.outs.values()) + ([self.pred[0]] if self.pred else [])


class Builder:
    """Records logical ops in program order. All shape inference lives here; lowering never guesses shapes."""

    def __init__(self, graph='main'):
        self.graph, self.ops, self.tensors = graph, [], []
        self.step, self.phase, self._scope, self._n = 0, 'fwd', [], itertools.count()
        self._remat, self._remat_ids = None, itertools.count()
        self._pred = None

    def scope(self, name):
        b = self

        class _S:
            def __enter__(s):
                b._scope.append(name)

            def __exit__(s, *a):
                b._scope.pop()
        return _S()

    def remat(self, enabled=True):
        """Context: forward ops emitted inside are recomputed just before their backward (activation checkpointing). Only the
        tensors the region READS from outside stay alive across the forward/backward gap."""
        b = self

        class _R:
            def __enter__(s):
                s.prev, b._remat = b._remat, (next(b._remat_ids) if enabled else None)

            def __exit__(s, *a):
                b._remat = s.prev
        return _R()

    # ---- device-side control: predicated ops, predicates, bounded while-loops (see FRONTEND.md "Data-dependent control")
    CMP_KINDS = {'<': 0, '<=': 1, '>': 2, '>=': 3, '==': 4, '!=': 5}

    def flags(self, name, n=1, init=0):
        """A persistent i32 tensor of n predicate/flag elements (1 x n), every element initially `init`."""
        return self.tensor(name, 1, n, 'state', buf=name, dtype='i32', init=('fill', init))

    def when(self, pred, index=0, negate=False):
        """Context: ops emitted inside run on the device only if pred[index] != 0 (== 0 with negate=True). pred is an i32 tensor written
        by earlier ops (`pred_cmp`). A skipped op does not write its outputs: they keep their previous contents, so a predicated op should
        update persistent state in place (state tensors, or an output that an earlier op already wrote); otherwise the value is whatever the
        buffer held. Predicated ops cannot be differentiated and cannot nest."""
        self._check(pred.dtype == 'i32' and 0 <= index < pred.numel, 'when', 'pred must be an i32 tensor and index inside it', pred)
        b = self

        class _W:
            def __enter__(s):
                if b._pred is not None:
                    raise FrontendError('when: predicated regions cannot nest (combine the conditions with pred_cmp comb=and/or)')
                b._pred = (pred, index, bool(negate))

            def __exit__(s, *a):
                b._pred = None
        return _W()

    def pred_cmp(self, x, kind, thresh, index=0, out=None, out_index=0, prev=None, prev_index=0, comb=None):
        """out[out_index] = (x[index] <kind> thresh), optionally AND/OR-ed with prev[prev_index]; comb in (None, 'and', 'or').
        x is f32 (flat index). Returns the i32 predicate tensor (a fresh 1x1 one unless `out` is given)."""
        self._check(kind in self.CMP_KINDS and comb in (None, 'and', 'or') and (comb is None) == (prev is None), 'pred_cmp',
                    f'kind in {sorted(self.CMP_KINDS)}, comb in (None, and, or), prev exactly when comb is set', x)
        self._check(x.dtype == 'f32' and 0 <= index < x.numel, 'pred_cmp', 'x must be f32 with the index inside it', x)
        out = out or self._out('pred', 1, 1, dtype='i32')
        self._check(out.dtype == 'i32' and 0 <= out_index < out.numel, 'pred_cmp', 'out must be i32 with out_index inside it', out)
        ins = {'x': x, **({'prev': prev} if prev else {})}
        if prev is not None:
            self._check(prev.dtype == 'i32' and 0 <= prev_index < prev.numel, 'pred_cmp', 'prev must be i32 with prev_index inside it', prev)
        self.emit('pred_cmp', ins, {'out': out}, xi=index, oi=out_index, pi=prev_index, cmp=self.CMP_KINDS[kind],
                  comb={None: 0, 'and': 1, 'or': 2}[comb], thresh=float(thresh))
        return out

    def while_loop(self, max_iters, cond, body, name='loop'):
        """A device-side while loop with a data-dependent trip count, unrolled to `max_iters` iterations of predicated tasks.
        cond(b) emits nothing and returns (x, index, kind, thresh): the loop continues while x[index] <kind> thresh, evaluated on the device.
        body(b, i) emits the body of iteration i (state is updated in place). Iteration i runs only if the condition held before it and in
        every earlier iteration. Returns the i32 1 x (max_iters + 1) flags tensor: flags[i] = 1 iff iteration i ran (i < max_iters);
        flags[max_iters] = 1 iff the cap was reached with the condition still true. The trip count is sum(flags[:max_iters])."""
        self._check(max_iters >= 1, 'while_loop', 'max_iters must be >= 1', )
        fl = self.flags(name, max_iters + 1)
        for i in range(max_iters):
            x, xi, kind, th = cond(self)
            self.pred_cmp(x, kind, th, xi, out=fl, out_index=i, prev=fl if i else None, prev_index=i - 1 if i else 0, comb='and' if i else None)
            with self.when(fl, i):
                body(self, i)
        x, xi, kind, th = cond(self)
        self.pred_cmp(x, kind, th, xi, out=fl, out_index=max_iters, prev=fl, prev_index=max_iters - 1, comb='and')
        return fl

    def tensor(self, name, rows, cols, kind='act', **kw):
        t = Tensor('.'.join(self._scope + [name]), rows, cols, kind, **kw)
        self.tensors.append(t)
        return t

    def _out(self, name, rows, cols, **kw):
        return self.tensor(f'{name}{next(self._n)}', rows, cols, **kw)

    def emit(self, kind, ins, outs, **attrs):
        op = LogicalOp(kind, ins, outs, attrs, '.'.join(self._scope + [kind]), self.step, self.phase, self.graph,
                       self._remat if self.phase == 'fwd' else None, self._pred)
        if self.phase == 'fwd':
            for t in outs.values():
                t.rg = t.rg or (t.dtype in ('f32', 'bf16') and any(i.rg for i in ins.values()))
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

    def linear(self, x, w, bias=None, res=None, gelu=False, wt=False, bf16=False):
        """out = [gelu](x @ w + bias) [+ res]; one fused matmul_tile epilogue. With gelu the pre-activation is also kept.
        wt: w is stored [N, K] and used transposed (tied embeddings). bf16: operands rounded to bfloat16, fp32 accumulate."""
        K, N = (w.cols, w.rows) if wt else (w.rows, w.cols)
        self._check(x.cols == K, 'linear', f'inner dims {x.cols} != {K}', x, w)
        self._check('i32' not in (x.dtype, w.dtype) and (x.dtype == w.dtype == 'f32' or bf16), 'linear',
                    'bf16-storage operands need bf16=True; i32 operands are not matrices', x, w)
        self._check(all(t is None or t.dtype == 'f32' for t in (bias, res)), 'linear', 'bias and res must be f32', w)
        self._check(bias is None or bias.shape == (1, N), 'linear', 'bias must be 1 x N', w)
        self._check(res is None or res.shape == (x.rows, N), 'linear', 'res must match the output', x, w)
        ins = {'x': x, 'w': w, **({'bias': bias} if bias else {}), **({'res': res} if res else {})}
        outs = {'out': self._out('lin', x.rows, N)}
        if gelu:
            outs['pre'] = self._out('pre', x.rows, N)
        self.emit('linear', ins, outs, gelu=gelu, wt=wt, bf16=bf16)
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
        """Write n_new rows of knew/vnew (post-rope) into the per-head KV cache at positions lens[0].. (one sequence); `heads` = KV heads."""
        self.emit('kv_append', {'knew': knew, 'vnew': vnew, 'kc': kc, 'vc': vc, 'lens': lens}, {'kc': kc, 'vc': vc},
                  n_new=n_new, heads=heads, tmax=tmax)

    def attn_decode(self, q, kc, vc, lens, heads, tmax, kv_heads=None):
        """One query row attends to cache positions [0, lens[0]] inclusive (GQA: kv_heads cache heads)."""
        out = self._out('dec', 1, q.cols)
        self.emit('attn_decode', {'q': q, 'kc': kc, 'vc': vc, 'lens': lens}, {'out': out}, heads=heads, tmax=tmax,
                  kv_heads=kv_heads or heads)
        return out

    def attention(self, q, k, v, heads, seq, kv_heads=None, causal=True):
        """Multi-head attention over sequences of `seq` rows. GQA: k/v have kv_heads heads (heads % kv_heads == 0), each shared by
        heads/kv_heads query heads. causal=False attends to the whole sequence (ViT, encoders)."""
        kvh = kv_heads or heads
        hd = q.cols // heads
        self._check(q.cols % heads == 0 and heads % kvh == 0 and k.shape == v.shape and k.cols == kvh * hd and k.rows == q.rows
                    and q.rows % seq == 0, 'attention',
                    'q [B*T, H*hd], k/v [B*T, Hkv*hd], H % Hkv == 0, rows % seq == 0', q, k, v)
        out, lse = self._out('attn', q.rows, q.cols), self._out('lse', q.rows // seq * heads, seq)
        self.emit('attention', {'q': q, 'k': k, 'v': v}, {'out': out, 'lse': lse}, heads=heads, seq=seq, kv_heads=kvh, causal=causal)
        return out

    def ce_loss(self, logits, tgt, grad_scale=1.0):
        """Mean softmax cross-entropy over rows (loss[0,0]); loss[0,1] holds 1/count for the backward. grad_scale multiplies the
        backward only (loss scaling, 1/microbatches when gradients are accumulated)."""
        self._check(tgt.shape == (logits.rows, 1) and tgt.dtype == 'i32', 'ce_loss', 'tgt must be an i32 column', logits, tgt)
        outs = {'loss': self._out('loss', 1, 2), 'rowloss': self._out('rowloss', logits.rows, 1),
                'lse': self._out('lse', logits.rows, 1)}
        self.emit('ce_loss', {'logits': logits, 'tgt': tgt}, outs, grad_scale=grad_scale)
        return outs['loss']

    def mse_loss(self, pred, tgt, grad_scale=1.0):
        """Mean squared error over all elements (loss[0,0], a 1x1 tensor); tgt is an f32 tensor shaped like pred. Built only from
        existing device ops (scale_flat, residual_add, row_dot, colsum). grad_scale multiplies the backward only, as for ce_loss."""
        self._check(tgt.shape == pred.shape and tgt.dtype == 'f32' and pred.dtype == 'f32', 'mse_loss',
                    'tgt must be an f32 tensor of the same shape as pred', pred, tgt)
        outs = {'loss': self._out('loss', 1, 1), 'diff': self._out('diff', *pred.shape), 'neg': self._out('negtgt', *pred.shape),
                'rowsq': self._out('rowsq', pred.rows, 1)}
        self.emit('mse_loss', {'pred': pred, 'tgt': tgt}, outs, grad_scale=grad_scale)
        return outs['loss']

    def mse_bwd(self, fwd):
        """d(mean loss)/d pred = grad_scale * 2 (pred - tgt) / numel."""
        out = self._out('dpred', *fwd.ins['pred'].shape)
        self.emit('mse_bwd', {'diff': fwd.outs['diff']}, {'out': out}, grad_scale=fwd.attrs['grad_scale'])
        return out

    def ident_index(self, n):
        """i32 column 0..n-1 (a host-constant input buffer shared by every op that needs an identity row index)."""
        return self.tensor(f'ident{n}', n, 1, 'input', buf=f'ident.{n}', dtype='i32', init=('values', list(range(n))))

    def row_select(self, x, mask):
        """out[r,0] = dot(x[r,:], mask[r,:]): with a one-hot f32 mask, the entry of each row picked by an action id (Q(s,a))."""
        self._check(mask.shape == x.shape and mask.dtype == 'f32' and x.dtype == 'f32', 'row_select',
                    'mask must be an f32 tensor shaped like x', x, mask)
        ident = self.ident_index(x.rows)
        out = self._out('rowsel', x.rows, 1)
        self.emit('row_select', {'x': x, 'mask': mask, 'ident': ident}, {'out': out})
        return out

    def pg_loss(self, logits, tgt, adv, grad_scale=1.0):
        """Policy-gradient loss: mean over rows of adv[r] * (-log softmax(logits[r])[tgt[r]]); adv is an f32 column that gets no
        gradient (the advantage / return estimate is data). Minimising it is REINFORCE with that advantage."""
        self._check(tgt.shape == (logits.rows, 1) and tgt.dtype == 'i32', 'pg_loss', 'tgt must be an i32 column', logits, tgt)
        self._check(adv.shape == (logits.rows, 1) and adv.dtype == 'f32', 'pg_loss', 'adv must be an f32 column', logits, adv)
        ident = self.ident_index(logits.rows)
        outs = {'loss': self._out('loss', 1, 1), 'rowloss': self._out('rowloss', logits.rows, 1), 'lse': self._out('lse', logits.rows, 1),
                'ce': self._out('cetmp', 1, 2), 'wrow': self._out('wrow', logits.rows, 1)}
        self.emit('pg_loss', {'logits': logits, 'tgt': tgt, 'adv': adv, 'ident': ident}, outs, grad_scale=grad_scale)
        return outs['loss']

    def pg_bwd(self, fwd):
        i, o = fwd.ins, fwd.outs
        outs = {'out': self._out('dlogits', *i['logits'].shape), 'tmp': self._out('dce', *i['logits'].shape)}
        self.emit('pg_bwd', {'logits': i['logits'], 'tgt': i['tgt'], 'lse': o['lse'], 'ce': o['ce'], 'adv': i['adv'], 'ident': i['ident']},
                  outs, grad_scale=fwd.attrs['grad_scale'])
        return outs['out']

    def argmax(self, logits, out=None):
        out = out or self._out('tok', logits.rows, 1, dtype='i32')
        self.emit('argmax', {'logits': logits}, {'out': out})
        return out

    # ---- backward-capable building blocks (emitted by VJP rules; `into` accumulates in place into an existing gradient)
    def mm(self, a, b, ta=False, tb=False, into=None, bf16=False):
        """out = op(a) @ op(b) (device matmul_tile without epilogue)."""
        M, K = (a.cols, a.rows) if ta else (a.rows, a.cols)
        K2, N = (b.cols, b.rows) if tb else (b.rows, b.cols)
        self._check(K == K2, 'mm', f'inner dims {K} != {K2} (ta={ta}, tb={tb})', a, b)
        self._check('i32' not in (a.dtype, b.dtype) and (a.dtype == b.dtype == 'f32' or bf16), 'mm',
                    'bf16-storage operands need bf16=True; i32 operands are not matrices', a, b)
        out = into or self._out('mm', M, N)
        self._check(out.shape == (M, N), 'mm', 'into has the wrong shape', out)
        self.emit('mm', {'a': a, 'b': b, **({'into': into} if into else {})}, {'out': out}, ta=ta, tb=tb, bf16=bf16)
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
        outs = {'dq': self._out('dq', *q.shape), 'dk': self._out('dk', *k.shape), 'dv': self._out('dv', *v.shape)}
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
                  {'out': out}, grad_scale=fwd.attrs['grad_scale'])
        return out

    # ---- optimizer / bookkeeping
    def adamw(self, w, g, m, v, step, tok=None, gscale=None, lr_buf=None, **hp):
        """In-place AdamW on w (device op adamw). `tok` is the version token the update must wait for (see train.py). With
        lr_buf the learning rate is read from that scalar buffer on the device (adamw_lr) instead of the baked `lr` attribute."""
        self.emit('adamw', {'w': w, 'g': g, 'm': m, 'v': v, 'step': step, **({'tok': tok} if tok else {}),
                            **({'gscale': gscale} if gscale else {}), **({'lr_buf': lr_buf} if lr_buf else {})},
                  {'w': w, 'm': m, 'v': v}, **hp)

    def sgd(self, w, g, lr, tok=None):
        self.emit('sgd', {'w': w, 'g': g, **({'tok': tok} if tok else {})}, {'w': w}, lr=lr)

    def grad_clip(self, grads, max_norm, coef):
        """Global gradient norm over `grads` -> coef[0] = min(1, max_norm/norm), coef[1] = norm (coef is adamw's gscale)."""
        partial = self._out('gnpartial', sum(g.rows for g in grads), 1)      # >= chunks of any tiling (a chunk has >= 1 row)
        self.emit('grad_clip', {f'g{i:04d}': g for i, g in enumerate(grads)}, {'coef': coef, 'partial': partial}, max_norm=max_norm)
        return coef

    def select_window(self, src, step, rows, stride, off=0, name='win', cols=1):
        """dst[i] = src[step[0]*stride + off + i] over the rows x cols elements: a per-step window of a stream, selected on the
        device from a counter (stride and off are in elements)."""
        dst = self._out(name, rows, cols, dtype=src.dtype)
        self.emit('select_window', {'src': src, 'step': step}, {'dst': dst}, stride=stride, off=off)
        return dst

    def counter_inc(self, x, delta=1):
        self.emit('counter_inc', {'x': x}, {'x': x}, delta=delta)

    def barrier(self, reads, writes, publishes=None):
        """No-op task ordered after every region read and before every region written; may publish a version."""
        self.emit('barrier', {f'r{i}': t for i, t in enumerate(reads)}, {f'w{i}': t for i, t in enumerate(writes)}, publishes=publishes)

    # ---- activations, SwiGLU, softmax (extra pack)
    ACT_KINDS = {'relu': 0, 'silu': 1, 'tanh': 2, 'sigmoid': 3}

    def act(self, x, kind):
        self._check(kind in self.ACT_KINDS, 'act', f'unknown activation {kind!r}; known {sorted(self.ACT_KINDS)}', x)
        out = self._out(kind, x.rows, x.cols)
        self.emit('act', {'x': x}, {'out': out}, fn=kind)
        return out

    def act_bwd(self, x, dy, kind, into=None):
        out = into or self._out('d' + kind, x.rows, x.cols)
        self.emit('act_bwd', {'x': x, 'dy': dy, **({'into': into} if into else {})}, {'out': out}, fn=kind)
        return out

    def silu_mul(self, gate, up):
        """SwiGLU gating: silu(gate) * up."""
        self._check(gate.shape == up.shape, 'silu_mul', 'gate and up must match', gate, up)
        out = self._out('swiglu', gate.rows, gate.cols)
        self.emit('silu_mul', {'gate': gate, 'up': up}, {'out': out})
        return out

    def silu_mul_bwd(self, gate, up, dy):
        outs = {'dgate': self._out('dgate', *gate.shape), 'dup': self._out('dup', *gate.shape)}
        self.emit('silu_mul_bwd', {'gate': gate, 'up': up, 'dy': dy}, outs)
        return outs['dgate'], outs['dup']

    def softmax(self, x, scale=1.0):
        out = self._out('softmax', x.rows, x.cols)
        self.emit('softmax', {'x': x}, {'out': out}, scale=scale)
        return out

    def softmax_bwd(self, fwd, dp, into=None):
        out = into or self._out('dsoftmax', *dp.shape)
        self.emit('softmax_bwd', {'p': fwd.outs['out'], 'dp': dp, **({'into': into} if into else {})}, {'out': out}, scale=fwd.attrs['scale'])
        return out

    # ---- row gather / scatter-add (index buffers are i32 columns; idx0 offsets idx/scale so one table serves several experts)
    def gather(self, src, idx, n, idx0=0, scale=None, into=None):
        """out[r,:] (+)= scale[idx0+r] * src[idx[idx0+r],:] for r < n; an index outside src's rows (-1 = padding) gives zeros."""
        self._check(idx.dtype == 'i32' and idx.cols == 1 and idx0 + n <= idx.rows, 'gather', 'idx must be an i32 column covering idx0+n rows', idx)
        self._check(scale is None or (scale.cols == 1 and idx0 + n <= scale.rows), 'gather', 'scale must be a column covering idx0+n rows', idx)
        out = into or self._out('gather', n, src.cols)
        self._check(out.shape == (n, src.cols), 'gather', 'into has the wrong shape', out)
        self.emit('gather', {'src': src, 'idx': idx, **({'scale': scale} if scale else {}), **({'into': into} if into else {})},
                  {'out': out}, n=n, idx0=idx0)
        return out

    def scatter_add(self, src, idx, rows, idx0=0, scale=None, into=None):
        """out[v,:] (+)= sum over r with idx[idx0+r]==v of scale[idx0+r] * src[r,:] (out has `rows` rows; ascending r)."""
        self._check(idx.dtype == 'i32' and idx.cols == 1 and idx0 + src.rows <= idx.rows, 'scatter_add',
                    'idx must be an i32 column covering idx0+src.rows rows', idx)
        self._check(scale is None or (scale.cols == 1 and idx0 + src.rows <= scale.rows), 'scatter_add', 'scale must be a column', idx)
        out = into or self._out('scatter', rows, src.cols)
        self._check(out.shape == (rows, src.cols), 'scatter_add', 'into has the wrong shape', out)
        self.emit('scatter_add', {'src': src, 'idx': idx, **({'scale': scale} if scale else {}), **({'into': into} if into else {})},
                  {'out': out}, idx0=idx0)
        return out

    def row_dot(self, a, b, idx, idx0, out):
        """out[idx0+r] = dot(a[idx[idx0+r],:], b[r,:]) for r < b.rows: writes its own range of `out` (overwrite)."""
        self.emit('row_dot', {'a': a, 'b': b, 'idx': idx}, {'out': out}, idx0=idx0)
        return out

    # ---- mixture-of-experts routing
    def moe_route(self, probs, k, capacity):
        """Top-k expert choice per token with per-expert capacity (earlier tokens win). Returns (sel [N,k], slot [N,k], eidx [E*C,1])."""
        N, E = probs.shape
        self._check(0 < k <= E and capacity >= 1, 'moe_route', f'need 0 < k <= E ({E}) and capacity >= 1', probs)
        outs = {'sel': self._out('sel', N, k, dtype='i32'), 'slot': self._out('slot', N, k, dtype='i32'),
                'eidx': self._out('eidx', E * capacity, 1, dtype='i32')}
        self.emit('moe_route', {'probs': probs}, outs, k=k, capacity=capacity)
        return outs['sel'], outs['slot'], outs['eidx']

    def moe_gates(self, probs, sel, slot, rows, renorm=True):
        """Gate value of every expert slot: probs[t,e] (/ sum of the token's k chosen probs when renorm); 0 for empty slots."""
        out = self._out('gates', rows, 1)
        self.emit('moe_gates', {'probs': probs, 'sel': sel, 'slot': slot}, {'out': out}, renorm=renorm)
        return out

    def moe_balance(self, probs, sel, coef):
        """Load-balancing auxiliary loss coef * E * sum_e f_e P_e (a 1x1 scalar; its gradient is seeded with 1)."""
        out = self._out('aux', 1, 1)
        self.emit('moe_balance', {'probs': probs, 'sel': sel}, {'loss': out}, coef=coef)
        return out

    # ---- positions, precision, schedules
    def add_rowtile(self, x, pos, seq):
        """x + pos[r % seq]: learned absolute position embedding added to every sequence."""
        self._check(pos.shape == (seq, x.cols) and x.rows % seq == 0, 'add_rowtile', 'pos must be seq x d and rows % seq == 0', x, pos)
        out = self._out('addpos', x.rows, x.cols)
        self.emit('add_rowtile', {'x': x, 'pos': pos}, {'out': out}, seq=seq)
        return out

    def pos_bwd(self, fwd, dout, into=None):
        pos = fwd.ins['pos']
        out = into or self._out('dpos', *pos.shape)
        self.emit('pos_bwd', {'dout': dout, **({'into': into} if into else {})}, {'dpos': out}, seq=fwd.attrs['seq'])
        return out

    def cast(self, x, dtype='bf16', real=False, round_grad=True):
        """Round x to bfloat16 values. Default (real=False): the result stays an f32 tensor (bf16 values in fp32 containers, the
        emulated path). real=True: the result is a tensor of dtype 'bf16' stored in 2 bytes per element (KvmBuf dtype 2); it can
        feed linear/mm(bf16=True) and cast(.., 'f32') only. cast(x_bf16, 'f32') widens exactly back to an f32 tensor.
        Gradients pass straight through; with round_grad (default) the gradient is rounded to bf16 values like the forward value
        (use round_grad=False for a weight copy whose master stays f32)."""
        self._check(dtype in ('bf16', 'f32'), 'cast', f'unsupported dtype {dtype!r}; supported: bf16, f32', x)
        if dtype == 'f32':
            self._check(x.dtype == 'bf16', 'cast', "cast to f32 widens a bf16-storage tensor (x is not one)", x)
        else:
            self._check(x.dtype == 'f32', 'cast', "cast to bf16 needs an f32 tensor", x)
        out = self._out('cast', x.rows, x.cols, dtype='bf16' if real and dtype == 'bf16' else 'f32')
        self.emit('cast', {'x': x}, {'out': out}, dtype=dtype, real=bool(real) and dtype == 'bf16', round_grad=round_grad)
        return out

    def lr_schedule(self, step, lr, sched, base, warmup, total, min_ratio):
        """lr[0] = base * schedule(step[0]): the learning rate lives in a device buffer that a task updates every step."""
        self.emit('lr_schedule', {'step': step}, {'lr': lr}, sched=sched, base=base, warmup=warmup, total=total, min_ratio=min_ratio)

    def scale_scalar(self, x, alpha, elems=1):
        """x[:elems] *= alpha in place (the loss-scale unscale of the clip coefficient)."""
        self.emit('scale_flat', {'x': x}, {'x': x}, alpha=alpha, elems=elems)

    def moe_gates_bwd(self, fwd, dgs, into=None):
        out = into or self._out('dprobs', *fwd.ins['probs'].shape)
        ins = {n: fwd.ins[n] for n in ('probs', 'sel', 'slot')}
        self.emit('moe_gates_bwd', {**ins, 'dgs': dgs, **({'into': into} if into else {})}, {'dprobs': out}, renorm=fwd.attrs['renorm'])
        return out

    def moe_balance_bwd(self, fwd, into=None):
        probs = fwd.ins['probs']
        out = into or self._out('dprobs', *probs.shape)
        self.emit('moe_balance_bwd', {'probs': probs, 'sel': fwd.ins['sel'], **({'into': into} if into else {})}, {'out': out},
                  coef=fwd.attrs['coef'])
        return out

    def add(self, a, b):
        """out = a + b (elementwise, same shape)."""
        self._check(a.shape == b.shape, 'add', 'shape mismatch', a, b)
        out = self._out('sum', *a.shape)
        self.emit('add', {'a': a, 'b': b}, {'out': out})
        return out
