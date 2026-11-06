"""Tiling lowerer: every logical op -> independent tile tasks (device op, typed args, and the tensor regions each tile reads and
writes). Dependencies are NOT decided here: dataflow.py derives every edge from those regions. Tile sizes that the device ops fix
(64x64 matmul tile, 32-row attention blocks) are constants of the op set; everything else is a `Tiling` field, so retargeting a
GPU is a config change.

The declared regions are the contract with the device op: an under-declared region is a race (caught by the random-order test in
tests/frontend), an over-declared one only costs parallelism.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .ir import Builder, FrontendError, LogicalOp, Tensor
from .opset import ATTN_Q, F_ACC, F_GELU, F_PRECISE, F_TA, F_TB, MM_TILE, op as device_op


@dataclass(frozen=True)
class Tiling:
    rows: int = 64           # row block of row-wise ops (norms, rope, embedding, cross-entropy, argmax, attention delta)
    ew_rows: int = 64        # row block of flat elementwise ops (gelu_bwd, residual accumulate)
    cols: int = 128          # column block of column reductions (colsum, norm parameter gradients)
    vocab: int = 1024        # key block of the embedding gradient
    opt_elems: int = 16384   # AdamW chunk (rounded down to whole rows)
    precise: bool = False    # GEMM tiles use 3xTF32 (matmul_tile flag 16, fp32 accuracy) instead of the default 1xTF32

    PRESETS = {
        'cpu': dict(rows=8, ew_rows=8, cols=32, vocab=16, opt_elems=256),
        'a100': dict(rows=128, ew_rows=128, cols=256, vocab=2048, opt_elems=65536),
        'b200': dict(rows=128, ew_rows=256, cols=512, vocab=4096, opt_elems=131072),
    }

    @classmethod
    def preset(cls, name, **override):
        if name not in cls.PRESETS:
            raise FrontendError(f'unknown tiling preset {name!r}; known: {sorted(cls.PRESETS)}')
        return cls(**{**cls.PRESETS[name], **override})

    def __post_init__(self):
        for k, v in vars(self).items():
            if k == 'precise':
                continue
            if not isinstance(v, int) or v < 1:
                raise FrontendError(f'Tiling.{k} must be a positive int, got {v!r}')


@dataclass
class TileTask:
    dev: str
    args: dict                                  # device arg -> Tensor | None (absent) | int | float
    reads: list = field(default_factory=list)   # Regions
    writes: list = field(default_factory=list)
    name: str = ''
    publishes: str | None = None
    op: LogicalOp | None = None                 # originating logical op (step, phase, graph)

    def flops_bytes(self):
        sc = {k: v for k, v in self.args.items() if not isinstance(v, (Tensor, type(None)))}
        d = device_op(self.dev)
        if d.fixed_us is not None:
            return 0.0, 0.0
        regs = {(r.t.uid, r.r0, r.r1, r.c0, r.c1): r for r in self.reads + self.writes}
        return d.flops(sc), 4.0 * sum((r.r1 - r.r0) * (r.c1 - r.c0) for r in regs.values())


LOWER: dict[str, Callable] = {}


def lowers(kind):
    def deco(fn):
        if kind in LOWER:
            raise ValueError(f'duplicate lowering for {kind}')
        LOWER[kind] = fn
        return fn
    return deco


def blocks(n, size):
    return [(a, min(a + size, n)) for a in range(0, n, size)]


def lower_op(op: LogicalOp, tc: Tiling) -> list[TileTask]:
    if op.kind not in LOWER:
        raise FrontendError(f'no lowering registered for logical op {op.kind!r} ({op.name}); registered: {sorted(LOWER)}')
    tasks = list(LOWER[op.kind](op, tc))
    if not tasks:
        raise FrontendError(f'{op.name}: lowering produced no tasks')
    for i, t in enumerate(tasks):
        t.op, t.name = op, f'{op.name}.{i}'
        d = device_op(t.dev)
        missing = [a for a, _ in d.sig if a not in t.args]
        if missing or len(t.args) != len(d.sig):
            raise FrontendError(f'{op.name}: lowering of {t.dev} must set exactly {[a for a, _ in d.sig]}, missing {missing}')
    return tasks


def _rows(t: Tensor, r0, r1):
    return t.region(r0, r1, 0, t.cols)


def _full(t: Tensor):
    return t.region()


def _task(dev, reads, writes, **args):
    return TileTask(dev, args, [r for r in reads if r is not None], [w for w in writes if w is not None])


# ------------------------------------------------------------------------------------------------ matmul family
def _mm_tiles(a, b, out, M, N, K, ta, tb, flags, bias=None, res=None, pre=None, acc=False, dev='matmul_tile', precise=False):
    flags |= F_PRECISE if precise else 0
    for m0 in range(0, M, MM_TILE):
        m1 = min(m0 + MM_TILE, M)
        for n0 in range(0, N, MM_TILE):
            n1 = min(n0 + MM_TILE, N)
            ra = a.region(0, K, m0, m1) if ta else a.region(m0, m1, 0, K)
            rb = b.region(n0, n1, 0, K) if tb else b.region(0, K, n0, n1)
            tile = out.region(m0, m1, n0, n1)
            reads = [ra, rb, bias.region(0, 1, n0, n1) if bias else None, res.region(m0, m1, n0, n1) if res else None,
                     tile if acc else None]
            yield _task(dev, reads, [tile, pre.region(m0, m1, n0, n1) if pre else None],
                        a=a, b=b, out=out, bias=bias, res=res, pre=pre, m0=m0, n0=n0, M=M, N=N, K=K, flags=flags, alpha=1.0)


def _mm_dev(op):
    return 'matmul_tile_bf16' if op.attrs.get('bf16') else 'matmul_tile'


@lowers('linear')
def _(op, tc):
    x, w, out = op.ins['x'], op.ins['w'], op.outs['out']
    wt = op.attrs['wt']
    yield from _mm_tiles(x, w, out, x.rows, out.cols, x.cols, False, wt, (F_GELU if op.attrs['gelu'] else 0) | (F_TB if wt else 0),
                         op.ins.get('bias'), op.ins.get('res'), op.outs.get('pre'), dev=_mm_dev(op), precise=tc.precise)


@lowers('mm')
def _(op, tc):
    a, b, out, ta, tb = op.ins['a'], op.ins['b'], op.outs['out'], op.attrs['ta'], op.attrs['tb']
    M, K = (a.cols, a.rows) if ta else (a.rows, a.cols)
    N = b.rows if tb else b.cols
    acc = 'into' in op.ins
    yield from _mm_tiles(a, b, out, M, N, K, ta, tb, (F_TA if ta else 0) | (F_TB if tb else 0) | (F_ACC if acc else 0), acc=acc,
                         dev=_mm_dev(op), precise=tc.precise)


# ------------------------------------------------------------------------------------------------ data / row-wise ops
@lowers('embedding')
def _(op, tc):
    ids, wte, out = op.ins['ids'], op.ins['wte'], op.outs['out']
    for r0, r1 in blocks(out.rows, tc.rows):
        yield _task('embedding_fwd', [_rows(ids, r0, r1), _full(wte)], [_rows(out, r0, r1)],
                    ids=ids, wte=wte, wpe=None, out=out, r0=r0, r1=r1, T=op.attrs['seq'], d=out.cols)


@lowers('embedding_bwd')
def _(op, tc):
    ids, dout, dtab = op.ins['ids'], op.ins['dout'], op.outs['dtab']
    acc = 'into' in op.ins
    for v0, v1 in blocks(dtab.rows, tc.vocab):
        yield _task('embedding_bwd', [_full(ids), _full(dout), _rows(dtab, v0, v1) if acc else None], [_rows(dtab, v0, v1)],
                    ids=ids, dout=dout, dtab=dtab, N=ids.rows, mod=0, v0=v0, v1=v1, d=dtab.cols, acc=int(acc))


def _norm(op, tc):
    x, out, rstd = op.ins['x'], op.outs['out'], op.outs['rstd']
    ln = op.kind == 'layernorm'
    for r0, r1 in blocks(x.rows, tc.rows):
        reads = [_rows(x, r0, r1), _full(op.ins['g'])] + ([_full(op.ins['beta'])] if ln else [])
        writes = [_rows(out, r0, r1), _rows(rstd, r0, r1)] + ([_rows(op.outs['mean'], r0, r1)] if ln else [])
        if ln:
            yield _task('layernorm_fwd', reads, writes, x=x, gamma=op.ins['g'], beta=op.ins['beta'], out=out,
                        mean=op.outs['mean'], rstd=rstd, r0=r0, r1=r1, d=x.cols, eps=op.attrs['eps'])
        else:
            yield _task('rmsnorm_fwd', reads, writes, x=x, gamma=op.ins['g'], out=out, rstd=rstd, r0=r0, r1=r1, d=x.cols,
                        eps=op.attrs['eps'])


lowers('layernorm')(_norm)
lowers('rmsnorm')(_norm)


def _norm_bwd(op, tc):
    ln = op.kind == 'layernorm_bwd'
    x, g, dy, rstd, dx = op.ins['x'], op.ins['g'], op.ins['dy'], op.ins['rstd'], op.outs['dx']
    mean = op.ins.get('mean')
    for r0, r1 in blocks(x.rows, tc.rows):
        reads = [_rows(x, r0, r1), _rows(dy, r0, r1), _rows(rstd, r0, r1), _full(g), _rows(mean, r0, r1) if ln else None,
                 _rows(dx, r0, r1) if op.attrs['acc_x'] else None]
        a = dict(x=x, dy=dy, gamma=g, rstd=rstd, dx=dx, r0=r0, r1=r1, d=x.cols, acc=int(op.attrs['acc_x']))
        if ln:
            a['mean'] = mean
        yield TileTask('layernorm_bwd_dx' if ln else 'rmsnorm_bwd_dx', a, [r for r in reads if r], [_rows(dx, r0, r1)])
    dg, db = op.outs['dg'], op.outs.get('db')
    for c0, c1 in blocks(x.cols, tc.cols):
        reads = [x.region(0, x.rows, c0, c1), dy.region(0, x.rows, c0, c1), _full(rstd), _full(mean) if ln else None]
        writes = [dg.region(0, 1, c0, c1), db.region(0, 1, c0, c1) if db else None]
        if op.attrs['acc_p']:
            reads += writes
        yield _task('norm_bwd_params', reads, writes, x=x, dy=dy, mean=mean, rstd=rstd, dgamma=dg, dbeta=db, R=x.rows, d=x.cols,
                    c0=c0, c1=c1, acc=int(op.attrs['acc_p']))


lowers('layernorm_bwd')(_norm_bwd)
lowers('rmsnorm_bwd')(_norm_bwd)


@lowers('rope')
def _(op, tc):
    x, out, pos = op.ins['x'], op.outs['out'], op.ins.get('pos')
    H, hd = op.attrs['heads'], x.cols // op.attrs['heads']
    for r0, r1 in blocks(x.rows, tc.rows):
        yield _task('rope_fwd', [_rows(x, r0, r1), _full(pos) if pos else None], [_rows(out, r0, r1)],
                    x=x, out=out, pos=pos, r0=r0, r1=r1, T=op.attrs['seq'], H=H, hd=hd, theta=op.attrs['theta'])


@lowers('rope_bwd')
def _(op, tc):
    dy, out = op.ins['dy'], op.outs['out']
    H, hd = op.attrs['heads'], dy.cols // op.attrs['heads']
    for r0, r1 in blocks(dy.rows, tc.rows):
        yield _task('rope_bwd', [_rows(dy, r0, r1)], [_rows(out, r0, r1)], x=dy, out=out, pos=None, r0=r0, r1=r1,
                    T=op.attrs['seq'], H=H, hd=hd, theta=op.attrs['theta'])


@lowers('gelu_bwd')
def _(op, tc):
    pre, dy, out = op.ins['pre'], op.ins['dy'], op.outs['out']
    for r0, r1 in blocks(pre.rows, tc.ew_rows):
        yield _task('gelu_bwd', [_rows(pre, r0, r1), _rows(dy, r0, r1)], [_rows(out, r0, r1)], x=pre, dy=dy, dx=out,
                    e0=r0 * pre.cols, e1=r1 * pre.cols, acc=0)


@lowers('add_into')
def _(op, tc):
    dst, src = op.ins['dst'], op.ins['src']
    for r0, r1 in blocks(dst.rows, tc.ew_rows):
        yield _task('residual_add', [_rows(dst, r0, r1), _rows(src, r0, r1)], [_rows(dst, r0, r1)], a=dst, b=src, out=dst,
                    e0=r0 * dst.cols, e1=r1 * dst.cols)


@lowers('colsum')
def _(op, tc):
    x, out, acc = op.ins['x'], op.outs['out'], 'into' in op.ins
    for c0, c1 in blocks(x.cols, tc.cols):
        w = out.region(0, 1, c0, c1)
        yield _task('colsum', [x.region(0, x.rows, c0, c1), w if acc else None], [w], x=x, out=out, R=x.rows, c0=c0, c1=c1, acc=int(acc))


# ------------------------------------------------------------------------------------------------ attention
def _seq_blocks(n, size):
    return [(a, min(a + size, n)) for a in range(0, n, size)]


def _attn_geom(op):
    q, k = op.ins['q'], op.ins['k']
    T, H, Hkv = op.attrs['seq'], op.attrs['heads'], op.attrs['kv_heads']
    return T, H, Hkv, q.cols // H, H // Hkv, op.attrs['causal']


def _kv_rows(b, T, q1, causal):
    """Rows of K/V a query block ending at q1 reads: the causal prefix, or the whole sequence."""
    return (b * T, b * T + (q1 if causal else T))


@lowers('attention')
def _(op, tc):
    q, k, v, out, lse = op.ins['q'], op.ins['k'], op.ins['v'], op.outs['out'], op.outs['lse']
    T, H, Hkv, hd, G, causal = _attn_geom(op)
    for b in range(q.rows // T):
        for h in range(H):
            hc, kc = (h * hd, (h + 1) * hd), (h // G * hd, (h // G + 1) * hd)
            for q0, q1 in _seq_blocks(T, ATTN_Q):
                R, KV = (b * T + q0, b * T + q1), _kv_rows(b, T, q1, causal)
                yield _task('attn_fwd' if causal else 'attn_full_fwd',
                            [q.region(*R, *hc), k.region(*KV, *kc), v.region(*KV, *kc)],
                            [out.region(*R, *hc), lse.region(b * H + h, b * H + h + 1, q0, q1)],
                            q=q, k=k, v=v, o=out, lse=lse, b=b, h=h, q0=q0, T=T, H=H, Hkv=Hkv, hd=hd, scale=hd ** -0.5)


@lowers('attention_bwd')
def _(op, tc):
    q, k, v, o, do, lse = (op.ins[n] for n in ('q', 'k', 'v', 'o', 'do', 'lse'))
    dq, dk, dv, delta = (op.outs[n] for n in ('dq', 'dk', 'dv', 'delta'))
    T, H, Hkv, hd, G, causal = _attn_geom(op)
    B = q.rows // T
    for b in range(B):                                            # delta = rowsum(do * o), per sequence, all heads
        for t0, t1 in blocks(T, tc.rows):
            yield _task('attn_bwd_delta', [o.region(b * T + t0, b * T + t1), do.region(b * T + t0, b * T + t1)],
                        [delta.region(b * H, (b + 1) * H, t0, t1)], o=o, do=do, delta=delta, r0=b * T + t0, r1=b * T + t1,
                        T=T, H=H, hd=hd)
    for b in range(B):
        for h in range(H):
            hc, kc, bh = (h * hd, (h + 1) * hd), (h // G * hd, (h // G + 1) * hd), (b * H + h, b * H + h + 1)
            for q0, q1 in _seq_blocks(T, ATTN_Q):
                R, KV = (b * T + q0, b * T + q1), _kv_rows(b, T, q1, causal)
                yield _task('attn_bwd_dq' if causal else 'attn_full_bwd_dq',
                            [q.region(*R, *hc), k.region(*KV, *kc), v.region(*KV, *kc), do.region(*R, *hc),
                             lse.region(*bh, q0, q1), delta.region(*bh, q0, q1)], [dq.region(*R, *hc)],
                            q=q, k=k, v=v, do=do, lse=lse, delta=delta, dq=dq, b=b, h=h, q0=q0, T=T, H=H, Hkv=Hkv, hd=hd,
                            scale=hd ** -0.5)
        for kvh in range(Hkv):                                    # dk/dv sum over the whole GQA group of query heads
            qc = (kvh * G * hd, (kvh + 1) * G * hd)
            kc, bh = (kvh * hd, (kvh + 1) * hd), (b * H + kvh * G, b * H + (kvh + 1) * G)
            for k0, k1 in _seq_blocks(T, ATTN_Q):
                lo = k0 if causal else 0
                Q, K = (b * T + lo, b * T + T), (b * T + k0, b * T + k1)
                yield _task('attn_bwd_dkv' if causal else 'attn_full_bwd_dkv',
                            [q.region(*Q, *qc), k.region(*K, *kc), v.region(*K, *kc), do.region(*Q, *qc),
                             lse.region(*bh, lo, T), delta.region(*bh, lo, T)],
                            [dk.region(*K, *kc), dv.region(*K, *kc)],
                            q=q, k=k, v=v, do=do, lse=lse, delta=delta, dk=dk, dv=dv, b=b, kvh=kvh, k0=k0, T=T, H=H, hd=hd,
                            scale=hd ** -0.5)


@lowers('kv_append')
def _(op, tc):
    knew, vnew, kc, vc, lens = (op.ins[n] for n in ('knew', 'vnew', 'kc', 'vc', 'lens'))
    H, hd = op.attrs['heads'], knew.cols // op.attrs['heads']
    yield _task('kv_append', [_full(knew), _full(vnew), _full(lens), _full(kc), _full(vc)], [_full(kc), _full(vc)],
                knew=knew, vnew=vnew, kc=kc, vc=vc, lens=lens, b0=0, b1=1, n_new=op.attrs['n_new'], Hkv=H, hd=hd,
                Tmax=op.attrs['tmax'])


@lowers('attn_decode')
def _(op, tc):
    q, kc, vc, lens, out = op.ins['q'], op.ins['kc'], op.ins['vc'], op.ins['lens'], op.outs['out']
    H, Hkv = op.attrs['heads'], op.attrs['kv_heads']
    hd = q.cols // H
    for h in range(H):
        yield _task('attn_decode', [q.region(0, 1, h * hd, (h + 1) * hd), _full(kc), _full(vc), _full(lens)],
                    [out.region(0, 1, h * hd, (h + 1) * hd)], q=q, kc=kc, vc=vc, lens=lens, o=out, b=0, h=h, H=H, Hkv=Hkv, hd=hd,
                    Tmax=op.attrs['tmax'], scale=hd ** -0.5)


# ------------------------------------------------------------------------------------------------ loss, sampling, optimizer
@lowers('ce_loss')
def _(op, tc):
    lg, tgt, loss, rl, lse = op.ins['logits'], op.ins['tgt'], op.outs['loss'], op.outs['rowloss'], op.outs['lse']
    for r0, r1 in blocks(lg.rows, tc.rows):
        yield _task('softmax_ce_fwd', [_rows(lg, r0, r1), _rows(tgt, r0, r1)], [_rows(rl, r0, r1), _rows(lse, r0, r1)],
                    logits=lg, tgt=tgt, rowloss=rl, lse=lse, r0=r0, r1=r1, V=lg.cols, ignore=-1)
    yield _task('softmax_ce_reduce', [_full(rl), _full(tgt)], [_full(loss)], rowloss=rl, tgt=tgt, loss=loss, N=lg.rows, ignore=-1)


@lowers('ce_bwd')
def _(op, tc):
    lg, tgt, lse, loss, out = op.ins['logits'], op.ins['tgt'], op.ins['lse'], op.ins['loss'], op.outs['out']
    for r0, r1 in blocks(lg.rows, tc.rows):
        yield _task('softmax_ce_bwd', [_rows(lg, r0, r1), _rows(tgt, r0, r1), _rows(lse, r0, r1), _full(loss)], [_rows(out, r0, r1)],
                    logits=lg, tgt=tgt, lse=lse, loss=loss, dlogits=out, r0=r0, r1=r1, V=lg.cols, ignore=-1, scale=op.attrs['grad_scale'])


@lowers('mse_loss')
def _(op, tc):
    pred, tgt, loss, diff, neg, rowsq = (op.ins['pred'], op.ins['tgt'], *(op.outs[n] for n in ('loss', 'diff', 'neg', 'rowsq')))
    for r0, r1, e0, e1 in _flat_blocks(pred, tc):          # diff = pred + (-1) * tgt
        yield _task('scale_flat', [_rows(tgt, r0, r1)], [_rows(neg, r0, r1)], x=tgt, out=neg, alpha=-1.0, e0=e0, e1=e1)
        yield _task('residual_add', [_rows(pred, r0, r1), _rows(neg, r0, r1)], [_rows(diff, r0, r1)], a=pred, b=neg, out=diff,
                    e0=e0, e1=e1)
    for r0, r1 in blocks(pred.rows, tc.rows):               # per-row sum of squares (idx absent: row r against row r)
        yield _task('row_dot', [_rows(diff, r0, r1)], [_rows(rowsq, r0, r1)], a=diff, b=diff, idx=None, out=rowsq, r0=r0, r1=r1,
                    d=pred.cols, acc=0, idx0=0)
    yield _task('colsum', [_full(rowsq)], [_full(loss)], x=rowsq, out=loss, R=pred.rows, c0=0, c1=1, acc=0)
    yield _task('scale_flat', [_full(loss)], [_full(loss)], x=loss, out=loss, alpha=1.0 / (pred.rows * pred.cols), e0=0, e1=1)


@lowers('mse_bwd')
def _(op, tc):
    diff, out = op.ins['diff'], op.outs['out']
    alpha = 2.0 * op.attrs['grad_scale'] / (diff.rows * diff.cols)
    for r0, r1, e0, e1 in _flat_blocks(diff, tc):
        yield _task('scale_flat', [_rows(diff, r0, r1)], [_rows(out, r0, r1)], x=diff, out=out, alpha=alpha, e0=e0, e1=e1)


@lowers('pg_loss')
def _(op, tc):
    lg, tgt, adv, ident = op.ins['logits'], op.ins['tgt'], op.ins['adv'], op.ins['ident']
    loss, rl, lse, ce, wrow = (op.outs[k] for k in ('loss', 'rowloss', 'lse', 'ce', 'wrow'))
    for r0, r1 in blocks(lg.rows, tc.rows):
        yield _task('softmax_ce_fwd', [_rows(lg, r0, r1), _rows(tgt, r0, r1)], [_rows(rl, r0, r1), _rows(lse, r0, r1)],
                    logits=lg, tgt=tgt, rowloss=rl, lse=lse, r0=r0, r1=r1, V=lg.cols, ignore=-1)
    yield _task('softmax_ce_reduce', [_full(rl), _full(tgt)], [_full(ce)], rowloss=rl, tgt=tgt, loss=ce, N=lg.rows, ignore=-1)
    for r0, r1 in blocks(lg.rows, tc.rows):               # wrow[r] = rowloss[r] * adv[r]  (row_dot over one column)
        yield _task('row_dot', [_full(rl), _rows(adv, r0, r1), _idx_rows(ident, r0, r1 - r0)], [_idx_rows(wrow, r0, r1 - r0)],
                    a=rl, b=adv, idx=ident, out=wrow, r0=r0, r1=r1, d=1, acc=0, idx0=0)
    yield _task('colsum', [_full(wrow)], [_full(loss)], x=wrow, out=loss, R=lg.rows, c0=0, c1=1, acc=0)
    yield _task('scale_flat', [_full(loss)], [_full(loss)], x=loss, out=loss, alpha=1.0 / lg.rows, e0=0, e1=1)


@lowers('pg_bwd')
def _(op, tc):
    lg, tgt, lse, ce, adv, ident = (op.ins[k] for k in ('logits', 'tgt', 'lse', 'ce', 'adv', 'ident'))
    out, tmp = op.outs['out'], op.outs['tmp']
    for r0, r1 in blocks(lg.rows, tc.rows):
        yield _task('softmax_ce_bwd', [_rows(lg, r0, r1), _rows(tgt, r0, r1), _rows(lse, r0, r1), _full(ce)], [_rows(tmp, r0, r1)],
                    logits=lg, tgt=tgt, lse=lse, loss=ce, dlogits=tmp, r0=r0, r1=r1, V=lg.cols, ignore=-1, scale=op.attrs['grad_scale'])
    for r0, r1 in blocks(lg.rows, tc.rows):               # out[r,:] = adv[r] * tmp[r,:]
        yield _task('gather_rows_scaled', [_rows(tmp, r0, r1), _idx_rows(ident, r0, r1 - r0), _idx_rows(adv, r0, r1 - r0)],
                    [_rows(out, r0, r1)], src=tmp, idx=ident, scale=adv, dst=out, r0=r0, r1=r1, d=lg.cols, acc=0, idx0=0)


@lowers('row_select')
def _(op, tc):
    x, mask, ident, out = op.ins['x'], op.ins['mask'], op.ins['ident'], op.outs['out']
    for r0, r1 in blocks(x.rows, tc.rows):
        yield _task('row_dot', [_full(x), _rows(mask, r0, r1), _idx_rows(ident, r0, r1 - r0)], [_idx_rows(out, r0, r1 - r0)],
                    a=x, b=mask, idx=ident, out=out, r0=r0, r1=r1, d=x.cols, acc=0, idx0=0)


@lowers('argmax')
def _(op, tc):
    lg, out = op.ins['logits'], op.outs['out']
    for r0, r1 in blocks(lg.rows, tc.rows):
        yield _task('argmax', [_rows(lg, r0, r1)], [_rows(out, r0, r1)], logits=lg, out=out, r0=r0, r1=r1, V=lg.cols)


@lowers('adamw')
def _(op, tc):
    w, g, m, v, step = (op.ins[n] for n in ('w', 'g', 'm', 'v', 'step'))
    a, gs, tok, lrb = op.attrs, op.ins.get('gscale'), op.ins.get('tok'), op.ins.get('lr_buf')
    for r0, r1 in _opt_chunks(w, tc):
        rw = [_rows(w, r0, r1), _rows(m, r0, r1), _rows(v, r0, r1)]
        reads = rw + [_rows(g, r0, r1), _full(step), _full(tok) if tok else None, _full(gs) if gs else None, _full(lrb) if lrb else None]
        hp = dict(p=w, g=g, m=m, v=v, gscale=gs, step=step, e0=r0 * w.cols, e1=r1 * w.cols, b1=a['b1'], b2=a['b2'], eps=a['eps'], wd=a['wd'])
        yield _task('adamw_lr', reads, rw, lr=lrb, **hp) if lrb else _task('adamw', reads, rw, lr=a['lr'], **hp)


def _opt_chunks(w, tc):
    return blocks(w.rows, max(1, tc.opt_elems // w.cols))


@lowers('grad_clip')
def _(op, tc):
    coef, partial = op.outs['coef'], op.outs['partial']
    slot = 0
    for k in sorted(op.ins):
        g = op.ins[k]
        for r0, r1 in _opt_chunks(g, tc):
            yield _task('gradnorm_partial', [_rows(g, r0, r1)], [partial.region(slot, slot + 1, 0, 1)],
                        g=g, partial=partial, slot=slot, e0=r0 * g.cols, e1=r1 * g.cols)
            slot += 1
    yield _task('gradnorm_finish', [_full(partial)], [_full(coef)], partial=partial, nslots=slot, out=coef,
                max_norm=op.attrs['max_norm'])


@lowers('select_window')
def _(op, tc):
    src, step, dst = op.ins['src'], op.ins['step'], op.outs['dst']
    for r0, r1 in blocks(dst.rows, tc.rows):
        yield _task('select_window', [_full(src), _full(step)], [_rows(dst, r0, r1)], src=src, step=step, dst=dst,
                    i0=r0 * dst.cols, i1=r1 * dst.cols, stride=op.attrs['stride'], off=op.attrs['off'])


@lowers('sgd')
def _(op, tc):
    w, g = op.ins['w'], op.ins['g']
    yield _task('sgd', [_full(w), _full(g), _full(op.ins['tok']) if 'tok' in op.ins else None], [_full(w)], w=w, g=g, lr=op.attrs['lr'], n=w.numel)


@lowers('counter_inc')
def _(op, tc):
    x = op.ins['x']
    yield _task('counter_inc', [_full(x)], [_full(x)], x=x, i0=0, i1=x.numel, delta=op.attrs['delta'])


@lowers('barrier')
def _(op, tc):
    yield TileTask('noop', {}, [t.region() for t in op.ins.values()], [t.region() for t in op.outs.values()],
                   publishes=op.attrs.get('publishes'))


# ------------------------------------------------------------------------------------------------ extra pack
def _flat_blocks(t, tc):
    for r0, r1 in blocks(t.rows, tc.ew_rows):
        yield r0, r1, r0 * t.cols, r1 * t.cols


@lowers('act')
def _(op, tc):
    x, out = op.ins['x'], op.outs['out']
    for r0, r1, e0, e1 in _flat_blocks(x, tc):
        yield _task('act_fwd', [_rows(x, r0, r1)], [_rows(out, r0, r1)], x=x, out=out, kind=Builder.ACT_KINDS[op.attrs['fn']], e0=e0, e1=e1)


@lowers('act_bwd')
def _(op, tc):
    x, dy, out, acc = op.ins['x'], op.ins['dy'], op.outs['out'], 'into' in op.ins
    for r0, r1, e0, e1 in _flat_blocks(x, tc):
        yield _task('act_bwd', [_rows(x, r0, r1), _rows(dy, r0, r1), _rows(out, r0, r1) if acc else None], [_rows(out, r0, r1)],
                    x=x, dy=dy, dx=out, kind=Builder.ACT_KINDS[op.attrs['fn']], e0=e0, e1=e1, acc=int(acc))


@lowers('silu_mul')
def _(op, tc):
    g, u, out = op.ins['gate'], op.ins['up'], op.outs['out']
    for r0, r1, e0, e1 in _flat_blocks(g, tc):
        yield _task('silu_mul_fwd', [_rows(g, r0, r1), _rows(u, r0, r1)], [_rows(out, r0, r1)], gate=g, up=u, out=out, e0=e0, e1=e1)


@lowers('silu_mul_bwd')
def _(op, tc):
    g, u, dy, dg, du = (*(op.ins[n] for n in ('gate', 'up', 'dy')), op.outs['dgate'], op.outs['dup'])
    for r0, r1, e0, e1 in _flat_blocks(g, tc):
        yield _task('silu_mul_bwd', [_rows(g, r0, r1), _rows(u, r0, r1), _rows(dy, r0, r1)], [_rows(dg, r0, r1), _rows(du, r0, r1)],
                    gate=g, up=u, dy=dy, dgate=dg, dup=du, e0=e0, e1=e1, acc=0)


@lowers('softmax')
def _(op, tc):
    x, out = op.ins['x'], op.outs['out']
    for r0, r1 in blocks(x.rows, tc.rows):
        yield _task('softmax_rows_fwd', [_rows(x, r0, r1)], [_rows(out, r0, r1)], x=x, out=out, r0=r0, r1=r1, d=x.cols,
                    scale=op.attrs['scale'])


@lowers('softmax_bwd')
def _(op, tc):
    p, dp, out, acc = op.ins['p'], op.ins['dp'], op.outs['out'], 'into' in op.ins
    for r0, r1 in blocks(p.rows, tc.rows):
        yield _task('softmax_rows_bwd', [_rows(p, r0, r1), _rows(dp, r0, r1), _rows(out, r0, r1) if acc else None], [_rows(out, r0, r1)],
                    p=p, dp=dp, dx=out, r0=r0, r1=r1, d=p.cols, scale=op.attrs['scale'], acc=int(acc))


def _idx_rows(t, idx0, n):
    return t.region(idx0, idx0 + n, 0, 1) if t is not None else None


@lowers('gather')
def _(op, tc):
    src, idx, sc, out, acc = op.ins['src'], op.ins['idx'], op.ins.get('scale'), op.outs['out'], 'into' in op.ins
    o = op.attrs['idx0']
    for r0, r1 in blocks(op.attrs['n'], tc.rows):
        yield _task('gather_rows_scaled', [_full(src), _idx_rows(idx, o + r0, r1 - r0), _idx_rows(sc, o + r0, r1 - r0),
                                           _rows(out, r0, r1) if acc else None], [_rows(out, r0, r1)],
                    src=src, idx=idx, scale=sc, dst=out, r0=r0, r1=r1, d=src.cols, acc=int(acc), idx0=o)


@lowers('scatter_add')
def _(op, tc):
    src, idx, sc, out, acc = op.ins['src'], op.ins['idx'], op.ins.get('scale'), op.outs['out'], 'into' in op.ins
    o, N = op.attrs['idx0'], src.rows
    for v0, v1 in blocks(out.rows, tc.rows):
        yield _task('scatter_add_rows', [_full(src), _idx_rows(idx, o, N), _idx_rows(sc, o, N), _rows(out, v0, v1) if acc else None],
                    [_rows(out, v0, v1)], src=src, idx=idx, scale=sc, dst=out, N=N, v0=v0, v1=v1, d=src.cols, acc=int(acc), idx0=o)


@lowers('row_dot')
def _(op, tc):
    a, b, idx, out, o = op.ins['a'], op.ins['b'], op.ins['idx'], op.outs['out'], op.attrs['idx0']
    for r0, r1 in blocks(b.rows, tc.rows):
        yield _task('row_dot', [_full(a), _rows(b, r0, r1), _idx_rows(idx, o + r0, r1 - r0)], [_idx_rows(out, o + r0, r1 - r0)],
                    a=a, b=b, idx=idx, out=out, r0=r0, r1=r1, d=b.cols, acc=0, idx0=o)


@lowers('moe_route')
def _(op, tc):
    probs, sel, slot, eidx = op.ins['probs'], op.outs['sel'], op.outs['slot'], op.outs['eidx']
    yield _task('moe_route', [_full(probs)], [_full(sel), _full(slot), _full(eidx)], probs=probs, sel=sel, slot=slot, eidx=eidx,
                N=probs.rows, E=probs.cols, k=sel.cols, C=op.attrs['capacity'])


@lowers('moe_gates')
def _(op, tc):
    probs, sel, slot, gs = op.ins['probs'], op.ins['sel'], op.ins['slot'], op.outs['out']
    yield _task('moe_gates_fwd', [_full(probs), _full(sel), _full(slot)], [_full(gs)], probs=probs, sel=sel, slot=slot, gs=gs,
                N=probs.rows, E=probs.cols, k=sel.cols, renorm=int(op.attrs['renorm']))


@lowers('moe_gates_bwd')
def _(op, tc):
    probs, sel, slot, dgs, out, acc = (*(op.ins[n] for n in ('probs', 'sel', 'slot', 'dgs')), op.outs['dprobs'], 'into' in op.ins)
    yield _task('moe_gates_bwd', [_full(probs), _full(sel), _full(slot), _full(dgs), _full(out) if acc else None], [_full(out)],
                probs=probs, sel=sel, slot=slot, dgs=dgs, dprobs=out, N=probs.rows, E=probs.cols, k=sel.cols,
                renorm=int(op.attrs['renorm']), acc=int(acc))


@lowers('moe_balance')
def _(op, tc):
    probs, sel, loss = op.ins['probs'], op.ins['sel'], op.outs['loss']
    yield _task('moe_balance_fwd', [_full(probs), _full(sel)], [_full(loss)], probs=probs, sel=sel, loss=loss, N=probs.rows,
                E=probs.cols, k=sel.cols, coef=op.attrs['coef'])


@lowers('moe_balance_bwd')
def _(op, tc):
    probs, sel, out, acc = op.ins['probs'], op.ins['sel'], op.outs['out'], 'into' in op.ins
    yield _task('moe_balance_bwd', [_full(probs), _full(sel), _full(out) if acc else None], [_full(out)], probs=probs, sel=sel,
                dprobs=out, N=probs.rows, E=probs.cols, k=sel.cols, coef=op.attrs['coef'], acc=int(acc))


@lowers('add_rowtile')
def _(op, tc):
    x, pos, out = op.ins['x'], op.ins['pos'], op.outs['out']
    for r0, r1 in blocks(x.rows, tc.rows):
        yield _task('add_rowtile', [_rows(x, r0, r1), _full(pos)], [_rows(out, r0, r1)], x=x, pos=pos, out=out, r0=r0, r1=r1,
                    T=op.attrs['seq'], d=x.cols)


@lowers('pos_bwd')
def _(op, tc):
    dout, dpos, acc = op.ins['dout'], op.outs['dpos'], 'into' in op.ins
    for v0, v1 in blocks(dpos.rows, tc.vocab):
        yield _task('embedding_bwd', [_full(dout), _rows(dpos, v0, v1) if acc else None], [_rows(dpos, v0, v1)], ids=None, dout=dout,
                    dtab=dpos, N=dout.rows, mod=op.attrs['seq'], v0=v0, v1=v1, d=dpos.cols, acc=int(acc))


@lowers('cast')
def _(op, tc):
    x, out = op.ins['x'], op.outs['out']
    for r0, r1, e0, e1 in _flat_blocks(x, tc):
        yield _task('round_bf16', [_rows(x, r0, r1)], [_rows(out, r0, r1)], x=x, out=out, e0=e0, e1=e1)


@lowers('lr_schedule')
def _(op, tc):
    step, lr, a = op.ins['step'], op.outs['lr'], op.attrs
    yield _task('lr_schedule', [_full(step)], [_full(lr)], step=step, lr=lr, kind=a['sched'], base=a['base'], warmup=a['warmup'],
                total=a['total'], min_ratio=a['min_ratio'])


@lowers('scale_flat')
def _(op, tc):
    x = op.ins['x']
    yield _task('scale_flat', [_full(x)], [_full(x)], x=x, out=x, alpha=op.attrs['alpha'], e0=0, e1=op.attrs['elems'])


# ------------------------------------------------------------------------------------------------ fused matmul epilogues (fuse.py)
def _mm_grid(a, b, M, N, K, ta, tb):
    for m0 in range(0, M, MM_TILE):
        m1 = min(m0 + MM_TILE, M)
        for n0 in range(0, N, MM_TILE):
            n1 = min(n0 + MM_TILE, N)
            ra = a.region(0, K, m0, m1) if ta else a.region(m0, m1, 0, K)
            rb = b.region(n0, n1, 0, K) if tb else b.region(0, K, n0, n1)
            yield m0, m1, n0, n1, ra, rb


@lowers('mm_dgelu')
def _(op, tc):
    a, b, pre, out, ta, tb = op.ins['a'], op.ins['b'], op.ins['pre'], op.outs['out'], op.attrs['ta'], op.attrs['tb']
    M, K = (a.cols, a.rows) if ta else (a.rows, a.cols)
    N = b.rows if tb else b.cols
    for m0, m1, n0, n1, ra, rb in _mm_grid(a, b, M, N, K, ta, tb):
        tile = out.region(m0, m1, n0, n1)
        yield _task('matmul_tile_dgelu', [ra, rb, pre.region(m0, m1, n0, n1)], [tile], a=a, b=b, out=out, pre=pre, m0=m0, n0=n0,
                    M=M, N=N, K=K, flags=(F_TA if ta else 0) | (F_TB if tb else 0) | (F_PRECISE if tc.precise else 0), alpha=1.0)


@lowers('linear_silu_mul')
def _(op, tc):
    x, w, gate, out, up = op.ins['x'], op.ins['w'], op.ins['gate'], op.outs['out'], op.outs['up']
    for m0, m1, n0, n1, ra, rb in _mm_grid(x, w, x.rows, w.cols, x.cols, False, False):
        t = (m0, m1, n0, n1)
        yield _task('matmul_tile_silu_mul', [ra, rb, gate.region(*t)], [out.region(*t), up.region(*t)], a=x, b=w, out=out, gate=gate,
                    up=up, m0=m0, n0=n0, M=x.rows, N=w.cols, K=x.cols, flags=F_PRECISE if tc.precise else 0, alpha=1.0)


@lowers('mm_silu_mul_bwd')
def _(op, tc):
    a, b, gate, up = op.ins['a'], op.ins['b'], op.ins['gate'], op.ins['up']
    dg, du, ta, tb = op.outs['dgate'], op.outs['dup'], op.attrs['ta'], op.attrs['tb']
    M, K = (a.cols, a.rows) if ta else (a.rows, a.cols)
    N = b.rows if tb else b.cols
    for m0, m1, n0, n1, ra, rb in _mm_grid(a, b, M, N, K, ta, tb):
        t = (m0, m1, n0, n1)
        yield _task('matmul_tile_silu_mul_bwd', [ra, rb, gate.region(*t), up.region(*t)], [dg.region(*t), du.region(*t)], a=a, b=b,
                    gate=gate, up=up, dgate=dg, dup=du, m0=m0, n0=n0, M=M, N=N, K=K,
                    flags=(F_TA if ta else 0) | (F_TB if tb else 0) | (F_PRECISE if tc.precise else 0), alpha=1.0)


@lowers('add')
def _(op, tc):
    a, b, out = op.ins['a'], op.ins['b'], op.outs['out']
    for r0, r1, e0, e1 in _flat_blocks(a, tc):
        yield _task('residual_add', [_rows(a, r0, r1), _rows(b, r0, r1)], [_rows(out, r0, r1)], a=a, b=b, out=out, e0=e0, e1=e1)
