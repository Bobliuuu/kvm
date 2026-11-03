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

from . import log
from .ir import FrontendError, LogicalOp, Region, Tensor
from .opset import ATTN_Q, F_ACC, F_GELU, F_TA, F_TB, MM_TILE, op as device_op


@dataclass(frozen=True)
class Tiling:
    rows: int = 64           # row block of row-wise ops (norms, rope, embedding, cross-entropy, argmax, attention delta)
    ew_rows: int = 64        # row block of flat elementwise ops (gelu_bwd, residual accumulate)
    cols: int = 128          # column block of column reductions (colsum, norm parameter gradients)
    vocab: int = 1024        # key block of the embedding gradient
    opt_elems: int = 16384   # AdamW chunk (rounded down to whole rows)

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
def _mm_tiles(a, b, out, M, N, K, ta, tb, flags, bias=None, res=None, pre=None, acc=False):
    for m0 in range(0, M, MM_TILE):
        m1 = min(m0 + MM_TILE, M)
        for n0 in range(0, N, MM_TILE):
            n1 = min(n0 + MM_TILE, N)
            ra = a.region(0, K, m0, m1) if ta else a.region(m0, m1, 0, K)
            rb = b.region(n0, n1, 0, K) if tb else b.region(0, K, n0, n1)
            tile = out.region(m0, m1, n0, n1)
            reads = [ra, rb, bias.region(0, 1, n0, n1) if bias else None, res.region(m0, m1, n0, n1) if res else None,
                     tile if acc else None]
            yield _task('matmul_tile', reads, [tile, pre.region(m0, m1, n0, n1) if pre else None],
                        a=a, b=b, out=out, bias=bias, res=res, pre=pre, m0=m0, n0=n0, M=M, N=N, K=K, flags=flags, alpha=1.0)


@lowers('linear')
def _(op, tc):
    x, w, out = op.ins['x'], op.ins['w'], op.outs['out']
    yield from _mm_tiles(x, w, out, x.rows, w.cols, x.cols, False, False, F_GELU if op.attrs['gelu'] else 0,
                         op.ins.get('bias'), op.ins.get('res'), op.outs.get('pre'))


@lowers('mm')
def _(op, tc):
    a, b, out, ta, tb = op.ins['a'], op.ins['b'], op.outs['out'], op.attrs['ta'], op.attrs['tb']
    M, K = (a.cols, a.rows) if ta else (a.rows, a.cols)
    N = b.rows if tb else b.cols
    acc = 'into' in op.ins
    yield from _mm_tiles(a, b, out, M, N, K, ta, tb, (F_TA if ta else 0) | (F_TB if tb else 0) | (F_ACC if acc else 0), acc=acc)


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


@lowers('attention')
def _(op, tc):
    q, k, v, out, lse = op.ins['q'], op.ins['k'], op.ins['v'], op.outs['out'], op.outs['lse']
    T, H = op.attrs['seq'], op.attrs['heads']
    hd = q.cols // H
    for b in range(q.rows // T):
        for h in range(H):
            hc = (h * hd, (h + 1) * hd)
            for q0, q1 in _seq_blocks(T, ATTN_Q):
                R, KV = (b * T + q0, b * T + q1), (b * T, b * T + q1)
                yield _task('attn_fwd', [q.region(*R, *hc), k.region(*KV, *hc), v.region(*KV, *hc)],
                            [out.region(*R, *hc), lse.region(b * H + h, b * H + h + 1, q0, q1)],
                            q=q, k=k, v=v, o=out, lse=lse, b=b, h=h, q0=q0, T=T, H=H, Hkv=H, hd=hd, scale=hd ** -0.5)


@lowers('attention_bwd')
def _(op, tc):
    q, k, v, o, do, lse = (op.ins[n] for n in ('q', 'k', 'v', 'o', 'do', 'lse'))
    dq, dk, dv, delta = (op.outs[n] for n in ('dq', 'dk', 'dv', 'delta'))
    T, H = op.attrs['seq'], op.attrs['heads']
    hd, B = q.cols // H, q.rows // op.attrs['seq']
    for b in range(B):                                            # delta = rowsum(do * o), per sequence, all heads
        for t0, t1 in blocks(T, tc.rows):
            yield _task('attn_bwd_delta', [o.region(b * T + t0, b * T + t1), do.region(b * T + t0, b * T + t1)],
                        [delta.region(b * H, (b + 1) * H, t0, t1)], o=o, do=do, delta=delta, r0=b * T + t0, r1=b * T + t1,
                        T=T, H=H, hd=hd)
    for b in range(B):
        for h in range(H):
            hc, bh = (h * hd, (h + 1) * hd), (b * H + h, b * H + h + 1)
            for q0, q1 in _seq_blocks(T, ATTN_Q):
                R, KV = (b * T + q0, b * T + q1), (b * T, b * T + q1)
                yield _task('attn_bwd_dq', [q.region(*R, *hc), k.region(*KV, *hc), v.region(*KV, *hc), do.region(*R, *hc),
                                            lse.region(*bh, q0, q1), delta.region(*bh, q0, q1)], [dq.region(*R, *hc)],
                            q=q, k=k, v=v, do=do, lse=lse, delta=delta, dq=dq, b=b, h=h, q0=q0, T=T, H=H, Hkv=H, hd=hd,
                            scale=hd ** -0.5)
            for k0, k1 in _seq_blocks(T, ATTN_Q):
                Q, K = (b * T + k0, b * T + T), (b * T + k0, b * T + k1)
                yield _task('attn_bwd_dkv', [q.region(*Q, *hc), k.region(*K, *hc), v.region(*K, *hc), do.region(*Q, *hc),
                                             lse.region(*bh, k0, T), delta.region(*bh, k0, T)],
                            [dk.region(*K, *hc), dv.region(*K, *hc)],
                            q=q, k=k, v=v, do=do, lse=lse, delta=delta, dk=dk, dv=dv, b=b, kvh=h, k0=k0, T=T, H=H, Hkv=H, hd=hd,
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
    H = op.attrs['heads']
    hd = q.cols // H
    for h in range(H):
        yield _task('attn_decode', [q.region(0, 1, h * hd, (h + 1) * hd), _full(kc), _full(vc), _full(lens)],
                    [out.region(0, 1, h * hd, (h + 1) * hd)], q=q, kc=kc, vc=vc, lens=lens, o=out, b=0, h=h, H=H, Hkv=H, hd=hd,
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
                    logits=lg, tgt=tgt, lse=lse, loss=loss, dlogits=out, r0=r0, r1=r1, V=lg.cols, ignore=-1, scale=1.0)


@lowers('argmax')
def _(op, tc):
    lg, out = op.ins['logits'], op.outs['out']
    for r0, r1 in blocks(lg.rows, tc.rows):
        yield _task('argmax', [_rows(lg, r0, r1)], [_rows(out, r0, r1)], logits=lg, out=out, r0=r0, r1=r1, V=lg.cols)


@lowers('adamw')
def _(op, tc):
    w, g, m, v, step = (op.ins[n] for n in ('w', 'g', 'm', 'v', 'step'))
    a = op.attrs
    for r0, r1 in blocks(w.rows, max(1, tc.opt_elems // w.cols)):
        rw = [_rows(w, r0, r1), _rows(m, r0, r1), _rows(v, r0, r1)]
        yield _task('adamw', rw + [_rows(g, r0, r1), _full(step), _full(op.ins['tok']) if 'tok' in op.ins else None], rw, p=w, g=g, m=m, v=v, gscale=None, step=step,
                    e0=r0 * w.cols, e1=r1 * w.cols, lr=a['lr'], b1=a['b1'], b2=a['b2'], eps=a['eps'], wd=a['wd'])


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
