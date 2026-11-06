"""Real bf16 STORAGE (ABI dtype 2) for programs built with precision 'bf16s'.

The frontend's 'bf16' precision rounds values but keeps them in fp32 containers. This pass, run on the finished (post-autodiff) op list,
makes the roundings real storage:

  * every `cast` output becomes a bf16 tensor (2 bytes per element). A cast whose input is the sole output of a bf16 matmul
    (`linear` / `mm`) is folded into it: the matmul epilogue (fp32 math) stores the rounded value directly (matmul_tile_bf16 accepts bf16
    a, b, out, res, pre). Any other cast becomes a narrowing `cast` op (round_bf16: f32 -> bf16).
  * each weight a bf16 matmul reads gets a bf16 shadow `wb.<name>` refreshed from the fp32 MASTER at the start of every step; the master
    (`w.<name>`, AdamW moments, gradients) stays fp32, so the optimizer is untouched and accumulates in fp32.
  * every other op reads and writes fp32: a bf16 tensor it touches goes through a fp32 scratch tensor (widen before, narrow after).
    The scratch tensors are short-lived and the memory planner reuses them, so the persistent activation footprint is bf16.

Results are bit-identical to precision 'bf16' (emulation) wherever the emulation already rounded, because bf16 values are exactly
representable; storage rounds only at the points listed above.
"""
from __future__ import annotations

from . import log
from .ir import LogicalOp, Tensor

CAPABLE = ('linear', 'mm')            # lowered to matmul_tile_bf16 when attrs['bf16'] (reads/writes bf16 storage natively)
# ins of a capable op that may be bf16 storage; everything else (bias) must be fp32
_NATIVE_IN = {'linear': ('x', 'w', 'res'), 'mm': ('a', 'b', 'into')}
_WEIGHT_IN = {'linear': ('w',), 'mm': ('a', 'b')}


def _capable(op):
    return op.kind in CAPABLE and op.attrs.get('bf16')


def _cast(x, out, like, name):
    return LogicalOp('cast', {'x': x}, {'out': out}, {'dtype': 'bf16'}, name, like.step, like.phase, like.graph)


def apply(ops: list[LogicalOp], weights: bool = True) -> tuple[list[LogicalOp], dict]:
    """weights=False keeps the fp32 masters as the matmul operands (the kernel rounds them on load): no bf16 weight shadows."""
    uses: dict = {}
    prod: dict = {}
    for i, op in enumerate(ops):
        for t in op.ins.values():
            uses[t.uid] = uses.get(t.uid, 0) + 1
        for t in op.outs.values():
            prod.setdefault(t.uid, []).append(i)
    stats = dict(folded=0, narrowed=0, widened=0, shadows=0)

    # 1. cast outputs become bf16; fold the cast into the producing bf16 matmul when that is its only consumer
    drop, out_of = set(), {}
    for i, op in enumerate(ops):
        if op.kind != 'cast' or op.attrs.get('dtype') != 'bf16':
            continue
        x, out = op.ins['x'], op.outs['out']
        out.dtype = 'bf16'
        p = prod.get(x.uid, [])
        m = ops[p[0]] if len(p) == 1 else None
        if (m is not None and _capable(m) and m.outs['out'] is x and 'into' not in m.ins and uses.get(x.uid) == 1
                and x.dtype == 'f32' and p[0] < i):
            out_of[p[0]] = out
            drop.add(i)
            stats['folded'] += 1
        else:
            stats['narrowed'] += 1
    ops = [op for i, op in enumerate(ops) if i not in drop] if not out_of else _fold(ops, drop, out_of)

    # 2. weight shadows (refreshed from the fp32 master at the start of each step)
    shadow: dict = {}
    for op in ops:
        if weights and _capable(op):
            for n in _WEIGHT_IN[op.kind]:
                t = op.ins.get(n)
                if t is not None and t.kind == 'param' and t.uid not in shadow:
                    shadow[t.uid] = Tensor('wb.' + t.name, t.rows, t.cols, 'state', buf='wb.' + t.name, init=('fill', 0.0), dtype='bf16')
    stats['shadows'] = len(shadow)

    # 3. rewrite: shadows for weights, widen/narrow around ops that are not storage-aware
    out, first_of_step, ntmp = [], {}, 0
    masters = {u: t for u, t in ((t.uid, t) for op in ops for t in op.ins.values()) if u in shadow}
    for op in ops:
        if op.step not in first_of_step:
            first_of_step[op.step] = True
            for u, sh in shadow.items():
                out.append(LogicalOp('cast', {'x': masters[u]}, {'out': sh}, {'dtype': 'bf16'}, 'shadow.' + sh.name, op.step, 'data', op.graph))
        if _capable(op):
            ins = dict(op.ins)
            for n in _WEIGHT_IN[op.kind]:
                if n in ins and ins[n].uid in shadow:
                    ins[n] = shadow[ins[n].uid]
            pre = []
            for n, t in list(ins.items()):                      # operands the kernel cannot read as bf16 (bias, ...) are widened
                if t.dtype == 'bf16' and n not in _NATIVE_IN[op.kind]:
                    tmp = Tensor(f'{t.name}.f32', t.rows, t.cols, 'act')
                    pre.append(_cast(t, tmp, op, f'widen.{t.name}'))
                    ins[n] = tmp
            stats['widened'] += len(pre)
            out += pre
            outs = dict(op.outs)
            out.append(LogicalOp(op.kind, ins, outs, op.attrs, op.name, op.step, op.phase, op.graph, op.remat))
            continue
        if op.kind == 'cast' and op.outs['out'].dtype == 'bf16' and op.ins['x'].dtype in ('f32', 'bf16'):
            out.append(op)                                       # a narrowing / copy cast: round_bf16 reads and writes storage natively
            continue
        tmps, pre, post = {}, [], []
        for t in (*op.ins.values(), *op.outs.values()):
            if t.dtype == 'bf16' and t.uid not in tmps:
                tmps[t.uid] = Tensor(f'{t.name}.f32', t.rows, t.cols, 'act')
        if not tmps:
            out.append(op)
            continue
        for t in op.ins.values():
            if t.uid in tmps and not any(p.outs['out'] is tmps[t.uid] for p in pre):
                pre.append(_cast(t, tmps[t.uid], op, f'widen.{t.name}'))
        for t in op.outs.values():
            if t.uid in tmps and not any(p.ins['x'] is tmps[t.uid] for p in post):
                post.append(_cast(tmps[t.uid], t, op, f'narrow.{t.name}'))
        sub = lambda d: {n: tmps.get(t.uid, t) for n, t in d.items()}
        out += pre
        out.append(LogicalOp(op.kind, sub(op.ins), sub(op.outs), op.attrs, op.name, op.step, op.phase, op.graph, op.remat))
        out += post
        stats['widened'] += len(pre)
        stats['narrowed'] += len(post)
    log.info('bf16 storage: %d ops -> %d (%s)', len(ops), len(out), stats)
    return out, stats


def _fold(ops, drop, out_of):
    res = []
    for i, op in enumerate(ops):
        if i in drop:
            continue
        if i in out_of:
            outs = dict(op.outs)
            outs['out'] = out_of[i]
            op = LogicalOp(op.kind, op.ins, outs, op.attrs, op.name, op.step, op.phase, op.graph, op.remat)
        res.append(op)
    return res
