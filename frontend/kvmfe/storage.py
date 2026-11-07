"""Real bf16 STORAGE (ABI dtype 2) for programs built with precision 'bf16s'.

The frontend's 'bf16' precision rounds values but keeps them in fp32 containers. This pass, run on the finished (post-autodiff) op list,
makes the roundings real storage:

  * every `cast` output becomes a bf16 tensor (2 bytes per element). A cast whose input is the sole output of a producer that can store
    bf16 itself is folded into that producer: a bf16 matmul (`linear` / `mm`, fp32 epilogue then one rounding), or a storage-aware op
    (NATIVE below: norms, rope, gelu_bwd, residual adds, embedding, cross entropy). Any other cast becomes a narrowing `cast` op (round_bf16).
  * a bf16 matmul whose operand is an fp32 activation reads an explicit bf16 copy of it (one `cast` per tensor, shared by all its consumers
    and folded into the producer when possible): the bf16 tensor-core tile (matmul_mma_bf16) needs bf16 storage on both operands, and the
    emulation already rounded such operands on load, so the numbers do not change.
  * each weight a bf16 matmul reads gets a bf16 shadow `wb.<name>` refreshed from the fp32 MASTER at the start of every step; the master
    (`w.<name>`, AdamW moments, gradients) stays fp32, so the optimizer is untouched and accumulates in fp32. Weight GRADIENTS stay fp32 as well.
  * every other op reads and writes fp32: a bf16 tensor it touches goes through a fp32 scratch tensor (widen before, narrow after).
    The scratch tensors are short-lived and the memory planner reuses them, so the persistent activation footprint is bf16.

Results are bit-identical to precision 'bf16' (emulation) wherever the emulation already rounded, because bf16 values are exactly
representable; storage rounds only at the points listed above.
"""
from __future__ import annotations

from . import log
from .ir import LogicalOp, Tensor
from .opset import MMA_TILE

CAPABLE = ('linear', 'mm')            # lowered to matmul_mma_bf16 / matmul_tile_bf16 when attrs['bf16'] (read and write bf16 storage natively)
# ins of a capable op that may be bf16 storage; everything else (bias) must be fp32
_NATIVE_IN = {'linear': ('x', 'w', 'res'), 'mm': ('a', 'b', 'into')}
_WEIGHT_IN = {'linear': ('w',), 'mm': ('a', 'b')}
_MMA_OPERANDS = {'linear': ('x', 'w'), 'mm': ('a', 'b')}   # operands that must both be bf16 storage for matmul_mma_bf16 (lower._mm_dev)
_OPERAND_IN = {'linear': ('x',), 'mm': ('a', 'b')}        # operands that get an explicit bf16 copy when they are fp32 activations

# Storage-aware (non-matmul) ops: logical op kind -> (in slots, out slots) that may hold bf16 storage. These mirror the device ops' ld_s/st_s
# arguments (include/kvm_ops_gpt.cuh) and runtime/ref_extra._BF16_STORAGE; every other slot of these ops is fp32 (statistics, parameters).
NATIVE = {
    'embedding': ((), ('out',)),
    'layernorm': (('x',), ('out',)),
    'rmsnorm': (('x',), ('out',)),
    'layernorm_bwd': (('x', 'dy'), ('dx',)),
    'rmsnorm_bwd': (('x', 'dy'), ('dx',)),
    'rope': (('x',), ('out',)),
    'rope_bwd': (('dy',), ('out',)),
    'gelu_bwd': (('pre', 'dy'), ('out',)),
    'add': (('a', 'b'), ('out',)),
    'add_into': (('dst', 'src'), ('dst',)),
    'colsum': (('x',), ()),
    'ce_loss': (('logits',), ()),
    'ce_bwd': (('logits',), ('out',)),
}


def _capable(op):
    return op.kind in CAPABLE and op.attrs.get('bf16')


def _cast(x, out, like, name):
    return LogicalOp('cast', {'x': x}, {'out': out}, {'dtype': 'bf16'}, name, like.step, like.phase, like.graph)


def _out_slot(op, x):
    """Slot of `op` through which it can store the bf16 value of its output tensor `x`, or None. Matmuls fold only when they do not
    accumulate into x (the epilogue would have to read an fp32 'into'); storage-aware ops only when x is not also an input (RMW)."""
    if _capable(op):
        return 'out' if op.outs.get('out') is x and 'into' not in op.ins else None
    native = NATIVE.get(op.kind)
    if native is None or any(t is x for t in op.ins.values()):
        return None
    return next((s for s in native[1] if op.outs.get(s) is x), None)


def _narrow_operands(ops, stats):
    """Insert an explicit bf16 copy of every fp32 activation a bf16 matmul reads as an operand (see module docstring). A copy is shared by
    all consumers of the same value of the tensor (a write to the tensor starts a new value)."""
    version: dict = {}
    copies: dict = {}
    out = []
    for op in ops:
        if _capable(op):
            ins = dict(op.ins)
            for n in _OPERAND_IN[op.kind]:
                t = ins.get(n)
                if t is None or t.dtype != 'f32' or t.kind == 'param':
                    continue
                key = (t.uid, version.get(t.uid, 0))
                if key not in copies:
                    copies[key] = Tensor(t.name + '.b16', t.rows, t.cols, 'act', dtype='bf16')
                    out.append(_cast(t, copies[key], op, 'narrow.' + t.name))
                    stats['operand_copies'] += 1
                ins[n] = copies[key]
            op = LogicalOp(op.kind, ins, op.outs, op.attrs, op.name, op.step, op.phase, op.graph, op.remat)
        for t in op.outs.values():
            version[t.uid] = version.get(t.uid, 0) + 1
        out.append(op)
    return out


def apply(ops: list[LogicalOp], weights: bool = True) -> tuple[list[LogicalOp], dict]:
    """weights=False keeps the fp32 masters as the matmul operands (the kernel rounds them on load): no bf16 weight shadows."""
    stats = dict(folded=0, narrowed=0, widened=0, shadows=0, operand_copies=0, bf16_pre=0)
    ops = _narrow_operands(ops, stats)
    uses: dict = {}
    prod: dict = {}
    for i, op in enumerate(ops):
        for t in op.ins.values():
            uses[t.uid] = uses.get(t.uid, 0) + 1
        for t in op.outs.values():
            prod.setdefault(t.uid, []).append(i)

    # 1. cast outputs become bf16; fold the cast into the producing op when that is its only consumer and the producer can store bf16
    drop, out_of = set(), {}
    for i, op in enumerate(ops):
        if op.kind != 'cast' or op.attrs.get('dtype') != 'bf16':
            continue
        x, out = op.ins['x'], op.outs['out']
        out.dtype = 'bf16'
        p = prod.get(x.uid, [])
        slot = _out_slot(ops[p[0]], x) if len(p) == 1 else None
        if slot is not None and uses.get(x.uid) == 1 and x.dtype == 'f32' and p[0] < i:
            out_of[p[0]] = (slot, out)
            drop.add(i)
            stats['folded'] += 1
        else:
            stats['narrowed'] += 1
    ops = [op for i, op in enumerate(ops) if i not in drop] if not out_of else _fold(ops, drop, out_of)

    # 1b. the pre-activation a gelu linear keeps for the backward is only read by gelu_bwd (storage-aware): store it bf16 as well. The forward
    # value is gelu of the fp32 accumulator; only the saved copy is rounded (torch autocast keeps the pre-activation in bf16 too).
    readers: dict = {}
    for op in ops:
        for n, t in op.ins.items():
            readers.setdefault(t.uid, []).append((op.kind, n))
    for op in ops:
        t = op.outs.get('pre') if _capable(op) else None
        if t is not None and t.dtype == 'f32' and readers.get(t.uid) and all(r == ('gelu_bwd', 'pre') for r in readers[t.uid]):
            t.dtype = 'bf16'
            stats['bf16_pre'] += 1

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
    out, first_of_step = [], {}
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
        native_in, native_out = NATIVE.get(op.kind, ((), ()))
        # a bf16 tensor goes through an fp32 scratch unless every slot it occupies in this op is storage-aware
        slots = [(n, t, False) for n, t in op.ins.items()] + [(n, t, True) for n, t in op.outs.items()]
        tmps, pre, post = {}, [], []
        for n, t, is_out in slots:
            if t.dtype == 'bf16' and n not in (native_out if is_out else native_in):
                tmps.setdefault(t.uid, Tensor(f'{t.name}.f32', t.rows, t.cols, 'act'))
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
    out = _align_optimizer(out)
    log.info('bf16 storage: %d ops -> %d (%s)', len(ops), len(out), stats)
    return out, stats


def _align_optimizer(ops):
    """Weight-gradient tiles of the bf16 tensor-core matmul are MMA_TILE[0] rows tall, not MM_TILE: the optimizer chunks (lower._opt_chunks) must
    round to that, or a chunk boundary splits a tile row and two chunks both wait on the same fan-in. Only when some matmul really lowers to it."""
    mma = any(op.kind in CAPABLE and op.attrs.get('bf16') and all(op.ins[n].dtype == 'bf16' for n in _MMA_OPERANDS[op.kind]) for op in ops)
    if not mma:
        return ops
    return [LogicalOp(op.kind, op.ins, op.outs, {**op.attrs, 'tile_rows': MMA_TILE[0]}, op.name, op.step, op.phase, op.graph, op.remat)
            if op.kind in ('adamw', 'grad_clip') else op for op in ops]


def _fold(ops, drop, out_of):
    res = []
    for i, op in enumerate(ops):
        if i in drop:
            continue
        if i in out_of:
            slot, tensor = out_of[i]
            outs = dict(op.outs)
            outs[slot] = tensor
            op = LogicalOp(op.kind, op.ins, outs, op.attrs, op.name, op.step, op.phase, op.graph, op.remat)
        res.append(op)
    return res
