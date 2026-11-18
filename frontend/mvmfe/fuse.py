"""Op fusion on the logical program (after autodiff, before lowering): producer-matmul + elementwise consumer pairs become one
logical op whose tiles run the epilogue in the GEMM task (mm_dgelu: matmul_tile flag 512; the SwiGLU pairs: the matmul_tile_* wrappers of the
extra pack): one task, one dependency hop fewer, no intermediate tensor. Only patterns whose intermediate has no other reader are fused, and every fused op computes exactly what
the two ops computed (the CPU reference of a fused tile is the unfused reference, so results are bitwise equal).

    gelu_bwd(pre, mm(dz, W^T))                 -> mm_dgelu            (backward of a GELU MLP)
    silu_mul(gate, linear(x, Wu))              -> linear_silu_mul     (SwiGLU forward, keeps `up` for the backward)
    silu_mul_bwd(gate, up, mm(dy, Wd^T))       -> mm_silu_mul_bwd     (SwiGLU backward)
    ce_bwd(ce_loss)                            -> ce_fused            (one cross-entropy step: forward and backward of a row block in one task, the reduce off the path)
"""
from __future__ import annotations

from collections import Counter

from . import log
from .ir import LogicalOp


def _uses(ops):
    c = Counter()
    for op in ops:
        for t in op.tensors():
            c[t.uid] += 1
    return c


def _plain_mm(op):
    return op.kind == 'mm' and 'into' not in op.ins and not op.attrs.get('bf16')


def _producers(ops):
    return {t.uid: i for i, op in enumerate(ops) for t in op.outs.values()}


DEFAULT_PATTERNS = ('ce_fused', 'mm_dgelu')     # applied in every lowering (lower_program): patterns whose device ops are in the LEAN op set


def fuse(ops: list[LogicalOp], patterns=None) -> tuple[list[LogicalOp], dict]:
    """Returns (new op list, {pattern: count}). Never mutates the input list's ops. patterns: which fused forms to build (default: all)."""
    uses, prod = _uses(ops), _producers(ops)
    drop, replace, stats = set(), {}, Counter()
    on = (lambda name: True) if patterns is None else (lambda name: name in patterns)
    for i, op in enumerate(ops):
        if op.kind == 'gelu_bwd' and on('mm_dgelu'):
            j = prod.get(op.ins['dy'].uid)
            mm = ops[j] if j is not None else None
            if mm and _plain_mm(mm) and uses[op.ins['dy'].uid] == 2 and j not in drop:
                f = LogicalOp('mm_dgelu', {'a': mm.ins['a'], 'b': mm.ins['b'], 'pre': op.ins['pre']}, {'out': op.outs['out']},
                              dict(mm.attrs), mm.name + '+dgelu', mm.step, mm.phase, mm.graph)
                replace[j], drop = f, drop | {i}
                stats['mm_dgelu'] += 1
        elif op.kind == 'silu_mul' and on('linear_silu_mul'):
            j = prod.get(op.ins['up'].uid)
            lin = ops[j] if j is not None else None
            if (lin and lin.kind == 'linear' and not (lin.attrs['gelu'] or lin.attrs['wt'] or lin.attrs['bf16'])
                    and not ({'bias', 'res'} & set(lin.ins)) and j not in drop):
                f = LogicalOp('linear_silu_mul', {'x': lin.ins['x'], 'w': lin.ins['w'], 'gate': op.ins['gate']},
                              {'out': op.outs['out'], 'up': op.ins['up']}, {}, op.name + '+up', op.step, op.phase, op.graph)
                replace[i], drop = f, drop | {j}
                stats['linear_silu_mul'] += 1
        elif op.kind == 'silu_mul_bwd' and on('mm_silu_mul_bwd'):
            j = prod.get(op.ins['dy'].uid)
            mm = ops[j] if j is not None else None
            if mm and _plain_mm(mm) and uses[op.ins['dy'].uid] == 2 and j not in drop:
                f = LogicalOp('mm_silu_mul_bwd', {'a': mm.ins['a'], 'b': mm.ins['b'], 'gate': op.ins['gate'], 'up': op.ins['up']},
                              dict(op.outs), dict(mm.attrs), mm.name + '+swiglu_bwd', mm.step, mm.phase, mm.graph)
                replace[j], drop = f, drop | {i}
                stats['mm_silu_mul_bwd'] += 1
        elif op.kind == 'ce_bwd' and on('ce_fused'):            # the forward and backward of one cross-entropy, adjacent on the chain
            j = prod.get(op.ins['lse'].uid)
            ce = ops[j] if j is not None else None
            if ce and ce.kind == 'ce_loss' and uses[op.ins['lse'].uid] == 2 and j not in drop:
                f = LogicalOp('ce_fused', {'logits': ce.ins['logits'], 'tgt': ce.ins['tgt']},
                              {'loss': ce.outs['loss'], 'rowloss': ce.outs['rowloss'], 'lse': ce.outs['lse'], 'out': op.outs['out']},
                              {'grad_scale': op.attrs['grad_scale']}, ce.name + '+bwd', ce.step, ce.phase, ce.graph)
                replace[j], drop = f, drop | {i}
                stats['ce_fused'] += 1
    out = [replace.get(i, op) for i, op in enumerate(ops) if i not in drop]
    log.info('fusion: %d logical ops -> %d (%s)', len(ops), len(out), dict(stats) or 'nothing fused')
    return out, dict(stats)
