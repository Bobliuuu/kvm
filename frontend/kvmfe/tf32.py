"""TF32 operand copies for 256x128-tile GEMMs (Tiling.tf32_preround).

A GEMM that rounds its operands to TF32 does it inside every tile that reads them: an element of A is rounded once per column of tiles
and one of B once per row of tiles (16 to 24 times at the 162M model's shapes), and on the 256x128 tile that work costs about a quarter
of its speed. This pass gives such a GEMM TF32 copies of its operands instead. Each copy is made once per value of the operand by a
`cast` op (dtype 'tf32', lowered to the device op round_tf32) and shared by every GEMM that reads the same value (a write to the tensor
starts a new value), and the GEMM is marked `tf32_exact` (matmul_tile flag 1024: the tile hands the copies to the tensor core as they
are). The products are bit for bit those of the rounding GEMM; the price is one read and one write of each operand per value, and a
copy buffer the memory planner reuses once its last GEMM has read it.

Only GEMMs the lowering puts on the 256x128 tile take copies: their operands are re-read most, and a smaller GEMM pays more for an extra
task than the rounding costs it.
"""
from __future__ import annotations

from . import log
from .ir import LogicalOp, Tensor
from .lower import Tiling, mm_choice
from .opset import F_HUGE

OPERANDS = {'linear': ('x', 'w'), 'mm': ('a', 'b')}


def gemm_dims(op: LogicalOp) -> tuple[int, int, int]:
    """M, N, K of a linear or mm op, as the lowering computes them."""
    if op.kind == 'linear':
        x, out = op.ins['x'], op.outs['out']
        return x.rows, out.cols, x.cols
    a, b = op.ins['a'], op.ins['b']
    M, K = (a.cols, a.rows) if op.attrs['ta'] else (a.rows, a.cols)
    N = b.rows if op.attrs['tb'] else b.cols
    return M, N, K


def takes_copies(op: LogicalOp, tc: Tiling) -> bool:
    """A plain 1xTF32 GEMM (no bf16 mode, not already exact) that the lowering puts on the 256x128 tile."""
    if op.kind not in OPERANDS or op.attrs.get('bf16') or op.attrs.get('tf32_exact'):
        return False
    if any(op.ins[n].dtype != 'f32' for n in OPERANDS[op.kind]):
        return False
    return bool(mm_choice(*gemm_dims(op), tc) & F_HUGE)


def apply(ops: list[LogicalOp], tc: Tiling) -> tuple[list[LogicalOp], dict]:
    """The op list with a TF32 copy in front of every operand of each GEMM that takes copies, and those GEMMs marked tf32_exact."""
    version: dict = {}
    copies: dict = {}
    stats = {'gemms': 0, 'copies': 0, 'elements': 0}
    out = []
    for op in ops:
        if takes_copies(op, tc):
            ins = dict(op.ins)
            for n in OPERANDS[op.kind]:
                t = ins[n]
                key = (t.uid, version.get(t.uid, 0))
                if key not in copies:
                    copies[key] = Tensor(t.name + '.tf32', t.rows, t.cols, 'act')
                    out.append(LogicalOp('cast', {'x': t}, {'out': copies[key]}, {'dtype': 'tf32'}, 'tf32.' + t.name, op.step, op.phase, op.graph))
                    stats['copies'] += 1
                    stats['elements'] += t.rows * t.cols
                ins[n] = copies[key]
            op = LogicalOp(op.kind, ins, op.outs, {**op.attrs, 'tf32_exact': True}, op.name, op.step, op.phase, op.graph, op.remat)
            stats['gemms'] += 1
        for t in op.outs.values():
            version[t.uid] = version.get(t.uid, 0) + 1
        out.append(op)
    log.info('tf32 copies: %d GEMMs read %d copies (%.1fM elements rounded once each)', stats['gemms'], stats['copies'], stats['elements'] / 1e6)
    return out, stats
