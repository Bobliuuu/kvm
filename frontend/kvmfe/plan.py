"""Buffer allocation and lifetime planning.

Tensors with a pinned buffer (weights, optimizer state, corpus, losses) keep it. Every other root tensor lives from its first to
its last logical op; tensors of identical shape whose lifetimes are separated by more than `gap` logical ops share one buffer
slot. Sharing is always safe: the dataflow analysis turns the reuse into tile-level WAR/WAW edges, it never relies on this module
for ordering. `gap` trades memory for overlap: reuse right after the last reader serializes consecutive steps, a larger gap keeps
them parallel.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field

from . import log
from .ir import FrontendError, LogicalOp, Tensor


DT_BYTES = {'f32': 4, 'i32': 4, 'bf16': 2}      # element size of every buffer dtype


@dataclass
class BufferInfo:
    name: str
    rows: int
    cols: int
    pinned: bool
    init: tuple | None = None
    tensors: int = 1
    dtype: str = 'f32'


@dataclass
class Plan:
    buffers: dict
    of: dict                                 # root tensor uid -> buffer name
    stats: dict = field(default_factory=dict)

    def buf(self, t: Tensor) -> str:
        return self.of[t.uid]


def plan_memory(ops: list[LogicalOp], reuse: str = 'shape', gap: int = 0, prefix: str = 'act') -> Plan:
    if reuse not in ('none', 'shape'):
        raise FrontendError(f"reuse must be 'none' or 'shape', got {reuse!r}")
    span: dict = {}
    roots: dict = {}
    for i, op in enumerate(ops):
        for t in op.tensors():
            roots[t.uid] = t
            span.setdefault(t.uid, [i, i])[1] = i
    buffers: dict = {}
    of: dict = {}
    for uid, r in roots.items():
        if r.buf is None:
            continue
        of[uid] = r.buf
        b = buffers.get(r.buf)
        if b is None:
            buffers[r.buf] = BufferInfo(r.buf, r.rows, r.cols, True, r.init, 1, r.dtype)
            continue
        if (b.rows, b.cols, b.dtype) != (r.rows, r.cols, r.dtype):
            raise FrontendError(f'pinned buffer {r.buf!r} shared by tensors of different shapes: {(b.rows, b.cols)} vs {r.shape}')
        b.tensors += 1
        if b.init is None or (b.init[0] == 'fill' and r.init and r.init[0] != 'fill'):
            b.init = r.init
    pinned_bytes = sum(b.rows * b.cols * DT_BYTES[b.dtype] for b in buffers.values())
    free: dict = {}
    active: list = []                         # heap of (last use, uid, shape, slot)
    n_slots, total, pool = 0, 0, 0
    for uid in sorted((u for u, r in roots.items() if r.buf is None), key=lambda u: span[u][0]):
        r, (first, last) = roots[uid], span[uid]
        total += r.numel * DT_BYTES[r.dtype]
        while active and active[0][0] + gap < first:
            _, _, key, slot = heapq.heappop(active)
            free.setdefault(key, []).append(slot)
        key = (r.shape, r.dtype)
        slot = free[key].pop() if reuse == 'shape' and free.get(key) else None
        if slot is None:
            slot = f'{prefix}{n_slots}'
            n_slots += 1
            buffers[slot] = BufferInfo(slot, r.rows, r.cols, False, None, 0, r.dtype)
            pool += r.numel * DT_BYTES[r.dtype]
        buffers[slot].tensors += 1
        of[uid] = slot
        heapq.heappush(active, (last, uid, key, slot))
    transient = sum(1 for r in roots.values() if r.buf is None)
    stats = dict(reuse=reuse, gap=gap, persistent_bytes=pinned_bytes, pool_bytes=pool, no_reuse_bytes=total, slots=n_slots,
                 transient_tensors=transient, reuse_factor=(total / pool) if pool else 1.0)
    log.info('memory plan: persistent %.1f MiB, activation pool %.1f MiB in %d slots (%d tensors; %.1f MiB without reuse)',
             pinned_bytes / 2**20, pool / 2**20, n_slots, transient, total / 2**20)
    return Plan(buffers, of, stats)
