"""Buffer allocation and lifetime planning.

Tensors with a pinned buffer (weights, optimizer state, corpus, losses) keep it. Every other root tensor lives from its first to
its last logical op; tensors of identical shape whose lifetimes are separated by more than `gap` logical ops share one buffer
slot. Sharing is always safe: the dataflow analysis turns the reuse into tile-level WAR/WAW edges, it never relies on this module
for ordering. `gap` trades memory for overlap: reuse right after the last reader serializes consecutive steps, a larger gap keeps
them parallel. Policies: 'shape' (reuse as soon as the gap allows), 'step' (a slot is reused only by a tensor of a LATER step: no
reuse inside a step, so no WAR/WAW edges inside it, at the price of one step's whole activation footprint), 'none'.
`local` names tensors that reuse slots inside a step under 'step' too (pre-rounding's TF32 copies, tf32.py: each lives from its cast to
its last GEMM, so under 'step' every copy would otherwise cost its full size). They get a pool of their own, so the other tensors' plan is
the policy's alone; a local slot is freed after its last reader plus `gap`, like any other. Under 'shape' they share the one pool with
everything else (a separate pool would only plan more), and under 'none' nothing is shared.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field

from . import log
from .ir import FrontendError, LogicalOp, Tensor


_ITEM = {'f32': 4, 'i32': 4, 'bf16': 2}      # bytes per element of each tensor dtype


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


REUSE_POLICIES = ('none', 'shape', 'step')


def plan_memory(ops: list[LogicalOp], reuse: str = 'shape', gap: int = 0, prefix: str = 'act', local: frozenset = frozenset()) -> Plan:
    if reuse not in REUSE_POLICIES:
        raise FrontendError(f"reuse must be one of {REUSE_POLICIES}, got {reuse!r}")
    step_start: dict = {}
    for i, op in enumerate(ops):
        step_start.setdefault(op.step, i)
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
    pinned_bytes = sum(b.rows * b.cols * _ITEM[b.dtype] for b in buffers.values())
    if reuse != 'step':
        local = frozenset()                   # only 'step' needs the local tensors apart (module docstring)
    n_slots, total, pool = 0, 0, 0
    for group, policy in ((False, reuse), (True, 'shape')):       # the local tensors last, in a pool of their own
        free: dict = {}
        active: list = []                     # heap of (last use, uid, shape, slot)
        for uid in sorted((u for u, r in roots.items() if r.buf is None and (u in local) == group), key=lambda u: span[u][0]):
            r, (first, last) = roots[uid], span[uid]
            total += r.numel * _ITEM[r.dtype]
            horizon = step_start[ops[first].step] if policy == 'step' else first         # 'step': only slots freed in earlier steps
            while active and active[0][0] + gap < first and active[0][0] < horizon:
                _, _, key, slot = heapq.heappop(active)
                free.setdefault(key, []).append(slot)
            key = (r.shape, r.dtype)
            slot = free[key].pop() if policy != 'none' and free.get(key) else None
            if slot is None:
                slot = f'{prefix}{n_slots}'
                n_slots += 1
                buffers[slot] = BufferInfo(slot, r.rows, r.cols, False, None, 0, r.dtype)
                pool += r.numel * _ITEM[r.dtype]
            buffers[slot].tensors += 1
            of[uid] = slot
            heapq.heappush(active, (last, uid, key, slot))
    transient = sum(1 for r in roots.values() if r.buf is None)
    stats = dict(reuse=reuse, gap=gap, persistent_bytes=pinned_bytes, pool_bytes=pool, no_reuse_bytes=total, slots=n_slots,
                 transient_tensors=transient, reuse_factor=(total / pool) if pool else 1.0)
    log.info('memory plan: persistent %.1f MiB, activation pool %.1f MiB in %d slots (%d tensors; %.1f MiB without reuse)',
             pinned_bytes / 2**20, pool / 2**20, n_slots, transient, total / 2**20)
    return Plan(buffers, of, stats)
