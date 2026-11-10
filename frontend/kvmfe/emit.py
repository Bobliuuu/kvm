"""Program -> tile tasks -> memory plan -> dataflow edges -> workload dict (the compiler's JSON schema)."""
from __future__ import annotations

import ctypes
import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

from . import log
from .dataflow import build_edges
from .ir import FrontendError, LogicalOp, Tensor
from .lower import TileTask, Tiling, lower_op
from .plan import Plan, plan_memory

def f32(x):
    return ctypes.c_float(x).value


def init_values(init, rows, cols):
    """Deterministic initial contents of a buffer (row-major list of float32-rounded numbers)."""
    n = rows * cols
    kind = init[0] if init else 'fill'
    if kind == 'fill':
        return [f32(init[1] if init else 0.0)] * n
    if kind == 'seeded':             # ('seeded', seed, std): the compiler's counter-based generator (runtime/initgen.py)
        from . import _runtime
        draws, bias, coef = _runtime.initgen.params('normal', f32(init[2]))
        return _runtime.initgen.seeded_values(init[1], n, draws, bias, coef)
    if kind == 'values':
        return [f32(v) for v in init[1]]
    raise FrontendError(f'unknown init {init!r}')


@dataclass
class GraphPolicy:
    priority: int = 0
    deadline_us: callable = lambda task: -1          # TileTask -> int (relative to admission, -1 none)
    weights_read: tuple | None = None               # (policy, version): declare a versioned read of 'weights' on weight readers


@dataclass
class Lowered:
    tasks: list
    plan: Plan
    preds: list
    ids: list
    tiling: Tiling
    data_files: dict = field(default_factory=dict)      # relative path -> bytes, referenced by `file` inits

    @property
    def edges(self):
        return sum(len(p) for p in self.preds)

    @property
    def joins(self):
        return sum(1 for t in self.tasks if t.name.startswith('join.'))


def lower_program(ops: list[LogicalOp], tiling: Tiling, reuse='shape', gap=0, prefix='act') -> Lowered:
    tasks = []
    for i, op in enumerate(ops):
        for t in lower_op(op, tiling):
            t.name = f's{op.step}/{i}:{t.name}'
            tasks.append(t)
    plan = plan_memory(ops, reuse, gap, prefix)
    rect = lambda r: (plan.buf(r.t), r.r0, r.r1, r.c0, r.c1)
    seq, preds = build_edges(join_min=tiling.join_min, accesses=[([rect(r) for r in t.reads], [rect(r) for r in t.writes]) for t in tasks])
    ordered, nxt = [], None
    for k in range(len(seq) - 1, -1, -1):                      # a join takes the op of the task that first needed it
        if seq[k] >= 0:
            nxt = tasks[seq[k]]
        ordered.append(nxt if seq[k] >= 0 else TileTask('noop', {}, name=f's{nxt.op.step}/join.{k}', op=nxt.op))
    ordered.reverse()
    log.info('lowered %d logical ops -> %d tile tasks (+%d joins), %d edges', len(ops), len(tasks), len(ordered) - len(tasks),
             sum(map(len, preds)))
    return Lowered(ordered, plan, preds, [t.name for t in ordered], tiling)


FILE_MIN = 4096            # value buffers larger than this travel as little-endian files, not JSON lists


def _init_entry(b, data_files):
    """JSON init of a buffer. Big explicit values are returned as a `file` init and recorded in data_files (path -> bytes)."""
    init = b.init or ('fill', 0.0)
    if init[0] == 'fill':
        return {'kind': 'fill', 'value': f32(init[1]) if b.dtype in ('f32', 'bf16') else int(init[1])}
    if init[0] == 'seeded':
        return {'kind': 'seeded', 'seed': init[1], 'scale': f32(init[2]), 'dist': 'normal'}
    vals = init_values(init, b.rows, b.cols)
    if len(vals) <= FILE_MIN:
        return {'kind': 'values', 'values': [int(v) for v in vals] if b.dtype == 'i32' else vals}
    path = 'data/' + ''.join(c if c.isalnum() or c in '._-' else '_' for c in b.name) + ('.i32' if b.dtype == 'i32' else '.f32')
    data_files[path] = struct.pack(f'<{len(vals)}{"i" if b.dtype == "i32" else "f"}', *([int(v) for v in vals] if b.dtype == 'i32' else vals))
    return {'kind': 'file', 'path': path}


def to_workload(low: Lowered, policies: dict[str, GraphPolicy]):
    """Build the workload dict. `policies` maps graph name -> GraphPolicy; every graph used by a task must be present."""
    plan, bufs = low.plan, []
    used = {plan.buf(r.t) for t in low.tasks for r in (*t.reads, *t.writes)}
    used |= {plan.buf(v) for t in low.tasks for v in t.args.values() if isinstance(v, Tensor)}
    used |= {plan.buf(t.pred[0]) for t in low.tasks if t.pred}
    for name, b in plan.buffers.items():
        if name not in used:
            continue
        init = _init_entry(b, low.data_files)
        entry = {'name': name, 'rows': b.rows, 'cols': b.cols, 'init': init}
        if b.dtype != 'f32':
            entry['dtype'] = b.dtype
        bufs.append(entry)
    weight_bufs = {n for n in plan.buffers if n.startswith('w.')}

    graphs: dict[str, list] = {}
    for i, t in enumerate(low.tasks):
        g = t.op.graph
        if g not in policies:
            raise FrontendError(f'task {t.name}: no GraphPolicy for graph {g!r}; have {sorted(policies)}')
        pol = policies[g]
        e = {'id': low.ids[i], 'op': t.dev}
        if t.args:
            e['args'] = {k: (plan.buf(v) if isinstance(v, Tensor) else v) for k, v in t.args.items()}
        if low.preds[i]:
            e['after'] = sorted(low.ids[j] for j in low.preds[i] if low.tasks[j].op.graph == g)
        if pol.priority:
            e['priority'] = pol.priority
        dl = pol.deadline_us(t)
        if dl >= 0:
            e['deadline_us'] = dl
        if t.pred:
            e['when'] = {'buf': plan.buf(t.pred[0]), 'index': t.pred[1], 'negate': t.pred[2]}
        if t.publishes:
            e['publishes'] = t.publishes
        if pol.weights_read and any(plan.buf(r.t) in weight_bufs for r in t.reads):
            e['reads'] = [{'var': 'weights', 'policy': pol.weights_read[0], 'version': pol.weights_read[1]}]
        graphs.setdefault(g, []).append(e)
    variables = sorted({e['publishes'] for es in graphs.values() for e in es if 'publishes' in e}
                       | {r['var'] for es in graphs.values() for e in es for r in e.get('reads', [])})
    cross = sum(1 for i in range(len(low.tasks)) for j in low.preds[i] if low.tasks[j].op.graph != low.tasks[i].op.graph)
    if cross:
        raise FrontendError(f'{cross} dependency edge(s) cross graphs; graphs must communicate only through weight versions')
    spec = {'variables': variables, 'buffers': bufs, 'graphs': [{'name': n, 'tasks': ts} for n, ts in graphs.items()]}
    return spec


def merge(*specs):
    """Union several workloads (same-named buffers must agree). Graph names must be unique."""
    out = {'variables': [], 'buffers': [], 'graphs': []}
    seen: dict[str, dict] = {}
    for s in specs:
        for v in s['variables']:
            if v not in out['variables']:
                out['variables'].append(v)
        for b in s['buffers']:
            old = seen.get(b['name'])
            if old is None:
                seen[b['name']] = b
                out['buffers'].append(b)
            elif {k: v for k, v in old.items() if k != 'init'} != {k: v for k, v in b.items() if k != 'init'}:
                raise FrontendError(f"buffer {b['name']!r} declared with different shapes/dtypes in two workloads")
            elif old.get('init') != b.get('init') and old.get('init', {}).get('kind') == 'fill' and b.get('init', {}).get('kind') == 'values':
                old['init'] = b['init']
        out['graphs'] += s['graphs']
    names = [g['name'] for g in out['graphs']]
    if len(set(names)) != len(names):
        raise FrontendError(f'duplicate graph names in merge: {names}')
    return out


def write_json(spec, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(spec, separators=(',', ':')) + '\n')
    log.info('wrote %s (%.1f MiB)', path, Path(path).stat().st_size / 2**20)
