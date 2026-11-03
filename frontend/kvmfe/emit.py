"""Program -> tile tasks -> memory plan -> dataflow edges -> workload dict (the compiler's JSON schema, plus buffer views)."""
from __future__ import annotations

import ctypes
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from . import log
from .dataflow import build_edges
from .ir import FrontendError, LogicalOp, Region, Tensor
from .lower import TileTask, Tiling, lower_op
from .plan import Plan, plan_memory

MATERIALIZE_LIMIT = 1 << 20          # elements per buffer whose initial values are written into the JSON


def f32(x):
    return ctypes.c_float(x).value


def init_values(init, rows, cols):
    """Deterministic initial contents of a buffer (row-major list of float32-rounded numbers)."""
    n = rows * cols
    kind = init[0] if init else 'fill'
    if kind == 'fill':
        return [f32(init[1] if init else 0.0)] * n
    if kind == 'param':
        _, spec, seed = init
        if spec[0] == 'fill':
            return [f32(spec[1])] * n
        rng = random.Random(seed)
        return [f32(rng.gauss(0.0, spec[1])) for _ in range(n)]
    if kind == 'values':
        return [f32(v) for v in init[1]]
    if kind == 'corpus':        # next token = 5*t+3 mod V: a learnable bigram chain (full period when V is a power of two)
        v, out, t = init[1], [], 1
        for _ in range(n):
            out.append(float(t))
            t = (5 * t + 3) % v
        return out
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

    @property
    def edges(self):
        return sum(len(p) for p in self.preds)


def lower_program(ops: list[LogicalOp], tiling: Tiling, reuse='shape', gap=0, prefix='act') -> Lowered:
    tasks = []
    for i, op in enumerate(ops):
        for t in lower_op(op, tiling):
            t.name = f's{op.step}/{i}:{t.name}'
            tasks.append(t)
    plan = plan_memory(ops, reuse, gap, prefix)
    rect = lambda r: (plan.buf(r.t), r.r0, r.r1, r.c0, r.c1)
    preds = build_edges([([rect(r) for r in t.reads], [rect(r) for r in t.writes]) for t in tasks])
    log.info('lowered %d logical ops -> %d tile tasks, %d edges', len(ops), len(tasks), sum(map(len, preds)))
    return Lowered(tasks, plan, preds, [t.name for t in tasks], tiling)


def _init_entry(b, limit):
    """JSON init of a buffer; (entry, 1 if too large to materialize else 0)."""
    init = b.init or ('fill', 0.0)
    fill = init[1] if init[0] == 'fill' else init[1][1] if init[0] == 'param' and init[1][0] == 'fill' else None
    if fill is not None:
        return {'kind': 'fill', 'value': f32(fill)}, 0
    if b.rows * b.cols > limit:
        return {'kind': 'fill', 'value': 0.0}, 1
    return {'kind': 'values', 'values': init_values(init, b.rows, b.cols)}, 0


def to_workload(low: Lowered, policies: dict[str, GraphPolicy], materialize_limit=MATERIALIZE_LIMIT, extra_buffers=()):
    """Build the workload dict. `policies` maps graph name -> GraphPolicy; every graph used by a task must be present."""
    plan, bufs = low.plan, []
    used = {plan.buf(r.t) for t in low.tasks for r in (*t.reads, *t.writes)}
    used |= {plan.buf(v) for t in low.tasks for v in t.args.values() if isinstance(v, Tensor)}
    skipped = 0
    for name, b in plan.buffers.items():
        if name not in used:
            continue
        init, big = _init_entry(b, materialize_limit)
        skipped += big
        entry = {'name': name, 'rows': b.rows, 'cols': b.cols, 'init': init}
        if b.dtype != 'f32':
            entry['dtype'] = b.dtype
        bufs.append(entry)
    if skipped:
        log.warning('%d large buffer(s) emitted with zero init (> %d elements): fill them from the host (no init-RNG op yet)', skipped,
                    materialize_limit)
    weight_bufs = {n for n in plan.buffers if n.startswith('w.')}

    graphs: dict[str, list] = {}
    pub_count: dict[str, int] = {}
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
