"""CPU golden executor for emitted workloads (float64 by default, optional float32 rounding).

Runs every task through the device-op reference semantics in opset.py, in program order or in a RANDOM topological order of the
declared `after` edges. A workload whose edges are complete gives identical results in every order, so the random order is the
test that the dataflow analysis found every hazard. This is the oracle the GPU run is diffed against.
"""
from __future__ import annotations

import random
import struct

from . import log
from .ir import FrontendError
from .opset import Buf, bf16_round, op as device_op


def initial_values(init, n, dtype, data_files):
    k = init['kind']
    if k == 'fill':
        return [init['value']] * n
    if k == 'values':
        return list(init['values'])
    if k == 'seeded':
        from . import _runtime
        draws, bias, coef = _runtime.initgen.params(init['dist'], init['scale'])
        return _runtime.initgen.seeded_values(init['seed'], n, draws, bias, coef)
    if k == 'file':
        raw = data_files[init['path']]
        if dtype == 'bf16':
            return [struct.unpack('<f', struct.pack('<I', h << 16))[0] for h in struct.unpack(f'<{n}H', raw)]
        return list(struct.unpack(f'<{n}{"i" if dtype == "i32" else "f"}', raw))
    raise FrontendError(f'unknown init kind {k!r}')


class Memory:
    def __init__(self, spec, f32=False, data_files=None):
        self.bufs = {}
        for b in spec['buffers']:
            init, n = b['init'], b['rows'] * b['cols']
            vals = initial_values(init, n, b.get('dtype', 'f32'), data_files or {})
            bf = b.get('dtype', 'f32') == 'bf16'
            if bf:
                vals = [bf16_round(v) for v in vals]
            self.bufs[b['name']] = Buf(vals, b['rows'], b['cols'], f32, bf)
        self.f32 = f32

    def matrix(self, name):
        b = self.bufs[name]
        return [b.row(r) for r in range(b.rows)]


def execute(spec, order='program', seed=0, f32=False, mem: Memory | None = None, data_files=None) -> Memory:
    mem = mem or Memory(spec, f32, data_files)
    tasks = []
    for g in spec['graphs']:
        base = len(tasks)
        index = {t['id']: base + i for i, t in enumerate(g['tasks'])}
        for t in g['tasks']:
            tasks.append((t, [index[a] for a in t.get('after', [])]))
    deps = [list(d) for _, d in tasks]
    _version_edges(tasks, deps)
    succ = [[] for _ in tasks]
    indeg = [len(d) for d in deps]
    for i, ds in enumerate(deps):
        for d in ds:
            succ[d].append(i)
    ready = [i for i, d in enumerate(indeg) if d == 0]
    rng, done = random.Random(seed), 0
    while ready:
        i = ready.pop(rng.randrange(len(ready))) if order == 'random' else ready.pop(0)
        _run_task(tasks[i][0], mem)
        done += 1
        for j in succ[i]:
            indeg[j] -= 1
            if indeg[j] == 0:
                ready.append(j)
        if order == 'program':
            ready.sort()
    if done != len(tasks):
        raise FrontendError(f'workload has a dependency cycle ({len(tasks) - done} tasks never ready)')
    log.info('refvm: executed %d tasks (%s order, %s)', done, order, 'f32' if mem.f32 else 'f64')
    return mem


def _version_edges(tasks, deps):
    """Mirror of the compiler's weight-version semantics: a read of version n waits for the n-th publish of the variable, and an
    `exact` read must also finish before publish n+1 (so the buffer it reads is still intact)."""
    pubs: dict = {}
    for i, (t, _) in enumerate(tasks):
        if t.get('publishes'):
            pubs.setdefault(t['publishes'], []).append(i)
    for i, (t, _) in enumerate(tasks):
        for r in t.get('reads', []):
            p = pubs.get(r['var'], [])
            if r['version'] > 0:
                deps[i].append(p[r['version'] - 1])
            if r['policy'] == 'exact' and r['version'] < len(p) and p[r['version']] != i:
                deps[p[r['version']]].append(i)


def _run_task(t, mem):
    d = device_op(t['op'])
    if d.ref is None:
        raise FrontendError(f"task {t['id']}: op {t['op']!r} has no CPU reference (opaque engine op)")
    w = t.get('when')
    if w and (mem.bufs[w['buf']].a[w['index']] != 0) == bool(w['negate']):
        return                                   # predicate false: the op body does not run, outputs keep their contents
    args = t.get('args', {})
    bufs = {a: (mem.bufs[args[a]] if args[a] is not None else None) for a, ty in d.sig if ty in ('buf', 'buf?')}
    d.ref(bufs, {a: args[a] for a, ty in d.sig if ty not in ('buf', 'buf?')})
