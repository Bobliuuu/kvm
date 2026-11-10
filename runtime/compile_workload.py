"""KVM host compiler: declarative workload JSON -> ABI v2 bundle (typed task descriptors + weight versioning).

    python3 runtime/compile_workload.py workload.json out_dir [--base-versions weight=3,...] [--emit-golden]

The operator schema comes from include/kvm_ops*.def and the descriptor layout from include/kvm_task_abi.h
(runtime/abi.py parses them); semantics for range checks come from runtime/ref_ops.py. See FRAMEWORK.md.
"""
import argparse
import gc
import os
import json
import logging
import math
import shutil
import struct
import sys
import time
from array import array
from itertools import chain
from dataclasses import dataclass, field
from pathlib import Path

import abi
import initgen
import ref_ops

log = logging.getLogger('kvm.compiler')
INT32_MAX = 2**31 - 1
INT32_MIN = -2**31
SEED_MAX = 2**63 - 1
POLICIES = ('exact', 'at_least')
DTYPES = {'f32': 0, 'i32': 1, 'bf16': 2}   # KvmBuf.dtype (bf16 = raw 16-bit storage; the init values are rounded to bf16)
_warned_ops: set = set()
_F32 = struct.Struct('<f')
_I32 = struct.Struct('<i')
_MISSING = object()


def _bf16_value(x):
    """float32 value rounded to the nearest bfloat16 (ties to even); inf/nan unchanged."""
    if x != x or x in (math.inf, -math.inf):
        return x
    bits = struct.unpack('<I', struct.pack('<f', x))[0]
    return struct.unpack('<f', struct.pack('<I', (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000))[0]


class CompileError(ValueError):
    """The workload is invalid; the message names the graph/task/argument at fault."""


def _need(cond, msg):
    if not cond:
        raise CompileError(msg)


def _is_int(x):
    return type(x) is int


def _keys(obj, where, allowed, required=()):
    _need(isinstance(obj, dict), f'{where}: expected an object, got {type(obj).__name__}')
    extra = sorted(set(obj) - set(allowed))
    _need(not extra, f'{where}: unknown key(s) {extra}; allowed: {sorted(allowed)}')
    missing = sorted(set(required) - set(obj))
    _need(not missing, f'{where}: missing key(s) {missing}')


def _le_bytes(a: array) -> bytes:
    if sys.byteorder == 'big':
        a = array(a.typecode, a)
        a.byteswap()
    return a.tobytes()


# ---------------------------------------------------------------------------------------------- program model

@dataclass
class Read:
    var: str
    policy: str
    version: int
    waits: bool                  # version 0 always holds; any other wait is kept even if <= base, because an earlier
                                 # admitted batch may not have published up to base yet
    gate: str | None = None      # name of the gate task carrying this wait, if it is not the task's own slot
    wait_ver: int = -1           # version the runtime waits for (>= version when ordering after earlier batches)
    implicit: bool = False       # compiler-added, not declared by the workload

    def __post_init__(self):
        if self.wait_ver < 0:
            self.wait_ver = self.version


@dataclass
class Buffer:
    name: str
    index: int
    rows: int
    cols: int
    dtype: str
    init: dict                   # bundle-metadata form of the initial contents
    values: list | None = None   # explicit contents for kind=values

    @property
    def numel(self):
        return self.rows * self.cols

    def materialize(self, data_dir=None) -> list:
        """Initial contents as a flat row-major list (floats for f32, ints for i32). Can be huge: call on demand."""
        k, n = self.init['kind'], self.numel
        if k == 'fill':
            return [self.init['value']] * n
        if k == 'values':
            return list(self.values)
        if k == 'seeded':
            i = self.init
            return initgen.seeded_values(i['seed'], n, i['draws'], i['bias'], i['coef'])
        if data_dir is None:
            raise CompileError(f"buffer {self.name!r}: init file {self.init['path']!r} needs a data_dir to be read")
        raw = (Path(data_dir) / self.init['path']).read_bytes()
        if self.dtype == 'bf16':
            _need(len(raw) == 2 * n, f"buffer {self.name!r}: file {self.init['path']!r} has {len(raw)} bytes, expected {2 * n}")
            return [struct.unpack('<f', struct.pack('<I', h << 16))[0] for h in struct.unpack(f'<{n}H', raw)]
        _need(len(raw) == 4 * n, f"buffer {self.name!r}: file {self.init['path']!r} has {len(raw)} bytes, expected {4 * n}")
        return list(struct.unpack(f"<{n}{'f' if self.dtype == 'f32' else 'i'}", raw))


@dataclass
class Variable:
    name: str
    id: int
    base_version: int
    final_version: int = 0


@dataclass
class Task:
    name: str
    graph: str
    workload_id: int
    op: abi.OpSpec
    args: dict                   # arg name -> buffer name | None (absent) | int | float32-rounded float
    words: list                  # int32 words, in operator argument order
    priority: int = 0
    deadline_us: int = -1
    wait: tuple | None = None    # (var name, version)
    pub: tuple | None = None     # (var name, version)
    reads: tuple = ()            # every declared read (including those carried by gates)
    gate: bool = False


@dataclass
class Compiled:
    metadata: dict
    tasks: bytes
    successors: bytes
    args: bytes
    program: list[Task]
    buffers: list[Buffer]
    variables: list[Variable]
    succ_off: array              # CSR of the final graph: successors of task i = succ[succ_off[i]:succ_off[i+1]]
    succ: array
    data_dir: Path | None = None

    def successors_of(self, i):
        return self.succ[self.succ_off[i]:self.succ_off[i + 1]]


# ---------------------------------------------------------------------------------------------- parsing

def _f32_value(x, where, allow_nonfinite):
    _need(isinstance(x, (int, float)) and not isinstance(x, bool), f'{where}: expected a number, got {x!r}')
    try:
        v = float(x)
    except OverflowError:
        raise CompileError(f'{where}: {x!r} is not representable as float32') from None
    if not math.isfinite(v):
        _need(allow_nonfinite, f'{where}: non-finite value {v} rejected (pass allow_nonfinite to permit)')
        return v
    _need(abs(v) <= ref_ops.F32_MAX, f'{where}: {v!r} overflows float32')
    return ref_ops.f32(v)


def _i32_value(x, where):
    _need(_is_int(x) or (isinstance(x, float) and x.is_integer()), f'{where}: i32 buffers hold integers, got {x!r}')
    _need(INT32_MIN <= x <= INT32_MAX, f'{where}: {x} overflows int32')
    return int(x)


def _parse_variables(names, base_versions):
    _need(isinstance(names, list) and all(isinstance(n, str) and n for n in names),
          "workload: 'variables' must be a list of non-empty strings")
    _need(len(set(names)) == len(names), f'workload: duplicate variable names in {names}')
    base_versions = dict(base_versions or {})
    unknown = sorted(set(base_versions) - set(names))
    _need(not unknown, f'base_versions names undeclared variable(s) {unknown}; declared: {names}')
    out = {}
    for i, n in enumerate(names):
        base = base_versions.get(n, 0)
        _need(_is_int(base) and 0 <= base <= INT32_MAX, f'base version of {n!r} must be an int32 >= 0, got {base!r}')
        out[n] = Variable(n, i, base, base)
    return out


def _parse_init(where, init, rows, cols, dtype, allow_nonfinite, data_dir):
    """Returns (metadata init dict, explicit values or None)."""
    _need(isinstance(init, dict) and 'kind' in init, f"{where}: init must be an object with a 'kind'")
    kind, n = init['kind'], rows * cols
    conv = (lambda x, w: _f32_value(x, w, allow_nonfinite)) if dtype == 'f32' else _i32_value
    if dtype == 'bf16':
        conv = lambda x, w: _bf16_value(_f32_value(x, w, allow_nonfinite))
    if kind == 'fill':
        _keys(init, f'{where} init', {'kind', 'value'}, {'kind', 'value'})
        v = conv(init['value'], f'{where} init.value')
        return {'kind': 'fill', 'value': v}, None
    if kind == 'values':
        _keys(init, f'{where} init', {'kind', 'values'}, {'kind', 'values'})
        raw = init['values']
        _need(isinstance(raw, list) and len(raw) == n, f"{where}: init.values must be a list of {n} numbers ({rows}x{cols})")
        values = [conv(x, f'{where} init.values[{j}]') for j, x in enumerate(raw)]
        return {'kind': 'values', 'values': values}, values
    if kind == 'seeded':
        _keys(init, f'{where} init', {'kind', 'seed', 'scale', 'dist'}, {'kind', 'seed', 'scale'})
        _need(dtype == 'f32', f'{where}: seeded init needs dtype f32 (bf16 buffers take fill / values / file)')
        seed, dist = init['seed'], init.get('dist', 'normal')
        _need(_is_int(seed) and 0 <= seed <= SEED_MAX, f'{where}: seed must be an int in [0, 2^63), got {seed!r}')
        scale = _f32_value(init['scale'], f'{where} init.scale', False)
        _need(scale >= 0, f'{where}: init.scale must be >= 0, got {scale}')
        _need(dist in initgen.DISTS, f'{where}: init.dist must be one of {initgen.DISTS}, got {dist!r}')
        draws, bias, coef = initgen.params(dist, scale)
        return {'kind': 'seeded', 'seed': seed, 'scale': scale, 'dist': dist, 'draws': draws, 'bias': bias, 'coef': coef}, None
    if kind == 'file':
        _keys(init, f'{where} init', {'kind', 'path'}, {'kind', 'path'})
        path = init['path']
        _need(isinstance(path, str) and path, f'{where}: init.path must be a non-empty string')
        p = Path(path)
        _need(not p.is_absolute() and '..' not in p.parts and '\\' not in path,
              f'{where}: init.path {path!r} must be relative to the bundle without ..')
        if data_dir is not None:
            f = Path(data_dir) / p
            _need(f.is_file(), f'{where}: init file {str(f)!r} not found')
            _need(f.stat().st_size == (2 if dtype == 'bf16' else 4) * n, f'{where}: init file {path!r} has {f.stat().st_size} bytes, expected {(2 if dtype == "bf16" else 4) * n} '
                                             f'({rows}x{cols} little-endian {dtype})')
        return {'kind': 'file', 'path': p.as_posix()}, None
    raise CompileError(f"{where}: init.kind must be one of 'fill', 'values', 'seeded', 'file', got {kind!r}")


def _parse_buffers(decls, allow_nonfinite, data_dir):
    _need(isinstance(decls, list), "workload: 'buffers' must be a list")
    out = {}
    for i, d in enumerate(decls):
        _keys(d, f'buffers[{i}]', {'name', 'rows', 'cols', 'dtype', 'init'}, {'name'})
        name = d['name']
        _need(isinstance(name, str) and name, f'buffers[{i}]: name must be a non-empty string')
        _need(name not in out, f'buffers[{i}]: duplicate buffer {name!r}')
        where = f'buffer {name!r}'
        rows, cols, dtype = d.get('rows', 1), d.get('cols', 1), d.get('dtype', 'f32')
        for label, dim in (('rows', rows), ('cols', cols)):
            _need(_is_int(dim) and dim >= 1, f'{where}: {label} must be an int >= 1, got {dim!r}')
        _need(rows * cols <= INT32_MAX, f'{where}: {rows}x{cols} elements overflow int32')
        _need(dtype in DTYPES, f'{where}: dtype must be one of {sorted(DTYPES)}, got {dtype!r}')
        meta, values = _parse_init(where, d.get('init', {'kind': 'fill', 'value': 0}), rows, cols, dtype, allow_nonfinite, data_dir)
        out[name] = Buffer(name, len(out), rows, cols, dtype, meta, values)
    return out


class _ArgPlan:
    """Per-operator argument decoding plan, built once: the per-task loop is the compiler's hot path."""
    __slots__ = ('op', 'ref', 'items', 'n', 'sig', 'dtype_checks')

    def __init__(self, op):
        self.op, self.ref = op, ref_ops.OPS.get(op.name)
        mins = self.ref.mins if self.ref else {}
        self.items = tuple((a.name, a.type, a.optional, mins.get(a.name)) for a in op.args)
        self.n = len(op.args)
        self.sig = ', '.join(f'{a.name}:{a.type}{"?" if a.optional else ""}' for a in op.args)
        self.dtype_checks = ()
        if self.ref and not self.ref.opaque:
            self.dtype_checks = tuple((a.name, 'i32' if a.name in self.ref.ints else 'f32')
                                      for a in op.args if a.type == 'buf' and a.name not in self.ref.anydt)


def _arg_error(where, plan, raw):
    """Raise the precise CompileError for args that failed the fast path (called only on the failure path)."""
    names = [i[0] for i in plan.items]
    missing = [n for n in names if n not in raw]
    extra = sorted(set(raw) - set(names))
    _need(not missing, f"{where}: op '{plan.op.name}' missing arg(s) {missing}; signature ({plan.sig})")
    _need(not extra, f"{where}: op '{plan.op.name}' has no arg(s) {extra}; takes {names}")


def _parse_args(where, plan, raw, buffers, allow_nonfinite):
    """Resolve a JSON args object to (values by arg name, int32 words) against the operator schema."""
    if raw.__class__ is not dict:
        raise CompileError(f'{where}: args must be an object, got {type(raw).__name__}')
    if len(raw) != plan.n:
        _arg_error(where, plan, raw)
    values, words = {}, []
    for name, typ, optional, lo in plan.items:
        v = raw.get(name, _MISSING)
        if v is _MISSING:
            _arg_error(where, plan, raw)
        if typ == 'buf':
            if v is None:
                if not optional:
                    raise CompileError(f"{where} arg '{name}': null is only allowed for optional buffers (buf?) and "
                                       f"'{name}' of op '{plan.op.name}' is required")
                word = abi.ABSENT_BUF
            else:
                buf = buffers.get(v) if v.__class__ is str else None
                if buf is None:
                    if v.__class__ is not str:
                        raise CompileError(f"{where} arg '{name}': expected a buffer name (string), got {v!r}")
                    raise CompileError(f"{where} arg '{name}': unknown buffer {v!r}; declared: {sorted(buffers)[:20]}")
                word = buf.index
        elif typ == 'i32':
            if v.__class__ is not int:
                raise CompileError(f"{where} arg '{name}': expected an integer, got {v!r}")
            if not INT32_MIN <= v <= INT32_MAX:
                raise CompileError(f"{where} arg '{name}': {v} overflows int32")
            if lo is not None and v < lo:
                raise CompileError(f"{where} arg '{name}': {v} below the minimum {lo} for {plan.op.name}")
            word = v
        else:
            if v.__class__ is not float and v.__class__ is not int:
                raise CompileError(f"{where} arg '{name}': expected a number, got {v!r}")
            try:
                packed = _F32.pack(v)
            except OverflowError:
                raise CompileError(f"{where} arg '{name}': {v!r} overflows float32") from None
            word = _I32.unpack(packed)[0]
            v = _F32.unpack(packed)[0]
            if (word >> 23) & 0xFF == 0xFF and not allow_nonfinite:
                raise CompileError(f"{where} arg '{name}': non-finite value {v} rejected (pass allow_nonfinite to permit)")
        values[name] = v
        words.append(word)
    return values, words


def _check_semantics(where, plan, values, buffers):
    """Dtype and extent checks of the buffer arguments against the operator's reference semantics."""
    ref, op = plan.ref, plan.op
    if ref is None:
        if op.name not in _warned_ops:
            _warned_ops.add(op.name)
            log.warning('op %r has no reference semantics: dtypes and extents are not checked for it', op.name)
        return
    for arg, want in plan.dtype_checks:
        name = values[arg]
        if name is not None and buffers[name].dtype != want:
            raise CompileError(f"{where} arg '{arg}': op '{op.name}' needs a {want} buffer, '{name}' is {buffers[name].dtype}")
    try:
        extents = ref.extents(values)
    except ValueError as e:
        raise CompileError(f'{where}: {e}') from None
    for arg, need in extents.items():
        name = values.get(arg)
        if name is None:
            continue
        buf = buffers[name]
        if need.__class__ is tuple:
            ok = buf.rows >= need[0] and buf.cols >= need[1]
            what = f'{need[0]}x{need[1]}'
        else:
            ok = buf.rows * buf.cols >= need
            what = f'{need} element(s)'
        if not ok:
            raise CompileError(f"{where} arg '{arg}': op '{op.name}' needs {what} but buffer '{name}' is {buf.rows}x{buf.cols}")


@dataclass
class _Decl:
    """A declared task before gates and edges are resolved."""
    task: Task
    after: list = field(default_factory=list)      # declared-task indices
    publishes: str | None = None
    reads: list = field(default_factory=list)      # (var, policy, version)


_TASK_KEYS = {'id', 'op', 'args', 'after', 'priority', 'deadline_us', 'publishes', 'reads'}
_READ_KEYS = {'var', 'policy', 'version'}


def _parse_tasks(graphs, ops, buffers, variables, allow_nonfinite):
    _need(isinstance(graphs, list) and graphs, "workload: 'graphs' must be a nonempty list")
    decls, graph_names, plans = [], set(), {}
    for gi, graph in enumerate(graphs):
        _keys(graph, f'graphs[{gi}]', {'name', 'tasks'}, {'name', 'tasks'})
        gname = graph['name']
        _need(isinstance(gname, str) and gname and '/' not in gname, f'graphs[{gi}]: name must be a non-empty string without "/"')
        _need(gname not in graph_names, f'graph {gname!r}: duplicate graph name')
        graph_names.add(gname)
        tasks = graph['tasks']
        _need(isinstance(tasks, list) and tasks, f'graph {gname!r}: tasks must be a nonempty list')
        base, index = len(decls), {}
        for ti, t in enumerate(tasks):
            if t.__class__ is not dict or not _TASK_KEYS >= t.keys() or 'id' not in t or 'op' not in t:
                _keys(t, f'graph {gname!r} tasks[{ti}]', _TASK_KEYS, ('id', 'op'))
            tid = t['id']
            if tid.__class__ is not str or not tid:
                raise CompileError(f'graph {gname!r} tasks[{ti}]: id must be a non-empty string')
            if tid in index:
                raise CompileError(f'graph {gname!r}: duplicate task id {tid!r}')
            index[tid] = base + ti
        for ti, t in enumerate(tasks):
            where = f"{gname}/{t['id']}"
            plan = plans.get(t['op'])
            if plan is None:
                op = ops.get(t['op'])
                _need(op is not None, f"{where}: unknown op {t['op']!r}; known: {sorted(ops)}")
                plan = plans[t['op']] = _ArgPlan(op)
            values, words = _parse_args(where, plan, t.get('args', {}), buffers, allow_nonfinite)
            _check_semantics(where, plan, values, buffers)
            prio, dl = t.get('priority', 0), t.get('deadline_us', -1)
            if prio.__class__ is not int or not INT32_MIN <= prio <= INT32_MAX:
                raise CompileError(f'{where}: priority must be an int32, got {prio!r}')
            if dl.__class__ is not int or not -1 <= dl <= INT32_MAX:
                raise CompileError(f'{where}: deadline_us must be -1 or an int32 >= 0, got {dl!r}')
            after = t.get('after', ())
            try:
                deps = [index[d] for d in after]
            except (KeyError, TypeError):
                _need(isinstance(after, (list, tuple)), f'{where}: after must be a list of task ids')
                bad = next(d for d in after if d not in index)
                raise CompileError(f"{where}: after names unknown task {bad!r} in graph {gname!r}") from None
            if deps:
                if len(set(deps)) != len(deps):
                    raise CompileError(f'{where}: duplicate dependency in after {list(after)[:8]}')
                if base + ti in deps:
                    raise CompileError(f'{where}: depends on itself')
            pub = t.get('publishes')
            if pub is not None and pub not in variables:
                raise CompileError(f"{where}: publishes undeclared variable {pub!r}; declared: {list(variables)}")
            reads = []
            raw_reads = t.get('reads', ())
            if raw_reads:
                _need(isinstance(raw_reads, (list, tuple)), f'{where}: reads must be a list')
                seen = set()
                for ri, r in enumerate(raw_reads):
                    _keys(r, f'{where} reads[{ri}]', _READ_KEYS, _READ_KEYS)
                    _need(r['var'] in variables, f"{where} reads[{ri}]: undeclared variable {r['var']!r}; declared: {list(variables)}")
                    _need(r['policy'] in POLICIES, f"{where} reads[{ri}]: policy must be one of {POLICIES}, got {r['policy']!r}")
                    _need(_is_int(r['version']) and 0 <= r['version'] <= INT32_MAX,
                          f"{where} reads[{ri}]: version must be an int32 >= 0, got {r['version']!r}")
                    _need(r['var'] not in seen, f"{where}: reads variable {r['var']!r} twice")
                    seen.add(r['var'])
                    reads.append((r['var'], r['policy'], r['version']))
            decls.append(_Decl(Task(where, gname, gi, plan.op, values, words, prio, dl), deps, pub, reads))
    return decls


# ---------------------------------------------------------------------------------------------- versioning

def _plan_versions(decls, variables):
    """Assign publish versions and Read records; return the exact-snapshot edges [(reader, next publisher)].

    Purely structural: whether the resulting graph is acyclic and deadlock free is checked on the built graph.
    """
    publishers = {v: [] for v in variables}                  # var -> [declared idx] in program order
    for i, d in enumerate(decls):
        if d.publishes:
            var = variables[d.publishes]
            publishers[d.publishes].append(i)
            var.final_version += 1
            _need(var.final_version <= INT32_MAX, f'{d.task.name}: version of {var.name!r} overflows int32')
            d.task.pub = (var.name, var.final_version)
    by_version = {v: {variables[v].base_version + k + 1: i for k, i in enumerate(idxs)} for v, idxs in publishers.items()}
    exact_edges = []
    for i, d in enumerate(decls):
        t, reads = d.task, []
        for var_name, policy, n in d.reads:
            var = variables[var_name]
            pubs = by_version[var_name]
            where = f"{t.name}: {policy} read of {var_name!r} at version {n}"
            if policy == 'exact':
                _need(n >= var.base_version,
                      f"{where} is stale: version {n + 1} is already published or admitted (base_version={var.base_version}), "
                      f'so version {n} can no longer be observed exactly; use at_least or read version {var.base_version}')
            if n > var.base_version:
                _need(n in pubs, f"{where} can never be satisfied: this batch publishes {var_name!r} only up to "
                                 f"version {var.final_version} (base_version={var.base_version})")
            nxt = pubs.get(n + 1)
            if policy == 'exact' and nxt is not None and nxt != i:   # nxt == i: reads N, publishes N+1, isolated
                exact_edges.append((i, nxt))                         # write-after-read: next update waits for this reader
            reads.append(Read(var_name, policy, n, waits=n > 0))
        t.reads = tuple(reads)
    for var_name, idxs in publishers.items():
        base = variables[var_name].base_version
        if idxs and base > 0:     # total order across admitted batches: the first publisher waits for the live head
            t = decls[idxs[0]].task
            own = next((r for r in t.reads if r.var == var_name), None)
            if own:
                own.wait_ver = max(own.wait_ver, base)
            else:
                t.reads += (Read(var_name, 'at_least', base, waits=True, implicit=True),)
    return publishers, by_version, exact_edges


def _lay_out(decls, ops, exact_edges):
    """Insert noop gate tasks for waiting reads beyond the first; return final tasks, final index of each declared task
    and the predecessor lists of the final graph (gates first, then declared edges, then exact-snapshot edges)."""
    noop = ops.get('noop')
    _need(noop is not None, "the operator registry defines no 'noop', needed for version gates")
    tasks, final, gates_of = [], [0] * len(decls), {}
    for i, d in enumerate(decls):
        t = d.task
        waiting = [r for r in t.reads if r.waits]
        for k, r in enumerate(waiting[1:], 1):
            gate = Task(f'__gate/{t.name}/{k}', t.graph, t.workload_id, noop, {}, [], t.priority, t.deadline_us,
                        wait=(r.var, r.wait_ver), gate=True)
            r.gate = gate.name
            gates_of.setdefault(i, []).append(len(tasks))
            tasks.append(gate)
        final[i] = len(tasks)
        if waiting:
            t.wait = (waiting[0].var, waiting[0].wait_ver)
        tasks.append(t)
    preds = [[] for _ in tasks]
    for i, d in enumerate(decls):
        preds[final[i]] = gates_of.get(i, []) + [final[j] for j in d.after]
    for reader, nxt in exact_edges:
        if final[reader] not in preds[final[nxt]]:
            preds[final[nxt]].append(final[reader])
    return tasks, final, preds


# ---------------------------------------------------------------------------------------------- graph analysis

class _Graph:
    """CSR successor graph with a topological position per node (None when the graph has a cycle)."""

    def __init__(self, preds):
        n = len(preds)
        outs = [[] for _ in range(n)]
        for i, ps in enumerate(preds):             # ascending destination => each successor list comes out sorted
            for p in ps:
                outs[p].append(i)
        off = array('i', [0])
        total = 0
        for o in outs:
            total += len(o)
            off.append(total)
        self.n, self.off, self.succ = n, off, array('i', chain.from_iterable(outs))
        self.indegree = array('i', map(len, preds))
        self.pos = self._topo()

    def _topo(self):
        off, succ, rem = self.off, self.succ, array('i', self.indegree)
        order = [i for i in range(self.n) if rem[i] == 0]
        head = 0
        while head < len(order):
            i = order[head]
            head += 1
            for k in range(off[i], off[i + 1]):
                j = succ[k]
                rem[j] -= 1
                if rem[j] == 0:
                    order.append(j)
        if len(order) != self.n:
            self.cyclic = [i for i in range(self.n) if rem[i] > 0]
            return None
        self.cyclic = []
        pos = array('i', bytes(4 * self.n))
        for p, i in enumerate(order):
            pos[i] = p
        return pos

    def reaches(self, a, b, skip=None):
        """True if a path a -> ... -> b exists (a == b counts). `skip` = one edge (u, v) to ignore."""
        if a == b:
            return True
        if self.pos is not None and self.pos[a] > self.pos[b]:
            return False
        off, succ, pos = self.off, self.succ, self.pos
        stack, seen = [a], {a}
        while stack:
            i = stack.pop()
            for k in range(off[i], off[i + 1]):
                j = succ[k]
                if j in seen or (skip is not None and (i, j) == skip):
                    continue
                if j == b:
                    return True
                if pos is None or pos[j] < pos[b]:
                    seen.add(j)
                    stack.append(j)
        return False


def _explain_cycle(tasks, preds, exact_edges, final, decls):
    """Raise the most specific CompileError for a cyclic final graph."""
    exact = {(final[r], final[n]) for r, n in exact_edges}
    user = [[p for p in ps if (p, i) not in exact] for i, ps in enumerate(preds)]
    ug = _Graph(user)
    if ug.pos is None:
        names = [tasks[i].name for i in ug.cyclic]
        raise CompileError(f'dependency cycle among {len(names)} task(s): {names[:8]}{"..." if len(names) > 8 else ""}')
    full = _Graph(preds)
    for r, n in exact_edges:
        if ug.reaches(final[n], final[r]):
            raise CompileError(
                f"exact read in {decls[r].task.name} can never hold: it is ordered after {decls[n].task.name}, which publishes "
                f"the next version of the variable it reads (the write-after-read edge would close a cycle)")
    names = [tasks[i].name for i in full.cyclic]
    raise CompileError(f'dependency cycle among {len(names)} task(s): {names[:8]}')


def _check_versions(graph, decls, final, variables, publishers, by_version):
    for var_name, idxs in publishers.items():
        for a, b in zip(idxs, idxs[1:]):
            _need(graph.reaches(final[a], final[b]),
                  f"unordered publishers of {var_name!r}: no dependency path from {decls[a].task.name} "
                  f"(version {decls[a].task.pub[1]}) to {decls[b].task.name} (version {decls[b].task.pub[1]}); "
                  f"add an 'after' chain so their order is defined")
    for i, d in enumerate(decls):                           # a reader must not precede the publisher it waits for
        for r in d.task.reads:
            if r.wait_ver > variables[r.var].base_version:
                p = by_version[r.var][r.wait_ver]
                _need(not graph.reaches(final[i], final[p]),
                      f"{d.task.name}: waits for {r.var!r} >= {r.wait_ver}, published by {decls[p].task.name}, "
                      f"which is ordered after it (deadlock)")


# ---------------------------------------------------------------------------------------------- compile

def compile_spec(spec, base_versions=None, allow_nonfinite=False, data_dir=None, ops=None) -> Compiled:
    """Compile a workload dict. `base_versions` maps variable -> live version for a later-admitted workload.
    `data_dir` is where `file` initial values are looked up (validated when given).
    `ops` overrides the operator registry (default: every registry file in include/).
    The cyclic GC is paused meanwhile: millions of acyclic containers make generational collection dominate otherwise."""
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        return _compile_spec(spec, base_versions, allow_nonfinite, data_dir, ops)
    finally:
        if was_enabled:
            gc.enable()


# Measured mean task times (us) of the 162M GPT step on an A100; unknown ops get the default.
_OP_US = {'matmul_mma_bf16': 130, 'matmul_tile': 160, 'attn_bwd_dkv': 205, 'attn_bwd_dq': 130, 'attn_fwd': 90, 'round_bf16': 110, 'adamw': 600,
          'colsum': 950, 'rope_bwd': 350, 'rope_fwd': 330, 'gelu_bwd': 575, 'norm_bwd_params': 2500, 'layernorm_bwd_dx': 235}


def _critical_priorities(tasks, graph):
    """KVM_CRIT_PRIO=1: priority = quantised longest remaining path (task cost + max successor rank), 0..30, so the ready queue (strict
    priority across its 32 buckets, FIFO within one) runs the critical chain first and the tail is not starved. A declared priority
    is kept when it is higher. Returns None when disabled."""
    if os.environ.get('KVM_CRIT_PRIO', '') in ('', '0') or graph.pos is None:
        return None
    n, off, succ = graph.n, graph.off, graph.succ
    order = sorted(range(n), key=lambda i: -graph.pos[i])
    bl = [0.0] * n
    for i in order:
        c = _OP_US.get(tasks[i].op.name, 100)
        m = 0.0
        for k in range(off[i], off[i + 1]):
            if bl[succ[k]] > m:
                m = bl[succ[k]]
        bl[i] = c + m
    top = max(bl) or 1.0
    q = int(os.environ['KVM_CRIT_PRIO']) if os.environ['KVM_CRIT_PRIO'].isdigit() and int(os.environ['KVM_CRIT_PRIO']) > 1 else 30
    return [max(tasks[i].priority, min(30, int(q * bl[i] / top))) for i in range(n)]


def _compile_spec(spec, base_versions, allow_nonfinite, data_dir, ops) -> Compiled:
    t0 = time.perf_counter()
    _keys(spec, 'workload', {'variables', 'buffers', 'graphs'}, {'graphs'})
    ops, task_abi = ops or abi.load_ops(), abi.load_task_abi()
    variables = _parse_variables(spec.get('variables', []), base_versions)
    buffers = _parse_buffers(spec.get('buffers', []), allow_nonfinite, data_dir)
    decls = _parse_tasks(spec['graphs'], ops, buffers, variables, allow_nonfinite)
    t1 = time.perf_counter()
    publishers, by_version, exact_edges = _plan_versions(decls, variables)
    tasks, final, preds = _lay_out(decls, ops, exact_edges)
    graph = _Graph(preds)
    if graph.pos is None:
        _explain_cycle(tasks, preds, exact_edges, final, decls)
    _check_versions(graph, decls, final, variables, publishers, by_version)
    t2 = time.perf_counter()

    prio = _critical_priorities(tasks, graph)
    var_id = {n: v.id for n, v in variables.items()}
    desc, flat_args, nargs = [], array('i'), 0
    pack, fields = task_abi.layout.pack, task_abi.fields
    no_var = task_abi.no_var
    for i, t in enumerate(tasks):
        wv, wn = (var_id[t.wait[0]], t.wait[1]) if t.wait else (no_var, 0)
        pv, pn = (var_id[t.pub[0]], t.pub[1]) if t.pub else (no_var, 0)
        values = dict(op_id=t.op.id, succ_off=graph.off[i], succ_cnt=graph.off[i + 1] - graph.off[i], indegree=graph.indegree[i],
                      priority=prio[i] if prio else t.priority, deadline_us=t.deadline_us, args_off=nargs, args_words=len(t.words),
                      wait_var=wv, wait_ver=wn, pub_var=pv, pub_ver=pn, workload_id=t.workload_id)
        desc.append(pack(*[values.get(f, 0) for f in fields]))
        flat_args.extend(t.words)
        nargs += len(t.words)
    _need(nargs <= INT32_MAX and len(graph.succ) <= INT32_MAX, 'workload too large for int32 offsets')

    index_of = {t.name: i for i, t in enumerate(tasks)}
    metadata = {
        'abi_version': task_abi.version, 'task_count': len(tasks), 'edge_count': len(graph.succ),
        'args_words': nargs, 'task_names': [t.name for t in tasks],
        'initial_ready': [i for i in range(len(tasks)) if graph.indegree[i] == 0],
        'operator_ids': {n: o.id for n, o in ops.items()},
        'buffers': [{'name': b.name, 'index': b.index, 'rows': b.rows, 'cols': b.cols, 'dtype': DTYPES[b.dtype], 'init': b.init}
                    for b in buffers.values()],
        'variables': [{'name': v.name, 'id': v.id, 'base_version': v.base_version} for v in variables.values()],
        'publishers': [{'task': i, 'name': t.name, 'var': var_id[t.pub[0]], 'version': t.pub[1]}
                       for i, t in enumerate(tasks) if t.pub],
        'reads': [{'task': i, 'name': t.name, 'var': var_id[r.var], 'policy': r.policy,
                   'version': r.version, 'observed': r.version if r.policy == 'exact' else None,
                   'gate': index_of[r.gate] if r.gate else None}
                  for i, t in enumerate(tasks) if not t.gate for r in t.reads if not r.implicit],
        'final_versions': {v.name: v.final_version for v in variables.values()},
    }
    t3 = time.perf_counter()
    log.info('compiled %d tasks (%d gates), %d edges, %d buffers, %d variables in %.2fs '
             '(parse %.2fs, graph+versions %.2fs, pack %.2fs)', len(tasks), sum(t.gate for t in tasks), len(graph.succ),
             len(buffers), len(variables), t3 - t0, t1 - t0, t2 - t1, t3 - t2)
    return Compiled(metadata, b''.join(desc), _le_bytes(graph.succ), _le_bytes(flat_args), tasks, list(buffers.values()),
                    list(variables.values()), graph.off, graph.succ, Path(data_dir) if data_dir else None)


def write_bundle(compiled: Compiled, out_dir) -> Path:
    """Write the bundle directory; `file` initial values are copied in under their relative path."""
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    files = [b for b in compiled.buffers if b.init['kind'] == 'file']
    if files:
        _need(compiled.data_dir is not None, f"buffers {[b.name for b in files][:4]} use file init: compile with a data_dir")
        for b in files:
            src, dst = compiled.data_dir / b.init['path'], dest / b.init['path']
            if src.resolve() != dst.resolve():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
    (dest / 'tasks.bin').write_bytes(compiled.tasks)
    (dest / 'successors.bin').write_bytes(compiled.successors)
    (dest / 'args.bin').write_bytes(compiled.args)
    with open(dest / 'metadata.json', 'w') as f:
        json.dump(compiled.metadata, f, indent=2)
        f.write('\n')
    log.info('wrote bundle to %s', dest)
    return dest


def parse_base_versions(text):
    out = {}
    for item in filter(None, (text or '').split(',')):
        name, sep, ver = item.partition('=')
        if not sep or not ver.lstrip('-').isdigit():
            raise CompileError(f"--base-versions entry {item!r} must look like name=INT")
        _need(name not in out, f'--base-versions repeats {name!r}')
        out[name] = int(ver)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('workload')
    p.add_argument('out_dir')
    p.add_argument('--base-versions', default='', help='live versions for a later-admitted workload: var=N,var=N')
    p.add_argument('--emit-golden', action='store_true', help='run the CPU reference and write golden.json')
    p.add_argument('--allow-nonfinite', action='store_true', help='accept NaN/inf float32 literals')
    p.add_argument('--allow-opaque', action='store_true', help='golden run: execute ops without CPU semantics as noops')
    p.add_argument('--golden-timeout', type=float, default=60, help='seconds the pure-Python golden run may take (large bundles need more)')
    p.add_argument('--data-dir', help='where `file` initial values live (default: the workload file directory)')
    p.add_argument('--log-level', default='INFO')
    args = p.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    data_dir = Path(args.data_dir) if args.data_dir else Path(args.workload).resolve().parent
    try:
        spec = json.loads(Path(args.workload).read_text())
        compiled = compile_spec(spec, parse_base_versions(args.base_versions), args.allow_nonfinite, data_dir)
        write_bundle(compiled, args.out_dir)
        if args.emit_golden:
            import workload_vm
            golden = workload_vm.run_golden(compiled, timeout=args.golden_timeout, allow_opaque=args.allow_opaque)
            with open(Path(args.out_dir) / 'golden.json', 'w') as f:
                json.dump(golden, f, indent=2)
                f.write('\n')
            log.info('wrote golden.json')
    except (CompileError, OSError, json.JSONDecodeError) as e:
        log.error('%s', e)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
