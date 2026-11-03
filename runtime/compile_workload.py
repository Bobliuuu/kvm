"""KVM host compiler: declarative workload JSON -> ABI v2 bundle (typed task descriptors + weight versioning).

    python3 runtime/compile_workload.py workload.json out_dir [--base-versions weight=3,...] [--emit-golden]

The operator schema comes from include/kvm_ops.def and the descriptor layout from include/kvm_task_abi.h
(runtime/abi.py parses both); semantics for range checks come from runtime/ref_ops.py. See FRAMEWORK.md.
"""
import argparse
import json
import logging
import math
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path

import abi
import ref_ops

log = logging.getLogger('kvm.compiler')
INT32_MAX = 2**31 - 1
INT32_MIN = -2**31
POLICIES = ('exact', 'at_least')


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


# ---------------------------------------------------------------------------------------------- program model

@dataclass
class Read:
    var: str
    policy: str
    version: int
    waits: bool                  # False when already satisfied at admission (version <= base)
    gate: str | None = None      # name of the gate task carrying this wait, if it is not the task's own slot


@dataclass
class Buffer:
    name: str
    index: int
    rows: int
    cols: int
    init: dict                   # metadata form: {kind: fill, value} | {kind: values, values}
    values: list                 # float32-rounded initial contents, row-major


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
    args: dict                   # arg name -> buffer name | int | float32-rounded float
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
    successor_lists: list[list[int]]


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


def _parse_buffers(decls, allow_nonfinite):
    _need(isinstance(decls, list), "workload: 'buffers' must be a list")
    out = {}
    for i, d in enumerate(decls):
        _keys(d, f'buffers[{i}]', {'name', 'rows', 'cols', 'init'}, {'name'})
        name = d['name']
        _need(isinstance(name, str) and name, f'buffers[{i}]: name must be a non-empty string')
        _need(name not in out, f'buffers[{i}]: duplicate buffer {name!r}')
        where = f'buffer {name!r}'
        rows, cols = d.get('rows', 1), d.get('cols', 1)
        for label, dim in (('rows', rows), ('cols', cols)):
            _need(_is_int(dim) and dim >= 1, f'{where}: {label} must be an int >= 1, got {dim!r}')
        _need(rows * cols <= INT32_MAX, f'{where}: {rows}x{cols} elements overflow int32')
        init = d.get('init', {'kind': 'fill', 'value': 0})
        _keys(init, f'{where} init', {'kind', 'value', 'values'}, {'kind'})
        if init['kind'] == 'fill':
            _keys(init, f'{where} init', {'kind', 'value'}, {'kind', 'value'})
            v = _f32_value(init['value'], f'{where} init.value', allow_nonfinite)
            values, meta = [v] * (rows * cols), {'kind': 'fill', 'value': v}
        elif init['kind'] == 'values':
            _keys(init, f'{where} init', {'kind', 'values'}, {'kind', 'values'})
            raw = init['values']
            _need(isinstance(raw, list) and len(raw) == rows * cols,
                  f"{where}: init.values must be a list of {rows * cols} numbers ({rows}x{cols})")
            values = [_f32_value(x, f'{where} init.values[{j}]', allow_nonfinite) for j, x in enumerate(raw)]
            meta = {'kind': 'values', 'values': values}
        else:
            raise CompileError(f"{where}: init.kind must be 'fill' or 'values', got {init['kind']!r}")
        out[name] = Buffer(name, len(out), rows, cols, meta, values)
    return out


def _parse_args(where, op, raw, buffers, allow_nonfinite):
    """Resolve a JSON args object to (values by arg name, int32 words) against the operator schema."""
    _need(isinstance(raw, dict), f'{where}: args must be an object, got {type(raw).__name__}')
    names = [a.name for a in op.args]
    missing = [n for n in names if n not in raw]
    extra = sorted(set(raw) - set(names))
    _need(not missing, f"{where}: op '{op.name}' missing arg(s) {missing}; signature "
                       f"({', '.join(f'{a.name}:{a.type}' for a in op.args)})")
    _need(not extra, f"{where}: op '{op.name}' has no arg(s) {extra}; takes {names}")
    mins = getattr(ref_ops.OPS.get(op.name), 'mins', {})
    values, words = {}, []
    for a in op.args:
        v, loc = raw[a.name], f"{where} arg '{a.name}'"
        if a.type == 'buf':
            _need(isinstance(v, str), f'{loc}: expected a buffer name (string), got {v!r}')
            _need(v in buffers, f"{loc}: unknown buffer {v!r}; declared: {sorted(buffers)}")
            values[a.name], word = v, buffers[v].index
        elif a.type == 'i32':
            _need(_is_int(v), f'{loc}: expected an integer, got {v!r}')
            _need(INT32_MIN <= v <= INT32_MAX, f'{loc}: {v} overflows int32')
            _need(v >= mins.get(a.name, INT32_MIN), f'{loc}: {v} below the minimum {mins[a.name]} for {op.name}')
            values[a.name], word = v, v
        else:
            v = _f32_value(v, loc, allow_nonfinite)
            values[a.name] = v
            word = struct.unpack('<i', struct.pack('<f', v))[0]
        words.append(word)
    return values, words


def _check_extents(where, op, values, buffers):
    ref = ref_ops.OPS.get(op.name)
    if ref is None:
        log.warning('%s: op %r has no reference semantics; extents not checked', where, op.name)
        return
    for arg, count in ref.extents(values).items():
        buf = buffers[values[arg]]
        _need(count <= buf.rows * buf.cols,
              f"{where} arg '{arg}': op '{op.name}' touches {count} element(s) but buffer '{buf.name}' "
              f"has {buf.rows * buf.cols} ({buf.rows}x{buf.cols})")


@dataclass
class _Decl:
    """A declared task before gates and edges are resolved."""
    task: Task
    after: list = field(default_factory=list)      # declared-task indices
    publishes: str | None = None
    reads: list = field(default_factory=list)      # (var, policy, version)


def _parse_tasks(graphs, ops, buffers, variables, allow_nonfinite):
    _need(isinstance(graphs, list) and graphs, "workload: 'graphs' must be a nonempty list")
    decls, graph_names = [], set()
    for gi, graph in enumerate(graphs):
        _keys(graph, f'graphs[{gi}]', {'name', 'tasks'}, {'name', 'tasks'})
        gname = graph['name']
        _need(isinstance(gname, str) and gname and '/' not in gname, f'graphs[{gi}]: name must be a non-empty string without "/"')
        _need(gname not in graph_names, f'graph {gname!r}: duplicate graph name')
        graph_names.add(gname)
        _need(isinstance(graph['tasks'], list) and graph['tasks'], f'graph {gname!r}: tasks must be a nonempty list')
        ids = set()
        for ti, t in enumerate(graph['tasks']):
            _keys(t, f'graph {gname!r} tasks[{ti}]', {'id', 'op', 'args', 'after', 'priority', 'deadline_us',
                                                      'publishes', 'reads'}, {'id', 'op'})
            tid = t['id']
            _need(isinstance(tid, str) and tid, f'graph {gname!r} tasks[{ti}]: id must be a non-empty string')
            _need(tid not in ids, f'graph {gname!r}: duplicate task id {tid!r}')
            ids.add(tid)
        # indices are assigned in a second pass so `after` may reference later tasks
        base = len(decls)
        index = {t['id']: base + i for i, t in enumerate(graph['tasks'])}
        for t in graph['tasks']:
            where = f"{gname}/{t['id']}"
            op = ops.get(t['op'])
            _need(op is not None, f"{where}: unknown op {t['op']!r}; known: {sorted(ops)}")
            values, words = _parse_args(where, op, t.get('args', {}), buffers, allow_nonfinite)
            _check_extents(where, op, values, buffers)
            prio, dl = t.get('priority', 0), t.get('deadline_us', -1)
            _need(_is_int(prio) and INT32_MIN <= prio <= INT32_MAX, f'{where}: priority must be an int32, got {prio!r}')
            _need(_is_int(dl) and -1 <= dl <= INT32_MAX, f'{where}: deadline_us must be -1 or an int32 >= 0, got {dl!r}')
            after = t.get('after', [])
            _need(isinstance(after, list), f'{where}: after must be a list of task ids')
            _need(len(set(map(str, after))) == len(after), f'{where}: duplicate dependency in after {after}')
            deps = []
            for dep in after:
                _need(dep in index, f"{where}: after names unknown task {dep!r} in graph {gname!r}")
                deps.append(index[dep])
            pub = t.get('publishes')
            _need(pub is None or pub in variables, f"{where}: publishes undeclared variable {pub!r}; declared: {list(variables)}")
            reads, seen = [], set()
            raw_reads = t.get('reads', [])
            _need(isinstance(raw_reads, list), f'{where}: reads must be a list')
            for ri, r in enumerate(raw_reads):
                _keys(r, f'{where} reads[{ri}]', {'var', 'policy', 'version'}, {'var', 'policy', 'version'})
                _need(r['var'] in variables, f"{where} reads[{ri}]: undeclared variable {r['var']!r}; declared: {list(variables)}")
                _need(r['policy'] in POLICIES, f"{where} reads[{ri}]: policy must be one of {POLICIES}, got {r['policy']!r}")
                _need(_is_int(r['version']) and 0 <= r['version'] <= INT32_MAX,
                      f"{where} reads[{ri}]: version must be an int32 >= 0, got {r['version']!r}")
                _need(r['var'] not in seen, f"{where}: reads variable {r['var']!r} twice")
                seen.add(r['var'])
                reads.append((r['var'], r['policy'], r['version']))
            task = Task(where, gname, gi, op, values, words, prio, dl)
            decls.append(_Decl(task, deps, pub, reads))
    return decls


# ---------------------------------------------------------------------------------------------- graph analysis

def _reaches(succ, src, dst):
    """True if a (possibly empty) path src -> ... -> dst exists."""
    stack, seen = [src], {src}
    while stack:
        n = stack.pop()
        if n == dst:
            return True
        for m in succ[n]:
            if m not in seen:
                seen.add(m)
                stack.append(m)
    return False


def _check_acyclic(names, succ):
    indeg = [0] * len(names)
    for outs in succ:
        for j in outs:
            indeg[j] += 1
    ready, done = [i for i, d in enumerate(indeg) if d == 0], 0
    while ready:
        i = ready.pop()
        done += 1
        for j in succ[i]:
            indeg[j] -= 1
            if indeg[j] == 0:
                ready.append(j)
    if done != len(names):
        stuck = [names[i] for i, d in enumerate(indeg) if d > 0]
        raise CompileError(f'dependency cycle among {len(stuck)} task(s): {stuck[:8]}{"..." if len(stuck) > 8 else ""}')


def _resolve_versions(decls, variables, succ):
    """Assign publish versions, add exact-snapshot edges, and validate every version constraint.

    Mutates `succ` (declared-task adjacency) and fills task.wait/pub/reads. Returns publisher lists.
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
                _need(not _reaches(succ, nxt, i),
                      f"{where} can never hold: {t.name} is ordered after {decls[nxt].task.name}, which publishes version {n + 1}")
                succ[i].add(nxt)         # write-after-read: the next update may not start before this reader ends
            reads.append(Read(var_name, policy, n, waits=n > var.base_version))
        t.reads = tuple(reads)

    for var_name, idxs in publishers.items():
        for a, b in zip(idxs, idxs[1:]):
            _need(_reaches(succ, a, b),
                  f"unordered publishers of {var_name!r}: no dependency path from {decls[a].task.name} "
                  f"(version {decls[a].task.pub[1]}) to {decls[b].task.name} (version {decls[b].task.pub[1]}); "
                  f"add an 'after' chain so their order is defined")

    for i, d in enumerate(decls):                           # a reader must not precede the publisher it waits for
        for r in d.task.reads:
            if r.waits:
                p = by_version[r.var][r.version]
                _need(not _reaches(succ, i, p),
                      f"{d.task.name}: waits for {r.var!r} >= {r.version}, published by {decls[p].task.name}, "
                      f"which is ordered after it (deadlock)")


def _lay_out(decls, succ, ops):
    """Insert noop gate tasks for reads beyond the first waiting one; return final tasks and adjacency."""
    noop = ops.get('noop')
    _need(noop is not None, "kvm_ops.def defines no 'noop' operator, needed for version gates")
    tasks, final, gate_of = [], {}, {}
    for i, d in enumerate(decls):
        t = d.task
        waiting = [r for r in t.reads if r.waits]
        for k, r in enumerate(waiting[1:], 1):
            gate = Task(f'__gate/{t.name}/{k}', t.graph, t.workload_id, noop, {}, [], t.priority, t.deadline_us,
                        wait=(r.var, r.version), gate=True)
            r.gate = gate.name
            gate_of.setdefault(i, []).append(len(tasks))
            tasks.append(gate)
        final[i] = len(tasks)
        if waiting:
            t.wait = (waiting[0].var, waiting[0].version)
        tasks.append(t)
    adj = [set() for _ in tasks]
    for i, outs in enumerate(succ):
        adj[final[i]].update(final[j] for j in outs)
    for i, gates in gate_of.items():
        for g in gates:
            adj[g].add(final[i])
    return tasks, [sorted(a) for a in adj], final


def compile_spec(spec, base_versions=None, allow_nonfinite=False) -> Compiled:
    """Compile a workload dict. `base_versions` maps variable -> live version for a later-admitted workload."""
    _keys(spec, 'workload', {'variables', 'buffers', 'graphs'}, {'graphs'})
    ops, task_abi = abi.load_ops(), abi.load_task_abi()
    variables = _parse_variables(spec.get('variables', []), base_versions)
    buffers = _parse_buffers(spec.get('buffers', []), allow_nonfinite)
    decls = _parse_tasks(spec['graphs'], ops, buffers, variables, allow_nonfinite)
    succ = [set() for _ in decls]
    for j, d in enumerate(decls):
        for i in d.after:
            _need(i != j, f'{decls[j].task.name}: depends on itself')
            succ[i].add(j)
    _check_acyclic([d.task.name for d in decls], succ)
    _resolve_versions(decls, variables, succ)
    tasks, adj, final = _lay_out(decls, succ, ops)
    _check_acyclic([t.name for t in tasks], adj)

    indegree = [0] * len(tasks)
    for outs in adj:
        for j in outs:
            indegree[j] += 1
    var_id = {n: v.id for n, v in variables.items()}
    desc, flat_succ, flat_args = bytearray(), [], []
    for i, t in enumerate(tasks):
        wv, wn = (var_id[t.wait[0]], t.wait[1]) if t.wait else (task_abi.no_var, 0)
        pv, pn = (var_id[t.pub[0]], t.pub[1]) if t.pub else (task_abi.no_var, 0)
        _need(len(t.words) == len(t.op.args), f'{t.name}: internal arity mismatch')
        fields = dict(op_id=t.op.id, succ_off=len(flat_succ), succ_cnt=len(adj[i]), indegree=indegree[i],
                      priority=t.priority, deadline_us=t.deadline_us, args_off=len(flat_args), args_words=len(t.words),
                      wait_var=wv, wait_ver=wn, pub_var=pv, pub_ver=pn, workload_id=t.workload_id)
        desc += task_abi.layout.pack(*(fields.get(f, 0) for f in task_abi.fields))
        flat_succ.extend(adj[i])
        flat_args.extend(t.words)
    _need(len(flat_args) <= INT32_MAX and len(flat_succ) <= INT32_MAX, 'workload too large for int32 offsets')

    index_of = {t.name: i for i, t in enumerate(tasks)}
    metadata = {
        'abi_version': task_abi.version, 'task_count': len(tasks), 'edge_count': len(flat_succ),
        'args_words': len(flat_args), 'task_names': [t.name for t in tasks],
        'initial_ready': [i for i, d in enumerate(indegree) if d == 0],
        'operator_ids': {n: o.id for n, o in ops.items()},
        'buffers': [{'name': b.name, 'index': b.index, 'rows': b.rows, 'cols': b.cols, 'init': b.init}
                    for b in buffers.values()],
        'variables': [{'name': v.name, 'id': v.id, 'base_version': v.base_version} for v in variables.values()],
        'publishers': [{'task': index_of[t.name], 'name': t.name, 'var': var_id[t.pub[0]], 'version': t.pub[1]}
                       for t in tasks if t.pub],
        'reads': [{'task': index_of[t.name], 'name': t.name, 'var': var_id[r.var], 'policy': r.policy,
                   'version': r.version, 'observed': r.version if r.policy == 'exact' else None,
                   'gate': index_of[r.gate] if r.gate else None}
                  for t in tasks if not t.gate for r in t.reads],
        'final_versions': {v.name: v.final_version for v in variables.values()},
    }
    log.info('compiled %d tasks (%d gates), %d edges, %d buffers, %d variables',
             len(tasks), sum(t.gate for t in tasks), len(flat_succ), len(buffers), len(variables))
    return Compiled(metadata, bytes(desc), struct.pack(f'<{len(flat_succ)}i', *flat_succ),
                    struct.pack(f'<{len(flat_args)}i', *flat_args), tasks, list(buffers.values()),
                    list(variables.values()), adj)


def write_bundle(compiled: Compiled, out_dir) -> Path:
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'tasks.bin').write_bytes(compiled.tasks)
    (dest / 'successors.bin').write_bytes(compiled.successors)
    (dest / 'args.bin').write_bytes(compiled.args)
    (dest / 'metadata.json').write_text(json.dumps(compiled.metadata, indent=2) + '\n')
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
    p.add_argument('--log-level', default='INFO')
    args = p.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    try:
        spec = json.loads(Path(args.workload).read_text())
        compiled = compile_spec(spec, parse_base_versions(args.base_versions), args.allow_nonfinite)
    except (CompileError, OSError, json.JSONDecodeError) as e:
        log.error('%s', e)
        return 1
    write_bundle(compiled, args.out_dir)
    if args.emit_golden:
        import workload_vm
        golden = workload_vm.run_golden(compiled)
        (Path(args.out_dir) / 'golden.json').write_text(json.dumps(golden, indent=2) + '\n')
        log.info('wrote golden.json')
    return 0


if __name__ == '__main__':
    sys.exit(main())
