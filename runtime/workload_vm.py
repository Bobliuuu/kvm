"""CPU reference VM: executes compiled KVM programs with the executor's semantics.

It runs the *compiler's output* (same tasks, edges, gates and wait/publish slots the GPU receives), so the two
cannot disagree about structure. Semantics implemented:
  * a task runs when all in-batch predecessors finished AND version[wait_var] >= wait_ver;
  * on completion it sets version[pub_var] = pub_ver;
  * `exact` reads are verified at start and again at finish (snapshot isolation violation is an error);
  * priority with aging (or a deterministic mode: one worker, highest priority, lowest submission order);
  * tasks may be submitted while workers run; each submission is compiled against the projected versions
    (live versions plus everything already admitted).
Every op follows runtime/ref_ops.py (float32 per operation).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import time
from dataclasses import dataclass

import compile_workload as cw
import ref_ops
from ref_types import Buf

log = logging.getLogger('kvm.vm')


class GraphError(ValueError):
    """A submission conflicts with the runtime's state (buffer shape clash, version mismatch)."""


@dataclass
class _Node:
    key: tuple
    task: cw.Task
    seq: int
    succ: list
    remaining: int
    state: str = 'pending'        # pending -> running -> done
    enqueued: float = 0.0


class Runtime:
    def __init__(self, workers=4, aging_per_second=20.0, deterministic=False, allow_opaque=False):
        if workers < 1 or aging_per_second <= 0 or (deterministic and workers != 1):
            raise ValueError('workers >= 1, aging_per_second > 0, deterministic requires workers == 1')
        self.workers, self.aging, self.deterministic = workers, aging_per_second, deterministic
        self.allow_opaque = allow_opaque
        self.cv = threading.Condition()
        self.buffers: dict[str, Buf] = {}
        self.shapes: dict[str, tuple] = {}
        self.versions: dict[str, int] = {}
        self.projected: dict[str, int] = {}
        self.nodes: dict[tuple, _Node] = {}
        self.ready: list[tuple] = []
        self.completed: list[str] = []
        self.errors: list[tuple[str, str]] = []
        self.trace: list[dict] = []
        self.threads: list[threading.Thread] = []
        self.batches = 0
        self.closed = self.started = False

    # ------------------------------------------------------------------ admission
    def submit(self, spec: dict, allow_nonfinite=False, data_dir=None, ops=None) -> cw.Compiled:
        """Compile `spec` against the projected versions and admit it (callable while running)."""
        with self.cv:
            declared = spec.get('variables', []) if isinstance(spec, dict) else []
            base = {v: self.projected[v] for v in declared if isinstance(v, str) and v in self.projected}
            compiled = cw.compile_spec(spec, base, allow_nonfinite, data_dir, ops)
            return self._admit(compiled)

    def submit_compiled(self, compiled: cw.Compiled) -> cw.Compiled:
        with self.cv:
            return self._admit(compiled)

    def _admit(self, compiled):
        if self.closed:
            raise RuntimeError('runtime closed')
        for v in compiled.variables:
            known = self.projected.get(v.name)
            if known is not None and known != v.base_version:
                raise GraphError(f'variable {v.name!r}: compiled against base_version {v.base_version}, '
                                 f'runtime projects {known}; recompile with base_versions')
        for b in compiled.buffers:
            if b.name in self.shapes and self.shapes[b.name] != (b.rows, b.cols, b.dtype):
                raise GraphError(f'buffer {b.name!r} already exists as {self.shapes[b.name]}, got {(b.rows, b.cols, b.dtype)}')
        missing = sorted({t.op.name for t in compiled.program if t.op.name not in ref_ops.OPS})
        if missing:
            raise GraphError(f'no CPU semantics for operator(s) {missing}: add them to runtime/ref_ops.py')
        opaque = sorted({t.op.name for t in compiled.program if ref_ops.OPS[t.op.name].opaque})
        if opaque and not self.allow_opaque:
            raise GraphError(f'operator(s) {opaque} have no CPU semantics (opaque); run with allow_opaque to execute them as noops')
        for b in compiled.buffers:
            if b.name not in self.buffers:
                self.buffers[b.name] = Buf(b.materialize(compiled.data_dir), b.rows, b.cols, b.dtype)
                self.shapes[b.name] = (b.rows, b.cols, b.dtype)
        for v in compiled.variables:
            self.versions.setdefault(v.name, v.base_version)
            self.projected[v.name] = v.final_version
        batch, now = self.batches, time.monotonic()
        self.batches += 1
        for i, t in enumerate(compiled.program):
            node = _Node((batch, i), t, len(self.nodes), [(batch, j) for j in compiled.successors_of(i)], 0)
            self.nodes[node.key] = node
        for j in compiled.succ:
            self.nodes[(batch, j)].remaining += 1
        for i in range(len(compiled.program)):
            n = self.nodes[(batch, i)]
            if n.remaining == 0:
                n.enqueued = now
                self.ready.append(n.key)
        log.info('admitted batch %d: %d tasks', batch, len(compiled.program))
        self.cv.notify_all()
        return compiled

    # ------------------------------------------------------------------ scheduling
    def _runnable(self, node):
        w = node.task.wait
        return w is None or self.versions[w[0]] >= w[1]

    def _pick(self):
        now = time.monotonic()
        cands = [self.nodes[k] for k in self.ready if self._runnable(self.nodes[k])]
        if not cands:
            return None
        if self.deterministic:
            node = max(cands, key=lambda n: (n.task.priority, -n.seq))
        else:
            node = max(cands, key=lambda n: (n.task.priority + self.aging * (now - n.enqueued),
                                             -(n.task.deadline_us if n.task.deadline_us >= 0 else float('inf')),
                                             -n.seq))
        self.ready.remove(node.key)
        return node

    def start(self):
        with self.cv:
            if self.started:
                return
            self.started = True
            self.threads = [threading.Thread(target=self._worker, name=f'vm-{i}', daemon=True) for i in range(self.workers)]
            for t in self.threads:
                t.start()

    def _worker(self):
        while True:
            with self.cv:
                while (node := self._pick()) is None:
                    if self.closed:
                        return
                    self.cv.wait()
                node.state = 'running'
                observed = {r.var: self.versions[r.var] for r in node.task.reads}
                started = time.monotonic()
            error = self._violation(node.task, observed, 'start')
            if error is None:
                try:
                    self._execute(node.task)
                except Exception as exc:                      # a task bug must not wedge the scheduler
                    error = repr(exc)
            with self.cv:
                finished = time.monotonic()
                error = error or self._violation(node.task, {r.var: self.versions[r.var] for r in node.task.reads}, 'finish')
                self._finish(node, observed, started, finished, error)
                self.cv.notify_all()

    @staticmethod
    def _violation(task, seen, when):
        for r in task.reads:
            if r.policy == 'exact' and seen[r.var] != r.version:
                return f'snapshot violated at {when}: exact read of {r.var!r} wants {r.version}, saw {seen[r.var]}'
            if r.policy == 'at_least' and seen[r.var] < r.wait_ver:
                return f'at_least violated at {when}: {r.var!r} wants >= {r.wait_ver}, saw {seen[r.var]}'
        return None

    def _execute(self, task):
        if task.when and (self.buffers[task.when[0]][task.when[1]] != 0) == task.when[2]:
            return                       # predicate false: no body, outputs untouched; the task still completes, publishes and releases successors
        ref = ref_ops.OPS[task.op.name]
        bufs = {a.name: (None if task.args[a.name] is None else self.buffers[task.args[a.name]])
                for a in task.op.args if a.type == 'buf'}
        scalars = {a.name: task.args[a.name] for a in task.op.args if a.type != 'buf'}
        ref.run(bufs, scalars)

    def _finish(self, node, observed, started, finished, error):
        t = node.task
        node.state = 'done'
        self.completed.append(t.name)
        self.trace.append({'task': t.name, 'op': t.op.name, 'observed': observed, 'started': started,
                           'finished': finished, 'error': error})
        log.debug('%s observed=%s%s', t.name, observed, f' ERROR {error}' if error else '')
        if error:
            log.error('%s failed: %s', t.name, error)
            self.errors.append((t.name, error))
            self._cancel_dependents(node)
            return
        if t.pub:
            var, ver = t.pub
            if self.versions[var] + 1 != ver:
                self.errors.append((t.name, f'publish order broken: {var!r} is {self.versions[var]}, publishing {ver}'))
                self._cancel_dependents(node)
                return
            self.versions[var] = ver
        for k in node.succ:
            s = self.nodes[k]
            s.remaining -= 1
            if s.remaining == 0:
                s.enqueued = finished
                self.ready.append(k)

    def _cancel_dependents(self, failed):
        """Cancel everything that can no longer run: DAG descendants and tasks waiting on a never-published version."""
        dead: dict[str, int] = {}
        def mark(n):
            if n.task.pub:
                dead[n.task.pub[0]] = min(dead.get(n.task.pub[0], n.task.pub[1]), n.task.pub[1])
        mark(failed)
        frontier, seen = [failed], {failed.key}
        while frontier:
            nxt = [self.nodes[k] for n in frontier for k in n.succ]
            nxt += [n for n in self.nodes.values() if n.state == 'pending' and n.task.wait
                    and n.task.wait[1] >= dead.get(n.task.wait[0], float('inf'))]
            frontier = []
            for n in nxt:
                if n.key in seen or n.state != 'pending':
                    continue
                seen.add(n.key)
                n.state = 'done'
                if n.key in self.ready:
                    self.ready.remove(n.key)
                self.completed.append(n.task.name)
                self.errors.append((n.task.name, 'dependency failed'))
                mark(n)
                frontier.append(n)

    # ------------------------------------------------------------------ lifecycle
    def wait(self, timeout=10):
        end = time.monotonic() + timeout
        with self.cv:
            while len(self.completed) < len(self.nodes):
                left = end - time.monotonic()
                if left <= 0:
                    pending = [n.task.name for n in self.nodes.values() if n.state != 'done']
                    raise TimeoutError(f'workload did not finish; {len(pending)} pending, e.g. {pending[:5]}')
                self.cv.wait(left)
            if self.errors:
                raise RuntimeError(f'{len(self.errors)} task error(s): {self.errors[:3]}')
            return list(self.trace)

    def shutdown(self):
        with self.cv:
            self.closed = True
            self.cv.notify_all()
        for t in self.threads:
            t.join(timeout=5)
            if t.is_alive():
                raise TimeoutError('worker did not stop')


def run_spec(spec, workers=4, deterministic=False, base_versions=None, timeout=30, allow_opaque=False, data_dir=None, ops=None):
    """Compile and run one workload to completion; returns (runtime, trace)."""
    compiled = cw.compile_spec(spec, base_versions, data_dir=data_dir, ops=ops)
    vm = Runtime(1 if deterministic else workers, deterministic=deterministic, allow_opaque=allow_opaque)
    vm.submit_compiled(compiled)
    vm.start()
    try:
        return vm, vm.wait(timeout)
    finally:
        vm.shutdown()


def run_golden(compiled: cw.Compiled, timeout=60, allow_opaque=False) -> dict:
    """Deterministic CPU run of a compiled bundle; the reference the GPU executor's output is diffed against."""
    vm = Runtime(1, deterministic=True, allow_opaque=allow_opaque)
    vm.submit_compiled(compiled)
    vm.start()
    try:
        trace = vm.wait(timeout)
    finally:
        vm.shutdown()
    seen = {e['task']: e['observed'] for e in trace}
    reads = [{'task': r['name'], 'var': compiled.variables[r['var']].name, 'policy': r['policy'],
              'version': r['version'], 'observed': seen[r['name']][compiled.variables[r['var']].name]}
             for r in compiled.metadata['reads']]
    return {'abi_version': compiled.metadata['abi_version'], 'schedule': 'deterministic',
            'buffers': {n: list(v) for n, v in sorted(vm.buffers.items())}, 'dtypes': {n: v.dtype for n, v in sorted(vm.buffers.items())},
            'versions': dict(vm.versions), 'reads': reads}


def main(argv=None):
    p = argparse.ArgumentParser(description='Run a workload on the CPU reference VM')
    p.add_argument('workload')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--deterministic', action='store_true', help='one worker, strict priority/submission order')
    p.add_argument('--base-versions', default='')
    p.add_argument('--allow-opaque', action='store_true', help='execute ops without CPU semantics as noops')
    p.add_argument('--data-dir', help='where `file` initial values live (default: the workload directory)')
    p.add_argument('--trace', action='store_true', help='log (task, observed versions) per task')
    p.add_argument('--log-level', default='INFO')
    args = p.parse_args(argv)
    logging.basicConfig(level='DEBUG' if args.trace else args.log_level.upper(),
                        format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    with open(args.workload) as f:
        spec = json.load(f)
    data_dir = args.data_dir or os.path.dirname(os.path.abspath(args.workload))
    vm, trace = run_spec(spec, args.workers, args.deterministic, cw.parse_base_versions(args.base_versions),
                         allow_opaque=args.allow_opaque, data_dir=data_dir)
    if args.trace:
        for e in trace:
            print(f"{e['task']}\tobserved={json.dumps(e['observed'], sort_keys=True)}")
    print(json.dumps({'buffers': vm.buffers, 'versions': vm.versions, 'tasks': len(trace)}, indent=2))


if __name__ == '__main__':
    main()
