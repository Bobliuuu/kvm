"""Declarative, operator-extensible CPU reference VM for mixed GPU workloads.

This is an executable specification for future CUDA lowering, NOT a GPU VM.
A workload is a JSON DAG of named ops. Nothing in the scheduler knows about
training, inference, or games. Tasks may be submitted while workers are active.
"""
from __future__ import annotations
import heapq
import json
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Any

class GraphError(ValueError):
    pass

@dataclass
class Task:
    id: str
    op: str
    args: dict
    after: tuple[str, ...] = ()
    priority: int = 0
    deadline: float | None = None
    label: str = ""

@dataclass
class Graph:
    name: str
    tasks: list[Task]

    @classmethod
    def from_dict(cls, spec: dict) -> 'Graph':
        name = str(spec['name'])
        tasks = [Task(id=str(t['id']), op=str(t['op']), args=dict(t.get('args', {})),
                      after=tuple(t.get('after', [])), priority=int(t.get('priority', 0)),
                      deadline=t.get('deadline'), label=str(t.get('label', '')))
                 for t in spec['tasks']]
        ids = [t.id for t in tasks]
        if len(set(ids)) != len(ids):
            raise GraphError('duplicate task id')
        if not tasks:
            raise GraphError('empty graph')
        indegree = {i: 0 for i in ids}
        outgoing = defaultdict(list)
        for t in tasks:
            if len(set(t.after)) != len(t.after):
                raise GraphError(f'duplicate prerequisite for {t.id}')
            for dep in t.after:
                if dep not in indegree:
                    raise GraphError(f'unknown prerequisite {dep}')
                outgoing[dep].append(t.id)
                indegree[t.id] += 1
        ready = [i for i, d in indegree.items() if d == 0]
        count = 0
        while ready:
            i = ready.pop()
            count += 1
            for j in outgoing[i]:
                indegree[j] -= 1
                if indegree[j] == 0:
                    ready.append(j)
        if count != len(tasks):
            raise GraphError('dependency cycle')
        return cls(name, tasks)

class Store:
    """Shared named tensors/scalars; task graph must order conflicting accesses."""
    def __init__(self, initial: dict | None = None):
        self._lock = threading.RLock()
        self.data = dict(initial or {})

    def get(self, key):
        with self._lock:
            return self.data[key]

    def put(self, key, value):
        with self._lock:
            self.data[key] = value

class Registry:
    def __init__(self):
        self.ops: dict[str, Callable[[Store, dict], Any]] = {}

    def register(self, name: str, fn: Callable[[Store, dict], Any]):
        if name in self.ops:
            raise ValueError('duplicate operator')
        self.ops[name] = fn

    def execute(self, op, store, args):
        return self.ops[op](store, args)

class Runtime:
    """Threaded, work-conserving ready DAG executor with aging and deadlines.

    A task's effective priority is computed when selecting work, so an old
    background task eventually outranks newly arriving high-priority tasks.
    Nonpreemptive: once a task starts, it runs until completion.
    """
    def __init__(self, registry: Registry, store: Store, workers=4,
                 aging_per_second=20.0):
        if workers < 1 or aging_per_second <= 0:
            raise ValueError('invalid scheduler configuration')
        self.registry, self.store = registry, store
        self.workers, self.aging = workers, aging_per_second
        self.cv = threading.Condition()
        self.ready: list[str] = []
        self.nodes: dict[str, Task] = {}
        self.remaining: dict[str, int] = {}
        self.successors: dict[str, list[str]] = defaultdict(list)
        self.enqueued: dict[str, float] = {}
        self.completed: list[str] = []
        self.errors: list[tuple[str, str]] = []
        self.inflight = 0
        self.closed = False
        self.threads = []
        self.trace: list[dict] = []
        self.started = False

    def start(self):
        with self.cv:
            if self.started:
                return
            self.started = True
            self.threads = [threading.Thread(target=self._worker, daemon=True)
                            for _ in range(self.workers)]
            for t in self.threads:
                t.start()

    def submit(self, graph: Graph):
        with self.cv:
            if self.closed:
                raise RuntimeError('runtime closed')
            # Validate the entire submission before mutating scheduler state.
            keys = [f'{graph.name}/{task.id}' for task in graph.tasks]
            if len(set(keys)) != len(keys):
                raise GraphError('duplicate task ids in submission')
            for task, key in zip(graph.tasks, keys):
                if key in self.nodes:
                    raise GraphError(f'duplicate submitted task {key}')
                if task.op not in self.registry.ops:
                    raise GraphError(f'unregistered operator {task.op}')
                if len(set(task.after)) != len(task.after):
                    raise GraphError(f'duplicate prerequisites for {key}')
                if any(f'{graph.name}/{dep}' not in keys for dep in task.after):
                    raise GraphError(f'unknown prerequisite for {key}')
            # Graph.from_dict validates cycles; enforce it even when callers
            # instantiate Graph directly rather than using the parser.
            Graph.from_dict({'name': graph.name, 'tasks': [
                {'id': t.id, 'op': t.op, 'after': list(t.after)}
                for t in graph.tasks]})
            now = time.monotonic()
            for task in graph.tasks:
                key = f'{graph.name}/{task.id}'
                self.nodes[key] = task
                self.remaining[key] = len(task.after)
                for dep in task.after:
                    self.successors[f'{graph.name}/{dep}'].append(key)
                if not task.after:
                    self.ready.append(key)
                    self.enqueued[key] = now
            self.cv.notify_all()

    def _pick(self):
        now = time.monotonic()
        def score(key):
            t = self.nodes[key]
            age = now - self.enqueued[key]
            # Deadline is a tie-breaker; priority+aging guarantees eventual
            # service for a continuously ready task under finite arrival rate.
            return (t.priority + self.aging * age,
                    -(t.deadline if t.deadline is not None else float('inf')),
                    -self.enqueued[key])
        key = max(self.ready, key=score)
        self.ready.remove(key)
        return key

    def _worker(self):
        while True:
            with self.cv:
                while not self.ready and not self.closed:
                    self.cv.wait()
                if self.closed and not self.ready:
                    return
                key = self._pick()
                self.inflight += 1
                started = time.monotonic()
            try:
                self.registry.execute(self.nodes[key].op, self.store,
                                      self.nodes[key].args)
            except Exception as exc:
                error = repr(exc)
            else:
                error = None
            finished = time.monotonic()
            with self.cv:
                self.inflight -= 1
                self.completed.append(key)
                self.trace.append({'task': key, 'op': self.nodes[key].op,
                                   'started': started, 'finished': finished,
                                   'error': error})
                if error:
                    self.errors.append((key, error))
                    # Cancel dependent tasks transitively. Do not run them
                    # against missing/invalid outputs.
                    pending = list(self.successors[key])
                    canceled = set()
                    while pending:
                        child = pending.pop()
                        if child in canceled:
                            continue
                        canceled.add(child)
                        # A descendant cannot be running because its failed
                        # ancestor was still executing; avoid duplicate cancels
                        # at diamond joins.
                        if child in self.ready:
                            self.ready.remove(child)
                        self.completed.append(child)
                        self.errors.append((child, 'dependency failed'))
                        pending.extend(self.successors[child])
                else:
                    for child in self.successors[key]:
                        self.remaining[child] -= 1
                        if self.remaining[child] == 0:
                            self.ready.append(child)
                            self.enqueued[child] = finished
                self.cv.notify_all()

    def wait(self, timeout=10):
        end = time.monotonic() + timeout
        with self.cv:
            while len(self.completed) < len(self.nodes):
                left = end - time.monotonic()
                if left <= 0:
                    raise TimeoutError('workload did not finish')
                self.cv.wait(left)
            if self.errors:
                raise RuntimeError(f'{len(self.errors)} task errors: {self.errors[:3]}')
            return list(self.trace)

    def shutdown(self):
        with self.cv:
            self.closed = True
            self.cv.notify_all()
        for t in self.threads:
            t.join(timeout=5)
            if t.is_alive():
                raise TimeoutError('worker did not stop')

def default_registry():
    """Small reference op library. Operators are registered, not hardcoded in VM."""
    r = Registry()
    r.register('copy', lambda s,a: s.put(a['out'], s.get(a['src'])))
    r.register('add', lambda s,a: s.put(a['out'], s.get(a['a']) + s.get(a['b'])))
    r.register('scale', lambda s,a: s.put(a['out'], s.get(a['src']) * a['factor']))
    r.register('relu', lambda s,a: s.put(a['out'], max(0, s.get(a['src']))))
    r.register('sgd', lambda s,a: s.put(a['weight'], s.get(a['weight']) - a['lr'] * s.get(a['grad'])))
    r.register('predict', lambda s,a: s.put(a['out'], s.get(a['input']) * s.get(a['weight'])))
    r.register('mse_grad', lambda s,a: (s.put(a['loss'], (s.get(a['pred']) - s.get(a['target']))**2),
                                      s.put(a['grad'], 2 * (s.get(a['pred']) - s.get(a['target'])) * s.get(a['input']))))
    return r

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('workload')
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    spec = json.load(open(args.workload))
    vm = Runtime(default_registry(), Store(spec.get('state', {})), args.workers)
    vm.start()
    try:
        for graph in spec['graphs']:
            vm.submit(Graph.from_dict(graph))
        vm.wait()
        print(json.dumps({'state': vm.store.data, 'tasks': len(vm.completed),
                          'execution_order': [t['task'] for t in vm.trace]}, indent=2))
    finally:
        vm.shutdown()
