"""Multi-device CPU simulator: runs ALL shards of a plan the way N executors would, with a RANDOM interleaving of their tasks.

It executes the compiled programs (the same tasks/edges/waits the GPU receives) with the reference semantics of runtime/
ref_ops.py + dist/commops.py. Cross-device effects are modeled faithfully:
  * a `put_signal` writes the peer's buffer list and then its inbox slot (data before signal, as on the device);
  * a remote publish becomes visible to the owner's waiters only when the owner's PUMP step runs (a separate schedulable
    action, like the admitter polling the inbox), so the simulator also explores "data landed, version not yet visible";
  * a `send_proxy` appends to the device's send queue; the "network" delivers per-pair messages in order, one at a time, as
    separate schedulable actions (data written to the receiver, then its inbox).
Different seeds give different interleavings; the result must not depend on them (bitwise, for deterministic reduction order).
"""
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'runtime'), str(ROOT / 'dist')]
import commops                  # noqa: E402
commops.install()
import partition as pt          # noqa: E402
import ref_ops                  # noqa: E402


class SimError(RuntimeError):
    pass


class RankKilled(SimError):
    """A rank died mid-step (fault injection). Carries the rank and how many scheduler actions had run."""
    def __init__(self, rank, at):
        super().__init__(f'rank {rank} killed after {at} scheduler actions')
        self.rank, self.at = rank, at


class Device:
    def __init__(self, shard):
        self.rank = shard.rank
        self.shard = shard
        self.c = pt.compile_shard(shard)
        self.buf = {b.name: b.materialize() for b in self.c.buffers}
        self.index = [b.name for b in self.c.buffers]
        self.vars = [v.name for v in self.c.variables]
        self.version = {v.name: (0 if v.name in shard.in_vars else v.base_version) for v in self.c.variables}
        self.inbox = [0] * max(1, len(self.vars))
        self.sendq = []
        n = len(self.c.program)
        self.indeg = [0] * n
        for i in range(n):
            for j in self.c.successors_of(i):
                self.indeg[j] += 1
        self.ready = [i for i in range(n) if self.indeg[i] == 0]
        self.done = 0
        self.violations = []


def runnable(dev, i):
    w = dev.c.program[i].wait
    return w is None or dev.version[w[0]] >= w[1]


class _Inbox(list):
    """The device inbox; remote writes mark the slot dirty so the pump only looks at slots that changed."""
    def __init__(self, vals, dirty):
        super().__init__(vals)
        self.dirty = dirty

    def __setitem__(self, k, v):
        super().__setitem__(k, v)
        self.dirty.add(k)


def _admit(d, i):
    w = d.c.program[i].wait
    if w is None or d.version[w[0]] >= w[1]:
        d.pos[i] = len(d.runnable)
        d.runnable.append(i)
    else:
        d.blocked.setdefault(w[0], []).append(i)


def _remove(d, i):
    k = d.pos.pop(i)
    last = d.runnable.pop()
    if last != i:
        d.runnable[k] = last
        d.pos[last] = k


def _wake(d, var):
    for i in d.blocked.pop(var, []):
        _admit(d, i)


def run(plan, seed, max_steps=10_000_000, pump_bias=0.3, check_fresh=True, init=None, kill=None, replace=None):
    """init: {rank: {buffer: values}} overrides initial buffer contents (state carried between steps / restored from a checkpoint).
    kill: (rank, n): raise RankKilled when n scheduler actions have run. Also returns the action count via run.last_steps."""
    rng = random.Random(seed)
    devs = [Device(s) for s in plan.shards]
    for r, bufs in (init or {}).items():
        for k, v in bufs.items():
            assert len(devs[r].buf[k]) == len(v), f'init for rank {r} buffer {k}: size mismatch'
            devs[r].buf[k] = list(v)
    by_rank = {d.rank: d for d in devs}
    net = {}                                   # (src, dst) -> list of messages (in order)
    nonempty = set()                           # net keys with queued messages
    total = sum(len(d.c.program) for d in devs)
    # incremental scheduler state (the naive "rebuild every action each step" is quadratic and made 32+ devices unusable)
    for d in devs:
        d.runnable, d.pos, d.blocked, d.dirty = [], {}, {}, set()
        d.inbox = _Inbox(d.inbox, d.dirty)
        d.var_idx = {v: k for k, v in enumerate(d.vars)}
        d.in_idx = [d.var_idx[v] for v in d.shard.in_vars]
        for i in list(d.ready):
            _admit(d, i)
    done_total = 0
    steps = 0
    while done_total < total:
        steps += 1
        run.last_steps = steps
        if kill is not None and steps > kill[1]:
            raise RankKilled(kill[0], steps - 1)
        if steps > max_steps:
            raise SimError('step limit')
        pumps = []
        for d in devs:
            if d.dirty:
                pend = [k for k in d.dirty if d.inbox[k] > d.version[d.vars[k]]]
                d.dirty.intersection_update(pend)
                if pend:
                    pumps.append(('pump', d, pend))
        others = pumps + [('net', k) for k in nonempty]
        ntask = sum(len(d.runnable) for d in devs)
        if not others and not ntask:
            blocked = [(d.rank, d.c.program[i].name, d.c.program[i].wait) for d in devs for v in d.blocked.values() for i in v][:5]
            raise SimError(f'deadlock: {total - done_total} tasks left; blocked {blocked}')
        if others and (not ntask or rng.random() < pump_bias):
            act = rng.choice(others)
        else:
            k = rng.randrange(ntask + len(others))
            if k >= ntask:
                act = others[k - ntask]
            else:
                for d in devs:
                    if k < len(d.runnable):
                        act = ('task', d, d.runnable[k]); break
                    k -= len(d.runnable)
        if act[0] == 'pump':
            _, d, pend = act
            for k in pend:                      # the admitter publishes the inbox value locally
                v = d.vars[k]
                d.version[v] = d.inbox[k]
                _wake(d, v)
            d.dirty.difference_update(pend)
        elif act[0] == 'net':
            key = act[1]
            m = net[key].pop(0)
            if not net[key]:
                nonempty.discard(key)
            dst = by_rank[m['dst_rank']]
            name = dst.index[m['dst_buf']]
            buf = dst.buf[name]
            for r in range(m['rows']):
                o = m['dst_off'] + r * m['dst_ld']
                buf[o:o + m['cols']] = m['data'][r * m['cols']:(r + 1) * m['cols']]
            if m['var'] >= 0:
                dst.inbox[m['var']] = max(dst.inbox[m['var']], m['ver'])
        else:
            _, d, i = act
            _remove(d, i)
            execute(d, i, by_rank)
            for m in d.sendq:
                key = (d.rank, m['dst_rank'])
                net.setdefault(key, []).append(m)
                nonempty.add(key)
            d.sendq.clear()
            t = d.c.program[i]
            if t.pub:
                d.version[t.pub[0]] = t.pub[1]
                _wake(d, t.pub[0])
            d.done += 1
            done_total += 1
            for j in d.c.successors_of(i):
                d.indeg[j] -= 1
                if d.indeg[j] == 0:
                    _admit(d, j)
    run.last_steps = steps
    return [dict(d.buf) for d in devs], devs


def execute(dev, i, by_rank):
    t = dev.c.program[i]
    ref = ref_ops.OPS[t.op.name]
    bufs, scal = {}, {}
    for a in t.op.args:
        if a.type != 'buf':
            scal[a.name] = t.args[a.name]
            continue
        name = t.args[a.name]
        rem = dev.shard.remotes.get(name)
        if name is None:                       # optional buffer argument left out
            bufs[a.name] = None
        elif rem is None:
            bufs[a.name] = dev.buf[name]
        elif rem['kind'] == 'peer':
            bufs[a.name] = by_rank[rem['rank']].buf[rem['buf']]
        elif rem['kind'] == 'inbox':
            bufs[a.name] = by_rank[rem['rank']].inbox
        else:
            bufs[a.name] = dev.sendq
    ref.run(bufs, scal)
