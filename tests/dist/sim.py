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


def run(plan, seed, max_steps=10_000_000, pump_bias=0.3, check_fresh=True):
    rng = random.Random(seed)
    devs = [Device(s) for s in plan.shards]
    by_rank = {d.rank: d for d in devs}
    net = {}                                   # (src, dst) -> list of messages (in order)
    total = sum(len(d.c.program) for d in devs)
    steps = 0
    while sum(d.done for d in devs) < total:
        steps += 1
        if steps > max_steps:
            raise SimError('step limit')
        actions = []
        for d in devs:
            for i in d.ready:
                if runnable(d, i):
                    actions.append(('task', d, i))
            pend = [v for v in d.shard.in_vars if d.inbox[d.vars.index(v)] > d.version[v]]
            if pend:
                actions.append(('pump', d, pend))
        for (s, t), q in net.items():
            if q:
                actions.append(('net', s, t))
        if not actions:
            stuck = [(d.rank, d.c.program[i].name) for d in devs for i in d.ready][:5]
            blocked = [(d.rank, d.c.program[i].name, d.c.program[i].wait) for d in devs for i in d.ready if not runnable(d, i)][:5]
            raise SimError(f'deadlock: {total - sum(d.done for d in devs)} tasks left; blocked {blocked}')
        pumps = [a for a in actions if a[0] != 'task']
        act = rng.choice(pumps) if pumps and rng.random() < pump_bias else rng.choice(actions)
        if act[0] == 'pump':
            _, d, pend = act
            for v in pend:                      # the admitter publishes the inbox value locally
                d.version[v] = d.inbox[d.vars.index(v)]
        elif act[0] == 'net':
            _, s, t = act
            m = net[(s, t)].pop(0)
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
            d.ready.remove(i)
            execute(d, i, by_rank)
            for m in d.sendq:
                net.setdefault((d.rank, m['dst_rank']), []).append(m)
            d.sendq.clear()
            t = d.c.program[i]
            if t.pub:
                d.version[t.pub[0]] = t.pub[1]
            d.done += 1
            for j in d.c.successors_of(i):
                d.indeg[j] -= 1
                if d.indeg[j] == 0:
                    d.ready.append(j)
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
        if rem is None:
            bufs[a.name] = dev.buf[name]
        elif rem['kind'] == 'peer':
            bufs[a.name] = by_rank[rem['rank']].buf[rem['buf']]
        elif rem['kind'] == 'inbox':
            bufs[a.name] = by_rank[rem['rank']].inbox
        else:
            bufs[a.name] = dev.sendq
    ref.run(bufs, scal)
