#!/usr/bin/env python3
"""Where the executor's idle time comes from, from `kvm_exec --trace-csv` plus the bundle that ran.

  python3 trace_idle.py TRACE.csv BUNDLE_DIR [STEPS]

Per worker block, the span between its first start and the run's end is cut into: run (op body), hop (gap before a task that was already ready when
the block went idle: pop latency, descriptor load, bookkeeping), starved (gap before a task that became ready only after the block went idle: the
block waited for a dependency), head (before the block's first task: admission ramp) and tail (after its last task). Plus: the admission rate,
the critical path (longest chain of run + release-to-start hops over the bundle's edges, using the measured times) and per-level cost.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_trace(path):
    rows = list(csv.DictReader(open(path)))
    f = lambda k: np.array([float(r[k]) for r in rows])
    return dict(op=[r['op'] for r in rows], admit=f('admit_us'), ready=f('ready_us'), start=f('start_us'), end=f('end_us'),
                block=np.array([int(r['block']) for r in rows]), cont=np.array([int(r['cont']) for r in rows]))


def main():
    tr = load_trace(sys.argv[1]); bundle = Path(sys.argv[2]); steps = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    ok = tr['start'] >= 0
    n = int(ok.sum()); S, E, R, A, B = tr['start'][ok], tr['end'][ok], tr['ready'][ok], tr['admit'][ok], tr['block'][ok]
    t0, t1 = S.min(), E.max(); span = t1 - t0
    print(f'{n} tasks, span {span / 1e3:.2f} ms ({span / steps / 1e3:.2f} ms/step), admission of the last task at {A.max() / 1e3:.2f} ms '
          f'(first at {A.min() / 1e3:.2f}), admit rate {n / max(A.max() - A.min(), 1e-9):.2f} tasks/us')
    blocks = np.unique(B); nb = len(blocks)
    run = (E - S).sum()
    hop = starved = head = tail = 0.0
    for b in blocks:
        m = B == b; s, e, r = S[m], E[m], R[m]; o = np.argsort(s); s, e, r = s[o], e[o], r[o]
        head += s[0] - t0; tail += t1 - e[-1]
        gap = s[1:] - e[:-1]
        late = r[1:] > e[:-1]                     # became ready only after this block went idle
        starved += np.where(late, np.maximum(r[1:] - e[:-1], 0), 0).sum()
        hop += np.where(late, np.minimum(gap, s[1:] - r[1:]).clip(0), gap).sum() if len(gap) else 0
    tot = span * nb
    print(f'worker-time {tot / 1e3:.1f} ms over {nb} blocks: run {run / tot:.1%}, hop(pop+bookkeeping after a ready task) {hop / tot:.1%}, '
          f'starved(waiting for a dependency) {starved / tot:.1%}, head(admission ramp) {head / tot:.1%}, tail {tail / tot:.1%}')
    # concurrency profile: mean number of tasks ready-or-running vs blocks
    ev = np.concatenate([np.stack([R, np.ones(n)], 1), np.stack([E, -np.ones(n)], 1)]); ev = ev[np.argsort(ev[:, 0], kind='stable')]
    dt = np.diff(ev[:, 0]); cnt = np.cumsum(ev[:, 1])[:-1]
    print(f'mean ready+running tasks {np.sum(cnt * dt) / span:.1f} (blocks {nb}); time with 0 ready/running {dt[cnt <= 0].sum() / span:.1%}, '
          f'with fewer than {nb} {dt[cnt < nb].sum() / span:.1%}')
    # critical path over the real edges
    t = np.frombuffer((bundle / 'tasks.bin').read_bytes(), dtype=np.int32).reshape(-1, 16) if (bundle / 'tasks.bin').exists() else None
    if t is not None:
        succ = np.frombuffer((bundle / 'successors.bin').read_bytes(), dtype=np.int32)
        ntask = len(t)
        # tasks.bin layout is the ABI's descriptor: succ_off, succ_cnt words (KvmTaskDesc words 1, 2)
        off, cnt_ = t[:, 1], t[:, 2]
        if tr['op'] and len(tr['op']) == ntask:
            full_S = tr['start']; full_E = tr['end']
            dur = np.where(full_S >= 0, full_E - full_S, 0.0)
            ready_after = np.zeros(ntask)   # earliest completion along the chain: longest path
            fin = np.zeros(ntask)
            best = np.zeros(ntask)
            order = range(ntask)
            for i in order:
                fin[i] = best[i] + dur[i]
                o, c = off[i], cnt_[i]
                if c > 0:
                    s_ = succ[o:o + c]
                    np.maximum.at(best, s_, fin[i])
            cp = fin.max()
            lvl = np.zeros(ntask, dtype=np.int64)
            for i in order:
                if cnt_[i] > 0: np.maximum.at(lvl, succ[off[i]:off[i] + cnt_[i]], lvl[i] + 1)
            print(f'critical path (sum of measured run times, no hops): {cp / 1e3:.2f} ms = {cp / span:.1%} of span; depth {lvl.max() + 1} tasks; '
                  f'(span - run-only path) / depth = {(span - cp) / (lvl.max() + 1):.1f} us per level of hop+idle')
    per = defaultdict(lambda: [0, 0.0])
    for op, s, e in zip(np.array(tr['op'])[ok], S, E): per[op][0] += 1; per[op][1] += e - s
    print(f'{"op":28s} {"tasks":>8s} {"mean us":>9s} {"sum ms":>9s}')
    for op, (c, s) in sorted(per.items(), key=lambda kv: -kv[1][1])[:10]: print(f'{op:28s} {c:8d} {s / c:9.1f} {s / 1e3:9.1f}')


if __name__ == '__main__':
    main()
