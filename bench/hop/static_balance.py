#!/usr/bin/env python3
"""Where the time of a static-schedule run went, worker by worker (kvm_exec --static --trace-csv FILE):

  static_balance.py TRACE.csv [STEPS]

For each worker: when its last record finished, the sum of op time (ready -> done) and of wait time (start -> ready), and what is left over (the fetch and the
publish between records). A run is as slow as its last worker: compare the workers' finish times with their busy time to separate imbalance from waiting.
"""
import csv
import sys
from collections import defaultdict

import numpy as np


def main() -> int:
    path = sys.argv[1]
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    w, start, ready, done, pub, nwait = [], [], [], [], [], []
    ops = defaultdict(lambda: [0, 0.0, 0.0])
    with open(path) as fh:
        for r in csv.DictReader(fh):
            s, rd, d, p = int(r['start_ns']), int(r['ready_ns']), int(r['done_ns']), int(r['published_ns'])
            w.append(int(r['worker'])); start.append(s); ready.append(rd); done.append(d); pub.append(p); nwait.append(int(r['nwait']))
            o = ops[r['op']]
            o[0] += 1; o[1] += (d - rd) / 1e3; o[2] += (rd - s) / 1e3
    w, start, ready, done, pub, nwait = (np.array(x, dtype=np.int64) for x in (w, start, ready, done, pub, nwait))
    span = (pub.max() - start.min()) / 1e3
    print(f'{len(w)} records, {steps} steps: span {span:.0f} us = {span / steps:.0f} us/step')
    n = w.max() + 1
    finish = np.zeros(n); run = np.zeros(n); wait = np.zeros(n); count = np.zeros(n)
    for k in range(n):
        m = w == k
        if not m.any():
            continue
        finish[k] = (pub[m].max() - start.min()) / 1e3
        run[k] = (done[m] - ready[m]).sum() / 1e3
        wait[k] = (ready[m] - start[m]).sum() / 1e3
        count[k] = m.sum()
    used = count > 0
    print(f'workers used {used.sum()}/{n}; finish time min {finish[used].min():.0f} p50 {np.median(finish[used]):.0f} max {finish[used].max():.0f} us (span {span:.0f})')
    print(f'op time per worker: min {run[used].min():.0f} p50 {np.median(run[used]):.0f} max {run[used].max():.0f} us; wait per worker: min {wait[used].min():.0f} p50 {np.median(wait[used]):.0f} max {wait[used].max():.0f} us')
    print(f'sum over workers: op {run.sum() / 1e3:.1f} ms, wait {wait.sum() / 1e3:.1f} ms, other {(span * used.sum() - run.sum() - wait.sum()) / 1e3:.1f} ms; '
          f'mean busy share {run[used].mean() / span:.1%}, mean wait share {wait[used].mean() / span:.1%}')
    print(f'{"op":<22}{"n":>9}{"run ms":>10}{"wait ms":>10}{"run us/record":>14}')
    for op, (c, r, wa) in sorted(ops.items(), key=lambda kv: -kv[1][1])[:12]:
        print(f'{op:<22}{c:>9}{r / 1e3:>10.1f}{wa / 1e3:>10.1f}{r / max(c, 1):>14.2f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
