#!/usr/bin/env python3
"""What the executor's time is made of, from kvm_exec --trace-csv: per op kind task count, summed run time, mean run time; the
fraction of worker-time spent running tasks (utilisation), and the mean ready-to-start wait.

  python3 trace_summary.py TRACE.csv STEPS      (workers = distinct blocks that ran a task)
"""
import csv
import sys
from collections import defaultdict


def main():
    path, steps = sys.argv[1], int(sys.argv[2])
    blocks = set()
    per = defaultdict(lambda: [0, 0.0, 0.0])
    start, end, wait, n = 1e18, 0.0, 0.0, 0
    for r in csv.DictReader(open(path)):
        run, q = float(r['run_us']), float(r['queue_wait_us'])
        if run < 0:
            continue
        s, e = float(r['start_us']), float(r['end_us'])
        blocks.add(r['block'])
        start, end, n = min(start, s), max(end, e), n + 1
        wait += max(q, 0.0)
        p = per[r['op']]
        p[0] += 1
        p[1] += run
        p[2] = max(p[2], run)
    span, workers = end - start, len(blocks)
    tot = sum(p[1] for p in per.values())
    print(f'{n} tasks, span {span / 1e3:.2f} ms = {span / steps:.0f} us/step, workers {workers}: task-run time {tot / 1e3:.1f} ms, '
          f'utilisation {tot / (span * workers):.1%}, mean ready->start wait {wait / n:.1f} us, mean task {tot / n:.1f} us')
    print(f'{"op":28s} {"tasks":>8s} {"sum run ms":>11s} {"mean us":>8s} {"max us":>8s} {"% run time":>10s}')
    for op, (c, s, m) in sorted(per.items(), key=lambda kv: -kv[1][1])[:12]:
        print(f'{op:28s} {c:8d} {s / 1e3:11.2f} {s / c:8.1f} {m:8.1f} {s / tot:10.1%}')


if __name__ == '__main__':
    main()
