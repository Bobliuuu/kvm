#!/usr/bin/env python3
"""The critical path of a kvm_exec run, from its --trace-csv and its bundle: how much of the path is op time and how much is the hop between tasks.

  python3 crit_path.py TRACE.csv BUNDLE_DIR [STEPS]

The path is walked backward from the task that ended last, each time to the predecessor that ended last (the one that released this task). A hop's gap is
start(task) - end(its last predecessor): release, the queue or a continuation, and the descriptor load. Times are the trace's (%globaltimer, 1,024 ns ticks: read sums, not single gaps).
"""
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    trace_path, bundle = sys.argv[1], Path(sys.argv[2])
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    tasks = np.fromfile(bundle / 'tasks.bin', dtype=np.int32).reshape(-1, 16)
    succ = np.fromfile(bundle / 'successors.bin', dtype=np.int32)
    n = len(tasks)
    start, end, ready, op, block = (np.zeros(n) for _ in range(5))
    cont = np.zeros(n, dtype=bool)
    with open(trace_path) as fh:
        for r in csv.DictReader(fh):
            s = int(r['slot'])
            start[s], end[s], ready[s], op[s], block[s] = float(r['start_us']), float(r['end_us']), float(r['ready_us']), hash(r['op']) & 0xffff, float(r['block'])
            cont[s] = int(r.get('cont', 0)) != 0
    # the predecessor of each task that finishes last
    pred_last = np.full(n, -1)
    for i in range(n):
        so, sc = tasks[i, 1], tasks[i, 2]
        for s in succ[so:so + sc]:
            if pred_last[s] < 0 or end[i] > end[pred_last[s]]:
                pred_last[s] = i
    gaps = np.array([start[i] - end[pred_last[i]] if pred_last[i] >= 0 else np.nan for i in range(n)])
    run = end - start
    i = int(np.argmax(end))
    path = [i]
    while pred_last[i] >= 0:
        i = int(pred_last[i])
        path.append(i)
    path.reverse()
    p = np.array(path)
    span = end[p[-1]] - start[p[0]]
    run_sum, gap_sum = run[p].sum(), gaps[p[1:]].sum()
    same = (block[p[1:]] == block[p[:-1]]).mean() if len(p) > 1 else 0.0
    print(f'{n} tasks, {steps} steps; critical path {len(p)} tasks ({len(p) / steps:.1f} per step), span {span:.0f} us ({span / steps:.1f} us/step)')
    print(f'  on the path: op time {run_sum:.0f} us ({run_sum / len(p):.2f} us/task), hop gaps {gap_sum:.0f} us ({gap_sum / max(1, len(p) - 1):.2f} us/hop); hops on the same block {same:.0%}')
    print(f'  all tasks: run p50 {np.nanmedian(run):.1f} us, gap p50 {np.nanmedian(gaps):.1f} us; continuation flag set on {cont.mean():.0%} of tasks')
    return 0


if __name__ == '__main__':
    sys.exit(main())
