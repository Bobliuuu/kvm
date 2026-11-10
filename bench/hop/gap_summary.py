#!/usr/bin/env python3
"""The dispatch-gap phases of a -DKVM_PROF_GAP build, from its --trace-csv (gp columns, written by kvm_write_trace_csv):

  gap_summary.py TRACE.csv

Each phase is measured by thread 0 of one block between two tasks it ran, in SM cycles (KVM_PG in kvm_exec.cuh). The clock is read from the trace itself, by comparing a block's
cycle counter with its %globaltimer across tasks, because the SM runs at one of two clocks (1095 or 1410 MHz) that the phases must be converted with. Tasks are split by how they
were started: continuation (the block's previous task released them) or queue (a pop).
"""
import csv
import sys
from collections import defaultdict

import numpy as np

PHASES = ('publish', 'release', 'sched_release', 'p3', 'p4', 'p5', 'pop_or_cont', 'desc', 'start_book')
SHOWN = (0, 1, 2, 6, 8)
# inside a phase: the cycle, after the phase began, at which the slowest lane finished the sub-step (columns gp9.. of the trace; 0 when no lane reached it)
SUBS = (('succ_id', 'sub_succ_id'), ('atomic', 'sub_atomic'), ('enqueue', 'sub_enqueue'), ('desc', 'sub_desc'), ('post', 'sub_post'), ('errv', 'sub_errv'))
MIN_SPAN_US = 50.0   # two tasks of one block this far apart (1,024 ns ticks of %globaltimer) give the clock to better than 1%


def load(path: str):
    rows, by_block = [], defaultdict(list)
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if int(r['gp_valid']) != 1:
                continue
            rows.append([float(r[f'gp{k}']) for k in range(9)] + [int(r['cont'])] + [float(r[name]) for _, name in SUBS])
            by_block[(int(r['smid']), int(r['block']))].append((float(r['start_us']), int(r['gp_cycle'])))
    return np.array(rows), by_block


def sm_ghz(by_block: dict) -> tuple[float, float, int]:
    """Median and spread of the clock over the blocks: cycles between a block's first and last task, over the microseconds between them."""
    rates = []
    for pts in by_block.values():
        (t0, c0), (t1, c1) = min(pts), max(pts)
        if t1 - t0 >= MIN_SPAN_US:
            rates.append(((c1 - c0) & 0xffffffff) / ((t1 - t0) * 1e3))
    if not rates:
        raise SystemExit('no two tasks of one block are far enough apart to read the clock')
    r = np.array(rates)
    return float(np.median(r)), float(np.percentile(r, 95) - np.percentile(r, 5)), len(r)


def main() -> int:
    a, by_block = load(sys.argv[1])
    ghz, spread, n = sm_ghz(by_block)
    print(f'{len(a)} tasks with a profiled gap; SM clock {ghz * 1e3:.0f} MHz (p5-p95 spread {spread * 1e3:.0f} MHz over {n} pairs)')
    for name, m in (('all', np.ones(len(a), bool)), ('continuation', a[:, 9] == 1), ('queue', a[:, 9] == 0)):
        if not m.any():
            continue
        us = a[m, :9] / (ghz * 1e3)
        cells = '  '.join(f'{PHASES[k]}={us[:, k].mean():.2f}' for k in SHOWN)
        print(f'  {name:<13} n={m.sum():>6}  {cells}  total={us.sum(axis=1).mean():.2f} us')
        subs = a[m, 10:] / (ghz * 1e3)
        print('      sub-steps (us after the phase began, slowest lane, mean over tasks where the sub-step ran): ' + '  '.join(f'{label}={subs[:, j][subs[:, j] > 0].mean():.2f}' if (subs[:, j] > 0).any() else f'{label}=-' for j, (label, _) in enumerate(SUBS)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
