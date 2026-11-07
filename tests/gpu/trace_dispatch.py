#!/usr/bin/env python3
"""Is the executor dispatch-bound? From kvm_exec --trace-csv (needs the whole run in the trace: --max-tasks >= tasks).

  python3 tests/gpu/trace_dispatch.py TRACE.csv

For every 5 ms bin of the run: tasks that were ready but not started (queue depth, time-averaged), tasks running (time-averaged,
= busy workers), starts per second. If workers are idle while the queue is deep, workers fail to take ready tasks: that is the
dispatch (pop) limit. Prints the stall-time view: among the bins where the average queue depth exceeded the number of workers,
the mean busy-worker count and the start rate (tasks/s); also the ready-to-start wait quantiles.
"""
import csv
import sys

import numpy as np


def main():
    rows = [r for r in csv.DictReader(open(sys.argv[1])) if float(r['run_us']) >= 0]
    ready = np.array([float(r['ready_us']) for r in rows]); start = np.array([float(r['start_us']) for r in rows])
    end = np.array([float(r['end_us']) for r in rows]); wait = start - ready
    workers = len({r['block'] for r in rows})
    t0, t1 = start.min(), end.max(); span = t1 - t0
    B = 5000.0; nb = int(span // B) + 1
    edges = t0 + B * np.arange(nb + 1)
    def occupancy(a, b):  # time-average number of intervals [a,b) overlapping each bin
        out = np.zeros(nb)
        for i in range(nb):
            lo, hi = edges[i], edges[i + 1]
            out[i] = np.clip(np.minimum(b, hi) - np.maximum(a, lo), 0, None).sum() / B
        return out
    queued, running = occupancy(ready, start), occupancy(start, end)
    starts = np.histogram(start, bins=edges)[0] / (B * 1e-6)
    print(f"tasks {len(rows)}, workers {workers}, span {span/1e3:.1f} ms, mean busy {running.mean():.1f}, mean queued {queued.mean():.1f}")
    q = np.quantile(wait, [0.5, 0.9, 0.99, 1.0]); print(f"ready->start wait us: p50 {q[0]:.0f} p90 {q[1]:.0f} p99 {q[2]:.0f} max {q[3]:.0f}")
    deep = queued > workers
    print(f"bins with queue depth > workers: {deep.sum()} of {nb}")
    if deep.any():
        print(f"  in those bins: mean queued {queued[deep].mean():.0f}, mean busy {running[deep].mean():.1f} of {workers}, starts/s {starts[deep].mean():.3g}")
    print(f"overall start rate {len(rows)/(span*1e-6):.3g} tasks/s ({span/len(rows):.2f} us/task)")


if __name__ == '__main__':
    main()
