#!/usr/bin/env python3
"""Where does the time go? From a kvm_exec trace CSV: per 5 ms bucket admitted/started/ended counts and mean tasks in flight,
plus the share of the span with 0 tasks running, split by 'something admitted-but-unstarted existed' vs 'nothing to run'."""
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
A = [float(r['admit_us']) for r in rows]; R = [float(r['ready_us']) for r in rows]; S = [float(r['start_us']) for r in rows]; E = [float(r['end_us']) for r in rows]
t0 = min(A); t1 = max(E); span = t1 - t0
ev = sorted([(s, 1) for s in S] + [(e, -1) for e in E])
# idle accounting by sweep
run = 0; last = t0; idle = 0.0; ev_i = 0
rd = sorted([(r, 1) for r in R] + [(s, -1) for s in S]); ready = 0
pts = sorted(set([x[0] for x in ev] + [x[0] for x in rd]))
import bisect
evt = [x[0] for x in ev]; cum = []; c = 0
for _, d in ev: c += d; cum.append(c)
rdt = [x[0] for x in rd]; cr = []; c = 0
for _, d in rd: c += d; cr.append(c)
idle_ready = idle_noready = 0.0
prev = t0
for p in pts:
    if p > prev:
        i = bisect.bisect_right(evt, prev) - 1; j = bisect.bisect_right(rdt, prev) - 1
        run = cum[i] if i >= 0 else 0; rdy = cr[j] if j >= 0 else 0
        if run == 0:
            if rdy > 0: idle_ready += p - prev
            else: idle_noready += p - prev
    prev = max(prev, p)
print('span %.1f us; time with 0 running: ready-but-not-started %.1f us (%.1f%%), nothing ready %.1f us (%.1f%%)' % (span, idle_ready, 100*idle_ready/span, idle_noready, 100*idle_noready/span))
print('mean tasks in flight %.2f' % (sum(e - s for s, e in zip(S, E)) / span))
for k in range(0, int(span), int(span // 17) + 1):
    lo, hi = t0 + k, t0 + k + int(span // 17) + 1
    f = lambda v: sum(1 for x in v if lo <= x < hi)
    print('t=%7.0f us  admitted %5d  ready %5d  started %5d  ended %5d' % (k, f(A), f(R), f(S), f(E)))
