#!/usr/bin/env python3
"""Critical-path / admission decomposition from a kvm_exec --trace-csv and its bundle (no recycling: slot == task index).
usage: hopan.py BUNDLE_DIR TRACE_CSV"""
import csv, struct, sys, statistics as st
b, tcsv = sys.argv[1], sys.argv[2]
raw = open(b + '/tasks.bin', 'rb').read(); n = len(raw) // 64
succ = struct.unpack('<%di' % (len(open(b + '/successors.bin', 'rb').read()) // 4), open(b + '/successors.bin', 'rb').read())
T = [struct.unpack_from('<16i', raw, 64 * i) for i in range(n)]
rows = {}
for r in csv.DictReader(open(tcsv)): rows[int(r['slot'])] = r
adm = [float(rows[i]['admit_us']) for i in range(n)]; rdy = [float(rows[i]['ready_us']) for i in range(n)]
sta = [float(rows[i]['start_us']) for i in range(n)]; end = [float(rows[i]['end_us']) for i in range(n)]
last = [-1] * n; lastend = [0.0] * n
for i in range(n):
    for k in range(T[i][2]):
        s = succ[T[i][1] + k]
        if end[i] >= lastend[s]: lastend[s] = end[i]; last[s] = i
roots = [i for i in range(n) if last[i] < 0]
nl = sum(1 for i in range(n) if last[i] >= 0)
admlim = sum(1 for i in range(n) if last[i] >= 0 and adm[i] > lastend[i])
run = [end[i] - sta[i] for i in range(n)]
c2r = [rdy[i] - lastend[i] for i in range(n) if last[i] >= 0]       # last pred done -> enqueued
r2s = [sta[i] - rdy[i] for i in range(n)]                           # enqueued -> started
hop = [sta[i] - lastend[i] for i in range(n) if last[i] >= 0]
print('tasks %d roots %d total_span_us %.1f' % (n, len(roots), max(end) - min(adm)))
print('tasks admitted AFTER their last predecessor finished: %d / %d (%.1f%%)' % (admlim, nl, 100.0 * admlim / max(nl, 1)))
q = lambda v, p: sorted(v)[int(p * (len(v) - 1))]
for nm, v in (('run_us', run), ('pred_done->enqueue_us', c2r), ('enqueue->start_us', r2s), ('hop_us(pred_done->start)', hop)):
    print('%-26s median %.2f  p90 %.2f  mean %.2f' % (nm, st.median(v), q(v, .9), st.mean(v)))
# critical path: walk back from the last-ending task
i = max(range(n), key=lambda k: end[k]); path = []
while i >= 0: path.append(i); i = last[i]
sr = sum(run[k] for k in path); sh = sum(sta[k] - lastend[k] for k in path if last[k] >= 0)
print('critical path: %d tasks, sum run %.1f us, sum hops %.1f us, sum %.1f us of span %.1f' % (len(path), sr, sh, sr + sh, max(end) - min(adm)))
print('admission rate: admit window %.1f us for %d tasks = %.2f us/task' % (max(adm) - min(adm), n, (max(adm) - min(adm)) / n))
