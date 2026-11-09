#!/usr/bin/env python3
"""Step-overlap timeline + stats from kvm.cu --timeline CSVs (tiles program).

  python3 plot_overlap.py L0.csv L2.csv --labels "overlap 0" "overlap 2" --out fig.png --stats stats.json [--steps a:b]

Rows = tile instructions grouped by phase (H1, Y, DH1, DW2, DW1) per step; x = %globaltimer ns relative to the first claim of the window.
Bar = instruction executing (deps ready -> end); thin light bar = the block holding the instruction while it waits for its dependencies.
Stats per step boundary s -> s+1: ov = end of the LAST tile of step s minus the FIRST ready of step s+1 (positive = step s+1 started
computing while step s still had a tile running = overlap), period = first ready of s+1 minus first ready of s.
"""
import argparse, json, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument('csv', nargs='+'); ap.add_argument('--labels', nargs='*'); ap.add_argument('--out', required=True)
ap.add_argument('--stats'); ap.add_argument('--steps', default=None)
a = ap.parse_args()
PH = ['H1', 'Y', 'DH1', 'DW2', 'DW1']
COL = ['#3b6fb6', '#e08a1e', '#2f9e6b', '#c0392b', '#8e5bb5']

def load(p):
    d = np.loadtxt(p, delimiter=',', comments='#')
    return d  # idx,step,tmpl,phase,run,smid,block,claim,ready,cend,end,mma0,mma1

def stats(d):
    steps = sorted(set(d[:, 1].astype(int)))
    per = {}
    for s in steps:
        x = d[d[:, 1] == s]
        per[s] = dict(first_ready=x[:, 8].min(), last_end=x[:, 10].max(), first_claim=x[:, 7].min())
    ov, period, span = [], [], []
    for s in steps[:-1]:
        ov.append(per[s]['last_end'] - per[s + 1]['first_ready']); period.append(per[s + 1]['first_ready'] - per[s]['first_ready'])
    for s in steps: span.append(per[s]['last_end'] - per[s]['first_ready'])
    ph = {}
    for p in range(5):
        x = d[d[:, 3] == p]
        ph[PH[p]] = dict(compute_us=float(np.median(x[:, 10] - x[:, 8]) / 1e3), wait_us=float(np.median(x[:, 8] - x[:, 7]) / 1e3))
    # per-step phase end-to-start hops: first_ready of phase p+1 minus last end of phase p (same step, interior steps)
    hops = {}
    for p in range(4):
        v = []
        for s in steps:
            x, y = d[(d[:, 1] == s) & (d[:, 3] == p)], d[(d[:, 1] == s) & (d[:, 3] == p + 1)]
            v.append(y[:, 8].min() - x[:, 10].max())
        hops[f'{PH[p]}->{PH[p+1]}'] = float(np.median(v) / 1e3)
    return dict(steps=[int(steps[0]), int(steps[-1])], overlap_us_median=float(np.median(ov) / 1e3), overlap_us_max=float(np.max(ov) / 1e3),
                period_us_median=float(np.median(period) / 1e3), step_span_us_median=float(np.median(span) / 1e3), phases=ph, hop_us_median=hops)

n = len(a.csv)
fig, axes = plt.subplots(n, 1, figsize=(13, 3.6 * n + 0.6), squeeze=False, sharex=False)
allst = {}
for k, (ax, path) in enumerate(zip(axes[:, 0], a.csv)):
    d = load(path)
    if a.steps:
        s0, s1 = map(int, a.steps.split(':')); d = d[(d[:, 1] >= s0) & (d[:, 1] < s1)]
    st = stats(d); lab = a.labels[k] if a.labels and k < len(a.labels) else path; allst[lab] = st
    t0 = d[:, 8].min() - 3000; d = d.copy(); d[:, 7:11] -= t0
    # lane = (step parity, phase) so successive steps stack: rows labelled by phase, colour by phase, step edges by alpha
    smin = int(d[:, 1].min())
    for r in d:
        s, ph = int(r[1]) - smin, int(r[3]); y = (s % 2) * 6 + (4 - ph) + (r[2] % 1) * 0
        lane = y + 0.15 * ((int(r[2]) % 5) / 5.0)
        ax.barh(lane, (r[8] - r[7]) / 1e3, left=r[7] / 1e3, height=0.08, color='#bbbbbb', linewidth=0)
        ax.barh(lane, (r[10] - r[8]) / 1e3, left=r[8] / 1e3, height=0.12, color=COL[ph], linewidth=0)
    ax.set_yticks([(p % 2) * 6 + (4 - p // 2 * 0) for p in range(0)])
    ticks, labs = [], []
    for par in range(2):
        for ph in range(5): ticks.append(par * 6 + 4 - ph + 0.06); labs.append(f'{PH[ph]} (step {"even" if par == 0 else "odd"})')
    ax.set_yticks(ticks); ax.set_yticklabels(labs, fontsize=7)
    ax.set_xlabel('us'); ax.set_title(f'{lab}: median overlap {st["overlap_us_median"]:.2f} us per step boundary, step period {st["period_us_median"]:.2f} us', fontsize=10)
    ax.grid(axis='x', alpha=.25)
mx = max(ax.get_xlim()[1] for ax in axes[:, 0])
for ax in axes[:, 0]: ax.set_xlim(0, mx)
fig.tight_layout(); fig.savefig(a.out, dpi=150)
if a.stats: json.dump(allst, open(a.stats, 'w'), indent=1)
print(json.dumps(allst, indent=1))
