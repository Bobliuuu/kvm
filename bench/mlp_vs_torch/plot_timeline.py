#!/usr/bin/env python3
"""Timeline figure (goal 4) from the VM's OWN per-instruction timestamps (kvm.cu --timeline).

  python3 plot_timeline.py timeline_R1_tiles.csv [more.csv ...] --out timeline.png [--steps 100:102]

One lane per (SM, block). Colored bar = instruction executing (deps ready -> end); light bar before it = the block holding the
instruction while it waited for its dependencies (claim -> ready). Times are %globaltimer ns relative to the run's first claim.
"""
import argparse
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument('csv', nargs='+')
ap.add_argument('--out', required=True)
ap.add_argument('--steps', default=None, help='a:b step window to draw (default: all rows in the file)')
a = ap.parse_args()
COL = ['#3b6fb6', '#e08a1e', '#2f9e6b', '#c0392b', '#8e5bb5', '#7f8c8d']


def load(path):
    head = open(path).readline()
    m = re.search(r'phases ((?:\d=\w+ ?)+)', head)
    phases = dict(kv.split('=') for kv in m.group(1).split()) if m else {}
    prog = re.search(r'prog=(\w+)', head)
    d = np.loadtxt(path, delimiter=',', comments='#')
    return d, phases, prog.group(1) if prog else '?'


fig, axes = plt.subplots(len(a.csv), 1, figsize=(13, 4.2 * len(a.csv)), squeeze=False)
for ax, path in zip(axes[:, 0], a.csv):
    d, phases, prog = load(path)
    if a.steps:
        s0, s1 = map(int, a.steps.split(':'))
        d = d[(d[:, 1] >= s0) & (d[:, 1] < s1)]
    smid, blk = d[:, 5].astype(int), d[:, 6].astype(int)
    lanes = {k: i for i, k in enumerate(sorted(set(zip(smid, blk))))}
    t0 = d[:, 7].min()
    for row in d:
        lane = lanes[(int(row[5]), int(row[6]))]
        claim, ready, end = (row[7] - t0) / 1e3, (row[8] - t0) / 1e3, (row[10] - t0) / 1e3
        ax.barh(lane, ready - claim, left=claim, height=0.8, color='#d5d8dc', linewidth=0)
        ax.barh(lane, end - ready, left=ready, height=0.8, color=COL[int(row[3]) % len(COL)], linewidth=0)
    ax.set_ylim(-1, len(lanes))
    ax.invert_yaxis()
    ax.set_xlabel('time since first claim in window (us)')
    ax.set_ylabel(f'{len(lanes)} lanes (SM, block)')
    ntask = len(d)
    span = (d[:, 10].max() - t0) / 1e3
    steps = sorted(set(d[:, 1].astype(int)))
    ax.set_title(f'{path.split("/")[-1]}: prog={prog}, steps {steps[0]}-{steps[-1]}, {ntask} instructions, window {span:.1f} us '
                 f'({span / len(steps):.1f} us per step)', fontsize=10, loc='left')
    handles = [plt.Rectangle((0, 0), 1, 1, color='#d5d8dc')] + [plt.Rectangle((0, 0), 1, 1, color=COL[int(k) % len(COL)]) for k in phases]
    ax.legend(handles, ['waiting for dependencies'] + [phases[k] for k in phases], loc='upper right', fontsize=8, ncol=3)
fig.tight_layout()
fig.savefig(a.out, dpi=130)
print('wrote', a.out)
