#!/usr/bin/env python3
"""Executor timeline: rows = SMs (workers), x = time, colour = op class, from the executor's own per-task trace.csv.

  python3 demo/viz/timeline.py [--trace results/2026-11-06-demo-tiny/trace.csv.gz] [--out-dir assets]

Writes timeline-light.png and timeline-dark.png (deterministic: no randomness, no clocks). Nothing is drawn that is not a task in the trace.
Each pixel column covers a time slice; a cell's opacity is the fraction of the slice that SM was busy (floor 65% so a single 12 us task stays visible),
its hue the busy-time-weighted mean of the class colours. The x axis is piecewise linear (see WARP): the first T1 ms hold all training windows and
serve requests, the rest is Doom alone, so a linear axis would squeeze everything but Doom into 8% of the width.
"""
import argparse
import csv
import gzip
import json
import sys
from pathlib import Path

import numpy as np

# class order = legend order. (workload 0 train, 1 serve, 2 game in the demo's run.json "classes")
CLASSES = ['train: matmul', 'train: attention', 'train: norm / elementwise / loss', 'train: optimizer (AdamW, grad-norm)',
           'serve (inference on versioned weights)', 'Doom: tick (game logic)', 'Doom: present / status']
COLORS = ['#0072B2', '#009E73', '#56B4E9', '#E69F00', '#D55E9A', '#CC3311', '#F0E442']   # Okabe-Ito based, readable on both backgrounds
ATTN = ('attn_', 'rope_')
OPT = ('adamw', 'gradnorm')
T1_US = 1.0           # axis break (us); set by configure_axis from the data
FRAC_A = 1.0          # share of the axis given to [0, T1]; 1.0 = plain linear axis


def configure_axis(cls, end, end_us):
    """Pick the axis from the data. If everything except Doom (classes 5, 6) is done in the first 40% of the run, give that busy span half the axis
    (break = busy end + 3 ms margin, rounded up to 5 ms); otherwise the axis is linear."""
    global T1_US, FRAC_A
    busy_end = float(end[cls <= 4].max()) if (cls <= 4).any() else end_us
    if busy_end < 0.4 * end_us:
        T1_US, FRAC_A = float(np.ceil((busy_end + 3000.0) / 5000.0) * 5000.0), 0.5
    else:
        T1_US, FRAC_A = end_us, 1.0


def axis_ticks_ms(end_us):
    if FRAC_A < 1.0:
        return [0, 100, 200, 300, int(T1_US / 1000)] + [t for t in (1000, 2000, 3000, 4000, 5000, 6000, 8000, 10000) if t * 1000 < end_us]
    step = 1000 if end_us <= 8e6 else 2000
    return list(range(0, int(end_us / 1000) + 1, step))


def classify(op, workload):
    if workload == 1:
        return 4
    if workload == 2:
        return 5 if op == 'doom_tick' else 6
    if op == 'matmul_tile':
        return 0
    if op.startswith(ATTN):
        return 1
    if op.startswith(OPT):
        return 3
    return 2


def warp(t_us, end_us):
    """Real time (us) -> axis position in [0, 1]; piecewise linear, continuous."""
    t = np.asarray(t_us, dtype=float)
    if FRAC_A >= 1.0:
        return t / T1_US
    return np.where(t <= T1_US, FRAC_A * t / T1_US, FRAC_A + (1 - FRAC_A) * (t - T1_US) / (end_us - T1_US))


def unwarp(x, end_us):
    x = np.asarray(x, dtype=float)
    if FRAC_A >= 1.0:
        return x * T1_US
    return np.where(x <= FRAC_A, x / FRAC_A * T1_US, T1_US + (x - FRAC_A) / (1 - FRAC_A) * (end_us - T1_US))


def load_trace(path):
    if str(path).endswith('.npz'):      # demo/viz/compact_trace.py output
        z = np.load(path)
        return z['start_us'].astype(float), z['end_us'].astype(float), z['smid'].astype(int), z['cls'].astype(int)
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', newline='') as fh:
        rows = [r for r in csv.DictReader(fh)]
    rows = [r for r in rows if float(r['start_us']) >= 0 and float(r['end_us']) >= 0]
    start = np.array([float(r['start_us']) for r in rows])
    end = np.array([float(r['end_us']) for r in rows])
    smid = np.array([int(r['smid']) for r in rows])
    cls = np.array([classify(r['op'], int(r['workload'])) for r in rows])
    return start, end, smid, cls


def rasterize(start, end, smid, cls, sm_ids, end_us, ncols):
    """busy[class, row, col] = fraction of that column's time slice the SM spent on that class."""
    edges_x = np.linspace(0, 1, ncols + 1)
    edges_t = unwarp(edges_x, end_us)
    row_of = {s: i for i, s in enumerate(sm_ids)}
    busy = np.zeros((len(CLASSES), len(sm_ids), ncols))
    for s, e, sm, c in zip(start, end, smid, cls):
        c0 = max(0, int(np.searchsorted(edges_t, s, side='right')) - 1)
        c1 = min(ncols - 1, int(np.searchsorted(edges_t, e, side='left')))
        r = row_of[sm]
        for col in range(c0, c1 + 1):
            ov = min(e, edges_t[col + 1]) - max(s, edges_t[col])
            if ov > 0:
                busy[c, r, col] += ov / (edges_t[col + 1] - edges_t[col])
    return np.clip(busy, 0, 1)


def compose(busy, bg):
    """RGB image: opacity from total busy fraction, hue from the class mix."""
    cols = np.array([[int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)] for h in COLORS])
    tot = busy.sum(axis=0)
    mix = np.tensordot(busy, cols, axes=(0, 0)) / np.maximum(tot, 1e-12)[..., None]
    alpha = np.where(tot > 0, np.clip(0.65 + 0.35 * np.minimum(tot, 1), 0, 1), 0)[..., None]
    return bg * (1 - alpha) + mix * alpha


THEMES = {'light': dict(bg='#ffffff', fg='#1f2328', grid='#d0d7de'), 'dark': dict(bg='#0d1117', fg='#e6edf3', grid='#30363d')}


def draw(theme, busy, sm_ids, end_us, meta, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    th = THEMES[theme]
    bg = np.array([int(th['bg'][i:i + 2], 16) / 255 for i in (1, 3, 5)])
    img = compose(busy, bg)
    fig, ax = plt.subplots(figsize=(13, 6.4), dpi=150, facecolor=th['bg'])
    ax.set_facecolor(th['bg'])
    ax.imshow(img, aspect='auto', interpolation='nearest', extent=(0, 1, len(sm_ids), 0))
    ticks_ms = axis_ticks_ms(end_us)
    ax.set_xticks([float(warp(t * 1000, end_us)) for t in ticks_ms])
    ax.set_xticklabels([str(t) for t in ticks_ms])
    if FRAC_A < 1.0:
        ax.axvline(FRAC_A, color=th['fg'], lw=1, ls=(0, (3, 3)))
        ax.text(FRAC_A, -1.2, 'axis scale changes here', color=th['fg'], fontsize=7.5, ha='center', va='bottom')
        ax.set_xlabel('time since first admission (ms); left half is 0-%.0f ms, right half %.0f-%.0f ms' % (T1_US / 1000, T1_US / 1000, end_us / 1000), color=th['fg'])
    else:
        ax.set_xlabel('time since first admission (ms), linear', color=th['fg'])
    ax.set_ylabel('worker = SM (%d rows, sorted by SM id)' % len(sm_ids), color=th['fg'])
    ax.tick_params(colors=th['fg'])
    for sp in ax.spines.values():
        sp.set_color(th['grid'])
    ax.set_title(meta['title'], color=th['fg'], fontsize=11, loc='left', pad=22)
    handles = [Patch(color=c, label='%s  (%s)' % (n, meta['counts'][i])) for i, (n, c) in enumerate(zip(CLASSES, COLORS))]
    leg = ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=8, frameon=False)
    for t in leg.get_texts():
        t.set_color(th['fg'])
    fig.text(0.01, 0.005, meta['foot'], color=th['fg'], fontsize=7, va='bottom')
    fig.subplots_adjust(left=0.07, right=0.99, top=0.9, bottom=0.27)
    fig.savefig(out, facecolor=th['bg'])
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    here = Path(__file__).resolve().parents[2]
    ap.add_argument('--trace', default=str(here / 'results/2026-11-06-demo-tiny/trace.csv.gz'))
    ap.add_argument('--run-json', default=str(here / 'results/2026-11-06-demo-tiny/run.json'))
    ap.add_argument('--out-dir', default=str(here / 'assets'))
    ap.add_argument('--cols', type=int, default=1300)
    ap.add_argument('--prefix', default='timeline', help='output file prefix: <prefix>-light.png / <prefix>-dark.png')
    ap.add_argument('--subtitle', default='(gpt_tiny: 200 AdamW steps in 20 windows; 5 serve requests; 700 Doom frames; the long bar is Doom boot tick)')
    ap.add_argument('--source', default='results/2026-11-06-demo-tiny/trace.csv.gz')
    a = ap.parse_args()
    start, end, smid, cls = load_trace(a.trace)
    run = json.load(open(a.run_json))
    end_us = float(end.max())
    configure_axis(cls, end, end_us)
    sm_ids = sorted(set(smid.tolist()))
    busy = rasterize(start, end, smid, cls, sm_ids, end_us, a.cols)
    counts = ['%d tasks' % int((cls == i).sum()) for i in range(len(CLASSES))]
    meta = dict(counts=counts,
                title=('One persistent kernel, three concurrent workloads: %s, %d SMs, %d tasks in %.2f s\n'
                       % (run['device'].replace('NVIDIA ', ''), run['sms'], run['tasks_done'], run['kernel_ms'] / 1000)) + a.subtitle,
                foot='Source: ' + a.source + ' (executor per-task timestamps; every task drawn). Opacity = busy fraction of the pixel column (floor 65%). Comm ops: none in this single-GPU run.')
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        draw(theme, busy, sm_ids, end_us, meta, out / ('%s-%s.png' % (a.prefix, theme)))
    print('wrote', out / 'timeline-light.png', out / 'timeline-dark.png', 'rows', len(sm_ids), 'tasks', len(start), file=sys.stderr)


if __name__ == '__main__':
    main()
