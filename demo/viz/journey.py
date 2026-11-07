#!/usr/bin/env python3
"""Optimization journey and where-the-time-goes figures from demo/viz/data/step_time_journey.json (light + dark PNG).

  python3 demo/viz/journey.py   ->  assets/step-time-journey-{light,dark}.png, assets/time-breakdown-{light,dark}.png

Executor bars are hatched when the change is on an unmerged branch. The 'combined' bar stays an empty dashed outline until its ms is filled in the JSON.
Nothing is measured here; every number is copied from the results README named in the JSON.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
THEMES = {'light': dict(bg='#ffffff', fg='#1f2328', grid='#d0d7de'), 'dark': dict(bg='#0d1117', fg='#e6edf3', grid='#30363d')}
BLUE, ORANGE, GREY, GREEN = '#0072B2', '#E69F00', '#8b949e', '#009E73'


def style(ax, th):
    ax.set_facecolor(th['bg'])
    ax.tick_params(colors=th['fg'], labelsize=8.5)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(th['grid'])
    ax.xaxis.label.set_color(th['fg'])


def journey(d, th, out):
    rows = [(e['label'], e['ms'], e) for e in d['executor']] + [(p['label'], p['ms'], None) for p in d['pytorch']]
    fig, ax = plt.subplots(figsize=(11, 5.6), dpi=150, facecolor=th['bg'])
    style(ax, th)
    ys = list(range(len(rows)))[::-1]
    base = d['executor'][0]['ms']
    for y, (label, ms, e) in zip(ys, rows):
        if ms is None:
            ax.barh(y, base, color='none', edgecolor=th['fg'], ls='--', lw=1, height=0.62)
            ax.text(base / 2, y, 'to be measured', va='center', ha='center', color=th['fg'], fontsize=8)
            continue
        if e is None:
            ax.barh(y, ms, color=GREY, height=0.62)
        elif e['merged']:
            ax.barh(y, ms, color=BLUE, height=0.62)
        else:
            ax.barh(y, ms, color=ORANGE, hatch='///', edgecolor=th['bg'], lw=0, height=0.62)
        txt = '%.0f ms' % ms if ms >= 100 else '%.1f ms' % ms
        if e is not None and e is not d['executor'][0]:
            txt += ('  (%.2fx vs main)' % (base / ms)) if ms < base * 0.98 else '  (no change vs main)'
        ax.text(ms + 12, y, txt, va='center', color=th['fg'], fontsize=8.5)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(0, base * 1.4)
    ax.set_xlabel('milliseconds per optimizer step, gpt_162m, %s (lower is better)' % d['gpu'])
    ax.axhline(len(d['pytorch']) - 0.5, color=th['grid'], lw=1)
    ax.set_title('The 162M GPT step: executor on main (alone vs combined) and unmerged branches, against PyTorch',
                 color=th['fg'], fontsize=10, loc='left')
    ax.legend(handles=[Patch(color=BLUE, label='executor, merged to main'),
                       Patch(facecolor=ORANGE, hatch='///', edgecolor=th['bg'], label='executor, branch, unmerged'),
                       Patch(color=GREY, label='PyTorch')], loc='lower right', frameon=False, fontsize=8, labelcolor=th['fg'])
    fig.text(0.01, 0.01, 'Data: demo/viz/data/step_time_journey.json (copied from each branch\'s results README). alone-rows come from separate holds; ratios are exact only inside one container.',
             color=th['fg'], fontsize=6.8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, facecolor=th['bg'])
    plt.close(fig)


def breakdown(d, th, out):
    w = d['workers']
    bars = d['breakdown']['bars']
    fig, ax = plt.subplots(figsize=(11, 3.6), dpi=150, facecolor=th['bg'])
    style(ax, th)
    colors = {'matmul tiles': BLUE, 'attention': GREEN, 'other tasks': ORANGE}
    for i, b in enumerate(bars[::-1]):
        total = w * b['step_ms'] / 1000
        left = 0
        for k, v in b['busy_cta_s'].items():
            ax.barh(i, v, left=left, color=colors[k], height=0.55, label=k if i == 0 else None)
            left += v
        ax.barh(i, total - left, left=left, color='none', edgecolor=GREY, hatch='...', height=0.55,
                label='idle: waiting on admission / ready-queue pop / dependencies' if i == 0 else None)
        ax.text(total + 1.5, i, '%.0f worker-s, %.0f%% busy%s' % (total, 100 * left / total, ' (approx.)' if b['approx'] else ''), va='center', color=th['fg'], fontsize=8.5)
    ax.set_yticks(range(len(bars)))
    ax.set_yticklabels([b['label'] for b in bars[::-1]])
    ax.set_xlim(0, w * bars[0]['step_ms'] / 1000 * 1.3)
    ax.set_xlabel('worker-seconds per step (%d workers x step time)' % w)
    ax.set_title('Where the time goes: a worker is busy under a fifth of the step; bf16 shrinks the matmul slice, not the idle one',
                 color=th['fg'], fontsize=10, loc='left')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.28), ncol=4, frameon=False, fontsize=8, labelcolor=th['fg'])
    fig.tight_layout()
    fig.savefig(out, facecolor=th['bg'])
    plt.close(fig)


def main():
    d = json.load(open(HERE / 'data/step_time_journey.json'))
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parents[1] / 'assets'
    out.mkdir(parents=True, exist_ok=True)
    for name, th in THEMES.items():
        journey(d, th, out / ('step-time-journey-%s.png' % name))
        breakdown(d, th, out / ('time-breakdown-%s.png' % name))
    print('wrote figures to', out, file=sys.stderr)


if __name__ == '__main__':
    main()
