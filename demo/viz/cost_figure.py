#!/usr/bin/env python3
"""Per-event costs that decide where a persistent-kernel VM wins or loses (assets/vm-costs.svg; transparent, readable on light and dark pages).

Every number is copied from a recorded result or from AGENTS.md; nothing is measured here. Costs are on different GPUs (labelled), so compare the
order of magnitude, not the digits.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SAVED, ADDED = '#2a9d8f', '#d1495b'
ROWS = [  # (label, microseconds, kind, source)
    ('kernel launch, stream (H100)', 2.1, SAVED, 'AGENTS.md'),
    ('kernel launch, CUDA graph (H100)', 1.3, SAVED, 'AGENTS.md'),
    ('dependent hop through global memory, kvm.cu (A100)', 5.0, ADDED, 'README, kvm.cu findings'),
    ('dependent hop, executor (A100)', 7.0, ADDED, '2026-11-04-a100-exec-recycling'),
    ('dispatch per task, 108 blocks (A100)', 1.38, ADDED, '2026-11-04-a100-exec-recycling'),
    ('admission per task, single admitter (A100)', 1.9, ADDED, '2026-11-04-a100-adaptive'),
]


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[2] / 'assets/vm-costs.svg')
    fg = '#8b949e'
    fig, ax = plt.subplots(figsize=(9, 3.4))
    fig.patch.set_alpha(0); ax.patch.set_alpha(0)
    ys = list(range(len(ROWS)))[::-1]
    ax.barh(ys, [r[1] for r in ROWS], color=[r[2] for r in ROWS], height=0.62)
    for y, r in zip(ys, ROWS):
        ax.text(r[1] + 0.1, y, '%.2g us   (%s)' % (r[1], r[3]), va='center', color=fg, fontsize=7.5)
    ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in ROWS], color=fg, fontsize=8.5)
    ax.set_xlim(0, 12); ax.set_xlabel('microseconds per event', color=fg, fontsize=8.5)
    ax.tick_params(colors=fg)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(fg)
    ax.set_title('Green: what a kernel boundary costs (a VM removes it). Red: what the VM adds.', color=fg, fontsize=9.5, loc='left')
    fig.tight_layout()
    fig.savefig(out, transparent=True, metadata={'Date': None})
    print('wrote', out, file=sys.stderr)


if __name__ == '__main__':
    main()
