#!/usr/bin/env python3
"""Assemble the results table from a run.sh log directory: python3 examples/models/report.py LOGDIR > table.md

Per case: model, steps, tasks, loss curve (GPU), max error vs the CPU golden, kvm_exec wall time (best of 4 launches: the
checked run plus three repeats; the time is the executor's own, from first admission to last completion).
"""
import json
import re
import sys
from pathlib import Path


def times(d):
    out = []
    for f in ('dump.txt', 'times.txt'):
        p = d / f
        if p.exists():
            out += [float(m.group(1)) for m in re.finditer(r'OK: \d+/\d+ tasks, ([\d.]+) ms', p.read_text()[-4000:] if f == 'dump.txt' else p.read_text())]
    return out


def curve(chk):
    names = sorted((n for n in chk['loss_gpu'] if n.startswith('loss.')), key=lambda n: [int(x) for x in n.split('.')[1:]])
    steps = sorted({int(n.split('.')[1]) for n in names})
    vals = [sum(chk['loss_gpu'][n] for n in names if int(n.split('.')[1]) == s) / sum(1 for n in names if int(n.split('.')[1]) == s) for s in steps]
    pick = sorted({0, len(vals) // 4, len(vals) // 2, 3 * len(vals) // 4, len(vals) - 1})
    return ' '.join(f'{vals[i]:.3f}' for i in pick), len(vals)


def main():
    root = Path(sys.argv[1]) / 'bundles'
    print('| case | model | steps | tasks | mode | loss (steps spread over the run) | max error vs CPU golden | exec time ms (min of runs) |')
    print('|---|---|---|---|---|---|---|---|')
    for d in sorted(root.iterdir()):
        if not (d / 'check.json').exists():
            continue
        chk, meta = json.loads((d / 'check.json').read_text()), json.loads((d / 'meta.json').read_text())
        c, n = curve(chk)
        err = chk.get('max_rel_l2') if chk['mode'] == 'precise' else chk.get('max_rel_step1')
        errs = f"{err:.1e} ({'rel L2, all buffers' if chk['mode'] == 'precise' else 'step-1 acts/grads, max|e|/max|ref|'})" if err is not None else '-'
        t = times(d)
        print(f"| {d.name} | {meta['model']} | {meta['steps']} | {meta['tasks']:,} | {chk['mode']}{'+fuse' if meta['fuse'] else ''} | {c} | {errs} | "
              f"{min(t):.2f} (n={len(t)}) |" if t else f"| {d.name} | {meta['model']} | {meta['steps']} | {meta['tasks']:,} | {chk['mode']} | {c} | {errs} | - |")


if __name__ == '__main__':
    main()
