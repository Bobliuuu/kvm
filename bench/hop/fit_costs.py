#!/usr/bin/env python3
"""Fit one global table of op costs for the static scheduler, from per-record traces (kvm_exec --static --trace-csv FILE).

  fit_costs.py --fit TRACE.csv:BUNDLE[,TRACE.csv:BUNDLE ...] [--check TRACE.csv:BUNDLE ...] [--out TABLE] [--sig include_dir]

Table format, one op per line: op_name fixed_us per_mflop_us. The cost of a task is fixed_us + per_mflop_us * MFLOP, with MFLOP = 2 * min(64, M) * min(64, N) * K / 1e6 for matmul_tile
(args 8, 9, 10) and 0 for every other op. --check prints, for rows the fit never saw, the mean absolute error of the predicted op time and the ratio of predicted to measured total.
Ops whose mean moves more than x2 across the fit's traces are named, with the i32 argument that best correlates with the run time.
"""
import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

MATMUL = 'matmul_tile'


def op_table(include: Path) -> dict:
    names, sig = {}, {}
    for f in include.glob('*.def'):
        for m in re.finditer(r'X\((\w+),\s*(\d+),\s*"([^"]*)"', f.read_text()):
            names[int(m.group(2))], sig[m.group(1)] = m.group(1), m.group(3).split(',')
    return names, sig


def load(trace: str, bundle: str, names: dict) -> dict:
    """op -> array of (run_us, mflop, per-arg matrix)."""
    tasks = np.fromfile(Path(bundle) / 'tasks.bin', dtype=np.int32).reshape(-1, 16)
    args = np.fromfile(Path(bundle) / 'args.bin', dtype=np.int32)
    per_op = defaultdict(list)
    with open(trace) as fh:
        for r in csv.DictReader(fh):
            if r['op'] == '(wait)':
                continue
            t = int(r['task'])
            per_op[r['op']].append((t, (int(r['done_ns']) - int(r['ready_ns'])) / 1e3))
    out = {}
    for op, rows in per_op.items():
        ids = np.array([x[0] for x in rows]); run = np.array([x[1] for x in rows])
        ao = tasks[ids, 6]
        words = int(tasks[ids[0], 7])
        arg = np.stack([args[ao + j] for j in range(words)], axis=1) if words else np.zeros((len(ids), 0), dtype=np.int32)
        mflop = np.zeros(len(ids))
        if op == MATMUL:
            mflop = 2.0 * np.minimum(64, arg[:, 8]) * np.minimum(64, arg[:, 9]) * arg[:, 10] / 1e6
        out[op] = (run, mflop, arg)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fit', required=True, help='TRACE:BUNDLE,TRACE:BUNDLE,... (the rows the table is fitted on)')
    ap.add_argument('--check', default='', help='TRACE:BUNDLE,... (rows not fitted; predictions are scored on them)')
    ap.add_argument('--out', default='', help='write the table here')
    ap.add_argument('--sig', default='', help='include dir with the *.def files (for argument names)')
    a = ap.parse_args()
    names, sig = op_table(Path(a.sig)) if a.sig else ({}, {})
    fit_sets = [load(*p.split(':'), names) for p in a.fit.split(',')]
    ops = sorted({op for s in fit_sets for op in s})
    table = {}
    print(f'{"op":<24}{"n":>9}{"fixed_us":>10}{"per_MFLOP":>11}{"mean_us":>9}   per-trace means')
    for op in ops:
        run = np.concatenate([s[op][0] for s in fit_sets if op in s]); mf = np.concatenate([s[op][1] for s in fit_sets if op in s])
        if op == MATMUL and mf.max() > mf.min():
            A = np.vstack([np.ones_like(mf), mf]).T
            fixed, slope = np.linalg.lstsq(A, run, rcond=None)[0]
            fixed, slope = float(max(fixed, 0.0)), float(max(slope, 0.0))
        else:
            fixed, slope = float(run.mean()), 0.0
        table[op] = (fixed, slope)
        means = [f'{s[op][0].mean():.1f}' for s in fit_sets if op in s]
        vals = [s[op][0].mean() for s in fit_sets if op in s]
        spread = max(vals) / max(min(vals), 1e-3) if len(vals) > 1 else 1.0
        print(f'{op:<24}{len(run):>9}{fixed:>10.2f}{slope:>11.3f}{run.mean():>9.1f}   {" ".join(means)}{"   MOVES x%.1f" % spread if spread > 2 else ""}')
        if spread > 2 and sig.get(op):
            arg = np.concatenate([s[op][2] for s in fit_sets if op in s])
            best, bj = 0.0, -1
            for j, name in enumerate(sig[op]):
                if ':i32' in name and j < arg.shape[1] and arg[:, j].std() > 0:
                    c = np.corrcoef(arg[:, j], run)[0, 1]
                    if abs(c) > best:
                        best, bj = abs(c), j
            if bj >= 0:
                print(f'    driven by arg {bj} ({sig[op][bj]}), corr {best:.2f}')
    if a.out:
        with open(a.out, 'w') as fh:
            for op, (fixed, slope) in sorted(table.items()):
                fh.write(f'{op} {fixed:.3f} {slope:.3f}\n')
        print(f'wrote {a.out} ({len(table)} ops)')
    for p in filter(None, a.check.split(',')):
        s = load(*p.split(':'), names)
        tot_pred, tot_meas, err, n = 0.0, 0.0, 0.0, 0
        worst = []
        for op, (run, mf, _arg) in s.items():
            fixed, slope = table.get(op, (run.mean(), 0.0))
            pred = fixed + slope * mf
            tot_pred += pred.sum(); tot_meas += run.sum(); err += np.abs(pred - run).sum(); n += len(run)
            worst.append((abs(pred.sum() - run.sum()), op, pred.sum() / max(run.sum(), 1e-6)))
        print(f'check {p}: {n} records, predicted/measured total op time {tot_pred / tot_meas:.3f}, mean abs err {err / n:.1f} us/record ({err / max(tot_meas, 1e-6):.1%} of op time)')
        for _d, op, r in sorted(worst, reverse=True)[:6]:
            print(f'    {op:<24} predicted/measured {r:.2f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
