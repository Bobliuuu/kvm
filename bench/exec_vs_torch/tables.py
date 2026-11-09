#!/usr/bin/env python3
"""Markdown tables from a box_job.sh output directory (the *.json files): best [median .. worst] us/step and variant / executor ratios.

  python3 tables.py RAW_DIR > tables.md
"""
import json
import sys
from pathlib import Path

ORDER = ['eager', 'compile', 'compile_graph', 'seq_graph', 'multi_graph', 'batched', 'batched_compile', 'graph', 'compile_mat', 'executor']
NAMES = {'seq_graph': 'graphs, 1 stream', 'multi_graph': 'graphs, R streams', 'batched': 'batched graph', 'batched_compile': 'batched compile+graph',
         'compile_graph': 'compile + graph', 'compile': 'compile (no graph)', 'compile_mat': 'compile (no graph)', 'graph': 'whole-step graph'}


def cell(d):
    return f'{d["best"]:.1f} [{d["median"]:.1f} .. {d["worst"]:.1f}]'


def table(rows, key):
    cols = [c for c in ORDER if any(c in r['us_per_step'] for r in rows)]
    out = ['| ' + key + ' | ' + ' | '.join(NAMES.get(c, c) for c in cols) + ' |', '|' + '---|' * (len(cols) + 1)]
    for r in rows:
        out.append(f'| {r[key]} | ' + ' | '.join(cell(r['us_per_step'][c]) if c in r['us_per_step'] else '-' for c in cols) + ' |')
    out += ['', 'Ratio = variant best / executor best (above 1: the executor is faster; below 1: the variant is faster).', '',
            '| ' + key + ' | ' + ' | '.join(NAMES.get(c, c) for c in cols if c != 'executor') + ' |', '|' + '---|' * len(cols)]
    for r in rows:
        ex = r['us_per_step']['executor']['best']
        out.append(f'| {r[key]} | ' + ' | '.join(f'{r["us_per_step"][c]["best"] / ex:.2f}x' if c in r['us_per_step'] else '-' for c in cols if c != 'executor') + ' |')
    return out


def main():
    d = Path(sys.argv[1])
    for f in sorted(d.glob('mlp_*.json')):
        j = json.loads(f.read_text())
        a = j['args']
        print(f'### {f.stem}: hidden {a["hidden"]}, depth {a["depth"]}, {a["steps"]} steps, best of {a["reps"]} [median .. worst], us per step (all R runs advance one step)\n')
        rows = j['rows']
        for r in rows:
            r['R'] = r['runs']
        print('\n'.join(table(rows, 'R')))
        g = j['gate']
        print(f'\nGate: executor vs eager loss max rel {g["exec_vs_eager_loss_max_rel"]:.2e}, weights max rel L2 {g["exec_vs_eager_weight_rel_l2_max"]:.2e}; '
              f'vs float64: executor {g["fp64_loss_max_rel"]["executor"]:.2e}, eager {g["fp64_loss_max_rel"]["eager"]:.2e}; '
              f'loss {g["loss_first_last"][0]:.4f} -> {g["loss_first_last"][1]:.4f}; tasks/step/run {rows[0]["tasks_per_step_per_run"]}\n')
    for f in sorted(d.glob('gpt_*.json')):
        j = json.loads(f.read_text())
        info = j['info']
        print(f'### {f.stem}: {info["params"]} params, {info["tokens_per_step"]} tokens/step, {j["steps"]} steps, us per optimizer step, best [median .. worst]\n')
        ex = j['us_per_step']['executor']['best']
        print('| implementation | us/step | tokens/s | executor / this |\n|---|---|---|---|')
        for m, v in j['us_per_step'].items():
            print(f'| {NAMES.get(m, m)} | {cell(v)} | {info["tokens_per_step"] / v["best"] * 1e6:,.0f} | {ex / v["best"]:.2f}x |')
        g = j['gate']
        print(f'\nGate: executor vs eager loss max rel {g["executor_vs_eager_max_rel"]:.2e}; modes vs eager {({k: float("%.2e" % v) for k, v in g["modes_vs_eager_max_rel"].items()})}\n')


if __name__ == '__main__':
    main()
