#!/usr/bin/env python3
"""Gate 2 at any copy count (run_mlp.py's gates() uses 4): the 3-step gate bundle of WORK run once and as N replicated copies by EXE; each
copy's every buffer must be bitwise the single run's: a run's arithmetic must depend on its shape alone, at any copy count.

  python3 -I rep_check.py TREE WORK EXE N --hidden H --depth D
Prints the fuse report line of each run and the copies that differ (buffer names), exit 1 if any does.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('tree', type=Path); ap.add_argument('work', type=Path); ap.add_argument('exe'); ap.add_argument('n', type=int)
ap.add_argument('--hidden', type=int, required=True); ap.add_argument('--depth', type=int, required=True)
a = ap.parse_args()
sys.path.insert(0, str(a.tree / 'bench/exec_vs_torch'))
from run_mlp import run_exec, exec_flags  # noqa: E402
from replicate import replicate  # noqa: E402
import json  # noqa: E402

gate = a.work / 'gate' / 'bundle'
if not gate.exists():
    subprocess.run([sys.executable, '-I', str(a.tree / 'bench/exec_vs_torch/gen_mlp.py'), str(a.work / 'gate'), '--steps', '3', '--no-golden', '--tiling', 'a100',
                    '--hidden', str(a.hidden), '--depth', str(a.depth)], check=True, capture_output=True)
meta = json.loads((gate / 'metadata.json').read_text())
names = [b['name'] for b in sorted(meta['buffers'], key=lambda b: b['index'])]
_, one = run_exec(a.exe, gate, [], dump=True)
rep = a.work / f'gate_rep{a.n}'
if not rep.exists():
    replicate(gate, rep, a.n)
_, d = run_exec(a.exe, rep, exec_flags(meta['task_count'] * a.n * 2), dump=True)
nb = len(one)
assert len(d) == nb * a.n, (len(d), nb)
bad = {}
for k in range(a.n):
    diff = [names[i] if i < len(names) else str(i) for i in range(nb) if d[k * nb + i] != one[i]]
    if diff:
        bad[k] = diff
print(f'{a.n} copies of h{a.hidden}-d{a.depth}: {len(bad)} differ from the single run' + (f'; copy 0 differs in {len(bad.get(0, []))} of {nb} buffers, first {bad.get(0, [])[:4]}' if bad else ' (every buffer bitwise)'))
sys.exit(1 if bad else 0)
