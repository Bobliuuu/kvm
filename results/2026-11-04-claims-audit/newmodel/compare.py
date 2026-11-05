#!/usr/bin/env python3
"""compare.py CASE_DIR DUMP: GPU buffers (kvm_exec --dump-buffers) vs the compiler's CPU golden: relative L2 per f32 buffer."""
import json, math, sys
from pathlib import Path
d = Path(sys.argv[1]); meta = json.loads((d / 'bundle/metadata.json').read_text()); gold = json.loads((d / 'bundle/golden.json').read_text())
names = [b['name'] for b in meta['buffers']]; gpu = {}
for line in Path(sys.argv[2]).read_text().splitlines():
    if line.startswith('buffer '):
        h, r = line.split(':', 1); gpu[names[int(h.split()[1])]] = [float(x) for x in r.split()]
rel = lambda x, r: math.sqrt(sum((a - b) ** 2 for a, b in zip(x, r)) / max(sum(b * b for b in r), 1e-30)) if r else 0.0
f32 = [n for n in gold['buffers'] if gold['dtypes'][n] == 'f32']
missing = [n for n in f32 if n not in gpu or len(gpu[n]) != len(gold['buffers'][n])]
res = {n: rel(gpu[n], gold['buffers'][n]) for n in f32 if n not in missing}
grp = lambda p: max([v for n, v in res.items() if n.startswith(p)] or [float('nan')])
loss = [gpu[f'loss.{s}'][0] for s in range(50) if f'loss.{s}' in gpu]; gl = [gold['buffers'][f'loss.{s}'][0] for s in range(50) if f'loss.{s}' in gold['buffers']]
print(f'{d.name}: f32 buffers {len(f32)}, missing/mis-sized {len(missing)}; max rel L2: weights {grp("w."):.2e}, loss.* {grp("loss."):.2e}, all {max(res.values()):.2e}')
print('  loss GPU', [round(x, 5) for x in loss], 'CPU', [round(x, 5) for x in gl])
ok = not missing and grp('w.') < 6e-3 and grp('loss.') < 6e-3 and loss[-1] < loss[0]
print('  PASS' if ok else '  FAIL'); sys.exit(0 if ok else 1)
