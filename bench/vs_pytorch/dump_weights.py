#!/usr/bin/env python3
"""Run kvm_exec --dump-buffers on a bundle; keep the buffers named w.* (parameters after the run) and loss.* as float32 files in OUT.
   usage: dumpw.py EXE BUNDLE OUT        compare: dumpw.py --compare OUT_A OUT_B"""
import json, re, subprocess, sys
from pathlib import Path
import numpy as np

if sys.argv[1] == '--compare':
    a, b = Path(sys.argv[2]), Path(sys.argv[3])
    worst = {}
    tot_n = 0
    for f in sorted(a.glob('*.f32')):
        x, y = np.fromfile(f, np.float32), np.fromfile(b / f.name, np.float32)
        d = np.abs(x.astype(np.float64) - y)
        worst[f.stem] = (float(d.max()), float(np.abs(x).max()), int((d > 0).sum()), x.size)
        tot_n += x.size
    allmax = max(v[0] for v in worst.values())
    print(f'{len(worst)} buffers, {tot_n} elements: max|diff| {allmax:.3e}; differing elements {sum(v[2] for v in worst.values())}')
    for k, v in sorted(worst.items(), key=lambda kv: -kv[1][0])[:8]:
        print(f'  {k:28s} max|diff| {v[0]:.3e} (max|w| {v[1]:.3e}) differing {v[2]}/{v[3]}')
    sys.exit(0)

exe, bundle, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
meta = json.loads((bundle / 'metadata.json').read_text())
want = {b['index']: b['name'] for b in meta['buffers'] if re.match(r'(w\.|loss\.)', b['name'])}
last = ''
with subprocess.Popen([exe, '--quiet', '--dump-buffers', str(bundle)], stdout=subprocess.PIPE, text=True, bufsize=1 << 20) as p:
    for line in p.stdout:
        if line.startswith('buffer '):
            i = int(line[7:line.index(':')])
            if i in want:
                np.fromstring(line[line.index(':') + 1:], dtype=np.float32, sep=' ').tofile(out / (want[i] + '.f32'))
        else:
            last = line.strip() or last
    rc = p.wait()
print(last, 'rc', rc, len(want), 'buffers kept')
sys.exit(rc)
