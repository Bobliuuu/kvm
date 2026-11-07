#!/usr/bin/env python3
"""Per-step losses of a kvm_exec run on a gen.py bundle and their difference to a torch_bench result, read with --dump-bin.

  python3 exec_losses_bin.py WORKLOAD_DIR ./kvm_exec [--torch torch.json --mode eager] [--out losses.json]

Same output as bench/vs_pytorch/exec_losses.py, but the executor writes only the pinned `loss.<step>` buffers (--dump-bin DIR indices) instead of
printing every buffer as text (about 11 bytes per element: tens of GB for a 162M-parameter model).
"""
import argparse
import json
import re
import struct
import subprocess
import tempfile
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('workload_dir')
    ap.add_argument('exe')
    ap.add_argument('--torch')
    ap.add_argument('--mode', default='eager')
    ap.add_argument('--out')
    a = ap.parse_args()
    bundle = Path(a.workload_dir) / 'bundle'
    meta = json.loads((bundle / 'metadata.json').read_text())
    idx = {b['index']: int(b['name'].split('.')[1]) for b in meta['buffers'] if re.fullmatch(r'loss\.\d+', b['name'])}
    with tempfile.TemporaryDirectory(prefix='losses_') as tmp:
        r = subprocess.run([a.exe, '--quiet', '--dump-bin', tmp, ','.join(map(str, sorted(idx))), str(bundle)], capture_output=True, text=True)
        last = (r.stdout.strip().splitlines() or [''])[-1]
        if r.returncode != 0 or not last.startswith('OK'):
            raise SystemExit(f'executor failed: rc {r.returncode}: {last} {r.stderr[-300:]}')
        got = {step: struct.unpack('<f', (Path(tmp) / f'{i}.bin').read_bytes()[:4])[0] for i, step in idx.items()}
    losses = [got[i] for i in sorted(got)]
    out = {'executor': losses, 'run': last}
    print('executor:', ' '.join(f'{x:.5f}' for x in losses), '|', last)
    if a.torch:
        t = json.loads(Path(a.torch).read_text())['modes'][a.mode]['losses']
        out['torch'] = t
        out['rel_diff'] = [abs(x - y) / abs(y) for x, y in zip(losses, t)]
        print('torch   :', ' '.join(f'{x:.5f}' for x in t))
        print('rel diff:', ' '.join(f'{x:.2e}' for x in out['rel_diff']))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
