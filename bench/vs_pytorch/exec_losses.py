#!/usr/bin/env python3
"""Per-step losses of a kvm_exec run on a gen.py bundle, and their difference to a torch_bench result.

  python3 exec_losses.py WORKLOAD_DIR ./kvm_exec [--torch torch_eager.json --mode eager] [--out losses.json]

kvm_exec --dump-buffers prints every buffer as text; this streams the output and keeps only the pinned `loss.<step>` buffers (element 0 is the
mean cross-entropy), so a 162M-parameter bundle needs no disk for the dump. Relative difference per step = |exec - torch| / torch.
"""
import argparse
import json
import re
import subprocess
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
    got, last = {}, ''
    with subprocess.Popen([a.exe, '--quiet', '--dump-buffers', str(bundle)], stdout=subprocess.PIPE, text=True) as p:
        for line in p.stdout:
            m = re.match(r'buffer (\d+):', line)
            if m:
                if int(m.group(1)) in idx:
                    got[idx[int(m.group(1))]] = float(line.split(':', 1)[1].split()[0])
            else:
                last = line.strip() or last
        rc = p.wait()
    if rc != 0 or not last.startswith('OK'):
        raise SystemExit(f'executor failed: rc {rc}: {last}')
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
