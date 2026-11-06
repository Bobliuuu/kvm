#!/usr/bin/env python3
"""Check a kvm_exec --dump-buffers run of a bf16-storage gpt_tiny bundle.

  check.py BUNDLE_DIR DUMP [--golden]      (BUNDLE_DIR holds workload.json and bundle/)

Always: every bf16 buffer holds only bf16-representable values. With --golden (short bundles): losses and the step-1 activation /
gradient buffers against the CPU golden (bf16 storage: a value one rounding-boundary away is one bf16 ulp, 2^-8, off).
Exit status 0 = PASS."""
import json
import math
import struct
import sys
from pathlib import Path

TOL_LOSS, TOL_ACT = 5e-3, 2e-2


def rel_l2(x, ref):
    num = sum((a - b) ** 2 for a, b in zip(x, ref))
    den = sum(b * b for b in ref)
    return math.sqrt(num / den) if den > 0 else math.sqrt(num)


def main():
    d = Path(sys.argv[1])
    spec = json.loads((d / 'workload.json').read_text())
    meta = json.loads((d / 'bundle' / 'metadata.json').read_text())
    names = [b['name'] for b in meta['buffers']]
    gpu = {}
    for line in Path(sys.argv[2]).read_text().splitlines():
        if line.startswith('buffer '):
            head, rest = line.split(':', 1)
            gpu[names[int(head.split()[1])]] = [float(x) for x in rest.split()]
    bf = [b['name'] for b in spec['buffers'] if b.get('dtype') == 'bf16']
    bad = [n for n in bf if any(struct.unpack('<I', struct.pack('<f', v))[0] & 0xFFFF for v in gpu[n])]
    fails = []
    print(f'{len(bf)} bf16 buffers; non-representable values in: {bad or "none"}')
    if bad:
        fails.append('bf16 representable')
    nsteps = sum(1 for n in gpu if n.startswith('loss.'))
    print('GPU losses: ' + ' '.join(f'{gpu[f"loss.{s}"][0]:.5f}' for s in range(min(nsteps, 60))))
    if '--golden' in sys.argv:
        gold = json.loads((d / 'bundle' / 'golden.json').read_text())['buffers']
        for s in range(nsteps):
            e = rel_l2(gpu[f'loss.{s}'], gold[f'loss.{s}'])
            print(f'  loss.{s} gpu {gpu[f"loss.{s}"][0]:.6f} golden {gold[f"loss.{s}"][0]:.6f} rel {e:.2e}')
            if e > TOL_LOSS:
                fails.append(f'loss.{s}')
        if nsteps == 1:
            skip = ('w.', 'm.', 'v.', 'wb.', 'opt_step', 'version_token', 'lr', 'stream_')
            errs = sorted(((rel_l2(gpu[n], gold[n]), n) for n in gold if n in gpu and not n.startswith(skip) and not n.startswith('loss') and n != 'clip_coef' and gpu[n] and any(gold[n])), reverse=True)
            print('  step-1 buffers vs golden, worst: ' + ', '.join(f'{n} {e:.2e}' for e, n in errs[:5]))
            if errs and errs[0][0] > TOL_ACT:
                fails.append('step-1 buffers')
    print('PASS' if not fails else 'FAIL: ' + ', '.join(fails))
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
