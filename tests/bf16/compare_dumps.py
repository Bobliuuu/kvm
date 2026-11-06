#!/usr/bin/env python3
"""Compare two kvm_exec --dump-buffers outputs by buffer NAME (indices differ between bundles): every buffer present in both
(weights, optimizer state, losses, gradients) must be bit-identical. Usage: compare_dumps.py BUNDLE_A DUMP_A BUNDLE_B DUMP_B [prefix...]"""
import json
import sys
from pathlib import Path


def load(bundle, dump):
    names = {b['index']: b['name'] for b in json.loads((Path(bundle) / 'metadata.json').read_text())['buffers']}
    out = {}
    for line in Path(dump).read_text().splitlines():
        if line.startswith('buffer '):
            head, vals = line.split(':', 1)
            out[names[int(head.split()[1])]] = vals.split()
    return out


def main():
    ba, da, bb, db, *prefixes = sys.argv[1:]
    prefixes = tuple(prefixes) or ('w.', 'loss.')
    a, b = load(ba, da), load(bb, db)
    common = sorted(n for n in a if n in b and n.startswith(prefixes))
    bad = [n for n in common if a[n] != b[n]]
    print(f'{len(common)} buffers compared ({", ".join(prefixes)}), {len(bad)} differ{": " + ", ".join(bad[:5]) if bad else ""}')
    return 1 if bad or not common else 0


if __name__ == '__main__':
    sys.exit(main())
