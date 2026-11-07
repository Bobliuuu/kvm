#!/usr/bin/env python3
"""Device bytes a compiled bundle allocates, by dtype (metadata.json: every buffer is rows x cols elements of its dtype 0 f32 / 1 i32 / 2 bf16).

  python3 bundle_bytes.py BUNDLE_DIR [BUNDLE_DIR ...] [--top N] [--json OUT]

The executor allocates each buffer once (the frontend's memory planner has already folded transient tensors onto shared buffers), so the sum is
the model's device footprint apart from the task ring and the scheduler state.
"""
import argparse
import json
from pathlib import Path

ITEM = {0: 4, 1: 4, 2: 2}
NAME = {0: 'f32', 1: 'i32', 2: 'bf16'}


def summarize(bundle, top):
    meta = json.loads((Path(bundle) / 'metadata.json').read_text())
    by, rows = {}, []
    for b in meta['buffers']:
        n = b['rows'] * b['cols'] * ITEM[b['dtype']]
        by[NAME[b['dtype']]] = by.get(NAME[b['dtype']], 0) + n
        rows.append((n, b['name'], NAME[b['dtype']], b['rows'], b['cols']))
    rows.sort(reverse=True)
    return {'bundle': str(bundle), 'buffers': len(rows), 'bytes_total': sum(by.values()), 'bytes_by_dtype': by,
            'largest': [dict(bytes=n, name=nm, dtype=dt, rows=r, cols=c) for n, nm, dt, r, c in rows[:top]]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bundles', nargs='+')
    ap.add_argument('--top', type=int, default=8)
    ap.add_argument('--json')
    a = ap.parse_args()
    out = [summarize(b, a.top) for b in a.bundles]
    for o in out:
        print(f"{o['bundle']}: {o['bytes_total'] / 2**30:.3f} GiB in {o['buffers']} buffers  " +
              '  '.join(f'{k} {v / 2**30:.3f} GiB' for k, v in sorted(o['bytes_by_dtype'].items())))
        for r in o['largest']:
            print(f"    {r['bytes'] / 2**20:9.1f} MiB  {r['dtype']:4s} {r['rows']}x{r['cols']}  {r['name']}")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
