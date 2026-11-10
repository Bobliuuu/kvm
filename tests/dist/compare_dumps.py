#!/usr/bin/env python3
"""Compare `test_dist plan --dump` outputs: every rank of a dp run must hold bit-identical state (replicas), and rank 0 must match
the reference run (one GPU, gradient accumulation over P microbatches) within float32 reduction-order noise.

    compare_dumps.py DP_DUMP_DIR REF_DUMP_DIR [--tol 1e-4] [--json OUT]
"""
import argparse
import json
import math
import struct
import sys
from pathlib import Path


def load(d):
    out = {}
    for line in (Path(d) / 'manifest.txt').read_text().splitlines():
        r, name, rows, cols, *_, fn = line.split()
        raw = (Path(d) / fn).read_bytes()
        out.setdefault(int(r), {})[name] = struct.unpack(f'<{len(raw) // 4}f', raw)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('dp')
    ap.add_argument('ref')
    ap.add_argument('--tol', type=float, default=1e-4)
    ap.add_argument('--json')
    a = ap.parse_args()
    dp, ref = load(a.dp), load(a.ref)
    ranks = sorted(dp)
    split = [n for r in ranks[1:] for n in dp[0] if dp[r].get(n) != dp[0][n]]
    worst, rep = (-1.0, ''), {}
    for n, v in dp[0].items():
        w = ref[0][n]
        den = max(abs(x) for x in w) or 1.0
        e = max(abs(x - y) for x, y in zip(v, w)) / den
        l2 = math.sqrt(sum((x - y) ** 2 for x, y in zip(v, w)) / max(1e-300, sum(y * y for y in w)))
        rep[n] = dict(max_rel=e, rel_l2=l2)
        worst = max(worst, (e, n))
    bad = [n for n in dp[0] if any(math.isnan(x) or math.isinf(x) for x in dp[0][n])]
    ok = not split and worst[0] <= a.tol and not bad
    print(f'{len(ranks)} ranks, {len(dp[0])} state buffers: replicas {"bit-identical" if not split else "DIFFER in " + str(split[:4])}; '
          f'worst max|err|/max|ref| vs reference {worst[0]:.2e} ({worst[1]}), tolerance {a.tol:g}; non-finite {bad[:3]}')
    if a.json:
        Path(a.json).write_text(json.dumps(dict(ranks=len(ranks), replicas_identical=not split, worst=worst, buffers=rep), indent=1))
    print('PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
