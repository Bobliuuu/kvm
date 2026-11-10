#!/usr/bin/env python3
"""Per-tensor executor vs CPU golden vs float64 reference for one generic bundle.
   python3 -I results/2026-11-09-encoder-reg-zero-grad/analyse.py WORKLOAD.json BUNDLE_DIR DUMP.txt"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'frontend')); sys.path.insert(0, str(ROOT / 'tests/gptdiff'))
from golden_check import read_dump, rel_l2
from kvmfe import refvm
wl, bundle, dump = sys.argv[1:4]
spec = json.load(open(wl)); meta = json.load(open(f'{bundle}/metadata.json')); g = json.load(open(f'{bundle}/golden.json'))
gold = g['buffers']; gpu = read_dump(dump, [b['name'] for b in meta['buffers']])
mem = refvm.execute(spec, f32=False)
f64 = {n: [x for r in range(b.rows) for x in b.row(r)] for n, b in mem.bufs.items()}
mx = lambda v: max(abs(x) for x in v)
print('name  max|exec| max|golden| max|f64|  | max abs diff exec-f64, golden-f64, exec-golden')
for n in ['act47', 'act53', 'act115', 'm.enc.k.b', 'v.enc.k.b', 'w.enc.k.b', 'act69']:
    d = lambda a, b: max(abs(x - y) for x, y in zip(a, b))
    print(f'{n:10s} {mx(gpu[n]):.2e} {mx(gold[n]):.2e} {mx(f64[n]):.2e} | {d(gpu[n], f64[n]):.2e} {d(gold[n], f64[n]):.2e} {d(gpu[n], gold[n]):.2e}')
print('tensors that are ~0 in float64 (max < 1e-9) yet nonzero in exec or golden:')
print([n for n in f64 if g['dtypes'].get(n) == 'f32' and mx(f64[n]) < 1e-9 and (mx(gpu[n]) > 0 or mx(gold[n]) > 0)])
