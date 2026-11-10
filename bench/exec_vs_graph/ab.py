#!/usr/bin/env python3
"""Interleaved A/B of executor variants on the same bundles: round r runs every variant once (order rotated each round), so clock and
thermal drift hit all variants alike.  Run under the GPU lock, from the scratch repo root:
  flock -w 600 ~/.gpu.lock python3 bench/exec_vs_graph/ab.py --reps 7 --out OUT.json  w_tiny w_small -- NAME=EXE[,ARG...] ...
Each variant is NAME=./kvm_exec_path[,extra,args] (comma-separated args, e.g. lean_cont=./bin/exec_lean_cont,--no-cont).
Prints and stores best / median us per optimizer step; a launch that does not end in `OK: n/n tasks` is recorded as failed.
"""
import argparse, json, re, statistics, subprocess, sys
from pathlib import Path

argv = sys.argv[1:]
vi = argv.index('--')
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('workloads', nargs='+'); ap.add_argument('--reps', type=int, default=7); ap.add_argument('--out')
a = ap.parse_args(argv[:vi])
variants = []
for v in argv[vi + 1:]:
    name, rest = v.split('=', 1); parts = rest.split(','); variants.append((name, parts[0], parts[1:]))
res = {}
for wl in a.workloads:
    info = json.loads((Path(wl) / 'info.json').read_text()); n = info['steps']
    ms = {v[0]: [] for v in variants}; failed = {v[0]: 0 for v in variants}
    for r in range(a.reps):
        order = variants[r % len(variants):] + variants[:r % len(variants)]
        for name, exe, args in order:
            p = subprocess.run([exe, '--quiet', *args, str(Path(wl) / 'bundle')], capture_output=True, text=True, timeout=1800)
            m = re.search(r'^(OK|FAIL): (\d+)/(\d+) tasks, ([\d.]+) ms', p.stdout, re.M)
            if p.returncode == 0 and m and m.group(1) == 'OK': ms[name].append(float(m.group(4)) / n * 1e3)
            else: failed[name] += 1; print('FAILED', name, wl, p.returncode, p.stdout[-200:], p.stderr[-200:], flush=True)
    res[wl] = {}
    for name, _, _ in variants:
        x = ms[name]
        res[wl][name] = dict(us_per_step=x, best=min(x) if x else None, median=statistics.median(x) if x else None, failed=failed[name])
        print(f'{wl:10s} {name:22s} best {min(x):7.1f}  median {statistics.median(x):7.1f}  worst {max(x):7.1f} us/step  failed {failed[name]}' if x else f'{wl} {name} no data', flush=True)
if a.out: Path(a.out).write_text(json.dumps(res, indent=1))
