#!/usr/bin/env python3
"""Sharded AdamW training on ONE GPU: every "rank" is a partition of the in-process multi-executor harness (tests/dist/test_dist.cu
`plan`), so ZeRO gather / reduce-scatter / tensor-parallel collectives run as ordinary KVM tasks between executors that share
the device. No NCCL, no second GPU: this shows the sharded optimizer and the generated collectives are numerically right and how
much device memory they take; it says nothing about NVLink or network speed.

    scale_gpu.py golden OUT_DIR               # small stacks, CPU golden expectations compared bitwise on the device
    scale_gpu.py pair OUT_DIR --d 1024 --layers 16 --ranks 4 --m 64 --steps 3     # unsharded run vs sharded run, same device data
    (both build the plan on the CPU, run test_dist under the caller's GPU lock, and write JSON + logs under OUT_DIR)

The caller wraps this in `flock ... timeout ...`; it never takes the GPU lock itself.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'dist'), str(ROOT / 'runtime')]
import scaleplan as sp          # noqa: E402
import partition as pt          # noqa: E402

TD = os.environ.get('KVM_TEST_DIST', str(ROOT / 'test_dist'))


def run_plan(plan_dir, out_dir, tag, timeout_s, tol=None, log=None):
    cmd = [TD, 'plan', str(plan_dir), '--blocks', '4', '--timeout', str(timeout_s)]
    if out_dir:
        cmd += ['--dump', str(out_dir)]
    if tol is not None:
        cmd += ['--tol', str(tol)]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s + 600)
    wall = time.time() - t0
    if log:
        Path(log).write_text(p.stdout + p.stderr)
    err = p.stderr
    info = dict(tag=tag, rc=p.returncode, wall_s=round(wall, 2))
    m = re.search(r'MEM total local buffer bytes (\d+)', err)
    info['mem_local_bytes_total'] = int(m.group(1)) if m else None
    info['mem_local_bytes_per_rank'] = [int(x) for x in re.findall(r'MEM rank \d+: \d+ local buffers, (\d+) bytes', err)]
    m = re.search(r'MEM device used after allocation .* (\d+) bytes', err)
    info['device_used_after_alloc_bytes'] = int(m.group(1)) if m else None
    info['kernel_ms'] = [float(x) for x in re.findall(r'rank \d+: \d+ tasks, ([\d.]+) ms kernel', err)]
    m = re.search(r'(\d+) values compared, (\d+) bitwise equal, worst relative difference ([\d.e+-]+)', err)
    if m:
        info.update(compared=int(m.group(1)), bitwise_equal=int(m.group(2)), worst_rel=float(m.group(3)))
    info['tail'] = err.strip().splitlines()[-3:]
    return info


def cmd_golden(a):
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for st, P, coll, k in sp.small_configs():
        spec, _, _ = sp.build_stack(st, 'values')
        strat = sp.strategy_for(st, coll, k)
        plan = pt.partition(copy.deepcopy(spec), pt.Cluster.uniform(P, 1), strat)
        import workload_vm
        import compile_workload as cw
        gold = workload_vm.run_golden(cw.compile_spec(copy.deepcopy(plan.spec)))['buffers']
        name = f'{st.strategy}_P{P}_L{st.L}_s{st.steps}_{coll}'
        pd = out / 'plans' / name
        pt.write_plan(plan, pd, gold)
        r = run_plan(pd, None, name, 120, tol=None if st.steps == 1 else 1e-4, log=out / f'{name}.log')
        r.update(strategy=st.strategy, P=P, dims=st.dims, steps=st.steps, collective=coll, exact_required=st.steps == 1)
        results.append(r)
        print(json.dumps({k_: v for k_, v in r.items() if k_ != 'tail'}), flush=True)
    (out / 'golden_results.json').write_text(json.dumps(results, indent=1))
    return 0 if all(r['rc'] == 0 for r in results) else 1


def load_dump(d):
    """{name: array (gr x gc)} reconstructed from manifest.txt (R: any rank; ROWS/COLS blocks placed; scalars named loss* summed)."""
    import numpy as np
    parts = {}
    for line in Path(d, 'manifest.txt').read_text().splitlines():
        r, name, rows, cols, r0, c0, gr, gc, fn = line.split()
        parts.setdefault(name, []).append((int(r), int(rows), int(cols), int(r0), int(c0), int(gr), int(gc), fn))
    out = {}
    for name, ps in parts.items():
        if name.startswith('loss'):
            out[name] = np.float32(sum(np.fromfile(Path(d, p[7]), dtype='<f4')[0] for p in ps))
            continue
        gr, gc = ps[0][5], ps[0][6]
        full = np.zeros((gr, gc), dtype='<f4')
        for _, rows, cols, r0, c0, _, _, fn in ps:
            full[r0:r0 + rows, c0:c0 + cols] = np.fromfile(Path(d, fn), dtype='<f4').reshape(rows, cols)
        out[name] = full
    return out


def cmd_pair(a):
    import numpy as np
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dims = [a.d] * (a.layers + 1)
    res = dict(d=a.d, layers=a.layers, m=a.m, steps=a.steps, ranks=a.ranks, collective=a.collective, lr=a.lr, logical_params=0)
    runs = {}
    for P in ([1] if a.skip_sharded else [1, a.ranks]) if not a.sharded_only else [a.ranks]:
        st = sp.Stack(dims, m=a.m, steps=a.steps, lr=a.lr, strategy='single' if P == 1 else 'zero')
        res['logical_params'] = st.params()
        spec, _, gen = sp.build_stack(st, 'seeded', seed=a.seed)
        dump = [f'W{l}' for l in (1, a.layers)] + [f'W{l}T' for l in (1, a.layers)] + [f'loss{t}' for t in range(1, a.steps + 1)]
        strat = sp.strategy_for(st, a.collective, 1, symbolic=True, gen=gen, dump=dump)
        t0 = time.time()
        plan = pt.partition(copy.deepcopy(spec), pt.Cluster.uniform(P, 1), strat)
        plan_s = time.time() - t0
        pred = sp.LayoutCounter(spec, P, strat).result()
        meas = sp.measure(plan)
        name = f'P{P}'
        pd = out / 'plans' / name
        pt.write_plan(plan, pd, None)
        dd = out / f'dump_{name}'
        dd.mkdir(exist_ok=True)
        info = run_plan(pd, dd, name, a.timeout, log=out / f'run_{name}.log')
        info.update(plan_build_s=round(plan_s, 2), tasks_per_rank=meas['tasks'], planner_predicted_local_bytes_per_rank=pred['local_bytes'],
                    planned_local_bytes_per_rank=meas['local_bytes'][0], planned_comm_bytes=meas['comm_bytes'], planned_flops_per_rank=meas['flops'][0],
                    predicted_comm_bytes=pred['comm_bytes'], predicted_flops_per_rank=pred['flops'])
        runs[name] = info
        print(json.dumps({k: v for k, v in info.items() if k != 'tail'}), flush=True)
        if info['rc'] != 0:
            print('\n'.join(info['tail']))
            res['runs'] = runs
            (out / 'result.json').write_text(json.dumps(res, indent=1))
            return 1
    res['runs'] = runs
    if 'P1' in runs and f'P{a.ranks}' in runs:
        A, B = load_dump(out / 'dump_P1'), load_dump(out / f'dump_P{a.ranks}')
        cmp = {}
        for name in A:
            if name.startswith('loss'):
                cmp[name] = dict(unsharded=float(A[name]), sharded=float(B[name]), rel=float(abs(A[name] - B[name]) / max(1e-30, abs(A[name]))))
            else:
                x, y = A[name], B[name]
                denom = np.maximum(1.0, np.abs(x))
                cmp[name] = dict(max_abs=float(np.abs(x - y).max()), max_rel=float((np.abs(x - y) / denom).max()),
                                 bitwise_equal_fraction=float((x.view('<u4') == y.view('<u4')).mean()))
        res['compare_final_state'] = cmp
        res['loss_unsharded'] = [float(A[f'loss{t}']) for t in range(1, a.steps + 1)]
        res['loss_sharded'] = [float(B[f'loss{t}']) for t in range(1, a.steps + 1)]
        # the same compare, free of the dumps' memory
        for d in ('dump_P1', f'dump_P{a.ranks}'):
            for f in Path(out, d).glob('*.bin'):
                f.unlink()
    (out / 'result.json').write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ('runs',)}, indent=1))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    g = sub.add_parser('golden')
    g.add_argument('out_dir')
    p = sub.add_parser('pair')
    p.add_argument('out_dir')
    p.add_argument('--d', type=int, default=256)
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--ranks', type=int, default=4)
    p.add_argument('--m', type=int, default=64)
    p.add_argument('--steps', type=int, default=3)
    p.add_argument('--lr', type=float, default=1e-2)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--collective', default='direct')
    p.add_argument('--timeout', type=int, default=900)
    p.add_argument('--sharded-only', action='store_true')
    p.add_argument('--skip-sharded', action='store_true')
    a = ap.parse_args()
    return cmd_golden(a) if a.cmd == 'golden' else cmd_pair(a)


if __name__ == '__main__':
    sys.exit(main())
