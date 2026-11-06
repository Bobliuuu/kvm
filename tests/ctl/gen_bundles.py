#!/usr/bin/env python3
"""Bundles for the device-side control GPU test (run.sh): the SAME task graph with different data, so the trip count / taken branch is
decided on the device. Usage: gen_bundles.py OUT_DIR   -> OUT_DIR/<case>/{bundle,expect.json}; golden.json comes from the CPU VM.

Cases: loop_<v>  while x > 1: x *= 0.5 (cap 12 iterations), x = v;   branch_pos / branch_neg  y = 2y if sum(u) > 0 else -y;
       cnn_loop  a loop whose body is a real GEMM (x = x @ W, until the device-computed max element exceeds a threshold)."""
import json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'frontend'))
from kvmfe.emit import GraphPolicy, lower_program, to_workload
from kvmfe.ir import Builder
from kvmfe.lower import Tiling


def spec_of(b):
    low = lower_program(b.ops, Tiling.preset('cpu'), 'none', 0)
    return to_workload(low, {'main': GraphPolicy()}), low.data_files


def loop_case(v, cap=12):
    b = Builder()
    x = b.tensor('x', 1, 1, 'state', buf='x', init=('values', [v]))
    b.while_loop(cap, lambda bb: (x, 0, '>', 1.0), lambda bb, i: bb.scale_scalar(x, 0.5))
    n, t = 0, v
    while n < cap and t > 1.0:
        t, n = t * 0.5, n + 1
    return b, {'trips': n, 'cap': cap, 'skipped_bodies': cap - n}


def branch_case(sign):
    b = Builder()
    vals = [sign * 1.5, sign * 0.25, sign * 2.0, 0.125 * sign]
    u = b.tensor('u', 4, 1, 'state', buf='u', init=('values', vals))
    y = b.tensor('y', 1, 1, 'state', buf='y', init=('values', [3.0]))
    p = b.pred_cmp(b.colsum(u), '>', 0.0)
    with b.when(p):
        b.scale_scalar(y, 2.0)
    with b.when(p, negate=True):
        b.scale_scalar(y, -1.0)
    return b, {'skipped_bodies': 1}


def gemm_loop_case(scale, cap=14):
    """x [8x64] <- x @ W repeated while max-element-proxy x[0,0] < 100 (a device-computed value); W = scale * I."""
    b = Builder()
    n = 64
    x = b.tensor('x', 8, n, 'state', buf='x', init=('values', [1.0] * (8 * n)))
    w = b.tensor('w', n, n, 'state', buf='w', init=('values', [scale if i // n == i % n else 0.0 for i in range(n * n)]))
    def body(bb, i):
        y = bb.linear(x, w)                       # y = x @ W (a real matmul tile task per 64x64 block, predicated)
        bb.add_into(x, y)                         # x += y (in place, predicated): x <- x (1 + scale)
    b.while_loop(cap, lambda bb: (x, 0, '<', 100.0), body)
    t, k = 1.0, 0
    while k < cap and t < 100.0:
        t, k = t * (1 + scale), k + 1
    return b, {'trips': k, 'cap': cap, 'skipped_bodies': cap - k}


def emit(out, name, b, expect):
    d = out / name
    spec, files = spec_of(b)
    d.mkdir(parents=True, exist_ok=True)
    for p, raw in files.items():
        (d / p).parent.mkdir(parents=True, exist_ok=True); (d / p).write_bytes(raw)
    (d / 'workload.json').write_text(json.dumps(spec, separators=(',', ':')))
    subprocess.run([sys.executable, str(ROOT / 'runtime/compile_workload.py'), '--emit-golden', str(d / 'workload.json'), str(d / 'bundle')],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    npred = sum(1 for g in spec['graphs'] for t in g['tasks'] if 'when' in t)
    expect['predicated_tasks'] = npred
    (d / 'expect.json').write_text(json.dumps(expect))
    print(name, 'tasks', sum(len(g['tasks']) for g in spec['graphs']), 'predicated', npred, expect)


if __name__ == '__main__':
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    for v in (0.5, 3.0, 10.0, 1000.0, 1e9):
        emit(out, f'loop_{v:g}', *loop_case(v))
    emit(out, 'branch_pos', *branch_case(1.0))
    emit(out, 'branch_neg', *branch_case(-1.0))
    for s in (0.5, 1.0, 200.0):
        emit(out, f'gemmloop_{s:g}', *gemm_loop_case(s))
