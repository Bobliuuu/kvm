#!/usr/bin/env python3
"""Executor vs PyTorch eager / whole-step CUDA graph / torch.compile on a frontend GPT (gen.py workload), reps interleaved in one driver.

  python3 -I run_gpt.py WORKLOAD_DIR --exe ./kvm_exec --modes eager,graph,compile_mat,compile_graph --reps 7 --tol 2e-3 --out OUT.json

Same bytes on both sides (initial weights, token stream, AdamW lr 3e-3 betas (0.9, 0.95) wd 0.01, clip 1.0: torch_bench.Trainer reads them
from the executor's workload.json). One driver process owns the GPU context and all PyTorch modes (built and warmed first); per rep every
mode is timed once (CUDA events around the N steps), then one `kvm_exec` child runs the whole N-step bundle (the kernel's own cudaEvent
ms: persistent-kernel start and admission of the whole tape included). Ratios come from one time window; best of reps with the spread.
Gate before timing: executor per-step loss vs eager (relative) within --tol, and every other mode vs eager.
"""
import argparse
import json
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'bench' / 'vs_pytorch'))


def exec_run(exe, bundle, extra=(), dump_idx=()):
    with tempfile.TemporaryDirectory() as td:
        cmd = [str(exe), '--quiet', *extra, *(['--dump-bin', td, ','.join(map(str, dump_idx))] if dump_idx else []), str(bundle)]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        m = re.search(r'^OK: (\d+)/(\d+) tasks, ([\d.]+) ms', p.stdout, re.M)
        if p.returncode != 0 or not m or m.group(1) != m.group(2):
            raise SystemExit(f'FAIL {" ".join(cmd)}\n{p.stdout[-600:]}\n{p.stderr[-600:]}')
        dumps = {i: np.fromfile(Path(td) / f'{i}.bin', dtype='<f4') for i in dump_idx}
    return float(m.group(3)), dumps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('workload_dir')
    ap.add_argument('--exe', default='./kvm_exec')
    ap.add_argument('--modes', default='eager,graph,compile_mat,compile_graph')
    ap.add_argument('--reps', type=int, default=7)
    ap.add_argument('--tol', type=float, default=2e-3, help='max relative per-step loss difference to eager that still counts as correct')
    ap.add_argument('--compile-tol', type=float, default=1e-2, help='the same for torch.compile modes (Triton TF32 matmuls round differently from cuBLAS)')
    ap.add_argument('--exec-args', default='', help='extra kvm_exec arguments')
    ap.add_argument('--out')
    a = ap.parse_args()
    import torch
    import torch_bench as tb
    tb.self_check_init()
    wl = Path(a.workload_dir)
    info = json.loads((wl / 'info.json').read_text())
    meta = json.loads((wl / 'bundle' / 'metadata.json').read_text())
    loss_idx = sorted((b['index'], int(b['name'].split('.')[1])) for b in meta['buffers'] if re.fullmatch(r'loss\.\d+', b['name']))
    n = info['steps']
    assert len(loss_idx) == n, (len(loss_idx), n)
    extra = a.exec_args.split()
    props = torch.cuda.get_device_properties(0)
    env = dict(torch=torch.__version__, cuda=torch.version.cuda, gpu=props.name, sms=props.multi_processor_count)
    print('env', env, flush=True)

    runs, losses = {}, {}
    for mode in a.modes.split(','):
        torch.manual_seed(0)
        t = tb.Trainer(a.workload_dir, 'tf32')
        t0 = time.time()
        run = {'eager': t.eager, 'graph': t.graphed, 'compile_mat': lambda: t.compiled('max-autotune-no-cudagraphs'),
               'compile_ro': lambda: t.compiled('reduce-overhead'),
               'compile_graph': lambda: t.compiled('max-autotune-no-cudagraphs', graph=True)}[mode]()
        t.reset()
        out = run()
        losses[mode] = np.array([float(x) for x in out])
        runs[mode] = (t, run)
        print(f'{mode}: built in {time.time() - t0:.0f} s, loss {losses[mode][0]:.5f} -> {losses[mode][-1]:.5f}', flush=True)
    ms0, d = exec_run(a.exe, wl / 'bundle', extra, [i for i, _ in loss_idx])
    ex_loss = np.array([d[i][0] for i, _ in loss_idx], dtype=np.float64)
    ref = losses.get('eager', next(iter(losses.values())))
    gate = dict(executor_vs_eager_max_rel=float(np.max(np.abs(ex_loss - ref) / np.abs(ref))),
                modes_vs_eager_max_rel={m: float(np.max(np.abs(l - ref) / np.abs(ref))) for m, l in losses.items()},
                executor_loss=ex_loss.tolist(), eager_loss=ref.tolist())
    print('GATE: executor vs eager loss max rel %.2e; modes vs eager %s' % (gate['executor_vs_eager_max_rel'],
          {m: '%.1e' % v for m, v in gate['modes_vs_eager_max_rel'].items()}), flush=True)
    bad = [m for m, v in gate['modes_vs_eager_max_rel'].items() if v > (a.compile_tol if 'compile' in m else a.tol)]
    if gate['executor_vs_eager_max_rel'] > a.tol or bad:
        raise SystemExit(f'GATE FAILED (tol {a.tol}): executor {gate["executor_vs_eager_max_rel"]:.2e}, modes {bad}; no timing reported')

    ts = {m: [] for m in runs}
    ts['executor'] = []
    for rep in range(a.reps):
        for m, (t, run) in runs.items():
            t.reset()
            torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record(); run(); e1.record(); torch.cuda.synchronize()
            ts[m].append(e0.elapsed_time(e1) * 1e3 / n)
        ms, _ = exec_run(a.exe, wl / 'bundle', extra)
        ts['executor'].append(ms * 1e3 / n)
    row = {m: dict(best=min(v), median=statistics.median(v), worst=max(v), all=[round(x, 1) for x in v]) for m, v in ts.items()}
    print('us/step best (median, worst) over %d interleaved reps; ratio = mode / executor' % a.reps)
    ex = row['executor']['best']
    for m, v in row.items():
        print(f'  {m:14s} {v["best"]:10.1f} ({v["median"]:.1f}, {v["worst"]:.1f})   {v["best"] / ex:6.2f}x')
    if a.out:
        Path(a.out).write_text(json.dumps(dict(env=env, info=info, gate=gate, steps=n, tol=a.tol, compile_tol=a.compile_tol, us_per_step=row), indent=1))


if __name__ == '__main__':
    main()
