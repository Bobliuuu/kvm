#!/usr/bin/env python3
"""Regime 1: many independent runs filling the GPU. VM (one kvm_exec launch, R replicated runs) vs torch baselines, one process.

  python3 -I run_runs.py WORK_DIR --steps 50 --runs 1,2,4,8,16 --reps 5 --vm base=./kvm_exec [--vm chain=./kvm_exec_chain ...]

WORK_DIR holds `gate/` (3-step bundle, golden.json) and `time/` (--steps bundle) made by gen_mlp.sh. Gate first (a number from a run
that did not pass is not a result): the R-copy bundle's every copy must equal the single-run final buffers byte for byte, and the
single run must match the CPU golden. Then, per rep, every variant is timed back to back, so ratios come from one time window; the
result is the best of reps (spread printed). VM time = the executor's own `kernel done` ms (includes admission, excludes process start).
Output: one JSON line per R plus a table.
"""
import argparse, json, re, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from replicate import replicate  # noqa: E402


def run_vm(binary, bundle, extra=(), dump=False, golden=False):
    cmd = [binary, '--quiet', *extra, *(['--dump-buffers'] if dump else []), *(['--golden', '--golden-tol', '1e-3'] if golden else []), str(bundle)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if p.returncode != 0:
        raise SystemExit(f'FAIL {" ".join(cmd)}\n{p.stdout[-800:]}\n{p.stderr[-800:]}')
    m = re.search(r'OK: (\d+)/(\d+) tasks, ([\d.]+) ms', p.stdout)
    dumps = [l for l in p.stdout.splitlines() if l.startswith('buffer ')]
    return float(m.group(3)), dumps


def gate(work, binaries, runs_check=4):
    one = Path(work) / 'gate' / 'bundle'
    for name, b in binaries.items():
        _, base = run_vm(b, one, dump=True)
        run_vm(b, one, golden=True)   # CPU golden within 1e-3 relative L2 (1xTF32 noise level of the audit)
        rep = Path(work) / 'gate_rep'
        replicate(one, rep, runs_check)
        _, d = run_vm(b, rep, dump=True)
        nb = len(base)
        assert len(d) == nb * runs_check, (len(d), nb, runs_check)
        for k in range(runs_check):
            for i in range(nb):
                body = d[k * nb + i].split(':', 1)[1]
                assert body == base[i].split(':', 1)[1], f'{name}: copy {k} buffer {i} differs from the single run'
        print(f'GATE {name}: single run matches golden; {runs_check} replicated copies bitwise equal to the single run ({nb} buffers each)', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('work'); ap.add_argument('--steps', type=int, default=50); ap.add_argument('--runs', default='1,2,4,8,16')
    ap.add_argument('--reps', type=int, default=5); ap.add_argument('--vm', action='append', required=True)
    ap.add_argument('--vm-args', default='', help='extra kvm_exec arguments, e.g. "--blocks 108"')
    ap.add_argument('--no-torch', action='store_true')
    a = ap.parse_args()
    binaries = dict(s.split('=', 1) for s in a.vm)
    gate(a.work, binaries)
    extra = a.vm_args.split()
    if not a.no_torch:
        import torch; from torch_runs import Variants
        print('torch', torch.__version__, torch.cuda.get_device_name(0), flush=True)
    results = []
    for r in [int(x) for x in a.runs.split(',')]:
        bundle = Path(a.work) / f'time_r{r}'
        replicate(Path(a.work) / 'time' / 'bundle', bundle, r)
        v = None if a.no_torch else Variants(r)
        names = ([] if v is None else list(v.fns)) + [f'vm_{n}' for n in binaries]
        best = {n: 1e18 for n in names}; allv = {n: [] for n in names}
        for rep in range(a.reps):
            for n in names:
                if n.startswith('vm_'):
                    ms, _ = run_vm(binaries[n[3:]], bundle, extra); us = ms * 1e3 / a.steps
                else:
                    v.time(n, 3); us = v.time(n, a.steps)
                best[n] = min(best[n], us); allv[n].append(round(us, 1))
        row = {'runs': r, 'steps': a.steps, 'best_us_per_step': {k: round(x, 1) for k, x in best.items()}, 'all': allv}
        results.append(row); print(json.dumps(row), flush=True)
    print('\nbest-of-%d us per step (all R runs advance one step); ratio = variant / vm_%s  (>1: VM faster)' % (a.reps, next(iter(binaries))))
    cols = list(results[0]['best_us_per_step'])
    print('R'.rjust(4), *[c.rjust(12) for c in cols])
    for row in results:
        print(str(row['runs']).rjust(4), *[str(row['best_us_per_step'].get(c, '-')).rjust(12) for c in cols])
    ref = 'vm_' + next(iter(binaries))
    print('ratio vs ' + ref)
    for row in results:
        print(str(row['runs']).rjust(4), *[(('%.2fx' % (row['best_us_per_step'][c] / row['best_us_per_step'][ref])) if c in row['best_us_per_step'] else '-').rjust(12) for c in cols])


if __name__ == '__main__':
    main()
