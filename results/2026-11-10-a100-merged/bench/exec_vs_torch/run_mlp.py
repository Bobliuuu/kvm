#!/usr/bin/env python3
"""Executor (`kvm_exec`) vs PyTorch eager / whole-step CUDA graph / torch.compile on a small MLP trained for N steps, R independent runs.

  python3 -I run_mlp.py WORK_DIR --exe ./kvm_exec --hidden 72 --depth 2 --steps 50 --runs 1,2,4,8,16,32 --reps 7 --out OUT.json

One driver process owns the GPU context and the PyTorch variants; every executor measurement is a `kvm_exec` child launched from the same
driver, right after each PyTorch variant's timing, inside the same lock hold (kvm_exec has no in-process API). Per rep every variant is timed
once, in fixed order, so all ratios come from one time window; the result is the best of reps with the full spread.
Correctness gates run first and a failure aborts before any timing (a number from a run that did not pass is not a result):
  1. executor, 3-step bundle: final buffers within 1e-3 relative L2 of the CPU golden;
  2. executor, 4 replicated copies of that bundle: every copy bitwise equal to the single run;
  3. executor N-step single run vs PyTorch eager from the same bytes: per-step loss and final weights;
  4. every other PyTorch variant (all R runs) vs eager's loss trajectory.
Timings: PyTorch = CUDA events around N steps (steady state, state reset before each rep, graphs captured outside the timing);
executor = the kernel's own cudaEvent "ms" of one launch that runs all N steps of all R runs (persistent-kernel start and admission of
the whole tape included, process start and bundle load excluded).
"""
import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from replicate import replicate  # noqa: E402

COMPILE_TOL = 3e-2   # torch.compile (Triton TF32 matmuls) drifts from cuBLAS-TF32 eager by ~1e-2 late in a 13x loss decay; see the README
LOSS_TOL = 3e-3      # relative, per step: SGD lr 0.05 over N steps, 1xTF32 on both sides; the loss decays up to 500x, so the tail is the sensitive part (README)
WEIGHT_TOL = 1e-3    # relative L2 per weight buffer after N steps


def exec_flags(tasks_total):
    """Window the whole tape as ONE admission record (default ring/2 words admits ~1000 tasks per record, which serialises copies; vm-wins finding)."""
    p2 = lambda x: 1 << max(0, (int(x) - 1).bit_length())
    return ['--max-tasks', str(max(262144, p2(tasks_total * 1.25))), '--live-cap', str(max(131072, p2(tasks_total * 1.25))),
            '--ring-words', str(max(8388608, p2(tasks_total * 64)))]


def run_exec(exe, bundle, flags, dump=False, golden=False):
    cmd = [str(exe), '--quiet', *flags, *(['--dump-buffers'] if dump else []), *(['--golden', '--golden-tol', '1e-3'] if golden else []), str(bundle)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    m = re.search(r'^OK: (\d+)/(\d+) tasks, ([\d.]+) ms', p.stdout, re.M)
    if p.returncode != 0 or not m or m.group(1) != m.group(2):
        raise SystemExit(f'FAIL {" ".join(cmd)}\n{p.stdout[-800:]}\n{p.stderr[-800:]}')
    bufs = {}
    for line in p.stdout.splitlines():
        mm = re.match(r'buffer (\d+):(.*)', line)
        if mm:
            bufs[int(mm.group(1))] = mm.group(2).split()
    return float(m.group(3)), bufs


def named_buffers(bundle):
    return {b['name']: b['index'] for b in json.loads((Path(bundle) / 'metadata.json').read_text())['buffers']}


def gates(work, exe, flags4, golden=True):
    gate = Path(work) / 'gate' / 'bundle'
    _, one = run_exec(exe, gate, [], dump=True)
    if golden:
        run_exec(exe, gate, [], golden=True)
    rep = Path(work) / 'gate_rep'
    replicate(gate, rep, 4)
    _, d = run_exec(exe, rep, flags4, dump=True)
    nb = len(one)
    assert len(d) == nb * 4, (len(d), nb)
    for k in range(4):
        for i in range(nb):
            assert d[k * nb + i] == one[i], f'copy {k} buffer {i} differs from the single run'
    print(f'GATE 1+2 ok: {"3-step run within 1e-3 of the CPU golden; " if golden else "(no CPU golden at this size) "}4 replicated copies bitwise equal to the single run ({nb} buffers)', flush=True)


def record_failure(out, gate, measured):
    """A failed gate writes what it measured beside the stage's JSON (OUT.failed.json; no timing in it), so a reader can tell an executor fault from a problem PyTorch itself cannot reproduce."""
    if out:
        Path(out).with_name(Path(out).stem + '.failed.json').write_text(json.dumps(dict(failed=gate, gate=measured), indent=1, default=float))


def fp64_losses(wl, steps):
    """Independent reference: the same SGD problem in float64 (batch 0 at every step), on the GPU."""
    import torch
    from mlp_torch import LR, single_loss
    ps = [w.double().clone().requires_grad_() for w in wl.init]
    out = []
    for _ in range(steps):
        loss = single_loss(ps, wl.ids[0], wl.tgt[0])
        out.append(loss.item())
        gr = torch.autograd.grad(loss, ps)
        with torch.no_grad():
            torch._foreach_add_(ps, gr, alpha=-LR)
    return np.array(out)


def rel(a, b):
    return float(np.linalg.norm(a - b) / max(1e-30, np.linalg.norm(b)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('work')
    ap.add_argument('--exe', default='./kvm_exec')
    ap.add_argument('--hidden', type=int, default=72)
    ap.add_argument('--depth', type=int, default=2)
    ap.add_argument('--steps', type=int, default=50)
    ap.add_argument('--runs', default='1,2,4,8,16,32')
    ap.add_argument('--reps', type=int, default=7)
    ap.add_argument('--tiling', default='a100')
    ap.add_argument('--no-golden', action='store_true', help='the pure-Python CPU golden hangs on the larger models: gate 1 is skipped (gates 2-4 stay)')
    ap.add_argument('--skip', default='', help='comma list of torch variants to leave out')
    ap.add_argument('--out')
    a = ap.parse_args()
    work = Path(a.work)
    gen = [sys.executable, '-I', str(ROOT / 'bench/exec_vs_torch/gen_mlp.py')]
    shape = ['--tiling', a.tiling, '--hidden', str(a.hidden), '--depth', str(a.depth)]
    if not (work / 'gate' / 'bundle').exists():
        subprocess.run([*gen, str(work / 'gate'), '--steps', '3', *shape, *(['--no-golden'] if a.no_golden else [])], check=True, capture_output=True)
    wdir = work / f'steps{a.steps}'
    if not (wdir / 'bundle').exists():
        subprocess.run([*gen, str(wdir), '--steps', str(a.steps), '--no-golden', *shape], check=True, capture_output=True)
    info = json.loads(subprocess.run([sys.executable, '-c', f"import json;m=json.load(open('{wdir}/bundle/metadata.json'));print(json.dumps(dict(tasks=m['task_count'])))"],
                                     capture_output=True, text=True, check=True).stdout)
    tasks_step = info['tasks'] // a.steps
    runs = [int(x) for x in a.runs.split(',')]
    gates(work, a.exe, exec_flags(tasks_step * 3 * 4), golden=not a.no_golden)

    import torch
    from mlp_torch import Workload, build, default_variants, self_check_init
    from torch_bench import exec_warm_args, gpu_warm, is_capture_flake, with_capture_retries
    warm_args = exec_warm_args(a.exe)
    self_check_init()
    props = torch.cuda.get_device_properties(0)
    env = dict(torch=torch.__version__, cuda=torch.version.cuda, gpu=props.name, sms=props.multi_processor_count)
    print('env', env, flush=True)
    wl = Workload(wdir)
    skip = set(filter(None, a.skip.split(',')))

    # ---- gate 3: executor single run vs torch eager (loss per step, final weights)
    bundle1 = wdir / 'bundle'
    idx = named_buffers(bundle1)
    _, dump = run_exec(a.exe, bundle1, exec_flags(tasks_step * a.steps), dump=True)
    ex_loss = [float(dump[idx[f'loss.{s}']][0]) for s in range(a.steps)]
    ex_w = [np.array(dump[idx[n]], dtype=np.float64).reshape(w.shape) for n, w in zip(wl.names, wl.init)]
    ref = build(wl, 1, 'eager')
    ref.reset(); ref.run(a.steps)
    ref_loss = np.array([l[0] for l in ref.losses()])
    ref_w = ref.weights()
    ref64 = fp64_losses(wl, a.steps)
    rd = lambda l: float(np.max(np.abs(np.asarray(l) - ref64) / np.abs(ref64)))
    check = dict(fp64_loss_max_rel=dict(executor=rd(ex_loss), eager=rd(ref_loss)), exec_vs_eager_loss_max_rel=float(np.max(np.abs(np.array(ex_loss) - ref_loss) / np.abs(ref_loss))),
                 exec_vs_eager_weight_rel_l2={n: rel(e, w) for n, e, w in zip(wl.names, ex_w, ref_w)},
                 loss_first_last=[float(ref_loss[0]), float(ref_loss[-1])])
    check['exec_vs_eager_weight_rel_l2_max'] = max(check['exec_vs_eager_weight_rel_l2'].values())
    print('GATE 3: executor vs eager, loss max rel %.2e, weights max rel L2 %.2e, loss %.4f -> %.4f; vs float64: executor %.2e, eager %.2e' %
          (check['exec_vs_eager_loss_max_rel'], check['exec_vs_eager_weight_rel_l2_max'], *check['loss_first_last'], *check['fp64_loss_max_rel'].values()), flush=True)
    if check['exec_vs_eager_loss_max_rel'] > LOSS_TOL or check['exec_vs_eager_weight_rel_l2_max'] > WEIGHT_TOL:
        print('executor loss', ex_loss, '\neager loss   ', ref_loss.tolist())
        record_failure(a.out, 'gate3', check)
        raise SystemExit('GATE 3 FAILED: executor does not match PyTorch within tolerance; no timing is reported')
    del ref

    results = []
    skipped = []                                                        # variants a read could not capture (see below); recorded in the output
    for r in runs:
        names = [n for n in default_variants(r) if n not in skip]
        t_setup = time.time()
        variants = {}
        for n in list(names):
            try:
                v = with_capture_retries(lambda: build(wl, r, n), f'R={r} {n}')
            except RuntimeError as exc:                                 # (torch.AcceleratorError is a RuntimeError)
                if not (n.endswith('_whole') and is_capture_flake(exc)):
                    raise
                # a whole-run capture is an extra strength: when CUDA cannot capture it at this size the step-graph variants still stand for the row, and the skip is recorded
                print(f'  R={r} {n}: SKIPPED, its whole-run CUDA graph could not be captured here ({type(exc).__name__}): {str(exc)[:200]}', flush=True)
                names.remove(n)
                skipped.append(dict(variant=n, runs=r, error=str(exc)[:300]))
                continue
            v.reset(); v.run(a.steps)                                   # untimed rep: loss record for gate 4
            loss = np.array(v.losses())                                 # [step][run]
            d = float(np.max(np.abs(loss - ref_loss[:, None]) / np.abs(ref_loss[:, None])))
            d64 = rd(loss[:, 0])
            check['fp64_loss_max_rel'][f'{n}@R{r}'] = d64
            print(f'  R={r} {n}: loss vs eager {d:.2e}, vs fp64 {d64:.2e}', flush=True)
            if d > (COMPILE_TOL if 'compile' in n else LOSS_TOL):
                record_failure(a.out, f'gate4:{n}@R{r}', dict(check, conventional_vs_eager={f'{n}@R{r}': d}))
                raise SystemExit(f'GATE 4 FAILED: {n} (R={r}) loss differs from eager by {d:.2e}')
            variants[n] = (v, d)
        bundle = work / f'time_n{a.steps}_r{r}'
        replicate(bundle1, bundle, r)
        flags = exec_flags(tasks_step * a.steps * r)
        t_us = {n: [] for n in names}
        t_us['executor'] = []
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for rep in range(a.reps):
            for n in names:
                v = variants[n][0]
                gpu_warm()                                               # (bench_version 3: every timed window starts at the same clock state)
                v.reset(); torch.cuda.synchronize()
                e0.record(); v.run(a.steps); e1.record(); torch.cuda.synchronize()
                t_us[n].append(e0.elapsed_time(e1) * 1e3 / a.steps)
            gpu_warm()                                                   # (the executor child starts within the clock's hold time)
            ms, _ = run_exec(a.exe, bundle, [*flags, *warm_args])
            t_us['executor'].append(ms * 1e3 / a.steps)
        row = dict(hidden=a.hidden, depth=a.depth, steps=a.steps, runs=r, tasks_per_step_per_run=tasks_step, reps=a.reps,
                   loss_max_rel_vs_eager={n: variants[n][1] for n in names}, setup_s=round(time.time() - t_setup, 1),
                   us_per_step={n: dict(best=min(v), median=statistics.median(v), worst=max(v), all=[round(x, 1) for x in v]) for n, v in t_us.items()})
        results.append(row)
        print(json.dumps({k: row[k] for k in ('runs', 'steps')} | {n: round(d['best'], 1) for n, d in row['us_per_step'].items()}), flush=True)
        del variants
        torch.cuda.empty_cache()
    out = dict(env=env, gate=check, args=vars(a), rows=results, skipped_variants=skipped, exec_warm_args=warm_args)
    print('\nbest-of-%d us per step, all R runs advance one step; last block: variant / executor (>1: executor faster)' % a.reps)
    cols = list(results[0]['us_per_step'])
    for r_ in results:
        cols += [c for c in r_['us_per_step'] if c not in cols]
    print('R'.rjust(4), *[c.rjust(15) for c in cols])
    for row in results:
        print(str(row['runs']).rjust(4), *[('%.1f' % row['us_per_step'][c]['best'] if c in row['us_per_step'] else '-').rjust(15) for c in cols])
    print('ratio')
    for row in results:
        ex = row['us_per_step']['executor']['best']
        print(str(row['runs']).rjust(4), *[('%.2fx' % (row['us_per_step'][c]['best'] / ex) if c in row['us_per_step'] else '-').rjust(15) for c in cols])
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
