#!/usr/bin/env python3
"""Run the fused GPT step's device program (include/kvm_fused_gpt.cuh) alone on a GPU and judge it against the workload VM.

  python3 -I bench/fused_gpt/fgpt_bed.py WORKLOAD_DIR --bed BINARY [--steps 2] [--gangs 1,4] [--work DIR] [--tol 3e-4]

BINARY is bench/fused_gpt/fgpt_bed.cu built for the GPU at hand (nvcc -O3 -std=c++17 -arch=sm_XX -Iinclude). The workload is a GPT
bundle's workload directory (workload.json and its data/). This script:
  1. runs the bundle's own tasks for the first --steps steps on the workload VM (float32 a op), once: the result is cached in the work
     directory, keyed by the workload's hash and the step count;
  2. writes the model and its streams for the bed (meta.txt, p/m/v.bin, ids/tgt.bin) and runs the bed once a gang size;
  3. judges each run against the VM: every step's loss, and the last step's gradients and AdamW moments, each tensor's largest
     difference over its largest magnitude (a float32 GEMM's error scales with its operands); the weights are reported, not judged
     (AdamW's first update is about lr x sign(g), so float32 noise flips it where a gradient is near zero);
  4. compares each gang size with the first (the decomposition: only the order of a few sums changes with G).
Exit 0 when every check passes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import d2_reference as ref  # noqa: E402

cw = ref.cw


def log(msg: str) -> None:
    print(f"[fgpt_bed {time.strftime('%H:%M:%SZ', time.gmtime())}] {msg}", flush=True)


def write_f32(path: Path, values) -> None:
    path.write_bytes(struct.pack(f'<{len(values)}f', *values))


def read_f32(path: Path) -> list[float]:
    raw = path.read_bytes()
    return list(struct.unpack(f'<{len(raw) // 4}f', raw))


def window(spec: dict, dst: int | str) -> dict:
    """The first select_window task that fills `dst`: its source, stride and offset (every step's windows share them)."""
    for t in (t for g in spec['graphs'] for t in g['tasks']):
        if t['op'] == 'select_window' and t['args']['dst'] == dst:
            return t['args']
    raise SystemExit(f'no select_window fills {dst}')


def vm_reference(spec: dict, wdir: Path, steps: int, work: Path) -> dict:
    """The VM's buffers after `steps` steps (losses, parameters, moments, the last step's gradients), cached by workload and steps."""
    key = hashlib.sha256((wdir / 'workload.json').read_bytes()).hexdigest()[:16]
    cache = work / f'vm-{key}-k{steps}.json'
    if cache.exists():
        log(f'VM reference: cached {cache.name}')
        return json.loads(cache.read_text())
    log(f'VM reference: running the bundle\'s first {steps} step(s) on the workload VM (pure Python; minutes on gpt-small)')
    t0 = time.time()
    bufs, _ = ref.vm_run(spec, wdir, steps)
    grads = {}
    for t in (t for g in spec['graphs'] for t in g['tasks'] if t['op'] == 'adamw' and t['id'].startswith(f's{steps - 1}/')):
        grads[t['args']['p']] = t['args']['g']
    out = {'losses': [list(bufs[f'loss.{s}'][:2]) for s in range(steps)], 'grad_names': grads,
           'bufs': {n: list(v) for n, v in bufs.items() if n.startswith(('w.', 'm.', 'v.', 'g.')) or n in grads.values()}}
    cache.write_text(json.dumps(out))
    log(f'VM reference: done in {time.time() - t0:.0f} s, cached as {cache.name}')
    return out


def write_inputs(model: ref.Model, spec: dict, work: Path) -> tuple[list[str], list[int]]:
    """meta.txt and the binaries the bed reads; returns the parameter names and sizes in slot order."""
    names = model.params()
    emb = next(t['args'] for g in spec['graphs'] for t in g['tasks'] if t['op'] == 'embedding_fwd')
    wi = window(spec, emb['ids'])
    ce = next(t['args'] for g in spec['graphs'] for t in g['tasks'] if t['op'] == 'softmax_ce_fused')
    wt = window(spec, ce['tgt'])
    ids = [int(x) for x in model.init[wi['src']]]
    tgt = [int(x) for x in model.init[wt['src']]]
    assert all(float(x) == int(x) for x in model.init[wi['src']]), 'token stream is not integral'
    step0 = int(model.init[model.step_buf][0])
    lines = [f'N {model.N}', f'T {model.T}', f'B {model.B}', f'd {model.d}', f'H {model.H}', f'hd {model.hd}', f'F {model.F}',
             f'V {model.V}', f'L {model.L}', f'steps {model.steps}', f'ln_eps {model.ln_eps!r}', f'theta {model.theta!r}',
             f'scale {model.scale!r}', f'lr {model.lr!r}', f'beta1 {model.b1!r}', f'beta2 {model.b2!r}', f'adam_eps {model.eps!r}',
             f'max_norm {model.max_norm!r}', f'ids_stride {wi["stride"]}', f'ids_off {wi["off"]}', f'tgt_stride {wt["stride"]}',
             f'tgt_off {wt["off"]}', f'ids_len {len(ids)}', f'tgt_len {len(tgt)}', f'step0 {step0}']
    sizes = []
    for i, n in enumerate(names):
        r, c = model.shape[n]
        lines.append(f'slot {i} {r} {c} {model.wd[n]!r}')
        sizes.append(r * c)
    (work / 'meta.txt').write_text('\n'.join(lines) + '\n')
    for tag, prefix in (('p', 'w.'), ('m', 'm.'), ('v', 'v.')):
        vals = []
        for n in names:
            vals += list(model.init[n.replace('w.', prefix, 1)])
        write_f32(work / f'{tag}.bin', vals)
    (work / 'ids.bin').write_bytes(struct.pack(f'<{len(ids)}i', *ids))
    (work / 'tgt.bin').write_bytes(struct.pack(f'<{len(tgt)}i', *tgt))
    return names, sizes


def split(flat: list[float], names: list[str], sizes: list[int]) -> dict[str, list[float]]:
    out, at = {}, 0
    for n, k in zip(names, sizes):
        out[n] = flat[at:at + k]
        at += k
    return out


def rel(mine: list[float], theirs: list[float]) -> float:
    """Largest difference over the reference's largest magnitude (NaN in mine counts as a failure)."""
    scale = max((abs(x) for x in theirs), default=0.0) or 1.0
    worst = 0.0
    for x, y in zip(mine, theirs):
        if x != x:
            return float('inf')
        worst = max(worst, abs(x - y))
    return worst / scale


def judge(run: dict, vmr: dict, names: list[str], tol: float, lr: float) -> tuple[bool, list[str]]:
    lines, ok = [], True
    lw = max(abs(a[0] - b[0]) / abs(b[0]) for a, b in zip(run['losses'], vmr['losses']))
    inv = max(abs(a[1] - b[1]) for a, b in zip(run['losses'], vmr['losses']))
    good = lw < tol and inv < 1e-6
    ok &= good
    lines.append(f'losses: {", ".join(f"{a[0]:.6f}" for a in run["losses"])} (VM {", ".join(f"{b[0]:.6f}" for b in vmr["losses"])}): '
                 f'worst {lw:.2e} relative, 1/count off by {inv:.1e}: {"ok" if good else "FAIL"}')
    for what, key, limit in (('gradients', 'g', tol), ('first moments', 'm', tol), ('second moments', 'v', 2 * tol)):
        worst, wname = 0.0, ''
        for n in names:
            theirs = vmr['bufs'][vmr['grad_names'][n]] if key == 'g' else vmr['bufs'][n.replace('w.', f'{key}.', 1)]
            e = rel(run[key][n], theirs)
            if e > worst:
                worst, wname = e, n
        good = worst < limit
        ok &= good
        lines.append(f'{what} (last step): worst {worst:.2e} of its tensor\'s scale ({wname}): {"ok" if good else "FAIL"}')
    wl = 0.0
    wname = ''
    for n in names:
        e = max((abs(x - y) for x, y in zip(run['p'][n], vmr['bufs'][n])), default=0.0)
        if e != e or e > wl:
            wl, wname = e, n
    lines.append(f'weights (reported): largest difference {wl / lr:.2f} x lr ({wname})')
    return ok, lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('workload'); ap.add_argument('--bed', required=True); ap.add_argument('--steps', type=int, default=2)
    ap.add_argument('--gangs', default='1,4'); ap.add_argument('--work', default=None); ap.add_argument('--tol', type=float, default=3e-4)
    a = ap.parse_args()
    wdir = Path(a.workload).resolve()
    work = Path(a.work).resolve() if a.work else wdir.parent / f'fgpt-bed-{wdir.name}-k{a.steps}'
    work.mkdir(parents=True, exist_ok=True)
    spec = json.loads((wdir / 'workload.json').read_text())
    vmr = vm_reference(spec, wdir, a.steps, work)
    compiled = cw.compile_spec(spec, data_dir=wdir)
    model = ref.Model(spec, compiled, a.steps)
    log(f'model: L {model.L}, d {model.d}, heads {model.H} x {model.hd}, MLP {model.F}, V {model.V}, {model.B} x {model.T} rows; '
        f'{a.steps} steps; work {work}')
    names, sizes = write_inputs(model, spec, work)
    runs, fail = {}, 0
    for G in [int(x) for x in a.gangs.split(',')]:
        log(f'G {G}: {a.bed} {work} {G}')
        p = subprocess.run([a.bed, str(work), str(G)], capture_output=True, text=True, timeout=600)
        (work / f'bed-G{G}.log').write_text(p.stdout + p.stderr)
        for line in (p.stdout + p.stderr).splitlines():
            log(f'  {line}')
        if p.returncode:
            log(f'G {G}: the bed exited {p.returncode}: FAIL')
            fail += 1
            continue
        out = work / f'out-G{G}'
        flat = read_f32(out / 'losses.bin')
        run = {'losses': [flat[2 * s:2 * s + 2] for s in range(a.steps)]}
        for key in ('p', 'm', 'v', 'g'):
            run[key] = split(read_f32(out / f'{key}.bin'), names, sizes)
        runs[G] = run
        ok, lines = judge(run, vmr, names, a.tol, model.lr)
        for line in lines:
            log(f'G {G}: {line}')
        fail += not ok
    gangs = list(runs)
    for G in gangs[1:]:
        base = runs[gangs[0]]
        lw = max(abs(x[0] - y[0]) / abs(y[0]) for x, y in zip(runs[G]['losses'], base['losses']))
        gw = max(rel(runs[G]['g'][n], base['g'][n]) for n in names)
        good = lw < 1e-5 and gw < 1e-4
        fail += not good
        log(f'decomposition G {G} against G {gangs[0]}: losses {lw:.2e}, gradients {gw:.2e} of scale: {"ok" if good else "FAIL"}')
    log('OK' if not fail else f'FAILED: {fail}')
    return 1 if fail else 0


if __name__ == '__main__':
    sys.exit(main())
