#!/usr/bin/env python3
"""Compare a GPU run of a gpt_tiny AdamW bundle with the compiler's CPU golden, with an error model that fits AdamW.

  kvm_exec --dump-buffers BUNDLE > dump.txt ;  python3 golden_check.py WORKLOAD.json BUNDLE dump.txt --mode tf32|precise

Why not one relative-L2 tolerance on the final weights: AdamW's first update is u = g/(|g|+eps) ~ sign(g) (eps = 1e-8), a step
function of the gradient. With 1xTF32 GEMMs the gradients carry ~5e-4 relative error, so every element with |g| below that noise
may flip sign and then moves by 2*lr: the post-Adam weights of the two runs differ by up to 35% in L2 while every op is within its
documented tolerance (results/2026-11-03-a100-gptops/README.md). The checks that are meaningful:

  precise : GEMMs are 3xTF32 (fp32 accuracy): every f32 buffer, activations through weights and moments, within TOL_PRECISE.
  tf32    : (a) losses, (b) every buffer produced before the first optimizer task (activations, gradients) within TOL_TF32,
            (c) steps == 1 only: the adamw op reproduces the GPU weights/moments from the GPU's OWN gradients in float64
            (op correctness, independent of gradient noise), and (d) the CPU weights lie in the interval Adam maps the
            gradient tolerance to (u is monotone in g): no element may differ unless a sign flip within the noise explains it.
            After step 1 the runs follow different (equally valid) trajectories, so later weights are only reported.
Exit status 0 = PASS.
"""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'frontend'))

TOL_TF32 = 6e-3        # matmul 1xTF32, relative to max|ref| (gptops README tolerance table); measured 7e-4 rel L2 here
TOL_PRECISE = 2e-4     # measured 2.4e-5 (GPU 3xTF32 vs CPU f32) and 1.3e-5 (CPU f32 vs CPU f64) on 2 steps
TOL_LOSS = 1e-3
OPT = dict(lr=0.003, b1=0.9, b2=0.95, eps=1e-8)


def rel_l2(x, ref):
    num = sum((a - b) ** 2 for a, b in zip(x, ref))
    den = sum(b * b for b in ref)
    return math.sqrt(num / den) if den > 0 else math.sqrt(num)


def adam_step1(w0, g, coef, wd, lr, b1, b2, eps):
    """float64 AdamW at t=1 on gradient g*coef: returns (w1, m1, v1)."""
    w1, m1, v1 = [], [], []
    for a, x in zip(w0, g):
        x *= coef
        m, v = (1 - b1) * x, (1 - b2) * x * x
        w1.append(a * (1 - lr * wd) - lr * (m / (1 - b1)) / (math.sqrt(v / (1 - b2)) + eps))
        m1.append(m)
        v1.append(v)
    return w1, m1, v1


def read_dump(path, names):
    out = {}
    for line in Path(path).read_text().splitlines():
        if line.startswith('buffer '):
            head, rest = line.split(':', 1)
            out[names[int(head.split()[1])]] = [float(x) for x in rest.split()]
    return out


class Report:
    def __init__(self):
        self.failures = []

    def check(self, ok, what, detail):
        print(f'  {"ok  " if ok else "FAIL"} {what}: {detail}')
        if not ok:
            self.failures.append(what)


def first_opt_cut(tasks):
    """Buffers written by tasks that precede (in program order) every adamw of step 0 and are not optimizer inputs."""
    pre = set()
    for t in tasks:
        if t['op'] == 'adamw':
            break
        for k, v in t.get('args', {}).items():
            if isinstance(v, str) and not v.startswith(('w.', 'm.', 'v.')) and k in ('out', 'dst', 'o', 'dlogits', 'dx', 'dgamma', 'dbeta', 'dq', 'dk', 'dv', 'dtab',
                                            'rowloss', 'lse', 'loss', 'mean', 'rstd', 'pre', 'delta'):
                pre.add(v)
    return pre


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('workload')
    ap.add_argument('bundle')
    ap.add_argument('dump')
    ap.add_argument('--mode', choices=['tf32', 'precise'], required=True)
    a = ap.parse_args()
    spec = json.loads(Path(a.workload).read_text())
    meta = json.loads((Path(a.bundle) / 'metadata.json').read_text())
    golden = json.loads((Path(a.bundle) / 'golden.json').read_text())
    gold, dtypes = golden['buffers'], golden['dtypes']
    gpu = read_dump(a.dump, [b['name'] for b in meta['buffers']])
    tasks = spec['graphs'][0]['tasks']
    steps = sum(1 for t in tasks if t['op'] == 'counter_inc' and t['args']['x'] == 'opt_step')
    f32 = [n for n in gold if dtypes[n] == 'f32']
    rep = Report()
    print(f'{a.mode}: {steps} step(s), {len(f32)} f32 buffers')

    if a.mode == 'precise':
        errs = sorted(((rel_l2(gpu[n], gold[n]), n) for n in f32), reverse=True)
        print('  worst: ' + ', '.join(f'{n} {e:.2e}' for e, n in errs[:4]))
        rep.check(errs[0][0] <= TOL_PRECISE, f'all buffers rel L2 <= {TOL_PRECISE:g}', f'worst {errs[0][1]} {errs[0][0]:.2e}')
        return finish(rep)

    for n in (n for n in f32 if n.startswith('loss.')):
        e = rel_l2(gpu[n], gold[n])
        rep.check(e <= TOL_LOSS, f'{n} rel L2 <= {TOL_LOSS:g}', f'{e:.2e}')
    pre = sorted(n for n in first_opt_cut(tasks) if n in gold and dtypes[n] == 'f32' and not n.startswith('loss'))
    errs = sorted(((rel_l2(gpu[n], gold[n]), n) for n in pre), reverse=True)
    rep.check(errs[0][0] <= TOL_TF32, f'{len(pre)} step-1 activation/gradient buffers rel L2 <= {TOL_TF32:g}',
              f'worst {errs[0][1]} {errs[0][0]:.2e}')
    rep.check(rel_l2(gpu['clip_coef'], gold['clip_coef']) <= TOL_TF32 if steps == 1 else True, 'clip_coef (step 1)', '')

    if steps != 1:
        post = sorted(((rel_l2(gpu[n], gold[n]), n) for n in f32 if n.startswith(('w.', 'm.', 'v.'))), reverse=True)
        print(f'  info: post-Adam weights/moments after {steps} steps differ by up to {post[0][0]:.2e} ({post[0][1]}): '
              'different, equally valid trajectories under 1xTF32 gradient noise; see the 1-step model check')
        return finish(rep)

    from kvmfe.refvm import Memory
    mem = Memory(spec, False, {})
    init = {n: [x for r in range(b.rows) for x in b.row(r)] for n, b in mem.bufs.items()}
    cg, cc = gpu['clip_coef'][0], gold['clip_coef'][0]
    per = {}
    for t in tasks:
        if t['op'] != 'adamw':
            continue
        x = t['args']
        d = per.setdefault(x['p'], dict(g=x['g'], m=x['m'], v=x['v'], wd=x['wd'], idx=[]))
        d['idx'] += list(range(x['e0'], x['e1']))
    tot_flip = tot_bad = 0
    worst_op = worst_m = 0.0
    wd_err = flip_err2 = tot_err2 = 0.0
    for p, d in per.items():
        w0, gg, gc = init[p], gpu[d['g']], gold[d['g']]
        # (c) op correctness: GPU weights/moments from the GPU's own gradient
        w1, m1, v1 = adam_step1(w0, gg, cg, d['wd'], **OPT)
        worst_op = max(worst_op, rel_l2(gpu[p], w1))
        worst_m = max(worst_m, rel_l2(gpu[d['m']], m1), rel_l2(gpu[d['v']], v1))
        # (d) CPU weights must lie in the image of the gradient tolerance band
        tau = TOL_TF32 * max(abs(x * cc) for x in gc)
        lo, _, _ = adam_step1(w0, [(x * cc + tau) / cc for x in gc], cc, d['wd'], **OPT)
        hi, _, _ = adam_step1(w0, [(x * cc - tau) / cc for x in gc], cc, d['wd'], **OPT)
        for i, wg in enumerate(gpu[p]):
            slack = 4e-7 * (abs(w0[i]) + OPT['lr'])
            tot_err2 += (wg - gold[p][i]) ** 2
            if not lo[i] - slack <= wg <= hi[i] + slack:
                tot_bad += 1
            if (gg[i] > 0) != (gc[i] > 0):
                tot_flip += 1
                flip_err2 += (wg - gold[p][i]) ** 2
    rep.check(worst_op <= 1e-6, 'adamw op reproduces GPU w from GPU g (float64)', f'worst rel L2 {worst_op:.1e}')
    rep.check(worst_m <= 1e-6, 'adamw op reproduces GPU m, v from GPU g (float64)', f'worst rel L2 {worst_m:.1e}')
    rep.check(tot_bad == 0, f'GPU weights inside the Adam image of the +-{TOL_TF32:g}*max|g| band', f'{tot_bad} elements outside')
    print(f'  info: {tot_flip} gradient sign flips explain {flip_err2 / max(tot_err2, 1e-300):.1%} of the squared weight error')
    return finish(rep)


def finish(rep):
    print('PASS' if not rep.failures else 'FAIL: ' + '; '.join(rep.failures))
    return 0 if not rep.failures else 1


if __name__ == '__main__':
    sys.exit(main())
