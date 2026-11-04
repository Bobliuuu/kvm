#!/usr/bin/env python3
"""Compare a kvm_exec GPU run of a zoo-model bundle with the compiler's CPU golden.

  kvm_exec --dump-buffers BUNDLE > dump.txt ;  python3 check.py WORKLOAD.json BUNDLE dump.txt --mode precise|tf32 [--tol X] [--json OUT]

precise : GEMMs are 3xTF32 -> every f32 buffer (activations, gradients, moments, weights, losses) within --tol relative L2
          (default 1e-3), and every per-step loss within TOL_LOSS_PRECISE = 1e-5. The multi-step weight/moment error is not the GEMM
          error (that is ~1e-6): AdamW divides by sqrt(v), so a gradient element below the fp32 noise moves its weight by +-lr in one
          run and not in the other (results/2026-11-04-a100-gpt-golden); the loss is the sharp check.
lowp    : bf16 models. A bf16 rounding that flips between the GPU and the golden moves one element by 2^-9 relative and the flips
          compound over steps, so whole-trajectory buffers are only bounded loosely: per-step losses within TOL_LOSS_LOWP (1e-3),
          step-1 activations/gradients within TOL_TF32 of max|ref|, weights within TOL_W_LOWP (1e-2) relative L2.
Buffers whose golden L2 norm is below NULL_ABS (1e-8) carry no signal (e.g. the gradient of an attention KEY bias is exactly zero:
softmax is invariant to a per-query constant) and are skipped, together with the weights and second moments AdamW derives from
them (AdamW normalizes that noise to +-lr steps); they are listed in the output.
tf32    : GEMMs are 1xTF32. Per-step losses within TOL_LOSS, and every buffer produced before the first optimizer task of step 1
          (activations, gradients) within TOL_TF32 of max|ref|. Later steps follow a different but equally valid trajectory (the
          optimizer amplifies gradient noise; see results/2026-11-04-a100-gpt-golden), so weights are only reported.
Prints the worst errors; exit status 0 = PASS. --json writes the numbers for the results README.
"""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tests/gptdiff'))
from golden_check import read_dump, rel_l2  # noqa: E402

TOL_TF32 = 6e-3      # matmul 1xTF32 relative to max|ref| (results/2026-11-03-a100-gptops tolerance table)
TOL_LOSS = 2e-3
TOL_LOSS_PRECISE = 1e-5
TOL_LOSS_LOWP = 1e-3
TOL_W_LOWP = 1e-2
NULL_ABS = 1e-8
OPT_OPS = ('adamw', 'adamw_lr')
OUTS = ('out', 'dst', 'o', 'dlogits', 'dx', 'dgamma', 'dbeta', 'dq', 'dk', 'dv', 'dtab', 'rowloss', 'lse', 'loss', 'mean', 'rstd', 'pre',
        'delta', 'partial', 'up', 'dgate', 'dup', 'gs', 'sel', 'slot', 'eidx', 'dprobs')


def step1_buffers(tasks):
    """Buffers produced by step 0 before its first optimizer task and not rewritten by a later step (activations, gradients)."""
    written = lambda t: {v for k, v in t.get('args', {}).items() if isinstance(v, str) and k in OUTS}
    pre, later, seen_opt = set(), set(), False
    for t in tasks:
        seen_opt = seen_opt or t['op'] in OPT_OPS
        if t['id'].startswith('s0/') and not seen_opt:
            pre |= written(t)
        elif not t['id'].startswith('s0/'):
            later |= written(t)
    return {n for n in pre - later if not n.startswith(('w.', 'm.', 'v.', 'loss.', 'aux'))}


def max_rel(x, ref):
    den = max((abs(v) for v in ref), default=0.0)
    return max((abs(a - b) for a, b in zip(x, ref)), default=0.0) / den if den > 0 else max((abs(a) for a in x), default=0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('workload')
    ap.add_argument('bundle')
    ap.add_argument('dump')
    ap.add_argument('--mode', choices=['tf32', 'precise', 'lowp'], required=True)
    ap.add_argument('--tol', type=float, default=1e-3)
    ap.add_argument('--json')
    a = ap.parse_args()
    spec = json.loads(Path(a.workload).read_text())
    meta = json.loads((Path(a.bundle) / 'metadata.json').read_text())
    golden = json.loads((Path(a.bundle) / 'golden.json').read_text())
    gold, dtypes = golden['buffers'], golden['dtypes']
    gpu = read_dump(a.dump, [b['name'] for b in meta['buffers']])
    f32 = [n for n in gold if dtypes[n] == 'f32']
    tasks = spec['graphs'][0]['tasks']
    rep = dict(mode=a.mode, buffers=len(f32), failures=[])
    loss_names = sorted(n for n in f32 if n.startswith(('loss.', 'aux')))
    rep['loss_err'] = {n: rel_l2(gpu[n], gold[n]) for n in loss_names}
    rep['loss_gpu'] = {n: gpu[n][0] for n in loss_names}
    rep['loss_golden'] = {n: gold[n][0] for n in loss_names}

    def check(ok, what, detail):
        print(f'  {"ok  " if ok else "FAIL"} {what}: {detail}')
        if not ok:
            rep['failures'].append(what)

    norm = lambda v: math.sqrt(sum(x * x for x in v))
    null = {n for n in f32 if norm(gold[n]) < NULL_ABS and not n.startswith('loss.')}
    for n in list(null):
        if n.startswith('m.'):
            null |= {'w.' + n[2:], 'v.' + n[2:]} & set(f32)
    live = [n for n in f32 if n not in null]
    rep['null_buffers'] = sorted(null)
    print(f'{a.mode}: {len(f32)} f32 buffers ({len(null)} signal-free skipped: {sorted(null)[:6]}), {len(loss_names)} loss buffers')
    worst_loss = max(rep['loss_err'].values(), default=0.0)
    if a.mode == 'precise':
        errs = sorted(((rel_l2(gpu[n], gold[n]), n) for n in live), reverse=True)
        rep['worst'] = [(n, e) for e, n in errs[:5]]
        print('  worst: ' + ', '.join(f'{n} {e:.2e}' for e, n in errs[:4]))
        check(worst_loss <= TOL_LOSS_PRECISE, f'every per-step loss rel err <= {TOL_LOSS_PRECISE:g}', f'worst {worst_loss:.2e}')
        check(errs[0][0] <= a.tol, f'all buffers rel L2 <= {a.tol:g}', f'worst {errs[0][1]} {errs[0][0]:.2e}')
        rep['max_rel_l2'] = errs[0][0]
    else:
        tol_loss = TOL_LOSS_LOWP if a.mode == 'lowp' else TOL_LOSS
        check(worst_loss <= tol_loss, f'losses rel L2 <= {tol_loss:g}', f'worst {worst_loss:.2e}')
        pre = sorted(n for n in step1_buffers(tasks) if n in gold and dtypes[n] == 'f32' and n not in null)
        errs = sorted(((max_rel(gpu[n], gold[n]), n) for n in pre), reverse=True)
        check(bool(errs) and errs[0][0] <= TOL_TF32, f'{len(pre)} step-1 activation/gradient buffers max|err|/max|ref| <= {TOL_TF32:g}',
              f'worst {errs[0][1]} {errs[0][0]:.2e}' if errs else 'none found')
        rep['max_rel_step1'] = errs[0][0] if errs else None
        post = sorted(((rel_l2(gpu[n], gold[n]), n) for n in live if n.startswith(('w.', 'm.', 'v.'))), reverse=True)
        wts = [e for e, n in post if n.startswith('w.')]
        if post:
            print(f'  info: post-optimizer weights/moments differ by up to {post[0][0]:.2e} ({post[0][1]}) after the run')
            rep['post_opt_rel_l2'] = post[0][0]
        if a.mode == 'lowp' and wts:
            check(wts[0] <= TOL_W_LOWP, f'weights rel L2 <= {TOL_W_LOWP:g}', f'worst {wts[0]:.2e}')
    bad = [n for n in f32 if any(math.isnan(v) or math.isinf(v) for v in gpu[n])]
    check(not bad, 'no NaN/Inf in any GPU buffer', bad[:3])
    print('PASS' if not rep['failures'] else 'FAIL: ' + '; '.join(rep['failures']))
    if a.json:
        Path(a.json).write_text(json.dumps(rep, indent=1))
    return 0 if not rep['failures'] else 1


if __name__ == '__main__':
    sys.exit(main())
