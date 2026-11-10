#!/usr/bin/env python3
"""Compare a kvm_exec GPU run of a zoo-model bundle with the compiler's CPU golden.

  kvm_exec --dump-buffers BUNDLE > dump.txt ;  python3 check.py WORKLOAD.json BUNDLE dump.txt --mode precise|tf32 [--tol X] [--json OUT]

precise : GEMMs are 3xTF32 -> every f32 buffer (activations, gradients, moments, weights, losses) within --tol relative L2
          (default 1e-3), and every per-step loss within TOL_LOSS_PRECISE = 1e-5. The multi-step weight/moment error is not the GEMM
          error (that is ~1e-6): AdamW divides by sqrt(v), so a gradient element below the fp32 noise moves its weight by +-lr in one
          run and not in the other (results/2026-11-04-a100-gpt-golden); the loss is the sharp check.
lowp    : bf16 models. A bf16 rounding that flips between the GPU and the golden moves one element by 2^-9 relative and the flips
          compound over steps, so whole-trajectory buffers are only bounded loosely: per-step losses within TOL_LOSS_LOWP (1e-3),
          step-1 activations/gradients within TOL_TF32 of max|ref|, and the optimizer replay (AdamW in float64 on the GPU's own gradients reproduces the
          GPU's weights, TOL_REPLAY): the golden-vs-GPU weight divergence has no fixed size, it is the gradients' rounding difference amplified by AdamW.
Parameters with a structurally zero gradient (meta.json `invariant_params`, declared by the model: the key-projection bias of ViT
attention, softmax is invariant to a per-query constant) have no signal to compare: whatever the golden or the GPU holds there is
rounding noise, which AdamW's normalised step turns into +-lr moves (bf16 noise is far above any absolute cut-off, so a magnitude test
cannot find them). They are not compared relative to the golden but judged in absolute terms (check_invariant): the weight must stay
inside the optimizer's travel envelope, the moments at the noise level of the live gradients.
Buffers whose golden L2 norm is below NULL_ABS (1e-8) are skipped as well (bundles without that declaration); both lists are printed.
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
sys.path.insert(0, str(ROOT / 'examples/models'))
from golden_check import read_dump, rel_l2  # noqa: E402

TOL_TF32 = 6e-3      # matmul 1xTF32 relative to max|ref| (results/2026-11-03-a100-gptops tolerance table)
TOL_LOSS = 2e-3
TOL_LOSS_PRECISE = 1e-5
TOL_LOSS_LOWP = 1e-3
TOL_W_LOWP = 1e-2     # fallback only (bundles without meta.json config): see the optimizer replay below
TOL_REPLAY = 1e-5     # float32 executor vs float64 replay measured <= 1.6e-7 on 4 bf16 cases
NULL_ABS = 1e-8
NOISE_REL = 2.0 ** -6   # invariant parameters' moments (sqrt for the second): at most 4 bf16 ulps (2^-8 each) of the live gradients' scale
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


def invariant_buffers(name):
    return {f'{k}.{name}' for k in ('w', 'm', 'v')}


def check_invariant(names, gpu, gold, live, steps, optim):
    """Absolute checks of structurally-zero-gradient parameters. Returns [(ok, what, detail)].

    weight: AdamW moves a weight by at most lr * max(1, (1-b1)/sqrt(1-b2)) per step whatever the gradient (Kingma & Ba, sec. 2.1), so after
            `steps` steps GPU and golden both lie within T = steps * lr * that factor of the same start (the weight starts at the
            bias fill, 0): |w_gpu - w_golden| <= 2T and |w_gpu - w_start| <= T.
    moments: the gradient is rounding noise of the arithmetic that computed it, so |m_gpu| <= NOISE_REL * max|m| over the live
            parameters, and the same for sqrt(v)."""
    out = []
    step = optim['lr'] * max(1.0, (1 - optim['b1']) / math.sqrt(1 - optim['b2']))
    T = steps * step
    amax = lambda v: max((abs(x) for x in v), default=0.0)
    scale_m = max((amax(gold[n]) for n in live if n.startswith('m.')), default=0.0)
    scale_v = max((math.sqrt(amax(gold[n])) for n in live if n.startswith('v.')), default=0.0)
    for p in names:
        wn, mn, vn = (f'{k}.{p}' for k in ('w', 'm', 'v'))
        if wn not in gpu:
            continue
        d = max((abs(a - b) for a, b in zip(gpu[wn], gold[wn])), default=0.0)
        out.append((d <= 2 * T and amax(gpu[wn]) <= T * (1 + 1e-6), f'{p} weight (structurally zero gradient) within the optimizer envelope',
                    f'|w_gpu - w_golden| max {d:.3e}, |w_gpu| max {amax(gpu[wn]):.3e}, envelope T = {steps} steps * {step:.1e} = {T:.3e}'))
        mg, vg = amax(gpu[mn]), math.sqrt(amax(gpu[vn]))
        out.append((mg <= NOISE_REL * scale_m and vg <= NOISE_REL * scale_v, f'{p} moments at the noise level of the live gradients',
                    f'|m_gpu| max {mg:.3e} (golden {amax(gold[mn]):.3e}) vs {NOISE_REL * scale_m:.3e}, sqrt|v_gpu| max {vg:.3e} (golden {math.sqrt(amax(gold[vn])):.3e}) vs {NOISE_REL * scale_v:.3e}'))
    return out


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
    meta_file = Path(a.workload).parent / 'meta.json'
    info = json.loads(meta_file.read_text()) if meta_file.exists() else {}
    inv = [p for p in info.get('invariant_params', []) if f'w.{p}' in gold]
    inv_bufs = {b for p in inv for b in invariant_buffers(p)}
    # their gradient buffers carry the same rounding noise (they feed only the optimizer of that parameter); judged with the moments
    inv_grads = {t['args']['g'] for t in tasks if t['op'] in OPT_OPS and t['args'].get('p') in {f'w.{q}' for q in inv}} & set(f32)
    inv_bufs |= inv_grads
    null = {n for n in f32 if norm(gold[n]) < NULL_ABS and not n.startswith('loss.')} | inv_bufs
    for n in list(null):
        if n.startswith('m.'):
            null |= {'w.' + n[2:], 'v.' + n[2:]} & set(f32)
    live = [n for n in f32 if n not in null]
    rep['null_buffers'] = sorted(null)
    rep['invariant_params'] = inv
    print(f'{a.mode}: {len(f32)} f32 buffers ({len(null)} signal-free skipped: {sorted(null)[:6]}; {len(inv)} structurally-zero-gradient parameters '
          f'judged in absolute terms), {len(loss_names)} loss buffers')
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
        if a.mode == 'lowp' and wts and info.get('config'):
            # The golden-vs-GPU weight divergence is AdamW's normalised step applied to the two runs' (rounding-different) gradients: it has no
            # fixed size (san_bf16s, 2 steps: 1.7e-2, its 10-step sibling 8.5e-3). The sharp check is that the optimizer did the right thing with the
            # gradients it was given: replaying AdamW in float64 on the GPU's own gradients must reproduce the GPU's weights. The gradients
            # themselves are gated above (losses of every step, step-1 activations and gradients).
            import adam_amplification as aa
            r = aa.analyse(a.workload, a.bundle, a.dump, info, feedback=False, name=Path(a.workload).parent.name)
            rep['optimizer_replay'] = dict(gpu_weights=r['replay_gpu_vs_gpu'], golden_weights=r['replay_golden_vs_golden'], gpu_vs_golden=r['gpu_vs_golden'], grad_diff_by_step=r['grad_diff_by_step'])
            print(f'  info: weights differ from the golden by {r["gpu_vs_golden"][0]:.2e} ({r["gpu_vs_golden"][1]}); gradients differ per step by up to '
                  f'{max(r["grad_diff_by_step"]):.1e}, relative L2 of worst tensor, per step {[round(x, 4) for x in r["grad_diff_by_step"]][:6]}')
            check(r['replay_golden_vs_golden'][0] <= TOL_REPLAY, f'optimizer replay reproduces the golden weights (validates the replay) <= {TOL_REPLAY:g}', f'worst {r["replay_golden_vs_golden"][0]:.2e} ({r["replay_golden_vs_golden"][1]})')
            check(r['replay_gpu_vs_gpu'][0] <= TOL_REPLAY, f'AdamW replayed on the GPU\'s own gradients reproduces the GPU weights <= {TOL_REPLAY:g}', f'worst {r["replay_gpu_vs_gpu"][0]:.2e} ({r["replay_gpu_vs_gpu"][1]})')
        elif a.mode == 'lowp' and wts:
            check(wts[0] <= TOL_W_LOWP, f'weights rel L2 <= {TOL_W_LOWP:g}', f'worst {wts[0]:.2e}')
    if inv and not (info.get('steps') and info.get('optim')):
        check(False, 'invariant parameters need meta.json steps and optim', 'regenerate the bundle with gen_bundle.py')
    elif inv:
        for ok, what, detail in check_invariant(inv, gpu, gold, live, info['steps'], info['optim']):
            check(ok, what, detail)
        amax = lambda v: max((abs(x) for x in v), default=0.0)
        live_g = {t['args']['g'] for t in tasks if t['op'] in OPT_OPS} & set(live)
        scale_g = max((amax(gold[n]) for n in live_g), default=0.0)
        for n in sorted(inv_grads):
            check(amax(gpu[n]) <= NOISE_REL * scale_g, f'{n} (gradient of a structurally-zero-gradient parameter) at the noise level',
                  f'|g_gpu| max {amax(gpu[n]):.3e} (golden {amax(gold[n]):.3e}) vs {NOISE_REL * scale_g:.3e}')
    bad = [n for n in f32 if any(math.isnan(v) or math.isinf(v) for v in gpu[n])]
    check(not bad, 'no NaN/Inf in any GPU buffer', bad[:3])
    print('PASS' if not rep['failures'] else 'FAIL: ' + '; '.join(rep['failures']))
    if a.json:
        Path(a.json).write_text(json.dumps(rep, indent=1))
    return 0 if not rep['failures'] else 1


if __name__ == '__main__':
    sys.exit(main())
