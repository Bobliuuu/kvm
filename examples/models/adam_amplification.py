#!/usr/bin/env python3
"""Is a weight divergence between a GPU run and its CPU golden a defect, or AdamW amplifying gradient noise?

  adam_amplification.py CASE_DIR [--steps-from-meta] [--seed 0] [--json OUT]

CASE_DIR holds workload.json, meta.json, bundle/{metadata,golden}.json and dump.txt (kvm_exec --dump-buffers) of a zoo case with `reuse none`, so
the raw gradient buffer of every step survives in BOTH the golden and the dump. The test replays AdamW in float64 from the common initial weights:

  1. replay(golden gradients) must reproduce the golden's final weights (validates the replay: AdamW, clip coefficient, loss scale, schedule-free lr);
  2. replay(GPU gradients)    = what the optimizer does with the GPU's own gradients. If it reproduces the GPU's final weights, the optimizer ran correctly
     and the whole weight divergence is the gradients' (rounding) difference pushed through AdamW;
  3. control: replay(golden gradients x (1 + eps * N(0,1))) with eps = the measured per-tensor relative gradient difference between GPU and golden: the same
     spread appearing from pure synthetic noise at that level shows the divergence is amplification, not a systematic error.

Reports the weight relative L2 (per tensor, worst) for each. Relative L2 of the weights is the metric of examples/models/check.py (`weights rel L2`).
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tests' / 'gptdiff'))
from golden_check import read_dump  # noqa: E402

LR, B1, B2, EPS, WD = 3e-3, 0.9, 0.95, 1e-8, 0.01


def grads_of(spec, steps):
    """step -> {param: gradient buffer name}; the adamw tasks of that step."""
    out = [dict() for _ in range(steps)]
    for t in spec['graphs'][0]['tasks']:
        if t['op'] in ('adamw', 'adamw_lr') and t['id'].startswith('s'):
            s = int(t['id'].split('/')[0][1:])
            out[s][t['args']['p'][2:]] = t['args']['g']
    return out


def replay(w0, g_steps, decay, loss_scale, max_norm):
    """float64 AdamW with the executor's global-norm clip: gs = min(1, max_norm/(|g|+1e-6)) / loss_scale (max_norm = loss_scale * 1.0)."""
    w = {n: v.copy() for n, v in w0.items()}
    m = {n: np.zeros_like(v) for n, v in w0.items()}
    v = {n: np.zeros_like(x) for n, x in w0.items()}
    for t, g in enumerate(g_steps, 1):
        norm = math.sqrt(sum(float((x ** 2).sum()) for x in g.values()))
        gs = min(1.0, max_norm / (norm + 1e-6)) / loss_scale
        c1, c2 = 1 - B1 ** t, 1 - B2 ** t
        for n, x in g.items():
            x = x * gs
            m[n] = B1 * m[n] + (1 - B1) * x
            v[n] = B2 * v[n] + (1 - B2) * x * x
            w[n] = w[n] - LR * (WD if decay[n] else 0.0) * w[n] - LR * (m[n] / c1) / (np.sqrt(v[n] / c2) + EPS)
    return w


def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('case')
    ap.add_argument('--loss-scale', type=float, default=1024.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--trials', type=int, default=5)
    ap.add_argument('--json')
    a = ap.parse_args()
    d = Path(a.case)
    spec = json.loads((d / 'workload.json').read_text())
    meta = json.loads((d / 'bundle/metadata.json').read_text())
    info = json.loads((d / 'meta.json').read_text())
    gold = json.loads((d / 'bundle/golden.json').read_text())['buffers']
    gpu = read_dump(d / 'dump.txt', [b['name'] for b in meta['buffers']])
    steps = info['steps']
    names = grads_of(spec, steps)
    params = list(names[0])
    # initial weights: the workload's init is seeded/fill; recover them by replaying nothing: use the kvmfe initializer
    sys.path.insert(0, str(ROOT / 'frontend'))
    sys.path.insert(0, str(ROOT / 'tests' / 'frontend'))
    import oracle_models as om
    from kvmfe.pipeline import build_model
    import importlib.util
    s = importlib.util.spec_from_file_location('zoo_gen_bundle', ROOT / 'examples/models/gen_bundle.py')
    gb = importlib.util.module_from_spec(s)
    s.loader.exec_module(gb)
    cfg = dict(info['config'])
    if str(cfg.get('precision', '')).startswith('bf16'):
        cfg['precision'] = 'bf16'                         # bf16s differs from bf16 in storage only; parameters and init are identical
    model = build_model(gb.load_model(json.dumps(cfg)))
    w0 = {n: np.array(v, dtype=np.float32).astype(np.float64).reshape(-1) for n, v in om.params(model).items() if n in params}
    decay = {n: model.specs[n].decay for n in params}
    arr = lambda src, s_: {n: np.array(src[names[s_][n]], dtype=np.float64) for n in params}
    g_gold = [arr(gold, s_) for s_ in range(steps)]
    g_gpu = [arr(gpu, s_) for s_ in range(steps)]
    final = lambda src: {n: np.array(src['w.' + n], dtype=np.float64) for n in params}
    w_gold, w_gpu = final(gold), final(gpu)
    mx = a.loss_scale
    r_gold, r_gpu = replay(w0, g_gold, decay, mx, mx), replay(w0, g_gpu, decay, mx, mx)
    worst = lambda x, y: max(((rel(x[n], y[n]), n) for n in params))
    res = dict(case=d.name, steps=steps,
               replay_golden_vs_golden_weights=worst(r_gold, w_gold), replay_gpu_vs_gpu_weights=worst(r_gpu, w_gpu),
               gpu_vs_golden_weights=worst(w_gpu, w_gold), replay_of_gpu_grads_vs_replay_of_golden_grads=worst(r_gpu, r_gold))
    # per-step relative gradient difference, and the synthetic-noise control at that level
    eps = [{n: rel(g_gpu[s_][n], g_gold[s_][n]) for n in params} for s_ in range(steps)]
    res['grad_rel_diff_worst_by_step'] = [max(e.values()) for e in eps]
    rng = np.random.default_rng(a.seed)
    ctl = []
    for _ in range(a.trials):
        noisy = [{n: g_gold[s_][n] * (1 + eps[s_][n] * rng.standard_normal(g_gold[s_][n].shape)) for n in params} for s_ in range(steps)]
        ctl.append(worst(replay(w0, noisy, decay, mx, mx), r_gold)[0])
    res['control_noise_at_measured_level_weights_rel_l2'] = dict(min=min(ctl), max=max(ctl), trials=a.trials)
    for k, v in res.items():
        print(f'{k}: {v}')
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
