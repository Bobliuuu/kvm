#!/usr/bin/env python3
"""Is a weight divergence between a GPU run and its CPU golden a defect, or AdamW pushing the gradients' rounding difference through its normalised step?

  adam_amplification.py CASE_DIR [--json OUT]

CASE_DIR: workload.json, meta.json, bundle/{metadata,golden}.json, dump.txt (kvm_exec --dump-buffers) of a zoo case compiled with `reuse none`, so the raw
gradient buffer of every step survives in BOTH the golden and the dump. The optimizer is replayed in float64 from the common initial weights, with the
hyper-parameters, learning-rate schedule, loss scale and global-norm clip read from the workload's own tasks:

  replay_golden_vs_golden   replay(golden gradients) against the golden's final weights: validates the replay itself (~1e-7: float32 vs float64)
  replay_gpu_vs_gpu         replay(GPU gradients)    against the GPU's final weights: the optimizer ran correctly on whatever gradients it was given.
                            This is a sharp check (check.py gates on it); it does not depend on gradient noise.
  gpu_vs_golden             what check.py's `weights rel L2` measures
  grad_diff_by_step         relative L2 difference of the two runs' raw gradients, worst tensor, per step
  step2_*                   (>= 2 steps) the exact float64 gradient of step 2's batch at the two replayed step-1 weight sets: the part of the step-2 gradient
                            difference that the step-1 weight difference alone explains, against what was observed.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tests' / 'gptdiff'))
sys.path.insert(0, str(ROOT / 'examples' / 'models'))
from golden_check import read_dump  # noqa: E402


def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


class Optimizer:
    """AdamW exactly as the executor's tasks declare it (read from the workload)."""

    def __init__(self, spec, steps):
        tasks = spec['graphs'][0]['tasks']
        self.grad = [dict() for _ in range(steps)]          # step -> param -> gradient buffer
        self.hp, self.wd, self.sched, self.lr = None, {}, None, None
        self.max_norm, self.loss_scale = 0.0, 1.0
        for t in tasks:
            a = t.get('args', {})
            if t['op'] in ('adamw', 'adamw_lr'):
                self.grad[int(t['id'].split('/')[0][1:])][a['p'][2:]] = a['g']
                self.hp = dict(b1=a['b1'], b2=a['b2'], eps=a['eps'])
                self.wd[a['p'][2:]] = a['wd']
                if 'lr' in a and not isinstance(a['lr'], str):
                    self.lr = a['lr']
            elif t['op'] == 'lr_schedule':
                self.sched = a
            elif t['op'] == 'gradnorm_finish':
                self.max_norm = a['max_norm']
            elif t['op'] == 'scale_flat' and a['out'] == 'clip_coef':
                self.loss_scale = 1.0 / a['alpha']

    def lr_at(self, t):
        if self.sched is None:
            return self.lr
        s = self.sched
        warm = t / s['warmup'] if s['warmup'] > 0 and t <= s['warmup'] else 1.0
        prog = min(max((t - s['warmup']) / max(s['total'] - s['warmup'], 1), 0.0), 1.0)
        mr = s['min_ratio']
        f = {0: 1.0, 1: mr + (1 - mr) * 0.5 * (1 + math.cos(math.pi * prog)), 2: 1.0 - (1 - mr) * prog}[s['kind']]
        return s['base'] * warm * f

    def replay(self, w0, g_steps, upto=None):
        """float64 AdamW; gs = min(1, max_norm / (|g| + 1e-6)) / loss_scale where the norm is that of the (loss-scaled) gradients, as grad_clip + scale_flat."""
        b1, b2, eps = self.hp['b1'], self.hp['b2'], self.hp['eps']
        w = {n: v.copy() for n, v in w0.items()}
        m = {n: np.zeros_like(v) for n, v in w0.items()}
        v = {n: np.zeros_like(x) for n, x in w0.items()}
        for t, g in enumerate(g_steps[:upto], 1):
            norm = math.sqrt(sum(float((x ** 2).sum()) for x in g.values()))
            gs = min(1.0, self.max_norm / (norm + 1e-6)) / self.loss_scale if self.max_norm > 0 else 1.0
            lr, c1, c2 = self.lr_at(t), 1 - b1 ** t, 1 - b2 ** t
            for n, x in g.items():
                x = x * gs
                m[n] = b1 * m[n] + (1 - b1) * x
                v[n] = b2 * v[n] + (1 - b2) * x * x
                w[n] = w[n] - lr * self.wd[n] * w[n] - lr * (m[n] / c1) / (np.sqrt(v[n] / c2) + eps)
        return w


def load_model(info):
    sys.path.insert(0, str(ROOT / 'frontend'))
    sys.path.insert(0, str(ROOT / 'tests' / 'frontend'))
    import importlib.util
    from kvmfe.pipeline import build_model
    s = importlib.util.spec_from_file_location('zoo_gen_bundle', ROOT / 'examples/models/gen_bundle.py')
    gb = importlib.util.module_from_spec(s)
    s.loader.exec_module(gb)
    cfg = dict(info['config'])
    if str(cfg.get('precision', '')).startswith('bf16'):
        cfg['precision'] = 'bf16'                         # bf16s differs from bf16 in storage only; parameters and init are identical
    return build_model(gb.load_model(json.dumps(cfg)))


def analyse(workload, bundle, dump, info, feedback=True, name=''):
    sys.path.insert(0, str(ROOT / 'frontend'))
    sys.path.insert(0, str(ROOT / 'tests' / 'frontend'))
    import oracle_models as om
    spec = json.loads(Path(workload).read_text())
    meta = json.loads((Path(bundle) / 'metadata.json').read_text())
    gold = json.loads((Path(bundle) / 'golden.json').read_text())['buffers']
    gpu = read_dump(dump, [b['name'] for b in meta['buffers']])
    steps = info['steps']
    opt = Optimizer(spec, steps)
    params = list(opt.grad[0])
    model = load_model(info)
    w0 = {n: np.array(v, dtype=np.float32).astype(np.float64).reshape(-1) for n, v in om.params(model).items() if n in params}
    arr = lambda src, s_: {n: np.array(src[opt.grad[s_][n]], dtype=np.float64) for n in params}
    g_gold, g_gpu = [arr(gold, s_) for s_ in range(steps)], [arr(gpu, s_) for s_ in range(steps)]
    final = lambda src: {n: np.array(src['w.' + n], dtype=np.float64) for n in params}
    w_gold, w_gpu = final(gold), final(gpu)
    r_gold, r_gpu = opt.replay(w0, g_gold), opt.replay(w0, g_gpu)
    worst = lambda x, y: max((rel(x[n], y[n]), n) for n in params)
    res = dict(case=name, steps=steps, loss_scale=opt.loss_scale, max_norm=opt.max_norm, schedule=opt.sched,
               replay_golden_vs_golden=worst(r_gold, w_gold), replay_gpu_vs_gpu=worst(r_gpu, w_gpu), gpu_vs_golden=worst(w_gpu, w_gold),
               grad_diff_by_step=[max(rel(g_gpu[s_][n], g_gold[s_][n]) for n in params) for s_ in range(steps)])
    if feedback and steps >= 2:
        res.update(step2_feedback(model, opt, w0, g_gold, g_gpu, params))
    return res


def step2_feedback(model, opt, w0, g_gold, g_gpu, params):
    """Exact float64 gradient of step 2's batch at the two replayed step-1 weight sets (no bf16 rounding anywhere) vs the observed step-2 gradient difference."""
    import torch
    import torch_models as tm
    model.lp = False
    ws = {k: opt.replay(w0, g, upto=1) for k, g in (('gold', g_gold), ('gpu', g_gpu))}
    data = {}
    for st in model.batch_streams(2):
        n = st.rows * st.cols
        a = np.array(st.values[n:2 * n], dtype=np.float64 if st.dtype == 'f32' else np.int64).reshape(st.rows, st.cols)
        data[st.name] = torch.from_numpy(a if st.dtype == 'f32' else a[:, 0])
    grads = {}
    for k, w in ws.items():
        p = {n: torch.from_numpy(w[n].reshape(model.specs[n].rows, model.specs[n].cols)).requires_grad_(True) for n in params}
        sum(tm.FORWARDS[type(model).__name__](model, p, data).values()).backward()
        grads[k] = {n: q.grad.numpy().reshape(-1) for n, q in p.items()}
    return dict(step2_grad_diff_explained_by_step1_weights=max(rel(grads['gpu'][n], grads['gold'][n]) for n in params),
                step2_grad_diff_observed=max(rel(g_gpu[1][n], g_gold[1][n]) for n in params),
                step1_weight_diff_replayed=max(rel(ws['gpu'][n], ws['gold'][n]) for n in params))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('case')
    ap.add_argument('--json')
    a = ap.parse_args()
    d = Path(a.case)
    res = analyse(d / 'workload.json', d / 'bundle', d / 'dump.txt', json.loads((d / 'meta.json').read_text()), name=d.name)
    for k, v in res.items():
        print(f'{k}: {v}')
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
