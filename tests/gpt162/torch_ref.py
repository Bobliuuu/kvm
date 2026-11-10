#!/usr/bin/env python3
"""PyTorch reference for a kvmfe.demo training directory (any GPT preset; built for gpt_162m), from the executor's own bytes.

  torch_ref.py DEMO_DIR --model gpt_162m|JSON --steps N --lr LR --warmup W --out RESULT.json
               [--grads RUN_DIR/buffers]  [--curve] [--device cuda]

DEMO_DIR is what `python3 -m kvmfe.demo --out DEMO_DIR --train-steps N --window 1 ...` wrote. Initial weights and the token stream are
rebuilt from workloads/train.json exactly as the executor allocates them (counter-based generator in numpy, self-checked against
runtime/initgen.py; token stream from the data file), so both sides start from identical bytes. Optimiser: AdamW betas (0.9, 0.95),
eps 1e-8, weight decay 0.01 on matrices only, global grad-norm clip 1.0 (coef = min(1, 1/(norm + 1e-6))), device-side linear
warmup lr_t = lr * t/W for t <= W (t 1-based) then constant (lr_schedule op, include/kvm_ops_extra.cuh).

  --grads DIR   compare step 1 with a kvm_run --dump of the one-step run (reuse none): loss, per-parameter gradient rel L2 and
                global norm against torch fp64 (truth), torch 1xTF32 (same arithmetic class as the executor) and torch fp32-highest;
                m after step 1 (= (1-b1) * coef * g, linear in g so no sign-flip noise) and the weight update w1 - w0.
  --holdout N   with --curve: also the mean loss on N fresh batches (steps 10^6+i of the same stream, never trained on) at init, after the
                torch runs and, with --exec-weights DIR (a kvm_run --dump '^w\\.' of the N-step run), for the executor's final weights
  --curve       train N steps in torch (1xTF32 and fp32-highest) and print/record the loss curves; --obs FILE adds the executor's.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'bench' / 'vs_pytorch'))
import torch_bench as tb  # noqa: E402
from kvmfe.nn import GPT, ModelConfig  # noqa: E402

B1, B2, EPS, WD, CLIP = 0.9, 0.95, 1e-8, 0.01, 1.0


def set_precision(mode):
    tf = mode == 'tf32'
    torch.backends.cuda.matmul.allow_tf32 = tf
    torch.backends.cudnn.allow_tf32 = tf
    torch.set_float32_matmul_precision('high' if tf else 'highest')


class Ref:
    def __init__(self, demo, model, dev):
        self.cfg = ModelConfig.load(model)
        wl = Path(demo) / 'workloads'
        spec = json.loads((wl / 'train.json').read_text())
        bufs = {b['name']: b for b in spec['buffers']}
        self.specs = GPT(self.cfg).specs
        tb.self_check_init()
        self.init = {}
        for n, s in self.specs.items():
            v = tb.init_values(bufs['w.' + n], wl).reshape(s.rows, s.cols)
            self.init[n] = torch.from_numpy(np.ascontiguousarray(v)).to(dev)
        self.shape = {n: (v.shape if self.specs[n].rows > 1 else (v.shape[1],)) for n, v in self.init.items()}
        per = self.cfg.tokens
        self.ids = torch.from_numpy(tb.init_values(bufs['stream_ids'], wl).astype(np.int64)).to(dev).view(-1, per)
        self.tgt = torch.from_numpy(tb.init_values(bufs['stream_tgt'], wl).astype(np.int64)).to(dev).view(-1, per)
        self.dev = dev
        self.tm = tb.TorchGPT(self.cfg, self.specs, dev)

    def params(self, dtype):
        return {n: v.to(dtype).reshape(self.shape[n]).clone().requires_grad_(True) for n, v in self.init.items()}

    def loss_grads(self, p, step, dtype):
        if dtype == torch.float64:                         # exact rope tables for the truth run
            c = self.cfg
            hd = c.d_model // c.heads
            inv = c.rope_theta ** (-2.0 * torch.arange(hd // 2, dtype=torch.float64) / hd)
            ang = torch.arange(c.seq, dtype=torch.float64)[:, None] * inv[None]
            self.tm.cos, self.tm.sin = ang.cos().to(self.dev), ang.sin().to(self.dev)
        else:
            self.tm.cos, self.tm.sin = self.tm.cos.float(), self.tm.sin.float()
        for q in p.values():
            q.grad = None
        loss = self.tm.loss(p, self.ids[step], self.tgt[step])
        loss.backward()
        return loss.detach(), {n: q.grad.detach().reshape(self.init[n].shape) for n, q in p.items()}


def gnorm(g):
    return math.sqrt(sum(float((x.double() ** 2).sum()) for x in g.values()))


def rel(a, b):
    a, b = a.double(), b.double()
    return float(((a - b).norm() / b.norm().clamp_min(1e-300)))


def lr_at(t, lr, warmup):
    return lr * (t / warmup) if warmup > 0 and t <= warmup else lr


def adamw_step1(w0, g, coef, lr, decay):
    """float64 AdamW at t=1 on the clipped gradient."""
    x = g.double() * coef
    m, v = (1 - B1) * x, (1 - B2) * x * x
    w1 = w0.double() * (1 - lr * (WD if decay else 0.0)) - lr * (m / (1 - B1)) / ((v / (1 - B2)).sqrt() + EPS)
    return w1, m


def read_dump(d, name, shape, dev):
    return torch.from_numpy(np.fromfile(Path(d) / f'{name}.bin', dtype='<f4').reshape(shape)).to(dev)


def grads_mode(ref, a, out):
    """Executor step-1 gradients vs torch."""
    runs = {}
    for label, dtype, prec in (('fp64', torch.float64, 'precise'), ('tf32', torch.float32, 'tf32'), ('fp32', torch.float32, 'precise'),
                               ('bf16', torch.float32, 'precise')):
        set_precision(prec)
        ref.tm.autocast = label == 'bf16'          # torch.autocast(bfloat16): the arithmetic class of the executor's bf16 precision modes
        p = ref.params(dtype)
        loss, g = ref.loss_grads(p, 0, dtype)
        runs[label] = (float(loss), {n: x.cpu() for n, x in g.items()}, gnorm(g))
        del p, g
        torch.cuda.empty_cache()
        print(f'torch {label}: loss {runs[label][0]:.6f} global grad norm {runs[label][2]:.6f}', flush=True)
    spec = json.loads((Path(a.demo) / 'workloads' / 'train.json').read_text())
    gname = {}      # parameter -> executor gradient buffer, from the adamw tasks
    for gr in spec['graphs']:
        for t in gr['tasks']:
            if t['op'] in ('adamw', 'adamw_lr'):
                gname[t['args']['p'][2:]] = t['args']['g']
    d = Path(a.grads)
    eg = {n: read_dump(d, gname[n], ref.init[n].shape, 'cpu') for n in ref.specs}
    el = float(read_dump(d, 'loss.0', (1, 2), 'cpu').view(-1)[0])
    coef_e = float(read_dump(d, 'clip_coef', (1, 2), 'cpu').view(-1)[0])
    en = gnorm(eg)
    res = {'executor_loss': el, 'executor_grad_norm': en, 'executor_clip_coef': coef_e,
           'torch': {k: {'loss': v[0], 'grad_norm': v[2]} for k, v in runs.items()}, 'params': {}}
    truth = runs['fp64'][1]
    coef_t = min(1.0, CLIP / (runs['fp64'][2] + 1e-6))
    res['truth_clip_coef'] = coef_t
    print(f'executor: loss {el:.6f} global grad norm {en:.6f} clip_coef {coef_e:.6f}  (fp64 truth coef {coef_t:.6f})')
    print(f'loss vs fp64: executor {el - runs["fp64"][0]:+.2e}  torch-tf32 {runs["tf32"][0] - runs["fp64"][0]:+.2e}  '
          f'torch-fp32 {runs["fp32"][0] - runs["fp64"][0]:+.2e}  torch-autocast-bf16 {runs["bf16"][0] - runs["fp64"][0]:+.2e}')
    rows = []
    for n, s in ref.specs.items():
        r = {'shape': [s.rows, s.cols], 'norm_truth': float(truth[n].double().norm()),
             'exec_vs_fp64': rel(eg[n], truth[n]), 'tf32_vs_fp64': rel(runs['tf32'][1][n], truth[n]),
             'fp32_vs_fp64': rel(runs['fp32'][1][n], truth[n]), 'bf16_vs_fp64': rel(runs['bf16'][1][n], truth[n]),
             'exec_vs_tf32': rel(eg[n], runs['tf32'][1][n]), 'exec_vs_bf16': rel(eg[n], runs['bf16'][1][n])}
        res['params'][n] = r
        rows.append((n, r))
    # per-group aggregation (embedding, attention, mlp, norms, head)
    groups = {}
    for n, r in rows:
        k = ('wte' if n == 'gpt.wte' else 'head' if n == 'gpt.head.w' else 'final_norm' if n.startswith('gpt.nf') else
             'attn' if '.attn.' in n else 'mlp' if '.mlp.' in n else 'norm')
        groups.setdefault(k, []).append(n)
    res['groups'] = {}
    for k, names in groups.items():
        tn = math.sqrt(sum(float(truth[n].double().pow(2).sum()) for n in names))
        e = lambda src: math.sqrt(sum(float((src(n).double() - truth[n].double()).pow(2).sum()) for n in names)) / tn
        res['groups'][k] = {'n_tensors': len(names), 'grad_norm_truth': tn, 'grad_norm_exec': math.sqrt(sum(float(eg[n].double().pow(2).sum()) for n in names)),
                            'exec_vs_fp64': e(lambda n: eg[n]), 'tf32_vs_fp64': e(lambda n: runs['tf32'][1][n]), 'fp32_vs_fp64': e(lambda n: runs['fp32'][1][n]),
                            'bf16_vs_fp64': e(lambda n: runs['bf16'][1][n])}
        print(f'  group {k:10s} ({len(names):3d} tensors) |g| truth {tn:.4e} exec {res["groups"][k]["grad_norm_exec"]:.4e}  rel L2 vs fp64: '
              f'exec {res["groups"][k]["exec_vs_fp64"]:.2e}  torch-tf32 {res["groups"][k]["tf32_vs_fp64"]:.2e}  torch-fp32 {res["groups"][k]["fp32_vs_fp64"]:.2e}  '
              f'torch-autocast-bf16 {res["groups"][k]["bf16_vs_fp64"]:.2e}')
    worst = max(rows, key=lambda x: x[1]['exec_vs_fp64'])
    print(f'  worst tensor: {worst[0]} exec {worst[1]["exec_vs_fp64"]:.2e} (torch-tf32 on it {worst[1]["tf32_vs_fp64"]:.2e}, torch-bf16 {worst[1]["bf16_vs_fp64"]:.2e}); '
          f'all {len(rows)} tensors: max exec {worst[1]["exec_vs_fp64"]:.2e}, max torch-tf32 {max(r["tf32_vs_fp64"] for _, r in rows):.2e}, '
          f'max torch-autocast-bf16 {max(r["bf16_vs_fp64"] for _, r in rows):.2e}')
    # after the update: m (linear in g) and the weight delta
    lr1 = lr_at(1, a.lr, a.warmup)
    mm = {'m': [], 'dw': []}
    for n, s in ref.specs.items():
        w1, m_ref = adamw_step1(ref.init[n].cpu(), truth[n], coef_t, lr1, s.decay)
        try:
            m_e = read_dump(d, 'm.' + n, ref.init[n].shape, 'cpu')
            w_e = read_dump(d, 'w.' + n, ref.init[n].shape, 'cpu')
        except FileNotFoundError:
            continue
        mm['m'].append(rel(m_e, m_ref))
        # update quality: AdamW's first step is ~lr*sign(g), so tiny gradients flip; report the fraction of elements whose update sign differs
        du_e, du_r = w_e.double() - ref.init[n].cpu().double(), w1 - ref.init[n].cpu().double()
        mm['dw'].append((rel(du_e, du_r), float(((du_e > 0) != (du_r > 0)).double().mean())))
    if mm['m']:
        res['m_after_step1_rel_l2'] = {'max': max(mm['m']), 'median': float(np.median(mm['m']))}
        res['dw_after_step1'] = {'rel_l2_max': max(x[0] for x in mm['dw']), 'rel_l2_median': float(np.median([x[0] for x in mm['dw']])),
                                 'sign_flip_frac_max': max(x[1] for x in mm['dw']), 'sign_flip_frac_median': float(np.median([x[1] for x in mm['dw']]))}
        print('  m after step 1 vs fp64 reference (rel L2): ' + json.dumps(res['m_after_step1_rel_l2']))
        print('  w1 - w0 vs fp64 reference: ' + json.dumps(res['dw_after_step1']))
    out['grads'] = res


HOLDOUT_BASE = 10 ** 6


def holdout_loss(ref, p, n, dtype=torch.float32):
    """Mean loss of weights `p` on n fresh batches of the stream (steps HOLDOUT_BASE+i; the noise and starts differ from every training step)."""
    from kvmfe.train import batch_tokens
    tot = 0.0
    with torch.no_grad():
        for i in range(n):
            ids, tgt = batch_tokens(ref.cfg, HOLDOUT_BASE + i)
            tot += float(ref.tm.loss(p, torch.tensor(ids, device=ref.dev), torch.tensor(tgt, device=ref.dev)))
    return tot / n


def curve_mode(ref, a, out):
    cur = {}
    hold = {}
    if a.holdout:
        set_precision('tf32')
        hold['init'] = holdout_loss(ref, ref.params(torch.float32), a.holdout)
        print(f'holdout init: {hold["init"]:.4f}', flush=True)
    labels = a.curve_labels.split(',')
    for label, prec in (('tf32', 'tf32'), ('fp32', 'precise'), ('bf16', 'tf32')):
        if label not in labels:
            continue
        set_precision(prec)
        ref.tm.autocast = label == 'bf16'          # torch.autocast(bfloat16): fp32 master weights and AdamW, bf16 matmuls/attention
        p = ref.params(torch.float32)
        decay = [p[n] for n, s in ref.specs.items() if s.decay]
        nod = [p[n] for n, s in ref.specs.items() if not s.decay]
        opt = torch.optim.AdamW([{'params': decay, 'weight_decay': WD}, {'params': nod, 'weight_decay': 0.0}], lr=a.lr, betas=(B1, B2), eps=EPS)
        pl, losses = list(p.values()), []
        for t in range(a.steps):
            for g in opt.param_groups:
                g['lr'] = lr_at(t + 1, a.lr, a.warmup)
            loss, _ = ref.loss_grads(p, t, torch.float32)
            torch.nn.utils.clip_grad_norm_(pl, CLIP)
            opt.step()
            losses.append(float(loss))
        cur[label] = losses
        ref.tm.autocast = False
        if a.holdout:
            hold[label] = holdout_loss(ref, p, a.holdout)
            print(f'torch {label}: holdout {hold[label]:.4f}', flush=True)
        print(f'torch {label}: ' + ' '.join(f'{x:.4f}' for x in losses), flush=True)
        del p, opt
        torch.cuda.empty_cache()
    out['curve'] = cur
    if a.holdout and a.exec_weights:
        set_precision('tf32')
        pe = {n: read_dump(a.exec_weights, 'w.' + n, ref.init[n].shape, a.device).reshape(ref.shape[n]).clone() for n in ref.specs}
        hold['executor'] = holdout_loss(ref, pe, a.holdout)
        print(f'executor weights: holdout {hold["executor"]:.4f}', flush=True)
    if hold:
        out['holdout'] = hold
        out['holdout_batches'] = a.holdout
    if a.obs:
        ex = {}
        for line in Path(a.obs).read_text().splitlines():
            o = json.loads(line)
            if o['adm'].startswith('train_w') and 'loss.0' in o['buffers']:
                ex[int(o['adm'][len('train_w'):])] = o['buffers']['loss.0']['values'][0]
        e = [ex[i] for i in range(a.steps)]
        out['curve']['executor'] = e
        print('executor: ' + ' '.join(f'{x:.4f}' for x in e))
        cmp = {}
        for label in cur:
            if label == 'executor':
                continue
            d = [x - y for x, y in zip(e, cur[label])]
            k = max(1, a.steps // 5)
            cmp[label] = {'step1_diff': d[0], 'max_abs_diff': max(abs(x) for x in d), 'mean_abs_diff': sum(abs(x) for x in d) / len(d),
                          'mean_diff': sum(d) / len(d), 'first_step_over_1e-3': next((i + 1 for i, x in enumerate(d) if abs(x) > 1e-3), None),
                          'exec_last_mean': sum(e[-k:]) / k, 'torch_last_mean': sum(cur[label][-k:]) / k}
            print(f'executor - torch {label}: ' + json.dumps(cmp[label]))
        k = max(1, a.steps // 5)
        for x_, y_ in (('tf32', 'fp32'), ('bf16', 'tf32')):
            if x_ in cur and y_ in cur:
                cmp[f'torch_{x_}_vs_{y_}_max_abs_diff'] = max(abs(x - y) for x, y in zip(cur[x_], cur[y_]))
                print(f'torch {x_} vs torch {y_} (precision noise floor): max |diff| {cmp[f"torch_{x_}_vs_{y_}_max_abs_diff"]:.4f}')
        out['curve_compare'] = cmp


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('demo')
    ap.add_argument('--model', default='gpt_162m')
    ap.add_argument('--steps', type=int, default=1)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--warmup', type=int, default=0)
    ap.add_argument('--grads')
    ap.add_argument('--curve', action='store_true')
    ap.add_argument('--obs', help='kvm_run observations.jsonl of the same N-step window-1 run (adds executor losses and the comparison)')
    ap.add_argument('--holdout', type=int, default=0, help='with --curve: evaluate on this many fresh held-out batches')
    ap.add_argument('--exec-weights', help='directory of a kvm_run --dump of the executor weights (w.<param>.bin) for the hold-out evaluation')
    ap.add_argument('--curve-labels', default='tf32,fp32', help="with --curve: torch curves to run, of tf32, fp32, bf16 (autocast)")
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out')
    a = ap.parse_args()
    ref = Ref(a.demo, a.model, a.device)
    out = {'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0) if a.device == 'cuda' else 'cpu',
           'lr': a.lr, 'warmup': a.warmup, 'steps': a.steps}
    if a.grads:
        grads_mode(ref, a, out)
    if a.curve:
        curve_mode(ref, a, out)
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
