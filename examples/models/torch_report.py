#!/usr/bin/env python3
"""Tie the executor to PyTorch for every model family of the zoo (goal 2, independent reference: not the repo's CPU golden).

  torch_report.py cpu     [--json OUT]                    frontend program on the CPU float64 VM vs torch autograd (examples/models/torch_models.py)
  torch_report.py build   OUT_DIR                         one-step bundles of CASES (no CPU golden needed), for kvm_exec
  torch_report.py compare OUT_DIR [--json OUT] [--md OUT] executor step-1 loss and all gradients (kvm_exec --dump-buffers of each bundle,
                                                          OUT_DIR/CASE/dump.txt) vs torch float64 autograd on the same initial bytes and batch

GPU side, per case: `kvm_exec --quiet --dump-buffers OUT_DIR/CASE/bundle > OUT_DIR/CASE/dump.txt` (examples/models/torch_run.sh does it under the GPU lock).
Bundles use `reuse none`, so the step-0 gradient buffer of every parameter survives the run: it is the `g` of the step-0 adamw tasks.

Tolerances (relative L2 per tensor against torch float64; loss relative; see the table in the results README for the measured values):
  precise  3xTF32 tiles, fp32-class      loss 1e-5   gradients 3e-5   (measured on the A100: loss <= 4.8e-7, gradients <= 3.0e-6)
  tf32     1xTF32 tiles                  loss 5e-4   gradients 3e-3   (measured: loss <= 5.0e-5, gradients <= 8.6e-4; 10-bit mantissa operands, the class of results/2026-11-03-a100-gptops)
  lowp     bf16 emulation tiles          loss 1e-4   gradients 2e-2   (measured: loss <= 2.8e-8, gradients <= 6.8e-3, a bf16 ulp is 3.9e-3; the torch side emulates the same
                                                                      bf16 roundings, and gradients scaled by the loss scale are divided out)
The gradient bound of an exactly-zero-gradient tensor (ViT key bias) is absolute: |g| <= 1e-3 of the largest gradient norm of the model.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'tests' / 'gptdiff'))
sys.path.insert(0, str(ROOT / 'frontend'))

TOL = {'precise': dict(loss=1e-5, grad=3e-5), 'tf32': dict(loss=5e-4, grad=3e-3), 'lowp': dict(loss=1e-4, grad=2e-2)}
ZERO_GRAD_REL = 1e-3

# name, model (preset / JSON), mode, extra gen_bundle flags
BF16 = lambda fam, name, extra='': json.dumps({'family': fam, 'name': name, 'precision': 'bf16', **({} if not extra else json.loads(extra))})
CASES = [
    ('gpt_small_precise', 'gpt_small', 'precise', ''),
    ('gpt_small_tf32', 'gpt_small', 'tf32', ''),
    ('llama_precise', 'llama_tiny', 'precise', ''),
    ('llama_tf32', 'llama_tiny', 'tf32', ''),
    ('llama_gqa_tied_precise', 'llama_gqa_tied', 'precise', ''),
    ('moe_precise', 'moe_tiny', 'precise', ''),
    ('moe_tf32', 'moe_tiny', 'tf32', ''),
    ('vit_precise', 'vit_tiny', 'precise', ''),
    ('vit_tf32', 'vit_tiny', 'tf32', ''),
    ('mlp_precise', 'mlp_tiny', 'precise', ''),
    ('mlp_tf32', 'mlp_tiny', 'tf32', ''),
    ('llama_bf16', BF16('llama', 'llama_bf16'), 'lowp', '--loss-scale 1024'),
    ('mlp_bf16', BF16('mlp', 'mlp_bf16'), 'lowp', '--loss-scale 128'),
    ('vit_bf16', BF16('vit', 'vit_bf16'), 'lowp', '--loss-scale 128'),
]


def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b)) if np.linalg.norm(b) > 0 else float(np.linalg.norm(a))


def build_model_of(spec):
    import importlib.util
    from kvmfe.pipeline import build_model
    s = importlib.util.spec_from_file_location('zoo_gen_bundle', HERE / 'gen_bundle.py')       # by path: tests/gptdiff has a gen_bundle.py too
    gb = importlib.util.module_from_spec(s)
    s.loader.exec_module(gb)
    return build_model(gb.load_model(spec))


# ---------------------------------------------------------------------------------------------- cpu
def cmd_cpu(a):
    sys.path.insert(0, str(ROOT / 'tests' / 'frontend'))
    import test_torch_models as t
    rows = []
    for name, (cfg, tl, tg) in t.CASES.items():
        w = t.TorchReference().check(name)
        rows.append(dict(case=name, grad_tol=tg, worst_grad_rel_l2=w))
        print(f'{name:16s} worst gradient rel L2 vs torch autograd {w:.2e} (tolerance {tg:g})')
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1))


# ---------------------------------------------------------------------------------------------- build
def cmd_build(a):
    out = Path(a.out)
    for name, model, mode, extra in CASES:
        d = out / name
        flags = ['--model', model, '--steps', '1', '--no-golden', *(['--precise'] if mode == 'precise' else []), *extra.split()]
        subprocess.run([sys.executable, str(HERE / 'gen_bundle.py'), str(d), *flags], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run([sys.executable, str(ROOT / 'runtime/compile_workload.py'), str(d / 'workload.json'), str(d / 'bundle')],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        (d / 'case.json').write_text(json.dumps(dict(name=name, model=model, mode=mode, extra=extra)))
        print(f'{name}: bundle in {d / "bundle"}')


# ---------------------------------------------------------------------------------------------- compare
def read_dump(path, names):
    from golden_check import read_dump as rd
    return rd(path, names)


def compare_case(d: Path):
    import torch_models as tm
    case = json.loads((d / 'case.json').read_text())
    mode, extra = case['mode'], case['extra'].split()
    scale = float(extra[extra.index('--loss-scale') + 1]) if '--loss-scale' in extra else 1.0
    spec = json.loads((d / 'workload.json').read_text())
    meta = json.loads((d / 'bundle' / 'metadata.json').read_text())
    dump = read_dump(d / 'dump.txt', [b['name'] for b in meta['buffers']])
    model = build_model_of(case['model'])
    ref_loss, ref_grad = tm.reference(model)
    g_buf = {}
    for t in spec['graphs'][0]['tasks']:
        if t['op'] in ('adamw', 'adamw_lr') and t['id'].startswith('s0/'):
            g_buf[t['args']['p'][2:]] = t['args']['g']
    tol = TOL[mode]
    res = dict(case=case['name'], mode=mode, loss_scale=scale, params=len(ref_grad), losses={}, grads={}, failures=[])
    for k, v in ref_loss.items():                           # 'loss' -> loss.0 ; 'auxN' -> auxN.0
        buf = 'loss.0' if k == 'loss' else f'{k}.0'
        got = dump[buf][0]
        res['losses'][k] = dict(executor=got, torch=v, rel=abs(got - v) / max(abs(v), 1e-30))
        if res['losses'][k]['rel'] > tol['loss']:
            res['failures'].append(f'{k} rel {res["losses"][k]["rel"]:.2e} > {tol["loss"]:g}')
    norms = {n: float(np.linalg.norm(g)) for n, g in ref_grad.items()}
    gmax = max(norms.values())
    worst = ('', 0.0)
    for n, g in ref_grad.items():
        e = np.array(dump[g_buf[n]], dtype=np.float64).reshape(g.shape) / scale
        if model.specs[n].invariant:                        # declared structurally zero (and zero in torch: asserted by tests/frontend/test_models.py): absolute
            r = float(np.abs(e).max()) / (ZERO_GRAD_REL * gmax)
            res['grads'][n] = dict(zero_grad=True, executor_max_abs=float(np.abs(e).max()), bound=ZERO_GRAD_REL * gmax)
            if r > 1:
                res['failures'].append(f'{n}: structurally zero gradient, executor {np.abs(e).max():.2e} > {ZERO_GRAD_REL * gmax:.2e}')
            continue
        r = rel_l2(e, g)
        res['grads'][n] = dict(rel_l2=r, norm_torch=norms[n])
        if r > worst[1]:
            worst = (n, r)
        if r > tol['grad']:
            res['failures'].append(f'{n}: gradient rel L2 {r:.2e} > {tol["grad"]:g}')
    gn_t = math.sqrt(sum(v * v for v in norms.values()))
    gn_e = math.sqrt(sum(float(np.sum((np.array(dump[g_buf[n]]) / scale) ** 2)) for n in ref_grad))
    res.update(worst_param=worst[0], worst_grad_rel_l2=worst[1], global_norm_torch=gn_t, global_norm_executor=gn_e,
               global_norm_rel=abs(gn_e - gn_t) / gn_t, tol=tol, ok=not res['failures'])
    return res


def cmd_compare(a):
    out = Path(a.out)
    rows = []
    for name, *_ in CASES:
        if not (out / name / 'dump.txt').exists():
            print(f'{name}: no dump.txt, skipped')
            continue
        r = compare_case(out / name)
        rows.append(r)
        loss = ', '.join(f'{k} {v["rel"]:.1e}' for k, v in r['losses'].items())
        print(f'{"PASS" if r["ok"] else "FAIL"} {name:24s} {r["mode"]:7s} loss rel {loss}; {r["params"]} grads, worst {r["worst_grad_rel_l2"]:.2e} ({r["worst_param"]}), '
              f'global norm rel {r["global_norm_rel"]:.1e}; tol loss {r["tol"]["loss"]:g} grad {r["tol"]["grad"]:g}' + ('' if r['ok'] else ' :: ' + '; '.join(r['failures'][:3])))
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1))
    if a.md:
        lines = ['| case | mode | params | loss rel err vs torch f64 | worst gradient rel L2 (tensor) | global grad norm rel | tolerance (loss / grad) | verdict |', '|---|---|---|---|---|---|---|---|']
        for r in rows:
            lines.append(f'| {r["case"]} | {r["mode"]} | {r["params"]} | ' + ', '.join(f'{v["rel"]:.1e}' for v in r['losses'].values()) +
                         f' | {r["worst_grad_rel_l2"]:.2e} ({r["worst_param"]}) | {r["global_norm_rel"]:.1e} | {r["tol"]["loss"]:g} / {r["tol"]["grad"]:g} | {"PASS" if r["ok"] else "FAIL"} |')
        Path(a.md).write_text('\n'.join(lines) + '\n')
    return 0 if rows and all(r['ok'] for r in rows) else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('cpu')
    p.add_argument('--json')
    p.set_defaults(fn=cmd_cpu)
    p = sub.add_parser('build')
    p.add_argument('out')
    p.set_defaults(fn=cmd_build)
    p = sub.add_parser('compare')
    p.add_argument('out')
    p.add_argument('--json')
    p.add_argument('--md')
    p.set_defaults(fn=cmd_compare)
    a = ap.parse_args()
    return a.fn(a) or 0


if __name__ == '__main__':
    sys.exit(main())
