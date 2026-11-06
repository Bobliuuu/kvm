#!/usr/bin/env python3
"""Merge the PyTorch and executor results of one configuration into a markdown table with loss agreement and roofline floors.

  python3 report.py RESULT_DIR [RESULT_DIR ...]     (each holds info.json, torch.json, kvm*.json)

Floor per step = max(FLOPs / peak, bytes / HBM bandwidth, logical dependency depth x one hop). A100-SXM4-40GB: 156 TFLOP/s TF32
tensor (52 for 3xTF32), 1555 GB/s. Hop: the executor's measured dependent hop is 7.0 us (results/2026-11-04-a100-exec-recycling);
a CUDA-graph kernel-to-kernel edge is taken as 1.3 us (AGENTS.md, measured on an H100: not re-measured here).
"""
import json
import sys
from pathlib import Path

PEAK = {'tf32': 156e12, 'precise': 52e12}
BW = 1555e9
HOP_KVM, HOP_GRAPH = 7.0e-6, 1.3e-6


def load(p):
    return json.loads(Path(p).read_text())


def agree(a, b):
    """max over steps of |a-b| / max(|b|, 1e-9) and the same relative to the loss drop."""
    return max(abs(x - y) / max(abs(y), 1e-9) for x, y in zip(a, b))


def main():
    for d in map(Path, sys.argv[1:]):
        info, torch_r = load(d / 'info.json'), load(d / 'torch.json')
        kvm = {p.stem: load(p) for p in sorted(d.glob('kvm*.json'))}
        mode = info['mode']
        t_fb = max(info['flops_per_step'] / PEAK[mode], info['min_bytes_per_step'] / BW)
        floor_kvm, floor_graph = max(t_fb, info['logical_depth'] * HOP_KVM), max(t_fb, info['logical_depth'] * HOP_GRAPH)
        tok = info['tokens_per_step']
        rows = [(m, r) for m, r in torch_r['modes'].items() if 'error' not in r] + [(k, v) for k, v in kvm.items()]
        ref = torch_r['modes']['eager']['losses']
        print(f'### {d.name}: {info["model"]["name"]}  {info["params"]:,} params, {tok} tokens/step, {info["steps"]} steps, {mode}, '
              f'{info["tasks"]} tasks, depth {info["logical_depth"]} ops\n')
        print(f'roofline per step: compute/bytes {t_fb * 1e6:.1f} us; chain at graph edge {floor_graph * 1e6:.1f} us; chain at executor hop '
              f'{floor_kvm * 1e6:.1f} us\n')
        print('| run | us/step best | median | worst | tokens/s (best) | vs eager | vs graph | x floor (own chain) | max loss diff vs eager |')
        print('|---|---|---|---|---|---|---|---|---|')
        eb, gb = torch_r['modes']['eager']['ms_per_step_best'], torch_r['modes'].get('graph', {}).get('ms_per_step_best')
        for name, r in rows:
            b = r['ms_per_step_best']
            fl = floor_kvm if name.startswith('kvm') else floor_graph
            print(f'| {name} | {b * 1e3:.1f} | {r["ms_per_step_median"] * 1e3:.1f} | {r["ms_per_step_worst"] * 1e3:.1f} | {r["tokens_per_s_best"]:,.0f} | '
                  f'{eb / b:.2f}x | {(gb / b if gb else float("nan")):.2f}x | {b * 1e-3 / fl:.1f} | {agree(r["losses"], ref):.1e} |')
        print()
        print('losses (eager): ' + ' '.join(f'{x:.4f}' for x in ref[:: max(1, len(ref) // 8)]) + f' ... {ref[-1]:.4f}\n')


if __name__ == '__main__':
    main()
