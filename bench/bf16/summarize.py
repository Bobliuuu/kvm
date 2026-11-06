#!/usr/bin/env python3
"""Markdown tables for results/2026-11-06-a100-bf16 from the files bench/bf16/run_box.sh wrote.

  python3 bench/bf16/summarize.py RESULT_DIR > tables.md

Reads (whatever exists): bench_mma_bf16.log, exec_f32.json, exec_bf16s.json, torch_tf32.json, torch_bf16.json, losses_vs_torch_{bf16,tf32}.json,
grads_{f32,bf16s}/grads.json, bundle_bytes.json, test_*.log. Missing pieces are skipped, never invented.
"""
import json
import re
import sys
from pathlib import Path


def load(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def gemm_table(log):
    rows = []
    for line in Path(log).read_text().splitlines():
        m = re.match(r'(\S+)\s+(mma|old64)\s+([NT])([NT])\s+(\d+)x\s*(\d+)x\s*(\d+) tile .*?kvm\s+([\d.]+) \[\s*([\d.]+)\.\.\s*([\d.]+)\] cuBLAS\s+([\d.]+) \[\s*([\d.]+)\.\.\s*([\d.]+)\] TFLOP/s ratio ([\d.]+) err ([\d.e+-]+) (\w+)', line)
        if m:
            g = m.groups()
            rows.append(dict(tag=g[0], kind=g[1], ta=g[2], tb=g[3], M=int(g[4]), N=int(g[5]), K=int(g[6]), kvm=float(g[7]), kvm_lo=float(g[8]), kvm_hi=float(g[9]),
                             cub=float(g[10]), cub_lo=float(g[11]), cub_hi=float(g[12]), ratio=float(g[13]), err=g[14], ok=g[15]))
    out = ['| shape (M x N x K) | op | matmul_mma_bf16 TFLOP/s [min..max] | cuBLAS bf16 TFLOP/s [min..max] | ratio | max err / max abs |', '|---|---|---|---|---|---|']
    for r in rows:
        name = r['tag'] if r['tag'] != '4096^3' else '4096^3'
        out.append(f"| {r['M']}x{r['N']}x{r['K']} ({name}) | {r['kind']} {r['ta']}{r['tb']} | {r['kvm']:.1f} [{r['kvm_lo']:.1f}..{r['kvm_hi']:.1f}] | "
                   f"{r['cub']:.1f} [{r['cub_lo']:.1f}..{r['cub_hi']:.1f}] | {r['ratio']:.3f} | {r['err']} {r['ok']} |")
    return '\n'.join(out)


def step_table(d):
    rows = []
    for name, f, label in (('executor f32 (1xTF32 tiles)', 'exec_f32.json', 'ms_per_step'), ('executor bf16s (bf16 tensor-core tiles)', 'exec_bf16s.json', 'ms_per_step')):
        j = load(Path(d) / f)
        if j and 'ms_per_step_best' in j:
            rows.append((name, j['ms_per_step_best'], j['ms_per_step_median'], j['ms_per_step_worst'], j.get('tasks')))
    for prec, f in (('tf32', 'torch_tf32.json'), ('autocast bf16', 'torch_bf16.json')):
        j = load(Path(d) / f)
        for mode, m in (j or {}).get('modes', {}).items():
            if 'ms_per_step_best' in m:
                rows.append((f'PyTorch {prec} {mode}', m['ms_per_step_best'], m['ms_per_step_median'], m['ms_per_step_worst'], None))
    out = ['| implementation | ms / step best (median / worst) | tasks |', '|---|---|---|']
    for n, b, md, w, t in rows:
        out.append(f'| {n} | {b:.1f} ({md:.1f} / {w:.1f}) | {t if t else ""} |')
    return '\n'.join(out)


def main():
    d = Path(sys.argv[1])
    if (d / 'bench_mma_bf16.log').exists():
        print('### GEMM\n\n' + gemm_table(d / 'bench_mma_bf16.log') + '\n')
    print('### Step time\n\n' + step_table(d) + '\n')
    for name in ('losses_vs_torch_bf16.json', 'losses_vs_torch_tf32.json'):
        j = load(d / name)
        if j:
            print(f'### {name}\n\nexecutor {j["executor"]}\n\ntorch    {j.get("torch")}\n\nrel diff {j.get("rel_diff")}\n')
    for name in ('grads_f32', 'grads_bf16s'):
        j = load(d / name / 'grads.json')
        if j and 'grads' in j:
            g = j['grads']
            print(f'### {name}\n\nexecutor loss {g["executor_loss"]:.6f} fp64 {g["torch"]["fp64"]["loss"]:.6f} tf32 {g["torch"]["tf32"]["loss"]:.6f} '
                  f'autocast-bf16 {g["torch"].get("bf16", {}).get("loss", float("nan")):.6f}')
            print('\n| group | exec vs fp64 | torch tf32 vs fp64 | torch autocast-bf16 vs fp64 |\n|---|---|---|---|')
            for k, v in g['groups'].items():
                print(f"| {k} | {v['exec_vs_fp64']:.2e} | {v['tf32_vs_fp64']:.2e} | {v.get('bf16_vs_fp64', float('nan')):.2e} |")
            print()


if __name__ == '__main__':
    main()
