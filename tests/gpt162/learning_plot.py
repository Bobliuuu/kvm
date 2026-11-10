#!/usr/bin/env python3
"""Loss-curve figure, CSV and gap summary of a tests/gpt162/learning.sh run (curve.json holds executor, torch 1xTF32 and torch fp32).

  learning_plot.py RUN_DIR --model-json JSON [--png loss.png] [--csv loss.csv]

The per-step relative gap is |executor - torch| / torch for the same step (same init, same batch, same AdamW/LR/clip). The reference lines are
ln(vocab) (an untrained softmax), ln(K) (a model that learned only the token support) and the stream's exact entropy (the floor).
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'frontend'))
from kvmfe.nn import ModelConfig  # noqa: E402
from kvmfe.train import stream_entropy  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run_dir')
    ap.add_argument('--model-json', required=True)
    ap.add_argument('--png')
    ap.add_argument('--csv')
    a = ap.parse_args()
    d = json.loads((Path(a.run_dir) / 'curve.json').read_text())
    cfg = ModelConfig.load(a.model_json)
    c = d['curve']
    ex, tf, fp = c['executor'], c['tf32'], c['fp32']
    n = len(ex)
    gap = lambda x, y: [abs(p - q) / abs(q) for p, q in zip(x, y)]
    g_tf, g_fp, g_tt = gap(ex, tf), gap(ex, fp), gap(tf, fp)
    k = cfg.data_vocab or cfg.vocab
    h = stream_entropy(cfg)
    ref = {'ln_vocab': math.log(cfg.vocab), 'ln_K': math.log(k), 'entropy': h}
    if a.csv:
        with open(a.csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['step', 'executor', 'torch_tf32', 'torch_fp32', 'rel_gap_vs_tf32', 'rel_gap_vs_fp32', 'rel_gap_tf32_vs_fp32'])
            for i in range(n):
                w.writerow([i + 1, f'{ex[i]:.6f}', f'{tf[i]:.6f}', f'{fp[i]:.6f}', f'{g_tf[i]:.3e}', f'{g_fp[i]:.3e}', f'{g_tt[i]:.3e}'])
    last = max(1, n // 10)
    mean = lambda v: sum(v) / len(v)
    first = lambda v, thr: next((i + 1 for i, x in enumerate(v) if x < thr), None)
    summ = {
        'steps': n, **ref,
        'first_loss': {'executor': ex[0], 'torch_tf32': tf[0], 'torch_fp32': fp[0]},
        'last_mean': {'executor': mean(ex[-last:]), 'torch_tf32': mean(tf[-last:]), 'torch_fp32': mean(fp[-last:]), 'steps_averaged': last},
        'first_step_below_ln_K': {'executor': first(ex, ref['ln_K']), 'torch_tf32': first(tf, ref['ln_K'])},
        'first_step_below_entropy_plus_0.1': {'executor': first(ex, h + 0.1), 'torch_tf32': first(tf, h + 0.1)},
        'max_rel_gap_vs_tf32': {'value': max(g_tf), 'step': g_tf.index(max(g_tf)) + 1, 'median': sorted(g_tf)[n // 2], 'mean': mean(g_tf)},
        'max_rel_gap_vs_fp32': {'value': max(g_fp), 'step': g_fp.index(max(g_fp)) + 1},
        'max_rel_gap_tf32_vs_fp32 (noise floor)': {'value': max(g_tt), 'step': g_tt.index(max(g_tt)) + 1},
        'first_step_rel_gap_over_1pct_vs_tf32': next((i + 1 for i, x in enumerate(g_tf) if x > 0.01), None),
        'holdout': d.get('holdout'), 'holdout_batches': d.get('holdout_batches')}
    bf = c.get('bf16')   # torch autocast-bf16 curve (runs of a bf16s executor build), if the run asked for it
    if bf:
        g_bf, g_bt = gap(ex, bf), gap(bf, tf)
        summ['first_loss']['torch_bf16'] = bf[0]
        summ['last_mean']['torch_bf16'] = mean(bf[-last:])
        summ['first_step_below_ln_K']['torch_bf16'] = first(bf, ref['ln_K'])
        summ['first_step_below_entropy_plus_0.1']['torch_bf16'] = first(bf, h + 0.1)
        summ['max_rel_gap_vs_bf16'] = {'value': max(g_bf), 'step': g_bf.index(max(g_bf)) + 1, 'median': sorted(g_bf)[n // 2], 'mean': mean(g_bf)}
        summ['max_rel_gap_bf16_vs_tf32 (bf16 noise floor)'] = {'value': max(g_bt), 'step': g_bt.index(max(g_bt)) + 1, 'median': sorted(g_bt)[n // 2]}
    print(json.dumps(summ, indent=1))
    if a.run_dir:
        (Path(a.run_dir) / 'summary.json').write_text(json.dumps(summ, indent=1))
    if a.png:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        x = list(range(1, n + 1))
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True, gridspec_kw={'height_ratios': [3, 1.3]})
        ax.plot(x, fp, color='#bbbbbb', lw=3.5, label='PyTorch fp32 (highest)')
        ax.plot(x, tf, color='#d95f02', lw=1.8, label='PyTorch 1xTF32 AdamW')
        if bf:
            ax.plot(x, bf, color='#7570b3', lw=1.4, label='PyTorch autocast-bf16 AdamW')
        ax.plot(x, ex, color='#1b6ca8', lw=1.2, label='kvm executor 1xTF32')
        for nm, v, st in ((f'ln(vocab) = {ref["ln_vocab"]:.2f}', ref['ln_vocab'], ':'), (f'ln(K={k}) = {ref["ln_K"]:.2f}', ref['ln_K'], '--'),
                          (f'stream entropy = {h:.3f}', h, '-.')):
            ax.axhline(v, color='k', ls=st, lw=0.8)
            ax.text(n * 0.99, v, nm, ha='right', va='bottom', fontsize=8)
        ax.set_yscale('log')
        ax.set_ylabel('train loss (nats, log scale)')
        ax.set_title(f'gpt_162m ({cfg.params / 1e6:.0f}M) on the executor vs PyTorch, same init / stream / LR schedule\n'
                     f'noisy bigram over {k} of {cfg.vocab} tokens, noise {cfg.data_noise}, batch {cfg.batch}x{cfg.seq}, A100')
        ax.legend(loc='upper right', bbox_to_anchor=(1, 0.8))
        ax2.semilogy(x, [max(v, 1e-9) for v in g_tf], color='#1b6ca8', lw=1, label='|executor - torch tf32| / torch')
        if bf:
            ax2.semilogy(x, [max(v, 1e-9) for v in g_bf], color='#7570b3', lw=1, label='|executor - torch bf16| / torch')
        ax2.semilogy(x, [max(v, 1e-9) for v in g_tt], color='#999999', lw=1, label='|torch tf32 - torch fp32| / torch (noise floor)')
        ax2.set_ylabel('relative gap')
        ax2.set_xlabel('optimizer step')
        ax2.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(a.png, dpi=130)


if __name__ == '__main__':
    main()
