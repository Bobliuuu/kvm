#!/usr/bin/env python3
"""Emit one training workload for the executor-vs-PyTorch comparison: workload.json (the PyTorch side reads its initial weights,
token stream and hyper-parameters from it, so both sides train from identical bytes) and the executor bundle.

  python3 bench/vs_pytorch/gen.py OUT_DIR --model gpt_tiny|JSON --steps N --mode tf32|precise [--tiling a100] [--fuse]

--mode precise sets matmul_tile flag 16 (3xTF32, fp32 accuracy) on every GEMM; tf32 is the ops' default 1xTF32. Also writes
OUT_DIR/info.json: model config, parameter count, per-step FLOPs, minimum bytes and the logical dependency depth (roofline inputs).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'frontend'))
F_PRECISE = 16


def critical_depth(ops):
    """Longest chain of logical ops through tensor producers (ops are in program order; an op depends on the last writer of each
    tensor it reads or writes in place)."""
    last, depth = {}, []
    for op in ops:
        d = 1 + max([last.get(t.buf or t.name, 0) for t in list(op.ins.values()) + list(op.outs.values())] + [0])
        depth.append(d)
        for t in op.outs.values():
            last[t.buf or t.name] = d
    return max(depth)


def roofline_inputs(cfg, params, ops, steps):
    """FLOPs and minimum HBM bytes per optimizer step (fp32). GEMM FLOPs are 2*M*N*K: forward, then dgrad and wgrad (2x forward)."""
    T, B, d, L, V, H = cfg.seq, cfg.batch, cfg.d_model, cfg.layers, cfg.vocab, cfg.heads
    tok = B * T
    mm_fwd = L * (4 * 2 * tok * d * d + 2 * 2 * tok * d * cfg.ffn_mult * d) + 2 * tok * d * V
    attn_fwd = L * 2 * 2 * B * T * T * d / 2          # QK^T and PV, causal half
    flops = 3 * mm_fwd + 3.5 * attn_fwd     # attention backward recomputes P: 2.5x the forward on top of 1x
    # minimum traffic: weights read fwd and bwd, grad written and read (clip, adam), adam reads p,m,v, writes p,m,v; activations of the
    # forward written once and read once in the backward (a perfectly fused implementation).
    act = tok * (L * (d * 8 + cfg.ffn_mult * d * 2) + d * 2 + V * 2) * 4
    nbytes = params * 4 * (2 + 2 + 3 + 3) + 2 * act
    return {'flops_per_step': flops, 'min_bytes_per_step': nbytes, 'tokens_per_step': tok}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('out')
    ap.add_argument('--model', default='gpt_tiny')
    ap.add_argument('--steps', type=int, default=20)
    ap.add_argument('--mode', choices=['tf32', 'precise'], default='tf32')
    ap.add_argument('--tiling', default='a100')
    ap.add_argument('--fuse', action='store_true')
    ap.add_argument('--precision', default=None, help='f32 | bf16 | bf16s (model config override)')
    ap.add_argument('--reuse', default='step', help="buffer reuse: 'step' (default), 'shape' or 'none' (every tensor keeps its own buffer: needed to read gradients back)")
    a = ap.parse_args()
    from dataclasses import asdict
    from kvmfe.lower import Tiling
    from kvmfe.nn import ModelConfig
    from kvmfe.pipeline import compile_training
    from kvmfe.train import TrainConfig
    cfg = ModelConfig.load(a.model)
    if a.precision:
        from dataclasses import replace
        cfg = replace(cfg, precision=a.precision)
    c = compile_training(cfg, TrainConfig(steps=a.steps), Tiling.preset(a.tiling), reuse=a.reuse, fuse=a.fuse)
    if a.mode == 'precise':
        for g in c.spec['graphs']:
            for t in g['tasks']:
                if t['op'] == 'matmul_tile':
                    t['args']['flags'] |= F_PRECISE
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for rel, data in c.low.data_files.items():              # token streams above 4096 values travel as files next to the workload
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        (out / rel).write_bytes(data)
    (out / 'workload.json').write_text(json.dumps(c.spec, separators=(',', ':')))
    subprocess.run([sys.executable, str(ROOT / 'runtime/compile_workload.py'), str(out / 'workload.json'), str(out / 'bundle')], check=True)
    params = cfg.params
    info = {'model': asdict(cfg), 'steps': a.steps, 'mode': a.mode, 'tiling': a.tiling, 'fuse': a.fuse, 'params': params,
            'tasks': sum(len(g['tasks']) for g in c.spec['graphs']), 'logical_ops': len(c.program.b.ops),
            'logical_depth': critical_depth(c.program.b.ops), **roofline_inputs(cfg, params, c.program.b.ops, a.steps)}
    (out / 'info.json').write_text(json.dumps(info, indent=1))
    print(json.dumps(info))


if __name__ == '__main__':
    main()
