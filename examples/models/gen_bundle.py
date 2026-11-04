#!/usr/bin/env python3
"""Emit a training bundle of any zoo model (plus the compiler's float32 CPU golden) for a GPU run through kvm_exec.

  python3 examples/models/gen_bundle.py OUT_DIR --model moe_tiny --steps 5 [--precise] [--fuse] [--accum K] [--tiling cpu|a100]
         [--schedule cosine --warmup 2] [--loss-scale 1024] [--clip 1.0]

--model: a preset (llama_tiny, llama_gqa_tied, moe_tiny, mlp_tiny, vit_tiny, gpt_tiny, ...), a JSON file or a JSON string of a model
config (the class is picked by its fields). Buffers are never reused (--reuse none) so step-1 activations and gradients survive for
the comparison. --precise sets matmul flag 16 (3xTF32, fp32 accuracy) on every GEMM tile op: with it the GPU matches the golden to
fp32 level; without it the GEMMs are 1xTF32 (see tests/gptdiff for why only the first step is comparable then).
Writes OUT_DIR/workload.json, OUT_DIR/bundle/ (tasks.bin, ..., golden.json) and OUT_DIR/meta.json (task counts, config).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'frontend'))
F_PRECISE = 16
GEMM_OPS = ('matmul_tile', 'matmul_tile_dgelu', 'matmul_tile_silu_mul', 'matmul_tile_silu_mul_bwd')   # bf16 tiles ignore the flag


def load_model(spec):
    from kvmfe import models, nn
    from kvmfe.ir import FrontendError
    if isinstance(spec, str) and spec in nn.PRESETS:
        return nn.PRESETS[spec]
    try:
        return models.PRESETS[spec] if spec in models.PRESETS else models.build(spec).cfg
    except FrontendError:
        return nn.ModelConfig.load(spec)


def build(model, steps, tiling='cpu', precise=False, fuse=False, accum=1, schedule=None, warmup=0, loss_scale=1.0, clip=1.0, reuse='none'):
    from kvmfe.lower import Tiling
    from kvmfe.pipeline import compile_training
    from kvmfe.train import Schedule, TrainConfig
    cfg = load_model(model)
    tc = TrainConfig(steps=steps, accum=accum, clip=clip, loss_scale=loss_scale,
                     schedule=Schedule(schedule, warmup=warmup) if schedule else None)
    c = compile_training(cfg, tc, Tiling.preset(tiling), reuse=reuse, fuse=fuse)
    if precise:
        for g in c.spec['graphs']:
            for t in g['tasks']:
                if t['op'] in GEMM_OPS:
                    t['args']['flags'] |= F_PRECISE
    return c, cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('out')
    ap.add_argument('--model', required=True)
    ap.add_argument('--steps', type=int, default=3)
    ap.add_argument('--tiling', default='cpu')
    ap.add_argument('--precise', action='store_true')
    ap.add_argument('--fuse', action='store_true')
    ap.add_argument('--accum', type=int, default=1)
    ap.add_argument('--schedule', choices=['const', 'cosine', 'linear'])
    ap.add_argument('--warmup', type=int, default=0)
    ap.add_argument('--loss-scale', type=float, default=1.0)
    ap.add_argument('--clip', type=float, default=1.0)
    ap.add_argument('--no-golden', action='store_true', help='write workload.json only (skip the compiler golden)')
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    c, cfg = build(a.model, a.steps, a.tiling, a.precise, a.fuse, a.accum, a.schedule, a.warmup, a.loss_scale, a.clip)
    (out / 'workload.json').write_text(json.dumps(c.spec, separators=(',', ':')))
    for rel, data in c.low.data_files.items():          # big token / feature streams travel as files next to the workload
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        (out / rel).write_bytes(data)
    ops = {}
    for t in c.spec['graphs'][0]['tasks']:
        ops[t['op']] = ops.get(t['op'], 0) + 1
    meta = dict(model=getattr(cfg, 'name', a.model), config=json.loads(json.dumps(cfg.__dict__, default=list)), steps=a.steps,
                accum=a.accum, tiling=a.tiling, precise=a.precise, fuse=a.fuse, tasks=len(c.low.tasks), edges=c.low.edges,
                ops=dict(sorted(ops.items(), key=lambda kv: -kv[1])), plan=c.low.plan.stats)
    (out / 'meta.json').write_text(json.dumps(meta, indent=1, default=str))
    if not a.no_golden:
        subprocess.run([sys.executable, str(ROOT / 'runtime/compile_workload.py'), '--emit-golden', '--golden-timeout', '7200',
                        str(out / 'workload.json'), str(out / 'bundle')], check=True, stderr=subprocess.DEVNULL)
    print(f'{out}: {meta["model"]} {a.steps} steps, {meta["tasks"]} tasks, {meta["edges"]} edges, '
          f'{"3xTF32" if a.precise else "1xTF32"}{", fused" if a.fuse else ""}')


if __name__ == '__main__':
    main()
