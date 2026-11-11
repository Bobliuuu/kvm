#!/usr/bin/env python3
"""Emit one zoo training workload of the claim's hold-out (Llama / grouped-query or ViT): workload.json (the PyTorch side reads its initial weights, data streams and hyper-parameters from it, so both
sides train from identical bytes), the executor bundle and info.json (with "family", which torch_bench.Trainer dispatches on).

  python3 -I bench/vs_pytorch/gen_zoo.py OUT_DIR --model '{"family": "llama", "d_model": 128, "heads": 4, "kv_heads": 2, "layers": 2, "ffn": 192, "seq": 32, "batch": 8, "vocab": 256}' --steps 50 [--tiling a100]
"""
import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'frontend'))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('out')
    ap.add_argument('--model', required=True, help='a JSON model config with "family" (llama, vit)')
    ap.add_argument('--steps', type=int, default=50)
    ap.add_argument('--tiling', default='a100')
    a = ap.parse_args()
    from kvmfe import models
    from kvmfe.lower import Tiling
    from kvmfe.pipeline import compile_training
    from kvmfe.train import TrainConfig
    spec = json.loads(a.model)
    family = spec['family']
    cfg = models.build(spec).cfg
    c = compile_training(cfg, TrainConfig(steps=a.steps), Tiling.preset(a.tiling), reuse='step', fuse=False)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for rel, data in c.low.data_files.items():              # big streams travel as files next to the workload
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        (out / rel).write_bytes(data)
    (out / 'workload.json').write_text(json.dumps(c.spec, separators=(',', ':')))
    subprocess.run([sys.executable, str(ROOT / 'runtime/compile_workload.py'), str(out / 'workload.json'), str(out / 'bundle')], check=True)
    info = {'family': family, 'model': asdict(cfg), 'steps': a.steps, 'mode': 'tf32', 'tiling': a.tiling, 'tasks': sum(len(g['tasks']) for g in c.spec['graphs']), 'logical_ops': len(c.program.b.ops)}
    (out / 'info.json').write_text(json.dumps(info, indent=1))
    print(json.dumps(info))


if __name__ == '__main__':
    main()
