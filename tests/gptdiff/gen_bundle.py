#!/usr/bin/env python3
"""Emit a gpt_tiny AdamW training bundle plus the compiler's CPU golden, for the GPU golden check.

  python3 tests/gptdiff/gen_bundle.py OUT_DIR --steps 2 [--precise]

--reuse none (every activation/gradient has its own buffer, so step-1 gradients survive to the end of the run and can be compared).
--precise sets matmul_tile flag 16 (3xTF32, fp32 accuracy) on every GEMM; default is the ops' 1xTF32. Writes OUT_DIR/workload.json
and the bundle (tasks.bin, ..., golden.json) in OUT_DIR/bundle.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'frontend'))
F_PRECISE = 16


def build_spec(steps: int, precise: bool, model: str = 'gpt_tiny', tiling: str = 'cpu') -> dict:
    from kvmfe.lower import Tiling
    from kvmfe.nn import ModelConfig
    from kvmfe.pipeline import compile_training
    from kvmfe.train import TrainConfig
    c = compile_training(ModelConfig.load(model), TrainConfig(steps=steps), Tiling.preset(tiling), reuse='none')
    if c.low.data_files:
        raise SystemExit('gen_bundle: file-initialised buffers are not supported here')
    if precise:
        for g in c.spec['graphs']:
            for t in g['tasks']:
                if t['op'] == 'matmul_tile':
                    t['args']['flags'] |= F_PRECISE
    return c.spec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('out')
    ap.add_argument('--steps', type=int, default=2)
    ap.add_argument('--precise', action='store_true')
    ap.add_argument('--model', default='gpt_tiny')
    ap.add_argument('--tiling', default='cpu', help='Tiling preset (cpu: the 544-task 2-step gpt_tiny bundle of the original report)')
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    spec = build_spec(a.steps, a.precise, a.model, a.tiling)
    (out / 'workload.json').write_text(json.dumps(spec, separators=(',', ':')))
    subprocess.run([sys.executable, str(ROOT / 'runtime/compile_workload.py'), '--emit-golden', str(out / 'workload.json'),
                    str(out / 'bundle')], check=True, stderr=subprocess.DEVNULL)
    print(f'{out}/bundle: {a.steps} steps, {a.tiling} tiling, {"3xTF32" if a.precise else "1xTF32"}')


if __name__ == '__main__':
    main()
