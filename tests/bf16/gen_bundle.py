#!/usr/bin/env python3
"""Training bundles of one generic MLP in three precisions + the compiler's CPU golden, for the real-bf16-storage A100 check.

  python3 tests/bf16/gen_bundle.py OUT_DIR {f32|bf16|bf16-real} [--steps N] [--tiling cpu|a100] [--wide]

bf16 keeps bf16 values in fp32 containers (emulated); bf16-real stores weight copies and between-layer activations in 2-byte buffers
(KvmBuf dtype 2). Both must produce bit-identical weights and losses; --wide uses 256-wide layers so the memory difference is visible.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'frontend'))
from kvmfe.generic import MLP, CrossEntropy, ArrayData  # noqa: E402
from kvmfe.lower import Tiling  # noqa: E402
from kvmfe.optim import OptimConfig  # noqa: E402
from kvmfe.pipeline import compile_training  # noqa: E402


def build(precision, steps, tiling, wide=False):
    import random
    rng = random.Random(7)
    d, h, c, n = (64, 256, 10, 128) if wide else (20, 24, 5, 24)
    x = [[rng.gauss(0, 1) for _ in range(d)] for _ in range(n * 2)]
    y = [rng.randrange(c) for _ in range(n * 2)]
    data = ArrayData(n, {'x': x, 'y': y}, dtypes={'y': 'i32'})
    return compile_training(MLP('m', [d, h, h, c], act='gelu', precision=precision, std=0.2), steps=steps, tiling=Tiling.preset(tiling),
                            loss=CrossEntropy(), data=data, optimizer=OptimConfig(kind='sgd', lr=0.05), reuse='none')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('out')
    ap.add_argument('precision', choices=('f32', 'bf16', 'bf16-real'))
    ap.add_argument('--steps', type=int, default=3)
    ap.add_argument('--tiling', default='a100')
    ap.add_argument('--wide', action='store_true')
    a = ap.parse_args()
    comp = build(a.precision, a.steps, a.tiling, a.wide)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for rel, raw in comp.low.data_files.items():
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        (out / rel).write_bytes(raw)
    (out / 'workload.json').write_text(json.dumps(comp.spec, separators=(',', ':')))
    subprocess.run([sys.executable, str(ROOT / 'runtime/compile_workload.py'), '--emit-golden', '--golden-timeout', '900',
                    str(out / 'workload.json'), str(out / 'bundle')], check=True, stderr=subprocess.DEVNULL)
    bytes_ = {dt: sum(b.rows * b.cols * (2 if dt == 'bf16' else 4) for b in comp.low.plan.buffers.values() if b.dtype == dt)
              for dt in ('f32', 'i32', 'bf16')}
    print(f'{out}/bundle: {a.precision}, {a.steps} steps, {len(comp.low.tasks)} tasks, buffer bytes by dtype {bytes_}, '
          f'total {sum(bytes_.values())}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
