#!/usr/bin/env python3
"""Emit gpt_tiny AdamW bundles for the bf16-storage GPU check.

  python3 tests/bf16/gen.py OUT_DIR

Cases (each OUT_DIR/<case>/{workload.json,bundle}); golden.json (CPU VM) only for the short ones:
  f32_s50, bf16s_s50    50-step loss curves, fp32 against real bf16 storage (no golden: the curves are compared GPU against GPU)
  bf16s_s1, bf16s_s5    1 and 5 steps with the CPU golden, reuse none (every activation and gradient keeps its buffer)
"""
import dataclasses
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'frontend'))
from kvmfe.lower import Tiling  # noqa: E402
from kvmfe.nn import PRESETS  # noqa: E402
from kvmfe.pipeline import compile_training  # noqa: E402
from kvmfe.train import TrainConfig  # noqa: E402

CASES = [('f32_s50', 'f32', 50, 'shape', False), ('bf16s_s50', 'bf16s', 50, 'shape', False),
         ('bf16s_s1', 'bf16s', 1, 'none', True), ('bf16s_s5', 'bf16s', 5, 'none', True)]


def main():
    out = Path(sys.argv[1])
    for name, prec, steps, reuse, golden in CASES:
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        c = compile_training(dataclasses.replace(PRESETS['gpt_tiny'], precision=prec), TrainConfig(steps=steps), Tiling.preset('cpu'), reuse=reuse)
        if c.low.data_files:
            raise SystemExit('file-initialised buffers are not supported here')
        (d / 'workload.json').write_text(json.dumps(c.spec, separators=(',', ':')))
        subprocess.run([sys.executable, str(ROOT / 'runtime/compile_workload.py'), *(['--emit-golden', '--golden-timeout', '900'] if golden else []),
                        str(d / 'workload.json'), str(d / 'bundle')], check=True, stderr=subprocess.DEVNULL)
        st = c.low.plan.stats
        print(f'{name}: {steps} steps, persistent {st["persistent_bytes"]} B + pool {st["pool_bytes"]} B, {len(c.spec["buffers"])} buffers')


if __name__ == '__main__':
    main()
