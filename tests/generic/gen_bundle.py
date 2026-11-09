#!/usr/bin/env python3
"""Emit one never-shipped model (tests/generic/zoo.py) as a training bundle + the compiler's CPU golden, or as a MANIFEST demo.

  python3 tests/generic/gen_bundle.py OUT_DIR CASE [--steps N] [--tiling cpu|a100] [--precise]      OUT/workload.json, OUT/bundle/
  python3 tests/generic/gen_bundle.py OUT_DIR CASE --manifest [--window W] ...                      OUT/MANIFEST.json, workloads/, bundles/
  python3 tests/generic/gen_bundle.py --list

--reuse none keeps every activation and gradient in its own buffer, so they can all be compared with the golden (golden_check.py).
--precise sets matmul_tile flag 16 (3xTF32, fp32 accuracy) on every GEMM; the default is the ops' 1xTF32.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import zoo  # noqa: E402
from kvmfe import demo  # noqa: E402
from kvmfe.lower import Tiling  # noqa: E402
from kvmfe.pipeline import compile_training  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('out', nargs='?')
    ap.add_argument('case', nargs='?', choices=sorted(zoo.CASES))
    ap.add_argument('--steps', type=int)
    ap.add_argument('--tiling', default='cpu')
    ap.add_argument('--precise', action='store_true')
    ap.add_argument('--manifest', action='store_true')
    ap.add_argument('--window', type=int)
    ap.add_argument('--list', action='store_true')
    a = ap.parse_args()
    if a.list:
        for n, f in zoo.CASES.items():
            print(f'{n}: {f().note}')
        return 0
    if not (a.out and a.case):
        ap.error('OUT_DIR and CASE are required')
    case, out = zoo.CASES[a.case](), Path(a.out)
    steps = a.steps or case.steps
    if a.manifest:
        demo.write_training_demo(out, case.module, loss=case.loss, data=case.data, optimizer=case.optimizer, steps=steps,
                                 window=a.window, tiling=a.tiling, reuse='none', precise=a.precise)
        print(f'{out}/MANIFEST.json: {a.case}, {steps} steps, window {a.window or steps}')
        return 0
    comp = compile_training(case.module, loss=case.loss, data=case.data, optimizer=case.optimizer, steps=steps,
                            tiling=Tiling.preset(a.tiling), reuse='none')
    if comp.low.data_files:
        raise SystemExit('gen_bundle: file-initialised buffers are not supported here')
    spec = demo.make_precise(comp.spec) if a.precise else comp.spec
    out.mkdir(parents=True, exist_ok=True)
    (out / 'workload.json').write_text(json.dumps(spec, separators=(',', ':')))
    subprocess.run([sys.executable, str(ROOT / 'runtime/compile_workload.py'), '--emit-golden', '--golden-timeout', '900',
                    str(out / 'workload.json'), str(out / 'bundle')], check=True, stderr=subprocess.DEVNULL)
    print(f'{out}/bundle: {a.case}, {steps} steps, {len(comp.low.tasks)} tasks, {a.tiling} tiling, {"3xTF32" if a.precise else "1xTF32"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
