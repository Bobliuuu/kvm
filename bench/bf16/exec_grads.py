#!/usr/bin/env python3
"""Step-1 numerics of a gen.py bundle on the executor against PyTorch fp64 / TF32 / fp32 / autocast-bf16 (tests/gpt162/torch_ref.py).

  python3 bench/bf16/exec_grads.py WORKLOAD_DIR ./kvm_exec --out OUT_DIR [--lr 3e-3] [--warmup 0] [--keep]

WORKLOAD_DIR comes from `bench/vs_pytorch/gen.py OUT --steps 1 --reuse none` (reuse none: every gradient buffer survives the step). The executor runs
once with `--dump-bin` for the loss, the clip coefficient and every parameter's gradient (with --update also the first moment and updated weight; raw little-endian f32; bf16
buffers are widened by the executor); torch_ref.py then rebuilds the same initial weights and token stream from workload.json and reports, per
parameter, the relative L2 error of the executor's gradient against the float64 PyTorch truth next to what PyTorch's own TF32, fp32 and
autocast-bf16 give. Everything lands in OUT_DIR: exec_grads.log, grads.json (torch_ref's result) and run.txt (the executor's summary line).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('workload_dir')
    ap.add_argument('exe')
    ap.add_argument('--out', required=True)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--warmup', type=int, default=0)
    ap.add_argument('--keep', action='store_true', help='keep the raw dump directory (about 0.7 GB at 162M parameters)')
    ap.add_argument('--update', action='store_true', help='also dump m and the updated weights (3x the disk: 2 GB at 162M parameters)')
    a = ap.parse_args()
    wl, out = Path(a.workload_dir).resolve(), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    info = json.loads((wl / 'info.json').read_text())
    meta = json.loads((wl / 'bundle' / 'metadata.json').read_text())
    spec = json.loads((wl / 'workload.json').read_text())
    index = {b['name']: b['index'] for b in meta['buffers']}
    want = {'loss.0', 'clip_coef'}
    for g in spec['graphs']:
        for t in g['tasks']:
            if t['op'] in ('adamw', 'adamw_lr'):
                p = t['args']['p']
                want |= {t['args']['g']} | ({'m.' + p[2:], p} if a.update else set())
    missing = sorted(want - set(index))
    if missing:
        raise SystemExit(f'buffers not in the bundle: {missing[:5]}')
    tmp = Path(tempfile.mkdtemp(prefix='exec_grads_', dir=os.environ.get('KVM_TMP')))
    try:
        dump = tmp / 'buffers'
        dump.mkdir()
        cmd = [a.exe, '--quiet', '--dump-bin', str(dump), ','.join(str(index[n]) for n in sorted(want)), str(wl / 'bundle')]
        print('+', ' '.join(cmd[:5]), '...', flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True)
        (out / 'run.txt').write_text(r.stdout + r.stderr)
        print((r.stdout.strip().splitlines() or [''])[-1], flush=True)
        if r.returncode != 0:
            raise SystemExit(f'executor failed rc={r.returncode}: {r.stdout[-300:]} {r.stderr[-300:]}')
        for n in want:                                   # torch_ref.py reads <buffer name>.bin
            (dump / f'{index[n]}.bin').rename(dump / f'{n}.bin')
        demo = tmp / 'demo' / 'workloads'                # the layout torch_ref.py expects around a workload
        demo.mkdir(parents=True)
        for e in wl.iterdir():
            if e.name != 'bundle':
                (demo / ('train.json' if e.name == 'workload.json' else e.name)).symlink_to(e)
        cmd = [sys.executable, str(ROOT / 'tests' / 'gpt162' / 'torch_ref.py'), str(tmp / 'demo'), '--model', json.dumps(info['model']), '--steps', '1',
               '--lr', str(a.lr), '--warmup', str(a.warmup), '--grads', str(dump), '--out', str(out / 'grads.json')]
        r = subprocess.run(cmd, capture_output=True, text=True)
        (out / 'exec_grads.log').write_text(r.stdout + r.stderr)
        print(r.stdout, r.stderr[-2000:], flush=True)
        if r.returncode != 0:
            raise SystemExit(f'torch_ref.py failed rc={r.returncode}')
        if a.keep:
            shutil.copytree(dump, out / 'buffers')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
