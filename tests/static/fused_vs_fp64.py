#!/usr/bin/env python3
"""The fused MLP step's numerics, judged as the MLP bench judges a run (bench/exec_vs_torch/run_mlp.py, gate 3).

  KVM_PY tests/static/fused_vs_fp64.py WORKLOAD_DIR DUMP...

WORKLOAD_DIR is gen_mlp.py's output (workload.json, bundle/); each DUMP is the stdout of `kvm_exec --dump-buffers WORKLOAD_DIR/bundle`.
For each dump: the executor's per-step loss against a float64 run of the same SGD problem must sit within twice eager PyTorch's distance
plus 1e-4 (the float64 rule), and its per-step loss and final weights against eager within the bench's own tolerances.
Eager and float64 run once for all dumps. Every number is printed. Exit 0 when every dump is within all three bounds, 1 when one is not,
2 on bad input.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'bench/exec_vs_torch'))
from run_mlp import LOSS_TOL, WEIGHT_TOL, fp64_losses, named_buffers, rel  # noqa: E402  (the bench's gate code and tolerances)

FP64_MULT, FP64_FLOOR = 2.0, 1e-4   # the float64 rule: the executor at most 2x eager's distance to float64, plus 1e-4


def read_dump(path):
    bufs = {}
    for line in Path(path).read_text().splitlines():
        if line.startswith('buffer '):
            k, v = line[len('buffer '):].split(':', 1)
            bufs[int(k)] = v.split()
    return bufs


def main(argv):
    if len(argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    wdir, dumps = Path(argv[1]), argv[2:]
    if not (wdir / 'workload.json').is_file() or not (wdir / 'bundle' / 'metadata.json').is_file():
        print(f'fused_vs_fp64: {wdir} is not a gen_mlp.py output (workload.json, bundle/metadata.json)', file=sys.stderr)
        return 2
    from mlp_torch import Workload, build, self_check_init
    self_check_init()
    wl = Workload(wdir)
    idx = named_buffers(wdir / 'bundle')
    ref = build(wl, 1, 'eager')
    ref.reset()
    ref.run(wl.steps)
    ref_loss = np.array([l[0] for l in ref.losses()])
    ref_w = ref.weights()
    ref64 = fp64_losses(wl, wl.steps)
    to64 = lambda losses: float(np.max(np.abs(np.asarray(losses) - ref64) / np.abs(ref64)))
    eager64 = to64(ref_loss)
    bound64 = FP64_MULT * eager64 + FP64_FLOOR
    print(f'fused_vs_fp64: {wl.steps} steps; eager vs float64 {eager64:.2e}, so the executor may sit at {bound64:.2e}; '
          f'loss vs eager <= {LOSS_TOL:g}, weights vs eager <= {WEIGHT_TOL:g} relative L2')
    bad = 0
    for d in dumps:
        bufs = read_dump(d)
        missing = [n for n in [f'loss.{s}' for s in range(wl.steps)] + wl.names if n not in idx or idx[n] not in bufs]
        if missing:
            print(f'FAIL: {d}: no dumped buffer for {", ".join(missing[:4])}')
            bad += 1
            continue
        ex_loss = np.array([float(bufs[idx[f'loss.{s}']][0]) for s in range(wl.steps)])
        ex_w = [np.array(bufs[idx[n]], dtype=np.float64).reshape(tuple(w.shape)) for n, w in zip(wl.names, wl.init)]
        exec64 = to64(ex_loss)
        loss_eager = float(np.max(np.abs(ex_loss - ref_loss) / np.abs(ref_loss)))
        w_eager = max(rel(e, w) for e, w in zip(ex_w, ref_w))
        ok = exec64 <= bound64 and loss_eager <= LOSS_TOL and w_eager <= WEIGHT_TOL
        bad += not ok
        print(f'{"ok" if ok else "FAIL"}: {d}: vs float64 {exec64:.2e} (bound {bound64:.2e}); loss vs eager {loss_eager:.2e}; '
              f'weights vs eager {w_eager:.2e} relative L2')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
