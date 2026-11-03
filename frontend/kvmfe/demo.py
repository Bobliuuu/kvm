"""python3 -m kvmfe.demo --train-steps N --serve-every K --game-frames F --out DIR

Emits the three-workload demo from a declarative model config: `train.json` (N unrolled steps), `serve_v<k>.json` (a request
every K steps reading weights version k exactly), `game.json` (F paced frames of the registered game ops), `demo_combined.json`
(all graphs on one scheduler, sharing the `weights` variable) and `frontend.json` (sidecar: config, counts, buffer names).
If runtime/compile_workload.py is present each workload is also compiled to an ABI bundle under DIR/bundles/.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

from . import log
from .analyze import CostModel, analyze
from .emit import merge, write_json
from .game import GameConfig, compile_game
from .inspect import add_common
from .lower import Tiling
from .nn import GPT, ModelConfig
from .pipeline import compile_training
from .serve import ServeRequest, compile_serving
from .train import TrainConfig, weights_version

REPO = Path(__file__).resolve().parents[2]


def try_compile(path: Path, out: Path, compiler: Path):
    """Run the host compiler on one workload; failures are reported, never fatal (the JSON is the deliverable)."""
    if not compiler.exists():
        log.warning('host compiler %s not found: skipping bundle for %s', compiler, path.name)
        return None
    r = subprocess.run([sys.executable, str(compiler), str(path), str(out)], capture_output=True, text=True)
    if r.returncode:
        log.warning('host compiler rejected %s: %s', path.name, (r.stderr.strip().splitlines() or ['?'])[-1][:300])
        return None
    log.info('bundle %s: %s', out, r.stdout.strip())
    return out


def prompt_for(cfg, n=3):
    t, out = 1, []
    for _ in range(n):
        out.append(t % cfg.vocab)
        t = (5 * t + 3) % cfg.vocab
    return tuple(out)


def main(argv=None):
    p = argparse.ArgumentParser(prog='kvmfe.demo', description=__doc__)
    add_common(p)
    p.add_argument('--train-steps', type=int, default=4)
    p.add_argument('--serve-every', type=int, default=2, help='one serving request per K steps (0: none)')
    p.add_argument('--game-frames', type=int, default=0)
    p.add_argument('--defs-dir', default=None, help='directory with kvm_ops*.def (default: <repo>/include or $KVM_INCLUDE)')
    p.add_argument('--compiler', default=str(REPO / 'runtime' / 'compile_workload.py'))
    p.add_argument('--out', required=True)
    p.add_argument('--no-bundles', action='store_true')
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO, format='%(levelname)s %(message)s')
    if a.defs_dir:
        os.environ['KVM_INCLUDE'] = a.defs_dir
    cfg, tiling, out = ModelConfig.load(a.model), Tiling.preset(a.tiling), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    tr = compile_training(cfg, TrainConfig(steps=a.train_steps), tiling, a.reuse, a.gap)
    specs, files, side = {'train': tr.spec}, {}, {'model': cfg.__dict__, 'tiling': tiling.__dict__, 'reuse': a.reuse, 'gap': a.gap}
    side['loss_buffers'] = [t.buf for t in tr.program.losses]
    side['weight_buffers'] = {n: t.buf for n, t in tr.program.weights.items()}
    side['workloads'] = {'train': dict(tasks=len(tr.low.tasks), edges=tr.low.edges)}
    if a.serve_every:
        for k in range(0, a.train_steps + 1, a.serve_every):
            req = ServeRequest(f'serve_s{k}', prompt_for(cfg), max_new=3, version=weights_version(k))
            low, spec = compile_serving(GPT(cfg), req, tiling, a.reuse)
            specs[req.name] = spec
            side['workloads'][req.name] = dict(tasks=len(low.tasks), edges=low.edges, version=req.version, policy=req.policy,
                                               deadline_us=req.deadline_us, priority=req.priority)
    if a.game_frames:
        low, spec = compile_game(GameConfig(frames=a.game_frames), tiling)
        specs['game'] = spec
        side['workloads']['game'] = dict(tasks=len(low.tasks), edges=low.edges, frames=a.game_frames)
    specs['demo_combined'] = merge(*[s for n, s in specs.items() if n != 'demo_combined'])
    for name, spec in specs.items():
        files[name] = out / f'{name}.json'
        write_json(spec, files[name])
        if not a.no_bundles:
            try_compile(files[name], out / 'bundles' / name, Path(a.compiler))
    side['train_analysis'] = {k: v for k, v in analyze(tr.low, CostModel()).items() if k != 'widths'}
    (out / 'frontend.json').write_text(json.dumps(side, indent=1) + '\n')
    log.info('demo: %s', {n: (w['tasks'], w['edges']) for n, w in side['workloads'].items()})
    return 0


if __name__ == '__main__':
    sys.exit(main())
