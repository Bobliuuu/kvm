"""Regenerate tests/frontend/golden_models.json: the CPU float64 VM's loss curves (and aux losses) of every zoo preset, 5 steps.
python3 tests/frontend/make_golden_models.py   (review the diff: an intended numerical change updates this file in the same commit)"""
import json
from pathlib import Path

import _path  # noqa: F401
from kvmfe import models
from kvmfe.lower import Tiling
from kvmfe.pipeline import compile_training
from kvmfe.train import TrainConfig

STEPS = 5


def curves(cfg):
    c = compile_training(cfg, TrainConfig(steps=STEPS), Tiling.preset('cpu'), reuse='shape')
    mem = c.run()
    return {n: [mem.matrix(f'{n}.{s}')[0][0] for s in range(STEPS)] for n in c.program.losses[0]}


if __name__ == '__main__':
    out = {n: curves(cfg) for n, cfg in models.PRESETS.items()}
    Path(__file__).with_name('golden_models.json').write_text(json.dumps(out, indent=1) + '\n')
    print({n: round(v['loss'][-1], 4) for n, v in out.items()})
