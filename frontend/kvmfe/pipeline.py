"""One-call entry points: config -> workload dict (train / serve / game), used by the demo CLI, inspect and tests."""
from __future__ import annotations

from dataclasses import dataclass

from . import log
from .emit import GraphPolicy, Lowered, lower_program, to_workload
from .lower import Tiling
from .nn import GPT, ModelConfig
from .train import Program, TrainConfig, build_training


@dataclass
class Compiled:
    low: Lowered
    spec: dict
    program: Program | None = None

    def run(self, **kw):
        """Execute the workload on the CPU golden VM (float64)."""
        from .refvm import execute
        return execute(self.spec, data_files=self.low.data_files, **kw)


def build_model(cfg):
    """Config -> model object. A model object (anything with .specs) is returned unchanged."""
    if hasattr(cfg, 'specs'):
        return cfg
    if isinstance(cfg, ModelConfig):
        return GPT(cfg)
    from . import models
    return models.build(cfg)


def compile_training(model_cfg, tc: TrainConfig, tiling: Tiling, reuse='step', gap=0, with_spec=True, fuse=False) -> Compiled:
    """model_cfg: nn.ModelConfig (GPT), a models.* config, or a model object. fuse=True applies fuse.py before lowering."""
    model = build_model(model_cfg)
    n = sum(s.rows * s.cols for s in model.specs.values())
    log.info('model %s: %.2fM parameters, %d tensors', model.cfg.name, n / 1e6, len(model.specs))
    prog = build_training(model, tc)
    ops = prog.b.ops
    if fuse:
        from .fuse import fuse as run_fuse
        ops, _ = run_fuse(ops)
    if getattr(model.cfg, 'precision', 'f32') == 'bf16s':
        from . import storage
        ops, _ = storage.apply(ops)
    low = lower_program(ops, tiling, reuse, gap, prefix='act')
    spec = to_workload(low, {'train': GraphPolicy()}) if with_spec else None
    return Compiled(low, spec, prog)
