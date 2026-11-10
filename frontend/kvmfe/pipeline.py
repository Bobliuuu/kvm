"""One-call entry points: config -> workload dict (train / serve / game), used by the demo CLI, inspect and tests."""
from __future__ import annotations

from dataclasses import dataclass, replace

from . import log
from .emit import GraphPolicy, Lowered, lower_program, to_workload
from .generic import TrainableModule
from .ir import FrontendError
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


def compile_training(model, tc: TrainConfig | None = None, tiling: Tiling | None = None, reuse='step', gap=0, with_spec=True,
                     fuse=False, *, loss=None, data=None, optimizer=None, steps=None, seed=0) -> Compiled:
    """Compile an unrolled training program.

    Two ways to call it:
      * a model the frontend already knows (nn.ModelConfig / models.* config / model object with the training protocol):
        compile_training(cfg, TrainConfig(...), tiling)
      * ANY traced module (anything with `.specs` and `__call__(b, *inputs, w)`; see generic.py) with a pluggable loss, data
        source and optimizer: compile_training(module, loss=MSE(), data=ArrayData(...), optimizer=OptimConfig(kind='sgd'),
        steps=20, tiling=...). `tc` may still carry the other TrainConfig fields (accum, schedule, clip, ...); `steps` and
        `optimizer` override its fields.
    tiling defaults to the 'a100' preset. fuse=True applies fuse.py before lowering."""
    if loss is not None or data is not None:
        if loss is None or data is None:
            raise FrontendError('a generic module needs both loss= and data=')
        model = TrainableModule(model, loss, data, seed)
    elif optimizer is not None or steps is not None or seed:
        raise FrontendError('steps=, optimizer= and seed= belong to the generic form (loss= and data=); '
                            'for a known model put them in the TrainConfig')
    if tc is None:
        tc = TrainConfig(**({'steps': steps} if steps is not None else {}), **({'optim': optimizer} if optimizer else {}))
    elif loss is not None:
        tc = replace(tc, **({'steps': steps} if steps is not None else {}), **({'optim': optimizer} if optimizer else {}))
    model = build_model(model)
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
    low = lower_program(ops, tiling or Tiling.preset('a100'), reuse, gap, prefix='act')
    spec = to_workload(low, {'train': GraphPolicy()}) if with_spec else None
    return Compiled(low, spec, prog)
