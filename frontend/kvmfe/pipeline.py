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


def compile_training(model_cfg: ModelConfig, tc: TrainConfig, tiling: Tiling, reuse='shape', gap=0,
                     materialize=True, with_spec=True) -> Compiled:
    model = GPT(model_cfg)
    log.info('model %s: %.1fM parameters, %d layers, d=%d, heads=%d, seq=%d, batch=%d, vocab=%d', model_cfg.name,
             model_cfg.params / 1e6, model_cfg.layers, model_cfg.d_model, model_cfg.heads, model_cfg.seq, model_cfg.batch,
             model_cfg.vocab)
    prog = build_training(model, tc)
    low = lower_program(prog.b.ops, tiling, reuse, gap, prefix='act')
    spec = to_workload(low, {'train': GraphPolicy()}) if with_spec else None
    return Compiled(low, spec, prog)
