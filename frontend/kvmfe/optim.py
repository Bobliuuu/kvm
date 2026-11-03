"""Optimizer builders. AdamW uses the device op `adamw` (in place on the weight and its two moments, step read from a counter
buffer); SGD uses the legacy flat `sgd`. Both are ordered after the previous publish through the version token (train.py)."""
from __future__ import annotations

from dataclasses import dataclass

from .ir import Builder, FrontendError, Tensor


@dataclass(frozen=True)
class OptimConfig:
    kind: str = 'adamw'            # adamw | sgd
    lr: float = 3e-3
    b1: float = 0.9
    b2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.01

    def __post_init__(self):
        if self.kind not in ('adamw', 'sgd'):
            raise FrontendError(f'unknown optimizer {self.kind!r}; use adamw or sgd')
        if not self.lr > 0:
            raise FrontendError(f'lr must be positive, got {self.lr}')


class State:
    """Persistent optimizer state: AdamW moments per parameter and the 1-based step counter (i32)."""
    def __init__(self, b: Builder, specs, cfg: OptimConfig):
        mk = lambda p, tag: Tensor(f'{tag}.{p.name}', p.rows, p.cols, 'state', buf=f'{tag}.{p.name}', init=('fill', 0.0))
        self.m = {n: mk(p, 'm') for n, p in specs.items()} if cfg.kind == 'adamw' else {}
        self.v = {n: mk(p, 'v') for n, p in specs.items()} if cfg.kind == 'adamw' else {}
        self.step = Tensor('opt_step', 1, 1, 'state', buf='opt_step', init=('fill', 0.0), dtype='i32')
        b.tensors += [*self.m.values(), *self.v.values(), self.step]


def step_begin(b: Builder, cfg: OptimConfig, state: State):
    if cfg.kind == 'adamw':
        b.counter_inc(state.step)


def update(b: Builder, cfg: OptimConfig, spec, w: Tensor, g: Tensor, state: State, token: Tensor):
    with b.scope('opt'):
        if cfg.kind == 'sgd':
            b.sgd(w, g, cfg.lr, token)
        else:
            b.adamw(w, g, state.m[spec.name], state.v[spec.name], state.step, token, lr=cfg.lr, b1=cfg.b1, b2=cfg.b2,
                    eps=cfg.eps, wd=cfg.weight_decay if spec.decay else 0.0)
