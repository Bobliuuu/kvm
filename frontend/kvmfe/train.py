"""Multi-step training program: data -> forward -> loss -> autodiff backward -> optimizer -> publish, unrolled per step.

Steps are not sequenced by this module: step s+1 only depends on step s through tensor regions (weights, optimizer state, reused
activation memory), so dataflow lets its tiles start as soon as the tiles they need are done.

Weight versions (`weights` variable). The update is in place, so two publishes per step bracket it:
    version 2s     weights after s updates are stable           (published by update s-1, `post`)
    version 2s+1   update s is about to overwrite them          (published by `pre`, after backward finished)
    version 2s+2   weights after s+1 updates are stable          (`post`)
A serving request reading `exact` version 2s is ordered before `pre` of step s, i.e. before the first overwrite (the compiler adds
that edge); `at_least` just waits. A one-cell version token chains post(s-1) -> pre(s) -> updates(s) -> post(s).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import autodiff, log
from .ir import Builder, FrontendError, Tensor
from .nn import GPT, ModelConfig
from .optim import OptimConfig, State, step_begin, update

WEIGHTS_VAR = 'weights'


def weights_version(updates: int) -> int:
    """Variable version at which the weights after `updates` optimizer updates are stable."""
    return 2 * updates


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 20
    optim: OptimConfig = OptimConfig()


@dataclass
class Program:
    b: Builder
    model: GPT
    weights: dict = field(default_factory=dict)
    losses: list = field(default_factory=list)       # one 1x2 tensor per step; loss[0,0] is the mean cross-entropy
    state: State | None = None


def batch_tokens(cfg: ModelConfig, step: int):
    """Deterministic host-provided batch: per sequence a chain t -> (5t+3) mod V (a learnable bigram), next-token targets."""
    ids, tgt = [], []
    for b in range(cfg.batch):
        t = (step * cfg.batch + b) * 7919 % cfg.vocab
        chain = []
        for _ in range(cfg.seq + 1):
            chain.append(t)
            t = (5 * t + 3) % cfg.vocab
        ids += chain[:-1]
        tgt += chain[1:]
    return ids, tgt


def weight_tensor(spec, seed):
    return Tensor(spec.name, spec.rows, spec.cols, 'param', buf=f'w.{spec.name}', init=('param', spec.init, f'{seed}/{spec.name}'), rg=True)


def build_training(model: GPT, tc: TrainConfig) -> Program:
    cfg = model.cfg
    if tc.steps < 1:
        raise FrontendError('steps must be >= 1')
    b = Builder('train')
    w = {n: weight_tensor(s, cfg.seed) for n, s in model.specs.items()}
    token = Tensor('version_token', 1, 1, 'state', buf='version_token', init=('fill', 0.0))
    state = State(b, model.specs, tc.optim)
    b.tensors += [*w.values(), token]
    prog = Program(b, model, w, [], state)
    for s in range(tc.steps):
        b.step, b.phase = s, 'data'
        ids, tgt = (Tensor(f'{n}.s{s}', cfg.tokens, 1, 'input', buf=f'{n}.s{s}', dtype='i32', init=('values', v))
                    for n, v in zip(('ids', 'tgt'), batch_tokens(cfg, s)))
        b.tensors += [ids, tgt]
        b.phase, mark = 'fwd', len(b.ops)
        loss = b.ce_loss(model(b, ids, w), tgt)
        loss.buf = f'loss.{s}'                      # tiny and pinned: the loss curve stays readable after the run
        prog.losses.append(loss)
        grads = autodiff.backward(b, loss, b.ops[mark:], [w[n] for n in model.specs])
        b.phase = 'opt'
        step_begin(b, tc.optim, state)
        b.barrier(list(grads.values()) + [token], [token], publishes=WEIGHTS_VAR)          # pre: 2s+1
        for n, sp in model.specs.items():
            update(b, tc.optim, sp, w[n], grads[w[n]], state, token)
        b.barrier(list(w.values()) + [token], [token], publishes=WEIGHTS_VAR)               # post: 2s+2
        log.debug('step %d: %d logical ops so far', s, len(b.ops))
    return prog
