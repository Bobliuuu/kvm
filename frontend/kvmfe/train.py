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

import zlib
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
class Schedule:
    """Learning-rate schedule evaluated ON THE DEVICE by a task each step from the step counter (lr is a buffer, so the program
    text of every step is identical and window templates still work). kind: const | cosine | linear; linear warmup of
    `warmup` steps first; decays to min_ratio * base at `total` (default: the whole run)."""
    kind: str = 'cosine'
    warmup: int = 0
    total: int | None = None
    min_ratio: float = 0.1

    KINDS = {'const': 0, 'cosine': 1, 'linear': 2}

    def __post_init__(self):
        if self.kind not in self.KINDS or self.warmup < 0 or not 0 <= self.min_ratio <= 1:
            raise FrontendError(f'bad schedule {self}: kind in {sorted(self.KINDS)}, warmup >= 0, 0 <= min_ratio <= 1')


@dataclass(frozen=True)
class TrainConfig:
    """steps: optimizer steps unrolled in this program. total_steps: length of the token stream (>= first_step + steps; default
    first_step + steps). first_step > 0 builds a later window of a longer run: the device step counter already holds first_step,
    so the same program text works for any window (see window/loop options in demo.py).
    accum: microbatches per optimizer step (gradients accumulate on the device; each microbatch's loss gradient is scaled 1/accum).
    schedule: device-side LR schedule (AdamW). loss_scale: static loss scale S for low-precision runs: the backward starts from
    S * dL, and the clip coefficient (AdamW's gscale) is multiplied by 1/S, so weights see the unscaled gradient."""
    steps: int = 20
    optim: OptimConfig = OptimConfig()
    clip: float = 1.0              # global gradient-norm clip (AdamW only); <= 0 disables
    first_step: int = 0
    total_steps: int | None = None
    accum: int = 1
    schedule: Schedule | None = None
    loss_scale: float = 1.0


@dataclass
class Program:
    b: Builder
    model: object
    weights: dict = field(default_factory=dict)
    losses: list = field(default_factory=list)       # per (step, micro): {name: 1x2/1x1 tensor}; ['loss'][0,0] is the mean cross-entropy
    state: State | None = None
    clip_coef: Tensor | None = None
    lr: Tensor | None = None
    grads: dict = field(default_factory=dict)        # last step's {param name: gradient tensor} (pre-clip, loss-scaled)

    def loss_buffer(self, name, step, micro=0, accum=1):
        return f'{name}.{step}' if accum == 1 else f'{name}.{step}.{micro}'


def batch_stream(cfg: ModelConfig, steps: int):
    """The whole token stream: step s occupies elements [s*B*T, (s+1)*B*T) of ids and tgt (the device selects it from the counter)."""
    ids, tgt = [], []
    for s in range(steps):
        i, t = batch_tokens(cfg, s)
        ids += i
        tgt += t
    return ids, tgt


def batch_tokens(cfg: ModelConfig, step: int):
    """Deterministic host-provided batch: per sequence a chain t -> (5t+3) mod V (a learnable bigram), next-token targets."""
    ids, tgt = [], []
    if getattr(cfg, 'repeat_batch', False):        # only the GPT config has the flag (an overfitting check); other families train on the stream
        step = 0
    for b in range(cfg.batch):
        t = (step * cfg.batch + b) * 7919 % cfg.vocab
        chain = []
        for _ in range(cfg.seq + 1):
            chain.append(t)
            t = (5 * t + 3) % cfg.vocab
        ids += chain[:-1]
        tgt += chain[1:]
    return ids, tgt


def param_init(spec, seed):
    """Initial-value spec of a parameter: the compiler's counter-based generator, seeded from the parameter name."""
    return ('fill', spec.init[1]) if spec.init[0] == 'fill' else ('seeded', zlib.crc32(f'{seed}/{spec.name}'.encode()), spec.init[1])


def weight_tensor(spec, seed):
    return Tensor(spec.name, spec.rows, spec.cols, 'param', buf=f'w.{spec.name}', init=param_init(spec, seed), rg=True)


def build_training(model, tc: TrainConfig) -> Program:
    """Unrolled training program for any model implementing the protocol (nn.GPT is one; see models.py):
         model.cfg.seed, model.specs {name: ParamSpec}
         model.batch_streams(micros) -> [nn.Stream]           host data, one window per microbatch
         model.losses(b, batch {stream: tensor}, w {param: tensor}, grad_scale) -> [(name, 1x1|1x2 loss tensor)]
    The first loss is the primary one; every returned loss is a backward root with gradient 1."""
    cfg = model.cfg
    if tc.steps < 1 or tc.accum < 1:
        raise FrontendError('steps and accum must be >= 1')
    total = tc.total_steps or tc.first_step + tc.steps
    if total < tc.first_step + tc.steps:
        raise FrontendError(f'total_steps {total} < first_step + steps = {tc.first_step + tc.steps}')
    adamw = tc.optim.kind == 'adamw'
    if not adamw and (tc.schedule or tc.loss_scale != 1.0):
        raise FrontendError('schedule and loss_scale need the AdamW optimizer (sgd has a baked lr and no gradient scale input)')
    if not tc.loss_scale > 0:
        raise FrontendError(f'loss_scale must be positive, got {tc.loss_scale}')
    clip = adamw and (tc.clip > 0 or tc.loss_scale != 1.0)
    b = Builder('train')
    w = {n: weight_tensor(s, cfg.seed) for n, s in model.specs.items()}
    token = Tensor('version_token', 1, 1, 'state', buf='version_token', init=('fill', 0.0))
    state = State(b, model.specs, tc.optim)
    coef = Tensor('clip_coef', 1, 2, 'state', buf='clip_coef', init=('fill', 1.0)) if clip else None
    lr = Tensor('lr', 1, 1, 'state', buf='lr', init=('fill', tc.optim.lr)) if tc.schedule else None
    streams = model.batch_streams(total * tc.accum)
    src = [Tensor(f'stream_{st.name}', total * tc.accum * st.rows, st.cols, 'input', buf=f'stream_{st.name}', dtype=st.dtype,
                  init=('values', st.values)) for st in streams]
    b.tensors += [*w.values(), token, *src, *([coef] if clip else []), *([lr] if lr else [])]
    prog = Program(b, model, w, [], state, coef, lr)
    params = [w[n] for n in model.specs]
    for s in range(tc.steps):
        b.step, grads = s, None
        for m in range(tc.accum):
            b.phase = 'data'
            batch = {st.name: b.select_window(t, state.step, st.rows, tc.accum * st.rows * st.cols, m * st.rows * st.cols,
                                              f'{st.name}{m}', st.cols)
                     for st, t in zip(streams, src)}
            b.phase, mark = 'fwd', len(b.ops)
            roots = model.losses(b, batch, w, tc.loss_scale / tc.accum)
            for name, t in roots:
                t.buf = prog.loss_buffer(name, s, m, tc.accum)       # tiny and pinned: the curves stay readable after the run
            prog.losses.append(dict(roots))
            grads = autodiff.backward(b, [t for _, t in roots], b.ops[mark:], params, accumulate=grads)
        b.phase = 'opt'
        step_grads = [grads[p] for p in params]
        if clip:
            b.grad_clip(step_grads, tc.clip * tc.loss_scale if tc.clip > 0 else 1e30, coef)
            if tc.loss_scale != 1.0:
                b.scale_scalar(coef, 1.0 / tc.loss_scale)
        step_begin(b, tc.optim, state)                                 # step counter: s -> s + 1 (AdamW's 1-based step)
        if lr:
            sch = tc.schedule
            b.lr_schedule(state.step, lr, Schedule.KINDS[sch.kind], tc.optim.lr, sch.warmup, sch.total or total, sch.min_ratio)
        b.barrier(step_grads + [token], [token], publishes=WEIGHTS_VAR)          # pre: 2s+1
        for n, sp in model.specs.items():
            update(b, tc.optim, sp, w[n], grads[w[n]], state, token, coef, lr)
        b.barrier(list(w.values()) + [token], [token], publishes=WEIGHTS_VAR)               # post: 2s+2
        log.debug('step %d: %d logical ops so far', s, len(b.ops))
    prog.grads = {n: grads[w[n]] for n in model.specs}
    return prog
