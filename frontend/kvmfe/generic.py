"""Generic training: any traced module + a pluggable loss + a pluggable data source -> the unrolled training program.

    compile_training(module, loss=MSE(), data=ArrayData(...), optimizer=OptimConfig(kind='sgd'), steps=20)   # pipeline.py

A *module* is anything with `.specs` ({name: nn.ParamSpec}) and `__call__(b, *inputs, w) -> output tensor`, where `inputs` are the
batch tensors named by `data.inputs`, in that order. nn.Module subclasses (Linear, Norm, ...) compose as usual; nothing here knows
about attention geometry, token streams or softmax cross-entropy, so `ModelConfig`'s GPT-shaped validation never applies.
`TrainableModule` adapts (module, loss, data) to the protocol `train.build_training` consumes, so microbatch accumulation, LR
schedules, loss scaling, rematerialization and fusion work unchanged.

Reinforcement-learning pieces: `PolicyGradient` (REINFORCE with an advantage stream) and `QLearning` (TD/DQN on a one-hot action mask) losses; data
from logged or host-simulated rollouts through `GeneratedData` / `ArrayData`. An on-policy loop is a host loop around launches (roll
out with the latest weights on the host, feed the batch, run the bundle); one bundle never samples from its own policy.

Not supported (the program is a static DAG): data-dependent control flow, dynamic shapes, auxiliary losses from inside a module.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

from .ir import Builder, FrontendError, Tensor
from .nn import Linear, Module, Norm, Stream


# ====================================================================================================== losses
class Loss:
    """A loss turns (module output, target) into a scalar loss tensor whose loss[0,0] is the value reported per step.
    `n_targets` host streams feed it: `target` is one Stream/Tensor when it is 1, else a tuple in the order of `DataSource.target`
    (a tuple of stream names)."""
    target_dtype = 'f32'
    n_targets = 1

    def check(self, out: Tensor, target: Stream):
        raise NotImplementedError

    def __call__(self, b: Builder, out: Tensor, target: Tensor, grad_scale: float) -> Tensor:
        raise NotImplementedError


class CrossEntropy(Loss):
    """Mean softmax cross-entropy of `out` [N, classes] against an i32 class-index column [N, 1]."""
    target_dtype = 'i32'

    def check(self, out, target):
        if (target.rows, target.cols, target.dtype) != (out.rows, 1, 'i32'):
            raise FrontendError(f'CrossEntropy: target stream {target.name!r} must be an i32 column of {out.rows} rows '
                                f'(got {target.rows}x{target.cols} {target.dtype})')
        bad = [v for v in target.values if not 0 <= v < out.cols]
        if bad:
            raise FrontendError(f'CrossEntropy: target {target.name!r} has class ids outside [0, {out.cols}): e.g. {bad[:3]}')

    def __call__(self, b, out, target, grad_scale):
        return b.ce_loss(out, target, grad_scale)


class MSE(Loss):
    """Mean squared error over every element of `out` [N, D] against an f32 target of the same shape."""

    def check(self, out, target):
        if (target.rows, target.cols, target.dtype) != (out.rows, out.cols, 'f32'):
            raise FrontendError(f'MSE: target stream {target.name!r} must be f32 {out.rows}x{out.cols} '
                                f'(got {target.rows}x{target.cols} {target.dtype})')

    def __call__(self, b, out, target, grad_scale):
        return b.mse_loss(out, target, grad_scale)


class PolicyGradient(Loss):
    """REINFORCE with a given advantage: mean over rows of advantage * (-log pi(action | state)), pi = softmax of the module output
    [N, actions]. Targets, in this order: the taken action (i32 column, in [0, actions)) and the advantage / return (f32 column;
    data, so it gets no gradient). Data comes from a rollout function (GeneratedData) or logged episodes (ArrayData)."""
    n_targets = 2

    def check(self, out, target):
        act, adv = target
        if (act.rows, act.cols, act.dtype) != (out.rows, 1, 'i32') or (adv.rows, adv.cols, adv.dtype) != (out.rows, 1, 'f32'):
            raise FrontendError(f'PolicyGradient: targets must be (i32 action column, f32 advantage column) of {out.rows} rows; got '
                                f'{act.name!r} {act.rows}x{act.cols} {act.dtype}, {adv.name!r} {adv.rows}x{adv.cols} {adv.dtype}')
        bad = [v for v in act.values if not 0 <= v < out.cols]
        if bad:
            raise FrontendError(f'PolicyGradient: action ids outside [0, {out.cols}): e.g. {bad[:3]}')

    def __call__(self, b, out, target, grad_scale):
        return b.pg_loss(out, target[0], target[1], grad_scale)


class QLearning(Loss):
    """One-step TD / DQN loss: mean over rows of (Q(s, a) - y)^2 with Q = the module output [N, actions]. Targets, in this order: a
    one-hot f32 action mask [N, actions] (so Q(s,a) is a dot with the row) and the bootstrapped target y (f32 column, e.g.
    r + gamma * max_a' Q_target(s', a'), computed on the host from a frozen copy: data, no gradient)."""
    n_targets = 2

    def check(self, out, target):
        mask, y = target
        if (mask.rows, mask.cols, mask.dtype) != (out.rows, out.cols, 'f32') or (y.rows, y.cols, y.dtype) != (out.rows, 1, 'f32'):
            raise FrontendError(f'QLearning: targets must be (f32 one-hot mask {out.rows}x{out.cols}, f32 target column {out.rows}x1); got '
                                f'{mask.name!r} {mask.rows}x{mask.cols} {mask.dtype}, {y.name!r} {y.rows}x{y.cols} {y.dtype}')

    def __call__(self, b, out, target, grad_scale):
        return b.mse_loss(b.row_select(out, target[0]), target[1], grad_scale)


def one_hot(actions, n):
    """Rows of a one-hot f32 mask for action ids (a host helper for QLearning data)."""
    return [[1.0 if c == int(a) else 0.0 for c in range(n)] for a in actions]


# ====================================================================================================== data sources
class DataSource:
    """Host data consumed one window per microbatch. `inputs`: stream names passed to the module; `target`: the stream the loss reads."""
    inputs: tuple = ('x',)
    target: str | tuple = 'y'        # one stream name, or a tuple of names for a loss with several targets (PolicyGradient, QLearning)

    @property
    def targets(self) -> tuple:
        return (self.target,) if isinstance(self.target, str) else tuple(self.target)

    def streams(self, micros: int) -> list[Stream]:
        raise NotImplementedError


def _as_rows(name, data, dtype):
    rows = [list(r) if isinstance(r, (list, tuple)) else [r] for r in data]
    if not rows or not rows[0]:
        raise FrontendError(f'data array {name!r} is empty')
    width = len(rows[0])
    for i, r in enumerate(rows):
        if len(r) != width:
            raise FrontendError(f'data array {name!r}: row {i} has {len(r)} values, row 0 has {width}')
        for v in r:
            if not math.isfinite(v) or (dtype == 'i32' and v != int(v)):
                raise FrontendError(f'data array {name!r}: value {v!r} is not a finite {"integer" if dtype == "i32" else "number"}')
    return rows


class ArrayData(DataSource):
    """Fixed sample arrays served in order, `batch` samples per microbatch, wrapping around (epochs) when the samples run out.
    arrays: {name: list of rows (a row is a list of numbers, or one number)}; dtypes: {name: 'f32' | 'i32'} (default f32).
    Every array has the same number of samples."""

    def __init__(self, batch: int, arrays: dict, dtypes: dict | None = None, inputs=('x',), target='y'):
        if batch < 1:
            raise FrontendError(f'batch must be >= 1, got {batch}')
        self.inputs, self.target, self.batch = tuple(inputs), target, batch
        dtypes = dtypes or {}
        unknown = sorted(set(dtypes) - set(arrays))
        missing = sorted({*self.inputs, *self.targets} - set(arrays))
        if unknown or missing:
            raise FrontendError(f'ArrayData: dtypes for unknown arrays {unknown}; inputs/target missing from arrays: {missing}')
        self.dtypes = {n: dtypes.get(n, 'f32') for n in arrays}
        bad = {n: d for n, d in self.dtypes.items() if d not in ('f32', 'i32')}
        if bad:
            raise FrontendError(f'ArrayData: dtype must be f32 or i32, got {bad}')
        self.arrays = {n: _as_rows(n, a, self.dtypes[n]) for n, a in arrays.items()}
        sizes = {n: len(a) for n, a in self.arrays.items()}
        if len(set(sizes.values())) != 1:
            raise FrontendError(f'ArrayData: arrays have different sample counts {sizes}')
        self.samples = next(iter(sizes.values()))

    def streams(self, micros):
        out = []
        for n in (*self.inputs, *self.targets):
            a = self.arrays[n]
            vals = [v for m in range(micros) for i in range(self.batch) for v in a[(m * self.batch + i) % self.samples]]
            out.append(Stream(n, self.batch, len(a[0]), self.dtypes[n], vals))
        return out


class GeneratedData(DataSource):
    """Data from a function: make(micro) -> {name: list of rows} for microbatch number `micro` (0-based, deterministic).
    shapes: {name: (rows, cols, dtype)} per microbatch, checked against what make returns."""

    def __init__(self, make, shapes: dict, inputs=('x',), target='y'):
        self.make, self.shapes, self.inputs, self.target = make, dict(shapes), tuple(inputs), target
        missing = sorted({*self.inputs, *self.targets} - set(shapes))
        if missing:
            raise FrontendError(f'GeneratedData: no shape for {missing}')

    def streams(self, micros):
        cols = {n: [] for n in (*self.inputs, *self.targets)}
        for m in range(micros):
            got = self.make(m)
            for n in cols:
                rows, width, dtype = self.shapes[n]
                a = _as_rows(n, got[n], dtype)
                if (len(a), len(a[0])) != (rows, width):
                    raise FrontendError(f'GeneratedData: make({m})[{n!r}] is {len(a)}x{len(a[0])}, declared {rows}x{width}')
                cols[n] += [v for r in a for v in r]
        return [Stream(n, *self.shapes[n][:2], self.shapes[n][2], cols[n]) for n in cols]


# ====================================================================================================== the adapter
class TrainableModule:
    """(module, loss, data) as the model protocol of train.build_training: cfg.name / cfg.seed, specs, batch_streams, losses."""

    def __init__(self, module, loss: Loss, data: DataSource, seed: int = 0, name: str | None = None):
        specs = getattr(module, 'specs', None)
        if not specs or not callable(module):
            raise FrontendError(f'{type(module).__name__}: a trainable module needs a non-empty `.specs` and `__call__(b, *inputs, w)`')
        if not isinstance(loss, Loss):
            raise FrontendError(f'loss must be a generic.Loss (CrossEntropy, MSE), got {type(loss).__name__}')
        if not isinstance(data, DataSource):
            raise FrontendError(f'data must be a generic.DataSource (ArrayData, GeneratedData), got {type(data).__name__}')
        self.module, self.loss, self.data, self.specs = module, loss, data, specs
        self.cfg = SimpleNamespace(name=name or getattr(module, 'name', type(module).__name__), seed=seed)
        self._checked = False

    def batch_streams(self, micros):
        streams = self.data.streams(micros)
        by_name = {s.name: s for s in streams}
        for n in (*self.data.inputs, *self.data.targets):
            if n not in by_name:
                raise FrontendError(f'data source produced no stream {n!r}; has {sorted(by_name)}')
        return [by_name[n] for n in (*self.data.inputs, *self.data.targets)]

    def losses(self, b, batch, w, grad_scale=1.0):
        out = self.module(b, *(batch[n] for n in self.data.inputs), w)
        if not isinstance(out, Tensor):
            raise FrontendError(f'module {self.cfg.name!r} returned {type(out).__name__}, not a Tensor')
        names = self.data.targets
        if len(names) != self.loss.n_targets:
            raise FrontendError(f'{type(self.loss).__name__} needs {self.loss.n_targets} target stream(s), the data source names {len(names)}: {names}')
        if not self._checked:                                   # shapes are fixed for the whole run: check once, with a clear message
            by = {s.name: s for s in self.batch_streams(1)}
            self.loss.check(out, by[names[0]] if len(names) == 1 else tuple(by[n] for n in names))
            self._checked = True
        tg = batch[names[0]] if len(names) == 1 else tuple(batch[n] for n in names)
        return [('loss', self.loss(b, out, tg, grad_scale))]


# ====================================================================================================== building blocks
class Conv1D(Module):
    """1-D convolution over sequences of `length` rows (batch*length rows, `cin` channels), zero padding 'same', odd `taps`.
    Built from existing ops: tap j is a row gather shifted by j - taps//2 (out-of-sequence rows read as zeros) feeding a matmul
    that accumulates into the previous taps' output (the linear epilogue's residual input)."""

    def __init__(self, name, cin, cout, taps, length, std=0.1, bias=True):
        super().__init__(name)
        if taps < 1 or taps % 2 == 0 or length < 1:
            raise FrontendError(f'Conv1D {name}: taps must be odd and >= 1, length >= 1 (got taps={taps}, length={length})')
        self.cin, self.cout, self.taps, self.length = cin, cout, taps, length
        self.ws = [self.add(f'w{j}', cin, cout, ('normal', std), decay=True) for j in range(taps)]
        self.b = self.add('b', 1, cout, ('fill', 0.0)) if bias else None
        self._idx = {}

    def _shift_index(self, b, rows, shift):
        """i32 column: for output row r, the source row r + shift in the same sequence, or -1 (padding) outside it."""
        if shift == 0:
            return None
        key = (rows, shift)
        if key not in self._idx:
            idx = [r + shift if 0 <= (r % self.length) + shift < self.length else -1 for r in range(rows)]
            self._idx[key] = b.tensor(f'{self.name}.shift{shift:+d}', rows, 1, 'input', buf=f'{self.name}.shift{shift:+d}.{rows}',
                                      dtype='i32', init=('values', idx))
        return self._idx[key]

    def __call__(self, b, x, w):
        if x.cols != self.cin or x.rows % self.length:
            raise FrontendError(f'Conv1D {self.name}: input {x.shape} needs {self.cin} channels and rows % {self.length} == 0')
        out = None
        for j, wn in enumerate(self.ws):
            idx = self._shift_index(b, x.rows, j - self.taps // 2)
            tap = x if idx is None else b.gather(x, idx, x.rows)
            out = b.linear(tap, w[wn], w[self.b] if self.b and j == 0 else None, out)
        return out


class SequencePool(Module):
    """Mean over each sequence of `length` rows: [batch*length, d] -> [batch, d] (a scatter-add with scale 1/length)."""

    def __init__(self, name, length):
        super().__init__(name)
        self.length = length

    def __call__(self, b, x, w):
        n = x.rows
        if n % self.length:
            raise FrontendError(f'SequencePool {self.name}: {n} rows are not a multiple of length {self.length}')
        idx = b.tensor(f'{self.name}.idx', n, 1, 'input', buf=f'{self.name}.idx.{n}', dtype='i32',
                       init=('values', [r // self.length for r in range(n)]))
        inv = b.tensor(f'{self.name}.inv', n, 1, 'input', buf=f'{self.name}.inv.{n}', init=('values', [1.0 / self.length] * n))
        return b.scatter_add(x, idx, n // self.length, scale=inv)


class MLP(Module):
    """Feed-forward stack: Linear -> activation (relu | silu | tanh | sigmoid | gelu), `residual` adds a skip connection around
    every hidden layer whose width equals its input width, `norm` ('layernorm' | 'rmsnorm' | None) is applied after each hidden
    layer. dims = [in, hidden..., out]."""

    def __init__(self, name, dims, act='relu', norm=None, residual=False, bias=True, std=None, precision='f32'):
        super().__init__(name)
        if precision not in ('f32', 'bf16', 'bf16-real'):
            raise FrontendError(f"MLP {name}: precision must be 'f32', 'bf16' (emulated: bf16 values in fp32 containers) or "
                                f"'bf16-real' (2-byte bf16 storage for weight copies and activations), got {precision!r}")
        self.precision = precision
        if len(dims) < 2 or any(d < 1 for d in dims):
            raise FrontendError(f'MLP {name}: dims must be [in, hidden..., out] with positive widths, got {dims}')
        if act not in (*Builder.ACT_KINDS, 'gelu'):
            raise FrontendError(f'MLP {name}: unknown activation {act!r}')
        self.act, self.residual = act, residual
        self.layers = [Linear(f'{name}.fc{i}', a, c, bias, std if std is not None else math.sqrt(2.0 / a))
                       for i, (a, c) in enumerate(zip(dims, dims[1:]))]
        self.norms = [Norm(f'{name}.n{i}', c, norm) for i, c in enumerate(dims[1:-1])] if norm else []
        self.absorb(*self.layers, *self.norms)

    def __call__(self, b, x, w):
        last = len(self.layers) - 1
        lp = {'f32': False, 'bf16': True, 'bf16-real': 'real'}[self.precision]
        f32 = lambda t: b.cast(t, 'f32') if t.dtype == 'bf16' else t
        for i, fc in enumerate(self.layers):
            if i == last:
                return fc(b, x, w, bf16=lp, cast_out=False)
            skip = f32(x) if self.residual and self.specs[fc.w].cols == x.cols else None
            if self.act == 'gelu':
                x = fc(b, x, w, res=skip, gelu=True, bf16=lp, cast_out=False)    # fused epilogue: gelu(x W + b) + skip
            else:
                h = b.act(fc(b, x, w, bf16=lp, cast_out=False), self.act)
                x = b.add(h, skip) if skip is not None else h
            if self.norms:
                x = self.norms[i](b, x, w)
            if lp:
                x = b.cast(x, 'bf16', real=lp == 'real')               # the activation between layers is stored in bf16
        return x
