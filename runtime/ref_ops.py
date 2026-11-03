"""Reference (CPU, float32) semantics of every operator in include/kvm_ops.def.

The operator *table* (names, ids, argument types) lives in kvm_ops.def; this module only adds what a table
cannot say: what an operator does and which buffer extents / value ranges it needs. The compiler uses the
extents to reject out-of-bounds tasks at compile time; the CPU VM uses the functions to execute.

Float32 model: every arithmetic result is rounded to float32 (IEEE round-to-nearest-even) after each
operation, and reductions run in ascending index order, so integer-valued data is bit-exact against the GPU.
Multiply-add is NOT fused here: a GPU that fuses may differ by 1 ulp on non-integer data.
"""
import ctypes
from dataclasses import dataclass
from typing import Callable

F32_MAX = 3.4028234663852886e38
_C = ctypes.c_float


def f32(x: float) -> float:
    """Round a Python float to the nearest float32 (overflow gives +-inf)."""
    return _C(x).value


@dataclass(frozen=True)
class RefOp:
    run: Callable[[dict, dict], None]        # (buffers by arg name -> flat list, scalar args) -> None
    extents: Callable[[dict], dict]          # scalar args -> {buf arg name: elements touched}
    mins: dict                               # i32 arg name -> smallest legal value


def _copy(b, a):
    src, dst = b['src'], b['dst']
    for i in range(a['n']):
        dst[i] = src[i]


def _add(b, a):
    x, y, out = b['a'], b['b'], b['out']
    for i in range(a['n']):
        out[i] = f32(x[i] + y[i])


def _scale(b, a):
    x, out, alpha = b['x'], b['out'], a['alpha']
    for i in range(a['n']):
        out[i] = f32(x[i] * alpha)


def _relu(b, a):
    x, out = b['x'], b['out']
    for i in range(a['n']):
        out[i] = x[i] if x[i] > 0 else 0.0     # NaN -> 0, like fmaxf(x, 0)


def _sgd(b, a):
    w, g, lr = b['w'], b['g'], a['lr']
    for i in range(a['n']):
        w[i] = f32(w[i] - f32(lr * g[i]))


def _predict(b, a):
    x, w, out = b['x'], b['w'], b['out']
    for i in range(a['n']):
        out[i] = f32(w[0] * x[i])


def _mse_grad(b, a):
    pred, target, x, n = b['pred'], b['target'], b['x'], a['n']
    loss_acc = grad_acc = 0.0
    for i in range(n):
        d = f32(pred[i] - target[i])
        loss_acc = f32(loss_acc + f32(d * d))
        grad_acc = f32(grad_acc + f32(f32(2.0 * d) * x[i]))
    b['loss'][0] = f32(loss_acc / n)
    b['grad'][0] = f32(grad_acc / n)


def _matmul(b, a):
    x, y, m, n, k = b['a'], b['b'], a['m'], a['n'], a['k']
    out = [0.0] * (m * n)
    for i in range(m):
        for j in range(n):
            acc = 0.0
            for p in range(k):
                acc = f32(acc + f32(x[i * k + p] * y[p * n + j]))
            out[i * n + j] = acc
    b['out'][:m * n] = out


def _fill(b, a):
    x, v = b['x'], a['value']
    for i in range(a['n']):
        x[i] = v


def _spin(b, a):
    return None


_N = {'n': 1}
OPS: dict[str, RefOp] = {
    'noop': RefOp(lambda b, a: None, lambda a: {}, {}),
    'copy': RefOp(_copy, lambda a: {'src': a['n'], 'dst': a['n']}, _N),
    'add': RefOp(_add, lambda a: {'a': a['n'], 'b': a['n'], 'out': a['n']}, _N),
    'scale': RefOp(_scale, lambda a: {'x': a['n'], 'out': a['n']}, _N),
    'relu': RefOp(_relu, lambda a: {'x': a['n'], 'out': a['n']}, _N),
    'sgd': RefOp(_sgd, lambda a: {'w': a['n'], 'g': a['n']}, _N),
    'predict': RefOp(_predict, lambda a: {'x': a['n'], 'w': 1, 'out': a['n']}, _N),
    'mse_grad': RefOp(_mse_grad, lambda a: {'pred': a['n'], 'target': a['n'], 'x': a['n'], 'loss': 1, 'grad': 1}, _N),
    'matmul': RefOp(_matmul, lambda a: {'a': a['m'] * a['k'], 'b': a['k'] * a['n'], 'out': a['m'] * a['n']},
                    {'m': 1, 'n': 1, 'k': 1}),
    'fill': RefOp(_fill, lambda a: {'x': a['n']}, _N),
    'spin': RefOp(_spin, lambda a: {}, {'us': 0}),
}
