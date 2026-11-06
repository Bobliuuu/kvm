"""Shared types of the CPU reference operators (see ref_ops.py for the model)."""
import ctypes
from dataclasses import dataclass, field
from typing import Callable

F32_MAX = 3.4028234663852886e38
_C = ctypes.c_float


def f32(x: float) -> float:
    """Round a Python float to the nearest float32 (overflow gives +-inf)."""
    return _C(x).value


class Buf(list):
    """A device buffer as the reference sees it: flat row-major values plus shape and dtype ('f32' | 'i32')."""
    __slots__ = ('rows', 'cols', 'dtype')

    def __init__(self, values, rows, cols, dtype='f32'):
        super().__init__(values)
        self.rows, self.cols, self.dtype = rows, cols, dtype

    def at(self, r, c):
        return self[r * self.cols + c]


@dataclass(frozen=True)
class RefOp:
    """Reference semantics of one operator.

    run(buffers, scalars): buffers by arg name (a Buf, or None for an absent optional), scalars = i32/f32 args.
    extents(scalars) -> {buf arg: n | (rows, cols)}: n = needs >= n elements; (r, c) = needs rows >= r and cols >= c.
      Raises ValueError for arguments no execution could accept (negative ranges, bad flags).
    mins: i32 arg -> smallest legal value.  ints: buf args that must be i32 buffers.  anydt: buf args of any dtype.
    opaque: no CPU semantics; it executes as a noop with the declared extents, only if the runtime allows opaque ops.
    """
    run: Callable
    extents: Callable
    mins: dict = field(default_factory=dict)
    ints: frozenset = frozenset()
    anydt: frozenset = frozenset()
    bf16: frozenset = frozenset()                # buf args that must be bf16-storage buffers (dtype 2)
    dtfn: Callable | None = None                 # scalars -> {buf arg: dtype} for args whose dtype depends on a flag word
    opaque: bool = False
