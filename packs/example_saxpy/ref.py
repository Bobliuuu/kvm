"""CPU reference of packs/example_saxpy (the compiler's golden and extent checks)."""
from ref_types import RefOp, f32


def _saxpy(b, a):
    for i in range(a['n']):
        b['out'][i] = f32(f32(a['alpha'] * b['x'][i]) + b['y'][i])


OPS = {'saxpy': RefOp(_saxpy, lambda a: {'x': a['n'], 'y': a['n'], 'out': a['n']}, {'n': 1})}
