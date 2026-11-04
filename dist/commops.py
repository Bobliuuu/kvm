"""Reference (CPU) semantics of the communication operators.

include/kvm_ops_comm.def is the single source of truth for ids/signatures (parsed by runtime/abi.py, never copied here).
`install()` replaces the opaque placeholders for them in runtime/ref_ops.py by real semantics + extents, so shards compile to
the very bytes the GPU executor receives and the simulator can run them.
Remote buffers, inboxes and send queues are resolved by the caller (tests/dist/sim.py) to ordinary Python lists, so a
`put` into a peer is a plain list write: the reference semantics do not know where a buffer lives.
"""
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime'))
import abi          # noqa: E402
import ref_ops      # noqa: E402

COMM_DEF = ROOT / 'include' / 'kvm_ops_comm.def'
_installed = False


def merged_ops():
    """Every operator the compiler knows (the registries of include/, comm ops included: runtime/abi.py loads them all)."""
    ops = abi.load_ops()
    missing = {n for n, o in ops.items() if 64 <= o.id <= 95} - set(_ref())
    if missing:
        raise abi.AbiError(f'comm ops without reference semantics: {sorted(missing)}')
    return ops


def f32(x):
    return ref_ops.f32(x)


def bf16_bits(x: float) -> int:
    u = struct.unpack('<I', struct.pack('<f', x))[0]
    if (u & 0x7fffffff) > 0x7f800000:
        return (u >> 16) | 0x40
    return ((u + 0x7fff + ((u >> 16) & 1)) >> 16) & 0xffff


def _word(lo: int, hi: int) -> float:
    return struct.unpack('<f', struct.pack('<I', (hi << 16) | lo))[0]


def _bits(w: float) -> int:
    return struct.unpack('<I', struct.pack('<f', w))[0]


def _tile_ext(off, rows, cols, ld):
    return off + (rows - 1) * ld + cols


def _put(b, a):
    src, dst = b['src'], b['dst']
    for r in range(a['rows']):
        so, do = a['src_off'] + r * a['src_ld'], a['dst_off'] + r * a['dst_ld']
        dst[do:do + a['cols']] = src[so:so + a['cols']]


def _signal_to(flag, var, ver):
    flag[var] = ver                      # single-writer chain, strictly increasing (see DISTRIBUTED.md)


def _put_signal(b, a):
    _put(b, a)
    _signal_to(b['flag'], a['var'], a['ver'])


def _signal(b, a):
    _signal_to(b['flag'], a['var'], a['ver'])


def _reduce2(b, a):
    x, y, o, n = b['a'], b['b'], b['out'], a['n']
    for i in range(n):
        o[a['o_off'] + i] = f32(x[a['a_off'] + i] + y[a['b_off'] + i])


def _reduce_put_signal(b, a):
    x, y, d, n = b['a'], b['b'], b['dst'], a['n']
    for i in range(n):
        d[a['d_off'] + i] = f32(x[a['a_off'] + i] + y[a['b_off'] + i])
    _signal_to(b['flag'], a['var'], a['ver'])


def _send_proxy(b, a):
    src, q = b['src'], b['q']
    data = []
    for r in range(a['rows']):
        o = a['src_off'] + r * a['src_ld']
        data.extend(src[o:o + a['cols']])
    q.append(dict(dst_rank=a['dst_rank'], dst_buf=a['dst_buf'], dst_off=a['dst_off'], dst_ld=a['dst_ld'],
                  rows=a['rows'], cols=a['cols'], var=a['var'], ver=a['ver'], data=data))


def _pack_bf16(b, a):
    s, d, n = b['src'], b['dst'], a['n']
    for i in range((n + 1) // 2):
        lo = bf16_bits(s[a['src_off'] + 2 * i])
        hi = bf16_bits(s[a['src_off'] + 2 * i + 1]) if 2 * i + 1 < n else 0
        d[a['dst_off'] + i] = _word(lo, hi)


def _unpack_bf16(b, a):
    s, d, n = b['src'], b['dst'], a['n']
    for i in range(n):
        w = _bits(s[a['src_off'] + i // 2])
        v = struct.unpack('<f', struct.pack('<I', (w & 0xffff0000) if i & 1 else (w << 16) & 0xffffffff))[0]
        d[a['dst_off'] + i] = f32(d[a['dst_off'] + i] + v) if a['acc'] else v


_TILE = {'rows': 1, 'cols': 1, 'src_off': 0, 'dst_off': 0, 'src_ld': 1, 'dst_ld': 1}
_RO = ref_ops.RefOp


def _ref():
    t = lambda a, off, ld: _tile_ext(a[off], a['rows'], a['cols'], a[ld])
    return {
        'put': _RO(_put, lambda a: {'src': t(a, 'src_off', 'src_ld'), 'dst': t(a, 'dst_off', 'dst_ld')}, _TILE),
        'put_signal': _RO(_put_signal, lambda a: {'src': t(a, 'src_off', 'src_ld'), 'dst': t(a, 'dst_off', 'dst_ld'),
                                                  'flag': a['var'] + 1}, {**_TILE, 'var': 0, 'ver': 1}),
        'signal': _RO(_signal, lambda a: {'flag': a['var'] + 1}, {'var': 0, 'ver': 1}),
        'reduce2': _RO(_reduce2, lambda a: {'a': a['a_off'] + a['n'], 'b': a['b_off'] + a['n'], 'out': a['o_off'] + a['n']},
                       {'n': 1, 'a_off': 0, 'b_off': 0, 'o_off': 0}),
        'reduce_put_signal': _RO(_reduce_put_signal, lambda a: {'a': a['a_off'] + a['n'], 'b': a['b_off'] + a['n'],
                                                                'dst': a['d_off'] + a['n'], 'flag': a['var'] + 1},
                                 {'n': 1, 'a_off': 0, 'b_off': 0, 'd_off': 0, 'var': 0, 'ver': 1}),
        'send_proxy': _RO(_send_proxy, lambda a: {'src': _tile_ext(a['src_off'], a['rows'], a['cols'], a['src_ld'])},
                          {'rows': 1, 'cols': 1, 'src_off': 0, 'src_ld': 1, 'dst_rank': 0, 'dst_buf': 0, 'dst_off': 0,
                           'dst_ld': 1, 'var': -1, 'ver': 0}),
        'pack_bf16': _RO(_pack_bf16, lambda a: {'src': a['src_off'] + a['n'], 'dst': a['dst_off'] + (a['n'] + 1) // 2},
                         {'n': 1, 'src_off': 0, 'dst_off': 0}),
        'unpack_bf16': _RO(_unpack_bf16, lambda a: {'src': a['src_off'] + (a['n'] + 1) // 2, 'dst': a['dst_off'] + a['n']},
                           {'n': 1, 'src_off': 0, 'dst_off': 0, 'acc': 0}),
    }


def install():
    """Idempotent: replaces the opaque placeholders in ref_ops.OPS by the real comm-op semantics (extents, tile copies, signals)."""
    global _installed
    if _installed:
        return
    merged_ops()
    ref_ops.OPS.update(_ref())
    _installed = True
