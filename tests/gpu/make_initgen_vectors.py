#!/usr/bin/env python3
"""Test vectors for the device `seeded` initialiser, produced by the reference runtime/initgen.py.

    make_initgen_vectors.py [out]        write tests/gpu/initgen_vectors.txt
    make_initgen_vectors.py --check      regenerate and fail if the committed file differs

Line pairs: `seed draws bias_bits coef_bits n` then n hex words (float32 bit patterns).
"""
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'runtime'))
import initgen  # noqa: E402

SEEDS = [0, 1, 42, 123456789, (1 << 63) + 5, (1 << 64) - 1]
CASES = [('uniform', 1.0, 1), ('uniform', 0.02, 300), ('normal', 1.0, 7), ('normal', 0.02, 300), ('normal', 0.5, 1025)]


def bits(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]


def render():
    out = []
    for seed in SEEDS:
        for dist, scale, n in CASES:
            draws, bias, coef = initgen.params(dist, scale)
            vals = initgen.seeded_values(seed, n, draws, bias, coef)
            out.append(f'{seed} {draws} {bias_hex(bias)} {bias_hex(coef)} {n}')
            out.append(' '.join(f'{bits(v):08x}' for v in vals))
    return '\n'.join(out) + '\n'


def bias_hex(x):
    return f'{bits(x):08x}'


if __name__ == '__main__':
    path = ROOT / 'tests' / 'gpu' / 'initgen_vectors.txt'
    text = render()
    if len(sys.argv) > 1 and sys.argv[1] == '--check':
        sys.exit(0 if path.read_text() == text else (print('initgen_vectors.txt is stale', file=sys.stderr) or 1))
    Path(sys.argv[1] if len(sys.argv) > 1 else path).write_text(text)
