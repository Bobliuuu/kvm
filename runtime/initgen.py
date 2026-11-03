"""Counter-based initial-value generator for `seeded` buffers: identical on the CPU reference, the host and the device.

Draw j of seed s is SplitMix64 evaluated at counter j (no sequential state, so any element can be produced independently):

    state = (s + (j + 1) * 0x9E3779B97F4A7C15) mod 2^64
    z = state; z = (z ^ z >> 30) * 0xBF58476D1CE4E5B9; z = (z ^ z >> 27) * 0x94D049BB133111EB; z ^= z >> 31   (mod 2^64)
    u = float(z >> 40) * 2^-24                                   (exact in float32, in [0, 1))

Element i uses `draws` consecutive draws j = i*draws .. i*draws+draws-1 and every step below is one correctly rounded
float32 operation (no fused multiply-add):

    acc   = u                              if draws == 1      (dist "uniform")
          = (u0 + u1) + (u2 + u3)          if draws == 4      (dist "normal": Irwin-Hall sum of 4, mean 2, variance 1/3)
    value = (acc - bias) * coef

with (draws, bias, coef) = (1, 0.5, f32(2*scale)) for uniform, i.e. U(-scale, scale), and (4, 2.0, f32(scale*sqrt(3))) for
normal, i.e. mean 0, standard deviation `scale`, bounded at 2*sqrt(3)*scale (not Gaussian tails). The compiler writes
draws, bias and coef into the bundle metadata so a consumer never re-derives them.
"""
import math

from ref_types import f32

M64 = (1 << 64) - 1
GOLDEN = 0x9E3779B97F4A7C15
DISTS = ('uniform', 'normal')


def params(dist: str, scale: float):
    """(draws, bias, coef) for a distribution; raises ValueError for an unknown one."""
    if dist == 'uniform':
        return 1, 0.5, f32(2.0 * scale)
    if dist == 'normal':
        return 4, 2.0, f32(scale * math.sqrt(3.0))
    raise ValueError(f'unknown dist {dist!r}; expected one of {DISTS}')


def draw(seed: int, j: int) -> int:
    z = (seed + (j + 1) * GOLDEN) & M64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return z ^ (z >> 31)


def _u(seed: int, j: int) -> float:
    return (draw(seed, j) >> 40) * (1.0 / 16777216.0)


def seeded_values(seed: int, n: int, draws: int, bias: float, coef: float) -> list:
    out = []
    for i in range(n):
        if draws == 1:
            acc = _u(seed, i)
        else:
            j = i * draws
            acc = f32(f32(_u(seed, j) + _u(seed, j + 1)) + f32(_u(seed, j + 2) + _u(seed, j + 3)))
        out.append(f32(f32(acc - bias) * coef))
    return out
