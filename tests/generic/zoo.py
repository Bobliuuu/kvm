"""Models the repo never shipped, built only from public frontend pieces (generic.py + nn.py + Builder ops). Shared by the CPU tests
(tests/frontend/test_generic.py) and the A100 golden diff (tests/generic/gen_bundle.py, run.sh).

Every dimension is deliberately off the beaten path: widths that are not tile multiples (13, 21, 37), a class count of 5, a conv
over 12-row sequences, an encoder with 2 heads of dim 16. None of them is expressible as a `ModelConfig`.
"""
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'frontend'))
from kvmfe.generic import (MLP, MSE, ArrayData, Conv1D, CrossEntropy, GeneratedData, PolicyGradient, QLearning, SequencePool,  # noqa: E402
                           one_hot)
from kvmfe.nn import Linear, Module, Norm  # noqa: E402
from kvmfe.optim import OptimConfig  # noqa: E402


@dataclass
class Case:
    name: str
    module: object
    loss: object
    data: object
    optimizer: OptimConfig
    steps: int
    note: str


def _gauss(rng, n, d, scale=1.0):
    return [[scale * rng.gauss(0, 1) for _ in range(d)] for _ in range(n)]


class CNN1D(Module):
    """conv(3 taps) -> relu -> conv(5 taps) -> silu -> mean pool over each sequence -> linear."""

    def __init__(self, cin, length, classes):
        super().__init__('cnn')
        self.c1, self.c2 = Conv1D('cnn.c1', cin, 10, 3, length), Conv1D('cnn.c2', 10, 8, 5, length)
        self.pool, self.head = SequencePool('cnn.pool', length), Linear('cnn.head', 8, classes, True, 0.3)
        self.absorb(self.c1, self.c2, self.head)

    def __call__(self, b, x, w):
        h = b.act(self.c1(b, x, w), 'relu')
        h = b.act(self.c2(b, h, w), 'silu')
        return self.head(b, self.pool(b, h, w), w)


class Encoder(Module):
    """Bidirectional transformer encoder regressor: embed -> [norm, MHA, norm, GELU MLP with residuals] -> norm -> mean pool -> linear."""

    def __init__(self, fin, d, heads, seq, out):
        super().__init__('enc')
        self.d, self.heads, self.seq = d, heads, seq
        self.embed = Linear('enc.embed', fin, d, True, 0.3)
        self.n1, self.n2, self.nf = Norm('enc.n1', d, 'layernorm'), Norm('enc.n2', d, 'layernorm'), Norm('enc.nf', d, 'layernorm')
        self.q, self.k, self.v, self.o = (Linear(f'enc.{n}', d, d, True, 0.2) for n in 'qkvo')
        self.fc1, self.fc2 = Linear('enc.fc1', d, 2 * d, True, 0.2), Linear('enc.fc2', 2 * d, d, True, 0.2)
        self.pool, self.head = SequencePool('enc.pool', seq), Linear('enc.head', d, out, True, 0.3)
        self.absorb(self.embed, self.n1, self.n2, self.nf, self.q, self.k, self.v, self.o, self.fc1, self.fc2, self.head)

    def __call__(self, b, x, w):
        x = self.embed(b, x, w)
        h = self.n1(b, x, w)
        a = b.attention(self.q(b, h, w), self.k(b, h, w), self.v(b, h, w), self.heads, self.seq, causal=False)
        x = self.o(b, a, w, res=x)
        x = self.fc2(b, self.fc1(b, self.n2(b, x, w), w, gelu=True), w, res=x)
        return self.head(b, self.pool(b, self.nf(b, x, w), w), w)


def reg_mlp():
    rng = random.Random(11)
    x = _gauss(rng, 18, 11)
    y = [[math.tanh(sum(r[:4])), sum(r[4:8]) * 0.3, r[8] * r[9]] for r in x]
    return Case('reg_mlp', MLP('reg', [11, 24, 13, 3], act='tanh'), MSE(), ArrayData(6, {'x': x, 'y': y}),
                OptimConfig(kind='sgd', lr=0.05), 3, 'MLP regression, MSE, SGD, widths 11-24-13-3')


def res_rms_mlp():
    rng = random.Random(12)
    x = _gauss(rng, 25, 20)
    y = [int(abs(sum(r[:5]) * 2)) % 5 for r in x]
    return Case('res_rms_mlp', MLP('rm', [20, 24, 24, 24, 5], act='gelu', norm='rmsnorm', residual=True), CrossEntropy(),
                ArrayData(5, {'x': x, 'y': y}, dtypes={'y': 'i32'}), OptimConfig(kind='adamw', lr=3e-3), 3,
                'residual MLP + RMSNorm, softmax-CE over 5 classes, AdamW')


def cnn1d():
    rng, length = random.Random(13), 12
    x = _gauss(rng, 16 * length, 3)
    y = [int(sum(sum(r) for r in x[s * length:(s + 1) * length]) > 0) + 2 * int(x[s * length][0] > 0) for s in range(16)]
    return Case('cnn1d', CNN1D(3, length, 4), CrossEntropy(),
                _seq_data(x, y, length, 4),
                OptimConfig(kind='sgd', lr=0.1), 3, 'conv(3)-relu-conv(5)-silu-mean-pool-linear over 12-row sequences, CE, SGD')


def _seq_data(x, y, length, batch):
    """x rows are grouped in sequences of `length`; a microbatch is `batch` whole sequences."""
    n = len(y)

    def make(m):
        seqs = [(m * batch + i) % n for i in range(batch)]
        return {'x': [x[s * length + t] for s in seqs for t in range(length)], 'y': [[y[s]] for s in seqs]}
    return GeneratedData(make, {'x': (batch * length, 3, 'f32'), 'y': (batch, 1, 'i32')})


def encoder_reg():
    rng, seq = random.Random(14), 32
    x = _gauss(rng, 6 * seq, 6)
    y = [[sum(r[0] for r in x[s * seq:(s + 1) * seq]) / seq, x[s * seq][1], 0.5] for s in range(6)]
    batch = 2

    def make(m):
        seqs = [(m * batch + i) % 6 for i in range(batch)]
        return {'x': [x[s * seq + t] for s in seqs for t in range(seq)], 'y': [y[s] for s in seqs]}
    return Case('encoder_reg', Encoder(6, 32, 2, seq, 3), MSE(), GeneratedData(make, {'x': (batch * seq, 6, 'f32'), 'y': (batch, 3, 'f32')}),
                OptimConfig(kind='adamw', lr=1e-3), 2, 'bidirectional transformer encoder (2 heads, hd 16, seq 32), MSE, AdamW + clip')


class TokMLP(Module):
    """The audit's mlp_ce / resmlp_rms / bad_dims: token embedding -> [linear+GELU (+ norm)] * depth -> linear; no attention anywhere."""

    def __init__(self, name, vocab, d, hidden, depth, norm=None):
        super().__init__(name)
        self.wte = self.add('wte', vocab, d, ('normal', 0.1))
        dims = [d] + [hidden] * depth
        self.fcs = [Linear(f'{name}.fc{i}', dims[i], dims[i + 1], True, 0.2) for i in range(depth)]
        self.norms = [Norm(f'{name}.n{i}', dims[i + 1], norm) for i in range(depth)] if norm else []
        self.head = Linear(f'{name}.head', hidden, vocab, False, 0.2)
        self.absorb(*self.fcs, *self.norms, self.head)

    def __call__(self, b, ids, w):
        x = b.embedding(ids, w[self.wte], ids.rows)
        for i, fc in enumerate(self.fcs):
            x = fc(b, x, w, gelu=True)
            if self.norms:
                x = self.norms[i](b, x, w)
        return self.head(b, x, w)


def _tok_case(name, hidden, depth, norm, note, tokens=96, vocab=100, d=48, steps=3):
    ids = [(t * 7 + 3) % vocab for t in range(vocab)]
    nxt = [(5 * t + 3) % vocab for t in ids]
    return Case(name, TokMLP('mlp', vocab, d, hidden, depth, norm), CrossEntropy(),
                ArrayData(tokens, {'x': [[v] for v in ids], 'y': [[v] for v in nxt]}, dtypes={'x': 'i32', 'y': 'i32'}),
                OptimConfig(kind='sgd', lr=0.05), steps, note)


def mlp_ce():
    return _tok_case('mlp_ce', 72, 2, None, 'audit: embedding -> 2x(linear+GELU) -> linear -> CE, no attention/norm/rope')


def resmlp_rms():
    return _tok_case('resmlp_rms', 72, 3, 'rmsnorm', 'audit: 3x(linear+GELU+rmsnorm) -> CE')


def bad_dims():
    return _tok_case('bad_dims', 7, 1, None, 'audit: MLP with hidden width 7 (not a tile multiple)')


def pg_policy():
    """Contextual bandit from logged data: state 7 -> softmax over 5 actions; behaviour = uniform; reward 1 when the action equals a
    hidden function of the state; advantage = reward - batch-free constant baseline 0.2. REINFORCE with the logged advantage."""
    rng, n, acts = random.Random(21), 40, 5
    x = _gauss(rng, n, 7)
    best = [int(abs(r[0] * 2 + r[1])) % acts for r in x]
    a = [rng.randrange(acts) for _ in range(n)]
    adv = [[(1.0 if a[i] == best[i] else 0.0) - 0.2] for i in range(n)]
    return Case('pg_policy', MLP('pg', [7, 16, acts], act='tanh'), PolicyGradient(),
                ArrayData(8, {'x': x, 'act': [[v] for v in a], 'adv': adv}, dtypes={'act': 'i32'}, target=('act', 'adv')),
                OptimConfig(kind='sgd', lr=0.3), 3, 'REINFORCE (policy-gradient loss, advantage stream), MLP policy, logged bandit data, SGD')


def dqn_q():
    """One-step TD on logged transitions: Q(s,.) 6 -> 4 actions; target y = r + 0.9 * max_a' Q0(s', a') from a frozen linear Q0."""
    rng, n, acts = random.Random(22), 36, 4
    s0, s1 = _gauss(rng, n, 6), _gauss(rng, n, 6)
    a = [rng.randrange(acts) for _ in range(n)]
    r = [math.tanh(s0[i][a[i]]) for i in range(n)]
    q0 = lambda s: max(sum(0.3 * s[k] * (1 + j) / acts for k in range(6)) for j in range(acts))
    y = [[r[i] + 0.9 * q0(s1[i])] for i in range(n)]
    return Case('dqn_q', MLP('dqn', [6, 20, acts], act='relu'), QLearning(),
                ArrayData(6, {'x': s0, 'mask': one_hot(a, acts), 'y': y}, target=('mask', 'y')),
                OptimConfig(kind='adamw', lr=5e-3), 3, 'DQN-style TD loss (one-hot action mask, bootstrapped target), MLP Q-net, AdamW')


def wide_mlp():
    """Shapes that take the 128x128 / 256x128 GEMM tiles under the a100_big tiling (batch 256, widths 256/384): tile-flag coverage."""
    rng = random.Random(23)
    x = _gauss(rng, 256, 32)
    y = [int(abs(sum(r[:6]) * 2)) % 6 for r in x]
    return Case('wide_mlp', MLP('wd', [32, 256, 384, 6], act='gelu'), CrossEntropy(), ArrayData(256, {'x': x, 'y': y}, dtypes={'y': 'i32'}),
                OptimConfig(kind='sgd', lr=0.05), 2, 'wide MLP: GEMMs large enough for the big tile flags (32 / 64) under --tiling a100_big')


CASES = {f().name: f for f in (reg_mlp, res_rms_mlp, cnn1d, encoder_reg, mlp_ce, resmlp_rms, bad_dims, pg_policy, dqn_q, wide_mlp)}
