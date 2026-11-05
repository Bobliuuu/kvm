"""Small workloads used by the partitioner tests (integer-valued so partitioned results can be compared bitwise)."""
import random


def _vals(rng, n, lo=-4, hi=4):
    return [float(rng.randint(lo, hi)) for _ in range(n)]


def mlp(P=4, m=8, d=8, h=16, o=8, seed=1, steps=1, lr=0.0):
    """y = relu(x@W1)@W2 ; loss-free (the check is on activations + an SGD-style update with integer lr)."""
    rng = random.Random(seed)
    B = lambda n, r, c, v=None: dict(name=n, rows=r, cols=c, init=dict(kind='values', values=v) if v else dict(kind='fill', value=0))
    bufs = [B('x', m, d, _vals(rng, m * d)), B('W1', d, h, _vals(rng, d * h)), B('W2', h, o, _vals(rng, h * o)),
            B('h', m, h), B('a', m, h), B('y', m, o), B('z', m, o), B('gW2', h, o, _vals(rng, h * o)), B('gW1', d, h, _vals(rng, d * h))]
    tasks = [dict(id='mm1', op='matmul', args=dict(a='x', b='W1', out='h', m=m, n=h, k=d)),
             dict(id='act', op='relu', args=dict(x='h', out='a', n=m * h), after=['mm1']),
             dict(id='mm2', op='matmul', args=dict(a='a', b='W2', out='y', m=m, n=o, k=h), after=['act']),
             dict(id='out', op='relu', args=dict(x='y', out='z', n=m * o), after=['mm2']),
             dict(id='upd2', op='sgd', args=dict(w='W2', g='gW2', lr=0.5, n=h * o), after=['mm2']),
             dict(id='upd1', op='sgd', args=dict(w='W1', g='gW1', lr=0.5, n=d * h), after=['mm2'])]
    return dict(buffers=bufs, graphs=[dict(name='fwd', tasks=tasks)])


def regress(n=16, seed=3):
    """The repository's toy training loop (predict -> mse_grad -> sgd) twice, over a batch of n points."""
    rng = random.Random(seed)
    x = _vals(rng, n)
    bufs = [dict(name='x', rows=n, cols=1, init=dict(kind='values', values=x)),
            dict(name='target', rows=n, cols=1, init=dict(kind='values', values=[3 * v for v in x])),
            dict(name='w', init=dict(kind='fill', value=1.0)), dict(name='pred', rows=n, cols=1),
            dict(name='loss'), dict(name='grad')]
    tasks = []
    prev = None
    for s in (1, 2):
        tasks += [dict(id=f'fwd{s}', op='predict', args=dict(x='x', w='w', out='pred', n=n), after=[prev] if prev else []),
                  dict(id=f'bwd{s}', op='mse_grad', args=dict(pred='pred', target='target', x='x', loss='loss', grad='grad', n=n), after=[f'fwd{s}']),
                  dict(id=f'upd{s}', op='sgd', args=dict(w='w', g='grad', lr=0.0625, n=1), after=[f'bwd{s}'])]
        prev = f'upd{s}'
    return dict(buffers=bufs, graphs=[dict(name='train', tasks=tasks)])


def chain(L=4, m=8, d=8, seed=5, fill=False):
    """L layers of relu(h @ W): the shape of a pipelined forward pass. Weights in {-1,0,1} keep integers exact."""
    rng = random.Random(seed)
    init = (lambda n, lo, hi: dict(kind='fill', value=1)) if fill else (lambda n, lo, hi: dict(kind='values', values=_vals(rng, n, lo, hi)))
    bufs = [dict(name='x', rows=m, cols=d, init=init(m * d, 0, 3))]
    tasks = []
    prev = 'x'
    for l in range(L):
        bufs += [dict(name=f'W{l}', rows=d, cols=d, init=init(d * d, -1, 1)),
                 dict(name=f'p{l}', rows=m, cols=d), dict(name=f'h{l}', rows=m, cols=d)]
        tasks += [dict(id=f'mm{l}', op='matmul', args=dict(a=prev, b=f'W{l}', out=f'p{l}', m=m, n=d, k=d), after=[f'act{l-1}'] if l else []),
                  dict(id=f'act{l}', op='relu', args=dict(x=f'p{l}', out=f'h{l}', n=m * d), after=[f'mm{l}'])]
        prev = f'h{l}'
    return dict(buffers=bufs, graphs=[dict(name='fwd', tasks=tasks)]), [f'h{l}' for l in range(L - 1)] + ['x', f'p{L-1}'] + [f'p{l}' for l in range(L - 1)]
