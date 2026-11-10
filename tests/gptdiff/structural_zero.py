"""Structurally-zero gradients in a golden comparison: an absolute, data-derived bound instead of a relative one.

A relative L2 bound is meaningless for a tensor whose exact value is 0: every implementation returns rounding noise, and noise against
noise is O(1) (or 1e17, for a squared moment). The category found in the generic training cases (tests/generic/zoo.py, `encoder_reg`):
the key-bias gradient of bidirectional attention. softmax(q.k + c.q) is shift invariant in k's bias, so sum_j dK_j = sum_i q_i (sum_j dS_ij) = 0
(each dS row sums to 0). The compiler lowers it to `colsum(dK)` -> bias gradient; `m`, `v` and `w` of that parameter inherit the noise.
(The T=1 attention dq/dk of tests/gpu_ops/test_ops.cu is the same idea with its own per-element bound.)

Detection is by an independent float64 evaluation (frontend refvm), never by the device's value: a `colsum` output is structurally zero
iff float64 returns |out_c| <= 1e-12 * sum_r |x_rc| for every column. The float64 run happens only if the CPU golden already shows a
colsum with |out_c| <= SCREEN * sum_r|x_rc| for every column (cheap), so ordinary workloads never pay for it.

Bound for such a gradient g (per column c, per run, from that run's OWN dK): |g_c| <= (R + 8) * 2^-23 * sum_r |dK_rc|.
R+8: R fp32 summands of the colsum plus the few-ulp error of each dK element (measured on the A100, 3xTF32: 0.42 * 2^-23 * mass, so the
bound is 90x above the noise (worst |g|/bound 0.011); golden CPU f32 is lower still). A real bug in the attention backward (dropped term, wrong scale,
a missing -delta) breaks the row-sum identity at O(|dK|) = 1e-3 here, i.e. 5e2 times over the bound (negative tests: test_structural_zero.py).
Inherited state, from the optimizer task's own parameters: |m| <= max bound (convex mix of clipped gradients, clip coefficient <= 1),
v <= max bound^2, and |w - (1 - lr*wd)^n w0| <= n * lr * U with U the Adam update cap. w is a WEAK check by construction (Adam
normalises, so a real gradient bug also moves w by about lr); the gradient bound is the one that detects bugs.
"""
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'frontend'))

SCREEN = 1e-3
F64_ZERO = 1e-12
EPS32 = 2.0 ** -23


def _cols(spec):
    return {b['name']: (b['rows'], b['cols']) for b in spec['buffers']}


def _mass(x, shape, R, c0, c1):
    """sum_r |x[r, c]| for c in [c0, c1) over the first R rows."""
    cols = shape[1]
    return [sum(abs(x[r * cols + c]) for r in range(R)) for c in range(c0, c1)]


def adam_cap(b1, b2):
    """Upper bound of |m_hat| / sqrt(v_hat) for a bias-corrected Adam update (known bound for b1^2 < b2; 1 at t=1)."""
    if b1 * b1 < b2:
        return max(1.0, (1 - b1) / math.sqrt(1 - b2) / math.sqrt(1 - b1 * b1 / b2))
    return 1.0 / (1 - b1)


def analyse(spec, gold, dtypes, reference=None):
    """Return a dict: grad name -> task info, plus 'chain' entries. Empty when the workload has no structurally-zero gradient.

    `reference` (name -> flat float64 list) is the float64 evaluation; computed here when a screening hit needs it."""
    shapes, tasks = _cols(spec), spec['graphs'][0]['tasks']
    cands = {}
    for t in tasks:
        a = t.get('args', {})
        if t['op'] != 'colsum' or a.get('acc', 0) != 0 or dtypes.get(a['out']) != 'f32':
            continue
        if a['R'] < 2:
            continue   # a single-row colsum is a copy
        mass = _mass(gold[a['x']], shapes[a['x']], a['R'], a['c0'], a['c1'])
        out = gold[a['out']][a['c0']:a['c1']]
        if all(abs(o) <= SCREEN * m for o, m in zip(out, mass)):
            cands[a['out']] = a
    if not cands:
        return {}
    if reference is None:
        from kvmfe import refvm
        mem = refvm.execute(spec, f32=False)
        reference = {n: [x for r in range(b.rows) for x in b.row(r)] for n, b in mem.bufs.items()}
    zero = {}
    for out, a in cands.items():
        mass = _mass(reference[a['x']], shapes[a['x']], a['R'], a['c0'], a['c1'])
        if all(abs(o) <= F64_ZERO * m for o, m in zip(reference[out][a['c0']:a['c1']], mass)):
            zero[out] = a
    steps = sum(1 for t in tasks if t['op'] == 'counter_inc' and t['args']['x'] == 'opt_step')
    chain = {}      # parameter -> dict(g=[grad names], m, v, wd, lr, b1, b2, e0, e1)
    for t in tasks:
        a = t.get('args', {})
        if t['op'] == 'adamw' and a['g'] in zero:
            d = chain.setdefault(a['p'], dict(g=[], m=a['m'], v=a['v'], wd=a['wd'], lr=a['lr'], b1=a['b1'], b2=a['b2']))
            d['g'].append(a['g'])
    return dict(zero=zero, chain=chain, steps=steps, shapes=shapes)


def names(sz):
    """Every buffer name excluded from the relative bound."""
    if not sz:
        return set()
    out = set(sz['zero'])
    for p, d in sz['chain'].items():
        out |= {p, d['m'], d['v']}
    return out


def check(sz, buf, w0):
    """Absolute checks of one run (`buf`: name -> flat values). Returns [(ok, what, detail)]. `w0`: name -> initial values."""
    res, bound = [], {}
    for out, a in sz['zero'].items():
        mass = _mass(buf[a['x']], sz['shapes'][a['x']], a['R'], a['c0'], a['c1'])
        bd = [(a['R'] + 8) * EPS32 * m for m in mass]
        bound[out] = max(bd)
        ratio = max((abs(o) / b if b > 0 else (0 if o == 0 else math.inf)) for o, b in zip(buf[out][a['c0']:a['c1']], bd))
        res.append((ratio <= 1, f'{out} (sum of {a["x"]}, structurally 0) within (R+8)*2^-23*L1', f'worst |g|/bound {ratio:.3f}, max|g| {max(abs(o) for o in buf[out]):.2e}'))
    for p, d in sz['chain'].items():
        B = max(bound[g] for g in d['g'])
        mm, vv = max(abs(x) for x in buf[d['m']]), max(abs(x) for x in buf[d['v']])
        res.append((mm <= B, f'{d["m"]} <= max gradient bound', f'{mm:.2e} vs {B:.2e}'))
        res.append((vv <= B * B, f'{d["v"]} <= max gradient bound^2', f'{vv:.2e} vs {B * B:.2e}'))
        n, cap = sz['steps'], adam_cap(d['b1'], d['b2'])
        dev = max(abs(w - (1 - d['lr'] * d['wd']) ** n * x0) for w, x0 in zip(buf[p], w0[p]))
        res.append((dev <= n * d['lr'] * cap * 1.0001, f'{p} within {n} Adam steps of its start (weak check)', f'{dev:.2e} vs {n * d["lr"] * cap:.2e}'))
    return res
