#!/usr/bin/env python3
"""Scale planner: parameters / memory / FLOPs / communication of a sharded training step, emitted as KVM task graphs, with exact
accounting that is checked against the real partitioner on CPU.

Two layers, kept apart on purpose:

  (A) the STACK (this file, `Stack`): a deep linear network trained with AdamW, small enough to be partitioned for real by
      runtime/partition.py (ZeRO/FSDP, data parallel, Megatron-style tensor parallel; one mesh axis) and executed.
      Its backward needs transposed operands and the registry has no transpose op, so every parameter W_l is stored twice
      (W_l and its transpose WT_l, both trained; see `build_stack`). There is no attention and no nonlinearity: the partitioner
      has no rule for them. `LayoutCounter` predicts, without building any task, the bytes sent, the matmul/optimizer FLOPs
      and the buffer bytes per device; `measure()` counts the same quantities from the tasks and buffers the partitioner
      actually emitted. tests/dist/test_scaleplan.py requires them to be EQUAL (integers) for every small configuration.
  (B) the TRANSFORMER accounting (`transformer`, below): closed-form bytes per GPU / communication volume / FLOPs / pipeline
      critical path for a dense GPT-style model split by tensor x pipeline x data x ZeRO, e.g. one trillion parameters on N GPUs.
      It is a MODEL. Its building blocks (ZeRO gather / reduce-scatter volume, tensor-parallel all-reduce volume) are the ones
      validated in (A); everything about 3-D composition, attention and 1F1B is arithmetic, not measurement.

    python3 dist/scaleplan.py transformer --params 1e12 --gpus 4096 --tp 8 --pp 16 [--zero 3] [--json]
    python3 dist/scaleplan.py check            # golden comparison on CPU (also run by tests/dist/test_scaleplan.py)
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'runtime'), str(ROOT / 'dist'), str(ROOT / 'tests' / 'dist')]
import partition as pt          # noqa: E402
import initgen                  # noqa: E402

R, ROWS, COLS, PART = pt.R, pt.ROWS, pt.COLS, pt.PART


# ====================================================================================================== (A) the stack
@dataclass
class Stack:
    dims: list                   # widths d0..dL: layer l is dims[l-1] x dims[l]
    m: int                       # tokens (rows of the batch)
    steps: int = 1
    lr: float = 1e-3
    eps: float = 1e-6
    wd: float = 0.0
    strategy: str = 'zero'       # 'zero' (params/optimizer ROWS, batch ROWS = ZeRO + DP) | 'tp' (Megatron pairs) | 'single'

    @property
    def L(self):
        return len(self.dims) - 1

    def params(self):            # logical parameters (each stored twice: W and WT)
        return sum(self.dims[i] * self.dims[i + 1] for i in range(self.L))


def build_stack(st: Stack, mode='values', seed=1):
    """-> (spec, layouts, gen). mode 'values': explicit small-integer values (CPU golden, integers keep the first step exact);
    'seeded': nothing materialised, `gen` describes the seeded blocks for the symbolic partitioner / GPU harness."""
    rng = random.Random(seed)
    L, m, d = st.L, st.m, st.dims
    bufs, gen, order = {}, {}, []

    def B(name, r, c, init=None, dtype='f32', g=None):
        bufs[name] = dict(name=name, rows=r, cols=c, dtype=dtype)
        if init is not None:
            bufs[name]['init'] = init
        if g is not None:
            gen[name] = g
        order.append(name)

    ints = lambda n, lo, hi: [float(rng.randint(lo, hi)) for _ in range(n)]
    seq = [0]

    def seeded(r, c, scale, transpose_of=None):
        seq[0] += 1
        s = seed * 1000003 + (transpose_of if transpose_of is not None else seq[0])
        dr, bi, co = initgen.params('uniform', scale)
        return s, dict(seed=s, draws=dr, bias=bi, coef=co, gr=r, gc=c, transpose=transpose_of is not None)

    def pair(name, r, c, scale, lo=-1, hi=1):
        """A matrix and its transpose, consistent."""
        if mode == 'values':
            v = ints(r * c, lo, hi)
            tv = [v[i * c + j] for j in range(c) for i in range(r)]
            B(name, r, c, dict(kind='values', values=v))
            B(name + 'T', c, r, dict(kind='values', values=tv))
        else:
            s, g = seeded(r, c, scale)
            B(name, r, c, dict(kind='fill', value=0), g=g)
            _, gt = seeded(c, r, scale, transpose_of=s)
            gt['seed'] = s
            gt['gr'], gt['gc'] = c, r
            B(name + 'T', c, r, dict(kind='fill', value=0), g=gt)

    pair('X', m, d[0], 1.0, 0, 2)
    # X is (m x d0) and XT its transpose; negT/negTT: minus the target
    pair('nT', m, d[L], 1.0, -2, 2)
    B('Z', m, d[L])
    for l in range(1, L + 1):
        pair(f'W{l}', d[l - 1], d[l], 1.0 / math.sqrt(d[l - 1]))
        for nm, r, c in ((f'mW{l}', d[l - 1], d[l]), (f'vW{l}', d[l - 1], d[l]), (f'mW{l}T', d[l], d[l - 1]), (f'vW{l}T', d[l], d[l - 1]),
                         (f'gW{l}', d[l - 1], d[l]), (f'gW{l}T', d[l], d[l - 1]),
                         (f'H{l}', m, d[l]), (f'H{l}T', d[l], m), (f'dH{l}', m, d[l]), (f'dH{l}T', d[l], m)):
            B(nm, r, c)
    for t in range(1, st.steps + 1):
        B(f'step{t}', 1, 1, dict(kind='fill', value=t), dtype='i32')
        B(f'loss{t}', 1, 1)
        B(f'gl{t}', 1, 1)

    tasks = []
    last_w, readers = {}, {}

    def task(op, args, reads, writes):
        after = set()
        for b in reads:
            if b in last_w:
                after.add(last_w[b])
        for b in writes:
            if b in last_w:
                after.add(last_w[b])
            after |= readers.get(b, set())
        tid = f't{len(tasks)}'
        after.discard(tid)
        tasks.append(dict(id=tid, op=op, args=args, after=sorted(after, key=lambda x: int(x[1:]))))
        for b in reads:
            readers.setdefault(b, set()).add(tid)
        for b in writes:
            last_w[b] = tid
            readers[b] = set()

    def mm(a, b, o, m_, n_, k_):
        task('matmul', dict(a=a, b=b, out=o, m=m_, n=n_, k=k_), [a, b], [o])

    for t in range(1, st.steps + 1):
        for l in range(1, L + 1):
            mm('X' if l == 1 else f'H{l-1}', f'W{l}', f'H{l}', m, d[l], d[l - 1])
            mm(f'W{l}T', 'XT' if l == 1 else f'H{l-1}T', f'H{l}T', d[l], m, d[l - 1])
        task('add', dict(a=f'H{L}', b='nT', out=f'dH{L}', n=m * d[L]), [f'H{L}', 'nT'], [f'dH{L}'])
        task('add', dict(a=f'H{L}T', b='nTT', out=f'dH{L}T', n=m * d[L]), [f'H{L}T', 'nTT'], [f'dH{L}T'])
        task('mse_grad', dict(pred=f'dH{L}', target='Z', x=f'dH{L}', loss=f'loss{t}', grad=f'gl{t}', n=m * d[L]),
             [f'dH{L}', 'Z'], [f'loss{t}', f'gl{t}'])
        for l in range(L, 0, -1):
            mm('XT' if l == 1 else f'H{l-1}T', f'dH{l}', f'gW{l}', d[l - 1], d[l], m)
            mm(f'dH{l}T', 'X' if l == 1 else f'H{l-1}', f'gW{l}T', d[l], d[l - 1], m)
            if l > 1:
                mm(f'dH{l}', f'W{l}T', f'dH{l-1}', m, d[l - 1], d[l])
                mm(f'W{l}', f'dH{l}T', f'dH{l-1}T', d[l - 1], m, d[l])
            for w, g in ((f'W{l}', f'gW{l}'), (f'W{l}T', f'gW{l}T')):
                n = bufs[w]['rows'] * bufs[w]['cols']
                task('adamw', dict(p=w, g=g, m='m' + w, v='v' + w, gscale=None, step=f'step{t}', e0=0, e1=n, lr=st.lr, b1=0.9, b2=0.999,
                                   eps=st.eps, wd=st.wd), [g, f'step{t}'], [w, 'm' + w, 'v' + w])
    # m*/v* names: mW1 / vW1 / mW1T / vW1T  (prefix + parameter name)
    spec = dict(buffers=[bufs[n] for n in order], graphs=[dict(name='train', tasks=tasks)])
    if mode == 'values':
        pass
    return spec, layouts_for(st), gen


def layouts_for(st: Stack):
    """strategy.layouts for the declared (leaf) buffers. Intermediates take whatever layout the rules give them."""
    L, lay = st.L, {}
    if st.strategy == 'single':
        return lay
    for l in range(1, L + 1):
        if st.strategy == 'zero':
            ws = {f'W{l}': 'rows', f'W{l}T': 'rows'}
        else:   # tp: odd layers column-parallel (W COLS, WT ROWS), even layers row-parallel
            ws = {f'W{l}': 'cols', f'W{l}T': 'rows'} if l % 2 else {f'W{l}': 'rows', f'W{l}T': 'cols'}
        for k, v in ws.items():
            lay[k] = v
            lay['m' + k] = v
            lay['v' + k] = v
    if st.strategy == 'zero':
        lay.update(X='rows', XT='cols', nT='rows', nTT='cols', Z='rows')
    return lay


def strategy_for(st: Stack, collective='direct', chunks=1, symbolic=False, gen=None, dump=()):
    s = {'mode': 'shard', 'layouts': layouts_for(st), 'collective': collective, 'chunks': chunks}
    if symbolic:
        s.update(symbolic=True, gen=gen or {}, dump=list(dump))
    return s


# ---------------------------------------------------------------------------------------------------- measured from the plan
def measure(plan):
    """Quantities counted from what the partitioner emitted (tasks and buffer tables), per device."""
    out = dict(comm_bytes=0, comm_msgs=0, flops=[], local_bytes=[], temp_bytes=[], persistent_bytes=[], tasks=[], by_op=[])
    logical = set(plan.layout)
    for s in plan.shards:
        fl, ops = 0, {}
        for t in s.spec['graphs'][0]['tasks']:
            a, op = t['args'], t['op']
            ops[op] = ops.get(op, 0) + 1
            if op == 'matmul':
                fl += 2 * a['m'] * a['n'] * a['k']
            elif op == 'adamw':
                fl += ADAMW_FLOPS * (a['e1'] - a['e0'])
            elif op in ('put_signal', 'send_proxy'):
                out['comm_bytes'] += a['rows'] * a['cols'] * 4
                out['comm_msgs'] += 1
            elif op == 'reduce_put_signal':
                out['comm_bytes'] += a['n'] * 4
                out['comm_msgs'] += 1
        loc = persistent = 0
        for b in s.spec['buffers']:
            if b['name'] in s.remotes:
                continue
            nb = b['rows'] * b['cols'] * 4
            loc += nb
            if b['name'] in logical:
                persistent += nb
        out['flops'].append(fl)
        out['local_bytes'].append(loc)
        out['persistent_bytes'].append(persistent)
        out['temp_bytes'].append(loc - persistent)
        out['tasks'].append(len(s.spec['graphs'][0]['tasks']))
        out['by_op'].append(ops)
    return out


ADAMW_FLOPS = 20          # per element, counted the same way in the counter and in `measure` (a convention, not a hardware count)


# ---------------------------------------------------------------------------------------------------- predicted without tasks
class LayoutCounter:
    """The partitioner's layout algebra (ShardPlanner: operand layouts, cheapest-resharding choice, per-write-epoch cache of
    resharded views) executed on counts only: O(ops), independent of P, no tasks, no values. Collective volumes are the
    closed forms of runtime/partition.py's collectives (documented at each use); the test requires equality with the emitted
    tasks, so a wrong formula or a wrong rule shows up as a failing check, not as a plausible number."""

    def __init__(self, spec, P, strategy):
        self.P, self.algo, self.k = P, strategy.get('collective', 'direct'), strategy.get('chunks', 1)
        self.shape = {b['name']: (b['rows'], b['cols']) for b in spec['buffers']}
        self.init = {k: pt.LAYOUT_NAMES[v] for k, v in strategy.get('layouts', {}).items()}
        self.layout, self.cache, self.epoch = {}, {}, {n: 0 for n in self.shape}
        self.comm = 0              # elements sent, all devices together
        self.flops = 0             # per device
        self.temp = 0              # elements per device
        self.tasks = []
        for t in spec['graphs'][0]['tasks']:
            self.plan(t)

    def n(self, name):
        r, c = self.shape[name]
        return r * c

    def scalar(self, name):
        return self.shape[name] == (1, 1)

    def phys(self, name, lay):
        r, c = self.shape[name]
        if lay == ROWS:
            return r // self.P, c
        if lay == COLS:
            return r, c // self.P
        return r, c

    def decide(self, name, lay):
        if name not in self.layout:
            self.layout[name] = R if (self.scalar(name) and lay in (ROWS, COLS)) else lay

    def ensure(self, name):
        self.decide(name, self.init.get(name, R))
        return self.layout[name]

    def cost(self, name, frm, to):
        if frm == to or (frm == R and to in (ROWS, COLS)):
            return 0
        n, P = self.n(name) * 4, self.P
        if to == R:
            return n * (P - 1) / P if frm in (ROWS, COLS) else 2 * n * (P - 1) / P
        if frm == PART:
            return n * (P - 1) / P if to == ROWS else 2 * n * (P - 1) / P
        return n * (P - 1) / P / P

    def reshard(self, name, frm, to):
        n, P = self.n(name), self.P
        if frm == R:                                          # slice_local: a fresh n/P buffer on each device
            self.temp += n // P
        elif to == R and frm == PART:
            self.all_reduce(n)
        elif to == R:                                         # all_gather: fresh full buffer per device; each block sent to P-1 peers
            self.comm += (P - 1) * n
            self.temp += n
        elif frm == PART and to == ROWS:
            self.reduce_scatter(n)
        elif frm == PART:                                     # PART -> COLS: all-reduce then local slice
            self.all_reduce(n)
            self.temp += n // P
        else:                                                 # all_to_all: P(P-1) tiles of n/P^2; fresh n/P buffer
            self.comm += (P - 1) * n // P
            self.temp += n // P

    def reduce_scatter(self, n):
        """ring (always): P-1 steps x P devices x n/P sent; per device (P-1) landing chunks + final chunk + the ROWS output."""
        P = self.P
        self.comm += (P - 1) * n
        self.temp += (P - 1) * (n // P) + (n // P) + n // P

    def all_reduce(self, n):
        P = self.P
        if self.algo == 'direct':     # every device sends the whole buffer to every other; (P-1) landings, (P-2) partial sums, result
            self.comm += P * (P - 1) * n
            self.temp += (P - 1) * n + max(P - 2, 0) * n + n
        else:                         # ring: reduce-scatter + ring all-gather
            self.comm += 2 * (P - 1) * n
            self.temp += (P - 1) * (n // P) + (n // P) + n

    def view(self, name, lay):
        cur = self.ensure(name)
        if self.scalar(name) and lay in (ROWS, COLS):
            lay = R
        if cur == lay:
            return
        key = (name, lay)
        if key not in self.cache or self.cache[key] != self.epoch[name]:
            self.reshard(name, cur, lay)
            self.cache[key] = self.epoch[name]

    def plan(self, t):
        op, a, P = t['op'], t['args'], self.P
        if op == 'matmul':
            A, B_, O = a['a'], a['b'], a['out']
            m, n, k = a['m'], a['n'], a['k']
            la0, lb0 = self.ensure(A), self.ensure(B_)
            want = self.layout.get(O) or self.init.get(O)
            best = None
            for (la, lb), lo in pt.ShardPlanner.MM_RULES:
                if (lo == ROWS and m % P) or (lo == COLS and n % P) or (lo == PART and k % P):
                    continue
                c = self.cost(A, la0, la) + self.cost(B_, lb0, lb) + (self.cost(O, lo, want) if want and want != lo else 0)
                if best is None or c < best[0]:
                    best = (c, la, lb, lo)
            _, la, lb, lo = best
            self.view(A, la)
            self.view(B_, lb)
            assert not (want and want != lo), 'declared matmul output layouts are not used by the stack'
            self.decide(O, lo)
            lm, ln, lk = (m // P if lo == ROWS else m), (n // P if lo == COLS else n), (k // P if lo == PART else k)
            self.flops += 2 * lm * ln * lk
            self.epoch[O] += 1
        elif op == 'add':
            ins, outs = [a['a'], a['b']], [a['out']]
            if outs[0] in self.layout or outs[0] in self.init:       # a decided output fixes the layout (ShardPlanner.common_layout)
                T = self.ensure(outs[0])
            else:
                lays = [self.ensure(i) for i in ins if not self.scalar(i)]
                non = [x for x in lays if x != PART]
                T = non[0] if non else PART
            ref = ins[0]
            nloc = self.n(ref) // P if T in (ROWS, COLS) else self.n(ref)
            for i in ins:
                self.view(i, T)
            self.decide(outs[0], T)
            self.epoch[outs[0]] += 1
        elif op == 'mse_grad':
            T = self.ensure(a['pred'])
            T = R if T == PART else T
            for k_ in ('pred', 'target', 'x'):
                self.view(a[k_], T)
            res = R if T == R else PART
            for nm in (a['loss'], a['grad']):
                self.decide(nm, res)
                self.epoch[nm] += 1
            if T != R:
                self.temp += 2                                  # tl, tg: one scalar each
        elif op == 'adamw':
            p = a['p']
            T = self.ensure(p)
            self.ensure(a['m']), self.ensure(a['v'])
            self.view(a['g'], T)
            self.view(a['step'], R)
            full = self.n(p)
            self.flops += ADAMW_FLOPS * (full // P if T in (ROWS, COLS) else full)
            for o in (p, a['m'], a['v']):
                self.epoch[o] += 1
        else:
            raise ValueError(op)

    def result(self):
        P = self.P
        persistent = 0
        for nm, lay in self.layout.items():
            r, c = self.phys(nm, lay)
            persistent += r * c
        return dict(comm_bytes=4 * self.comm, flops=self.flops, persistent_bytes=4 * persistent, temp_bytes=4 * self.temp,
                    local_bytes=4 * (persistent + self.temp))


def check_stack(st: Stack, P, collective='direct', chunks=1, seeds=(0,), exact=True, rel=1e-5):
    """Partition for real, count what was emitted, compare with the predictor and (CPU simulator) with the unpartitioned VM.
    Raises AssertionError on the first difference. Returns a report dict."""
    import verify
    spec, _, _ = build_stack(st, 'values')
    strat = strategy_for(st, collective, chunks)
    cl = pt.Cluster.uniform(P, 1)
    plan = pt.partition(copy.deepcopy(spec), cl, strat)
    meas = measure(plan)
    pred = LayoutCounter(spec, P, strat).result()
    rep = dict(P=P, strategy=st.strategy, collective=collective, tasks=meas['tasks'][0], measured=dict(
        comm_bytes=meas['comm_bytes'], flops=meas['flops'][0], local_bytes=meas['local_bytes'][0], persistent_bytes=meas['persistent_bytes'][0],
        temp_bytes=meas['temp_bytes'][0]), predicted=pred)
    for key in ('comm_bytes',):
        assert meas[key] == pred[key], f'{key}: emitted {meas[key]} predicted {pred[key]} ({st}, P={P}, {collective})'
    for dev in range(P):
        assert meas['flops'][dev] == pred['flops'], f'flops dev {dev}: emitted {meas["flops"][dev]} predicted {pred["flops"]}'
        assert meas['persistent_bytes'][dev] == pred['persistent_bytes'], \
            f'persistent bytes dev {dev}: emitted {meas["persistent_bytes"][dev]} predicted {pred["persistent_bytes"]}'
        assert meas['local_bytes'][dev] == pred['local_bytes'], \
            f'local bytes dev {dev}: emitted {meas["local_bytes"][dev]} predicted {pred["local_bytes"]}'
    import sim
    import workload_vm
    import compile_workload as cw
    gold = workload_vm.run_golden(cw.compile_spec(copy.deepcopy(plan.spec)))['buffers']
    worst = 0.0
    for seed in seeds:
        results, _ = sim.run(plan, seed)
        got = plan.reconstruct(results)
        for name in got:
            if name not in gold:
                continue
            for i, (g, w) in enumerate(zip(got[name], gold[name])):
                if g != w:
                    e = abs(g - w) / max(1.0, abs(w))
                    worst = max(worst, e)
                    if exact or e > rel:
                        raise AssertionError(f'seed {seed}: {name}[{i}] = {g!r}, unpartitioned {w!r}')
    rep['worst_rel_vs_unpartitioned'] = worst
    rep['sim_makespan_us_model'] = pt.estimate(plan)['makespan_us']
    return rep


# ====================================================================================================== (B) transformer
@dataclass
class Model:
    layers: int = 128
    d: int = 25600
    ffn_mult: float = 4.0
    vocab: int = 51200
    seq: int = 2048

    def layer_params(self):
        return 4 * self.d * self.d + 2 * self.ffn_mult * self.d * self.d      # qkv + out proj, two MLP matrices (biases/norms ignored)

    def params(self):
        return self.layers * self.layer_params() + self.vocab * self.d

    @staticmethod
    def for_params(target, layers=128, ffn_mult=4.0, vocab=51200, seq=2048):
        """Solve d for a parameter target at fixed depth (rounded to a multiple of 128)."""
        k = 4 + 2 * ffn_mult
        d = (-vocab + math.sqrt(vocab ** 2 + 4 * layers * k * target)) / (2 * layers * k)
        d = max(128, int(round(d / 128.0)) * 128)
        return Model(layers, d, ffn_mult, vocab, seq)


@dataclass
class Par:
    tp: int = 8
    pp: int = 16
    dp: int = 32
    zero: int = 3                # 0 none, 1 optimizer state, 2 + gradients, 3 + parameters (gathered per layer)
    microbatches: int = 64       # per data-parallel replica per step
    mb_seqs: int = 1             # sequences per microbatch
    recompute: bool = True
    wbytes: int = 2              # parameters as used by the GEMMs (bf16)
    gbytes: int = 2              # gradients
    obytes: int = 12             # fp32 master + Adam m + v
    abytes: int = 2              # activations
    sp: bool = True              # sequence-parallel tensor parallel (activations split over tp)


def transformer(model: Model, par: Par, peak_tflops=312.0, eff=0.5, hbm_gb=40.0):
    """Closed-form accounting. Everything is per GPU unless it says otherwise. See the module docstring: a model, not a run."""
    L, d, f, V, s = model.layers, model.d, model.ffn_mult, model.vocab, model.seq
    tp, pp, dp, M, b = par.tp, par.pp, par.dp, par.microbatches, par.mb_seqs
    assert L % pp == 0, 'layers must divide by pp'
    lps = L // pp                                              # layers per stage
    P_layer = model.layer_params()
    stage_params = lps * P_layer / tp + V * d / (tp * pp)   # embedding spread over stages (model simplification)
    p_gpu = stage_params
    # ---- memory bytes per GPU
    shard = (lambda x, on: x / dp if on else x)
    w_bytes = shard(p_gpu * par.wbytes, par.zero >= 3)
    g_bytes = shard(p_gpu * par.gbytes, par.zero >= 2)
    o_bytes = shard(p_gpu * par.obytes, par.zero >= 1)
    gathered = (2 if par.zero >= 3 else 0) * P_layer / tp * par.wbytes           # double-buffered layer gather window
    tok = b * s
    act_in = tok * d * par.abytes / (tp if par.sp else 1)                       # one layer-input checkpoint per microbatch
    inflight = min(M, pp)                                                       # 1F1B: stage 0 holds pp microbatches
    if par.recompute:
        act = inflight * lps * act_in + (34 * tok * d / tp) * 1.0               # checkpoints + one layer's full activations
    else:
        act = inflight * lps * 34 * tok * d / tp
    mem = dict(params=w_bytes, grads=g_bytes, optimizer=o_bytes, gathered_window=gathered, activations=act)
    mem['total'] = sum(mem.values())
    mem['fits_hbm'] = mem['total'] <= hbm_gb * 1e9
    # ---- FLOPs
    flops_tok_layer = 2 * P_layer + 4 * s * d                                   # fwd matmuls (2 per param) + attention scores
    fwd = tok * lps * flops_tok_layer / tp
    step_flops_mb = fwd * (4 if par.recompute else 3)                           # fwd + (recompute) + 2x bwd
    flops_step = M * step_flops_mb
    model_flops_step = M * dp * tok * 3 * (L * flops_tok_layer + 2 * V * d)     # useful (no recompute), whole job
    # ---- communication bytes per GPU per step (what leaves the GPU)
    ar = lambda nb, g: 2 * nb * (g - 1) / g if g > 1 else 0
    tp_vol = M * lps * (6 if par.recompute else 4) * ar(tok * d * par.abytes, tp)    # 2 all-reduces per layer per pass (SP: RS+AG, same bytes)
    pp_vol = M * 2 * tok * d * par.abytes / tp if pp > 1 else 0                 # fwd activation + bwd gradient across a boundary
    rs = lambda nb: nb * (dp - 1) / dp if dp > 1 else 0
    gparams = p_gpu
    if par.zero >= 3:
        dp_vol = 2 * rs(gparams * par.wbytes) + rs(gparams * par.gbytes)    # gather fwd + gather bwd (once per step each) + RS grads
    elif par.zero >= 1:
        dp_vol = rs(gparams * par.gbytes) + rs(gparams * par.wbytes)             # RS grads + all-gather updated params
    else:
        dp_vol = ar(gparams * par.gbytes, dp)
    comm = dict(tensor_parallel=tp_vol, pipeline=pp_vol, data_parallel=dp_vol)
    comm['total'] = sum(comm.values())
    # ---- per-step critical path (lower bound by arithmetic: perfect overlap, 1F1B bubble)
    t_mb = step_flops_mb / (peak_tflops * 1e12 * eff)
    bubble = (pp - 1) * t_mb
    ideal = M * t_mb
    crit = dict(compute_s=ideal, bubble_s=bubble, lower_bound_s=ideal + bubble)
    return dict(model=dict(layers=L, d=d, params=model.params(), layer_params=P_layer), par=par.__dict__, memory_bytes_per_gpu=mem,
                flops_per_gpu_per_step=flops_step, model_flops_per_step_whole_job=model_flops_step, comm_bytes_per_gpu_per_step=comm,
                critical_path=crit, gpus=tp * pp * dp,
                mfu_at_lower_bound=model_flops_step / (tp * pp * dp) / ((ideal + bubble) * peak_tflops * 1e12) if ideal else 0.0)


def emit_graph(model: Model, par: Par):
    """KVM task graph (JSON) of ONE microbatch through ONE layer of ONE representative GPU, plus the per-step tasks. Op names and
    argument names are the registry's (matmul m/n/k, put_signal/send_proxy rows/cols, adamw e0/e1); `repeat` says how many times
    the group runs per step on that GPU. Dependencies are in program order (`after` = previous task of the group unless noted).
    It is a description for the cost model and for review, not something partition.py can instantiate at this size."""
    d, f, s, tp, dp = model.d, model.ffn_mult, model.seq, par.tp, par.dp
    tok = par.mb_seqs * s
    lps = model.layers // par.pp
    ff = int(f * d)
    g = []

    def T(op, **a):
        g.append(dict(id=f't{len(g)}', op=op, args=a, after=[f't{len(g) - 1}'] if g else []))
    lay_bytes = model.layer_params() / tp * par.wbytes
    if par.zero >= 3:
        T('put_signal', note='ZeRO-3 all-gather of this layer weights: send own shard to dp-1 peers', bytes=lay_bytes / dp * (dp - 1))
    T('matmul', note='qkv', m=tok, n=3 * d // tp, k=d)
    T('matmul', note='attention scores + context (flops folded into one matmul-equivalent)', m=tok, n=2 * s // tp, k=d)
    T('matmul', note='attention output projection', m=tok, n=d, k=d // tp)
    if tp > 1:
        T('put_signal', note='tensor-parallel all-reduce (reduce-scatter + all-gather)', bytes=2 * tok * d * par.abytes * (tp - 1) / tp)
    T('matmul', note='mlp up', m=tok, n=ff // tp, k=d)
    T('matmul', note='mlp down', m=tok, n=d, k=ff // tp)
    if tp > 1:
        T('put_signal', note='tensor-parallel all-reduce', bytes=2 * tok * d * par.abytes * (tp - 1) / tp)
    layer = dict(group='layer_forward', repeat=lps * par.microbatches, tasks=g,
                 note='backward = 2x the matmul tasks (grad wrt input and weight) with the same all-reduces; recompute = a second forward')
    step = [dict(group='pipeline_boundary', repeat=2 * par.microbatches,
                 tasks=[dict(id='p0', op='send_proxy', args=dict(rows=tok, cols=d // tp), bytes=tok * d * par.abytes / tp, after=[])]),
            dict(group='gradient_reduce_scatter', repeat=1,
                 tasks=[dict(id='r0', op='reduce_put_signal', args=dict(n=int(model.layer_params() / tp * lps)), after=[],
                             note='ZeRO reduce-scatter of this GPU gradients over dp')]),
            dict(group='sharded_optimizer', repeat=1,
                 tasks=[dict(id='o0', op='adamw', args=dict(e0=0, e1=int((lps * model.layer_params() / tp) / (dp if par.zero >= 1 else 1)), lr=1e-4),
                             after=[])])]
    return dict(kind='kvm-task-graph-summary', model=model.__dict__, par=par.__dict__, groups=[layer] + step)


def small_configs():
    """The configurations the golden check runs (each partitions in under a few seconds on CPU)."""
    out = []
    for P in (2, 4):
        for coll in ('direct', 'ring'):
            out.append((Stack([8, 16, 8], m=8, steps=1, strategy='zero'), P, coll, 1))
    out.append((Stack([8, 8, 8, 8], m=8, steps=1, strategy='zero'), 4, 'ring', 2))
    out.append((Stack([16, 32, 16], m=8, steps=1, strategy='tp'), 2, 'direct', 1))
    out.append((Stack([16, 32, 16], m=8, steps=1, strategy='tp'), 4, 'ring', 1))
    out.append((Stack([8, 16, 8], m=8, steps=3, strategy='zero'), 4, 'direct', 1))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('check')
    c.add_argument('--json', action='store_true')
    t = sub.add_parser('transformer')
    t.add_argument('--params', type=float, default=1e12)
    t.add_argument('--layers', type=int, default=128)
    t.add_argument('--gpus', type=int, default=4096)
    t.add_argument('--tp', type=int, default=8)
    t.add_argument('--pp', type=int, default=16)
    t.add_argument('--zero', type=int, default=3)
    t.add_argument('--microbatches', type=int, default=64)
    t.add_argument('--peak-tflops', type=float, default=312.0)
    t.add_argument('--eff', type=float, default=0.5)
    t.add_argument('--hbm-gb', type=float, default=40.0)
    t.add_argument('--json', action='store_true')
    t.add_argument('--emit-graph', metavar='FILE', help='write the representative-GPU task graph summary as JSON')
    a = ap.parse_args(argv)
    if a.cmd == 'check':
        rows = []
        for st, P, coll, k in small_configs():
            exact = st.steps == 1
            r = check_stack(st, P, coll, k, seeds=range(3), exact=exact, rel=1e-4)
            rows.append(r)
            print(f"OK P={P} {st.strategy:5s} dims={st.dims} steps={st.steps} {coll:6s} chunks={k}: comm {r['measured']['comm_bytes']} B == predicted, "
                  f"flops/dev {r['measured']['flops']}, bytes/dev {r['measured']['local_bytes']} == predicted, worst rel vs unpartitioned {r['worst_rel_vs_unpartitioned']:.2g}")
        if a.json:
            print(json.dumps(rows, indent=1))
        return 0
    dp = a.gpus // (a.tp * a.pp)
    assert dp * a.tp * a.pp == a.gpus, 'gpus must equal tp*pp*dp'
    mdl, par = Model.for_params(a.params, a.layers), Par(a.tp, a.pp, dp, a.zero, a.microbatches)
    if a.emit_graph:
        Path(a.emit_graph).write_text(json.dumps(emit_graph(mdl, par), indent=1, default=float) + '\n')
    res = transformer(mdl, par, a.peak_tflops, a.eff, a.hbm_gb)
    print(json.dumps(res, indent=1, default=float) if a.json else json.dumps(res, indent=1, default=float))
    return 0


if __name__ == '__main__':
    sys.exit(main())
