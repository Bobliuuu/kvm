#!/usr/bin/env python3
"""The fused GPT step's data flow (design D2), member by member, on the CPU: the device program's executable specification.

  python3 -I bench/fused_gpt/d2_reference.py WORKLOAD_DIR [--steps 2] [--gangs 1,2,4] [--tol 3e-4]    (from the repo root; stdlib only)

A gang of G members trains the bundle's pre-norm GPT for the given steps. Member b holds:
  * its own rows of every N x d stream (rows b N/G .. (b+1) N/G - 1): LayerNorm forward and backward, the residual adds;
  * whole heads of q, k and v (head h on member h % G): RoPE and causal attention stay local, and proj's rows of those heads;
  * a contiguous block of fc1's columns (and fc2's rows, the same units), and of the vocabulary (the head's columns, wte's rows);
  * a copy of every replicated parameter (the LayerNorms' gains and biases, proj's and fc2's biases), updated identically by all.
Members exchange only through `Exchange` points: written in one phase, read in a later one, which is where the device's gang barrier sits.
An all-gather is own rows written, then every row read; a reduce-scatter is per-member partials written, then each member sums the
partials of its own rows in member order.

Checks, two levels:
  1. the decomposition: every G gives the same per-step losses and gradients as G = 1, to float64 rounding;
  2. the semantics: G = 1 against the workload VM (the bundle's own ops, float32 each) after the same steps: per-step losses, and every
     gradient element above a noise floor, within --tol. (Weights are not compared element by element: AdamW's first update is about
     lr x sign(g), so float32 noise flips it on gradients near zero.)
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'runtime'))

import compile_workload as cw  # noqa: E402
import workload_vm as vm  # noqa: E402


# ---------------------------------------------------------------------------------------------- the model, read from the workload
class Model:
    """Shapes, hyperparameters and initial values of a GPT workload, read by buffer and task names (this is a test tool: the matcher
    in include/kvm_fuse_gpt.h decides from the DAG alone)."""

    def __init__(self, spec: dict, compiled: cw.Compiled, steps: int):
        bufs = {b.name: b for b in compiled.buffers}
        self.init = {b.name: list(b.materialize(compiled.data_dir)) for b in compiled.buffers}
        self.shape = {b.name: (b.rows, b.cols) for b in compiled.buffers}
        tasks = [t for g in spec['graphs'] for t in g['tasks']]
        first = {}
        for t in tasks:
            first.setdefault(t['op'], t)
        emb = first['embedding_fwd']['args']
        self.T, self.d = emb['T'], emb['d']
        self.V = bufs['w.gpt.wte'].rows
        self.N = bufs[emb['out']].rows
        self.B = self.N // self.T
        self.L = len({n.split('.')[2] for n in bufs if re.match(r'w\.gpt\.h\d+\.', n)})
        rope = first['rope_fwd']['args']
        self.H, self.hd, self.theta = rope['H'], rope['hd'], rope['theta']
        self.F = bufs['w.gpt.h0.mlp.fc1.w'].cols
        self.scale = first['attn_fwd']['args']['scale']
        self.ln_eps = first['layernorm_fwd']['args']['eps']
        self.max_norm = first['gradnorm_finish']['args']['max_norm']
        ad = first['adamw']['args']
        self.lr, self.b1, self.b2, self.eps = ad['lr'], ad['b1'], ad['b2'], ad['eps']
        self.wd = {}
        for t in tasks:
            if t['op'] == 'adamw':
                self.wd[t['args']['p']] = t['args']['wd']
        sw = [t for t in tasks if t['op'] == 'select_window']
        self.ids_src = next(t['args']['src'] for t in sw if t['args']['dst'] == emb['ids'])
        self.tgt_src = next(t['args']['src'] for t in sw if t['args']['dst'] != emb['ids'])
        self.stride, self.off = sw[0]['args']['stride'], sw[0]['args']['off']
        self.step_buf = sw[0]['args']['step']
        self.steps = steps

    def params(self):
        """Every parameter name, the bundle's AdamW order."""
        names = ['w.gpt.wte']
        for l in range(self.L):
            p = f'w.gpt.h{l}'
            names += [f'{p}.n1.g', f'{p}.n1.b', f'{p}.attn.q.w', f'{p}.attn.q.b', f'{p}.attn.k.w', f'{p}.attn.k.b', f'{p}.attn.v.w',
                      f'{p}.attn.v.b', f'{p}.attn.proj.w', f'{p}.attn.proj.b', f'{p}.n2.g', f'{p}.n2.b', f'{p}.mlp.fc1.w', f'{p}.mlp.fc1.b',
                      f'{p}.mlp.fc2.w', f'{p}.mlp.fc2.b']
        return names + ['w.gpt.nf.g', 'w.gpt.nf.b', 'w.gpt.head.w']


# ---------------------------------------------------------------------------------------------- small dense helpers (row-major lists)
def mat(flat, rows, cols):
    return [list(flat[r * cols:(r + 1) * cols]) for r in range(rows)]


def flat(m):
    return [x for row in m for x in row]


def zeros(r, c):
    return [[0.0] * c for _ in range(r)]


def gelu(x):
    return 0.5 * x * (1.0 + math.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))


def gelu_grad(x):
    y = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    t = math.tanh(y)
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * 0.7978845608028654 * (1.0 + 3 * 0.044715 * x * x)


class Exchange:
    """One exchange point of a step: members write slots in one phase; reads happen only after the phase that wrote them (the barrier)."""

    def __init__(self, name):
        self.name, self.slots, self.sealed = name, {}, False

    def write(self, key, value):
        assert not self.sealed, f'{self.name}: written after its barrier'
        assert key not in self.slots, f'{self.name}: slot {key} written twice'
        self.slots[key] = value

    def barrier(self):
        self.sealed = True

    def read(self, key):
        assert self.sealed, f'{self.name}: read before its barrier'
        return self.slots[key]


# ---------------------------------------------------------------------------------------------- the gang
class Gang:
    def __init__(self, m: Model, G: int):
        self.m, self.G = m, G
        assert m.N % G == 0, f'N {m.N} must split over {G} members'
        self.rows = [list(range(b * m.N // G, (b + 1) * m.N // G)) for b in range(G)]
        self.heads = [[h for h in range(m.H) if h % G == b] for b in range(G)]
        self.fcols = self.split(m.F)
        self.vcols = self.split(m.V)
        P = {n: list(m.init[n]) for n in m.params()}
        self.p = P                                   # the members' parameter shards, held as whole tensors: each member updates only its own part
        self.mom = {n: ([0.0] * len(P[n]), [0.0] * len(P[n])) for n in P}
        for n in P:                                  # AdamW moments start from the bundle's initial m and v
            self.mom[n] = (list(m.init[n.replace('w.', 'm.', 1)]), list(m.init[n.replace('w.', 'v.', 1)]))
        self.t = int(m.init[m.step_buf][0])
        self.losses, self.grads = [], []

    def split(self, n):
        per = -(-n // self.G)
        return [list(range(b * per, min(n, (b + 1) * per))) for b in range(self.G)]

    def W(self, name):
        r, c = self.m.shape[name]
        return mat(self.p[name], r, c)

    # ---- one training step
    def step(self):
        m, G = self.m, self.G
        d, H, hd, F, V, N, T = m.d, m.H, m.hd, m.F, m.V, m.N, m.T
        ids = [int(m.init[m.ids_src][self.t * m.stride + m.off + i]) for i in range(N)]
        tgt = [int(m.init[m.tgt_src][self.t * m.stride + m.off + i]) for i in range(N)]
        wte = self.W('w.gpt.wte')
        # forward: the embedding on own rows
        x = zeros(N, d)
        for b in range(G):
            for r in self.rows[b]:
                x[r] = list(wte[ids[r]])
        saved = []
        for l in range(m.L):
            pre = f'w.gpt.h{l}'
            s = {'x': [list(row) for row in x]}
            h, s['m1'], s['r1'] = self.ln_rows(x, self.p[f'{pre}.n1.g'], self.p[f'{pre}.n1.b'])
            xh = self.all_gather('h', h)                 # every column GEMM reads every row
            Wq, Wk, Wv, Wo = self.W(f'{pre}.attn.q.w'), self.W(f'{pre}.attn.k.w'), self.W(f'{pre}.attn.v.w'), self.W(f'{pre}.attn.proj.w')
            bq, bk, bv = self.p[f'{pre}.attn.q.b'], self.p[f'{pre}.attn.k.b'], self.p[f'{pre}.attn.v.b']
            q, k, v, o, lse = zeros(N, d), zeros(N, d), zeros(N, d), zeros(N, d), {}
            for b in range(G):                           # head owners: q, k, v of their heads, RoPE, attention
                for hh in self.heads[b]:
                    cols = range(hh * hd, (hh + 1) * hd)
                    for r in range(N):
                        for c in cols:
                            q[r][c] = sum(xh[r][kk] * Wq[kk][c] for kk in range(d)) + bq[c]
                            k[r][c] = sum(xh[r][kk] * Wk[kk][c] for kk in range(d)) + bk[c]
                            v[r][c] = sum(xh[r][kk] * Wv[kk][c] for kk in range(d)) + bv[c]
                    self.rope(q, hh, +1); self.rope(k, hh, +1)
                    for sq in range(m.B):
                        self.attn_fwd(q, k, v, o, lse, sq, hh)
            s.update(q=q, k=k, v=v, o=o, lse=lse, h=xh)
            xm = self.reduce_scatter('proj', [self.part_rows_by_heads(o, Wo, b) for b in range(G)], x, self.p[f'{pre}.attn.proj.b'])
            s['xm'] = [list(row) for row in xm]
            h2, s['m2'], s['r2'] = self.ln_rows(xm, self.p[f'{pre}.n2.g'], self.p[f'{pre}.n2.b'])
            xh2 = self.all_gather('h2', h2)
            W1, b1, W2 = self.W(f'{pre}.mlp.fc1.w'), self.p[f'{pre}.mlp.fc1.b'], self.W(f'{pre}.mlp.fc2.w')
            u, zpre = zeros(N, F), zeros(N, F)
            for b in range(G):                           # fc1 owners: their columns, GELU
                for c in self.fcols[b]:
                    for r in range(N):
                        z = sum(xh2[r][kk] * W1[kk][c] for kk in range(d)) + b1[c]
                        zpre[r][c], u[r][c] = z, gelu(z)
            s.update(u=u, pre=zpre, h2=xh2)
            x = self.reduce_scatter('fc2', [self.part_rows(u, W2, self.fcols[b]) for b in range(G)], xm, self.p[f'{pre}.mlp.fc2.b'])
            saved.append(s)
        xL = [list(row) for row in x]
        hf, mf, rf = self.ln_rows(x, self.p['w.gpt.nf.g'], self.p['w.gpt.nf.b'])
        xhf = self.all_gather('hf', hf)
        Wh = self.W('w.gpt.head.w')
        logits = zeros(N, V)
        for b in range(G):
            for c in self.vcols[b]:
                for r in range(N):
                    logits[r][c] = sum(xhf[r][kk] * Wh[kk][c] for kk in range(d))
        # cross-entropy across vocabulary shards: each member's per-row (max, sum exp), one exchange, then every member gets each row's lse
        ce = Exchange('ce')
        for b in range(G):
            mx = [max(logits[r][c] for c in self.vcols[b]) for r in range(N)]
            se = [sum(math.exp(logits[r][c] - mx[r]) for c in self.vcols[b]) for r in range(N)]
            tl = {r: logits[r][tgt[r]] for r in range(N) if tgt[r] in self.vcols[b]}
            ce.write(b, (mx, se, tl))
        ce.barrier()
        lse_row, rowloss = [0.0] * N, [0.0] * N
        for r in range(N):
            M = max(ce.read(b)[0][r] for b in range(G))
            lse_row[r] = M + math.log(sum(ce.read(b)[1][r] * math.exp(ce.read(b)[0][r] - M) for b in range(G)))
            owner = next(b for b in range(G) if r in ce.read(b)[2])
            rowloss[r] = lse_row[r] - ce.read(owner)[2][r]
        self.losses.append(sum(rowloss) / N)
        dlog = zeros(N, V)
        for b in range(G):
            for c in self.vcols[b]:
                for r in range(N):
                    dlog[r][c] = (math.exp(logits[r][c] - lse_row[r]) - (1.0 if c == tgt[r] else 0.0)) / N
        # ---- backward
        grads = {n: [0.0] * len(self.p[n]) for n in self.p}
        gH = zeros(d, V)
        for b in range(G):
            for c in self.vcols[b]:
                for kk in range(d):
                    gH[kk][c] = sum(xhf[r][kk] * dlog[r][c] for r in range(N))
        grads['w.gpt.head.w'] = flat(gH)
        dhf = self.reduce_scatter('head_dx', [self.part_rows_t(dlog, Wh, self.vcols[b]) for b in range(G)], None, None)
        Gs, dgf, dbf = self.ln_bwd_rows(xL, dhf, self.p['w.gpt.nf.g'], mf, rf, None)
        grads['w.gpt.nf.g'], grads['w.gpt.nf.b'] = dgf, dbf
        for l in range(m.L - 1, -1, -1):
            pre, s = f'w.gpt.h{l}', saved[l]
            W1, W2, Wo = self.W(f'{pre}.mlp.fc1.w'), self.W(f'{pre}.mlp.fc2.w'), self.W(f'{pre}.attn.proj.w')
            Gg = self.all_gather('G_fc2', Gs)
            grads[f'{pre}.mlp.fc2.b'] = [sum(Gg[r][c] for r in range(N)) for c in range(d)]   # replicated: every member, the same order
            gW2, gW1, gb1 = zeros(F, d), zeros(d, F), [0.0] * F
            du = zeros(N, F)
            for b in range(G):                           # fc1 owners: fc2's rows and fc1's columns of their units
                for c in self.fcols[b]:
                    for j in range(d):
                        gW2[c][j] = sum(s['u'][r][c] * Gg[r][j] for r in range(N))
                    for r in range(N):
                        du[r][c] = sum(Gg[r][j] * W2[c][j] for j in range(d)) * gelu_grad(s['pre'][r][c])
                    gb1[c] = sum(du[r][c] for r in range(N))
                    for kk in range(d):
                        gW1[kk][c] = sum(s['h2'][r][kk] * du[r][c] for r in range(N))
            grads[f'{pre}.mlp.fc2.w'], grads[f'{pre}.mlp.fc1.w'], grads[f'{pre}.mlp.fc1.b'] = flat(gW2), flat(gW1), gb1
            dh2 = self.reduce_scatter('fc1_dx', [self.part_rows_t(du, W1, self.fcols[b]) for b in range(G)], None, None)
            dx, dg2, db2 = self.ln_bwd_rows(s['xm'], dh2, self.p[f'{pre}.n2.g'], s['m2'], s['r2'], None)
            Gs = [[Gs[r][c] + dx[r][c] for c in range(d)] for r in range(N)]
            grads[f'{pre}.n2.g'], grads[f'{pre}.n2.b'] = dg2, db2
            Gg = self.all_gather('G_proj', Gs)
            grads[f'{pre}.attn.proj.b'] = [sum(Gg[r][c] for r in range(N)) for c in range(d)]
            Wq, Wk, Wv = self.W(f'{pre}.attn.q.w'), self.W(f'{pre}.attn.k.w'), self.W(f'{pre}.attn.v.w')
            gWo, gWq, gWk, gWv = zeros(d, d), zeros(d, d), zeros(d, d), zeros(d, d)
            gbq, gbk, gbv = [0.0] * d, [0.0] * d, [0.0] * d
            dq_r, dk_r, dv = zeros(N, d), zeros(N, d), zeros(N, d)
            for b in range(G):                           # head owners: proj's rows, attention's backward, q, k, v of their heads
                for hh in self.heads[b]:
                    cols = range(hh * hd, (hh + 1) * hd)
                    for c in cols:
                        for j in range(d):
                            gWo[c][j] = sum(s['o'][r][c] * Gg[r][j] for r in range(N))
                    do = zeros(N, d)
                    for r in range(N):
                        for c in cols:
                            do[r][c] = sum(Gg[r][j] * Wo[c][j] for j in range(d))
                    dq, dk = zeros(N, d), zeros(N, d)
                    for sq in range(m.B):
                        self.attn_bwd(s, do, dq, dk, dv, sq, hh)
                    self.rope(dq, hh, -1); self.rope(dk, hh, -1)
                    for c in cols:
                        gbq[c] = sum(dq[r][c] for r in range(N)); gbk[c] = sum(dk[r][c] for r in range(N)); gbv[c] = sum(dv[r][c] for r in range(N))
                        for kk in range(d):
                            gWq[kk][c] = sum(s['h'][r][kk] * dq[r][c] for r in range(N))
                            gWk[kk][c] = sum(s['h'][r][kk] * dk[r][c] for r in range(N))
                            gWv[kk][c] = sum(s['h'][r][kk] * dv[r][c] for r in range(N))
                        for r in range(N):
                            dq_r[r][c], dk_r[r][c] = dq[r][c], dk[r][c]
            for n, g in ((f'{pre}.attn.proj.w', gWo), (f'{pre}.attn.q.w', gWq), (f'{pre}.attn.k.w', gWk), (f'{pre}.attn.v.w', gWv)):
                grads[n] = flat(g)
            grads[f'{pre}.attn.q.b'], grads[f'{pre}.attn.k.b'], grads[f'{pre}.attn.v.b'] = gbq, gbk, gbv
            parts = []
            for b in range(G):                           # dh = dq Wq^T + dk Wk^T + dv Wv^T, a member's heads' columns
                cols = [c for hh in self.heads[b] for c in range(hh * hd, (hh + 1) * hd)]
                parts.append([[sum(dq_r[r][c] * Wq[j][c] + dk_r[r][c] * Wk[j][c] + dv[r][c] * Wv[j][c] for c in cols) for j in range(d)]
                              for r in range(N)])
            dh = self.reduce_scatter('qkv_dx', parts, None, None)
            dx, dg1, db1 = self.ln_bwd_rows(s['x'], dh, self.p[f'{pre}.n1.g'], s['m1'], s['r1'], None)
            Gs = [[Gs[r][c] + dx[r][c] for c in range(d)] for r in range(N)]
            grads[f'{pre}.n1.g'], grads[f'{pre}.n1.b'] = dg1, db1
        Gg = self.all_gather('G_emb', Gs)
        gE = zeros(V, d)
        for b in range(G):                               # vocabulary owners: their rows of wte's gradient, rows in order
            own = set(self.vcols[b])
            for r in range(N):
                if ids[r] in own:
                    for c in range(d):
                        gE[ids[r]][c] += Gg[r][c]
        grads['w.gpt.wte'] = flat(gE)
        self.grads.append(grads)
        # ---- the clip: every member's sum of squares of what it owns (member 0 also the replicated parameters), one exchange
        norm = math.sqrt(sum(sum(g * g for g in grads[n]) for n in grads))
        coef = min(1.0, self.m.max_norm / (norm + 1e-6)) if self.m.max_norm > 0 else 1.0
        # ---- AdamW in place, with the counter advanced first (the bundle's order)
        self.t += 1
        bc1, bc2 = 1.0 - self.m.b1 ** self.t, 1.0 - self.m.b2 ** self.t
        for n, g in grads.items():
            p, (mm, vv) = self.p[n], self.mom[n]
            wd = self.m.wd[n]
            for i in range(len(p)):
                gi = g[i] * coef
                mm[i] = self.m.b1 * mm[i] + (1 - self.m.b1) * gi
                vv[i] = self.m.b2 * vv[i] + (1 - self.m.b2) * gi * gi
                p[i] = p[i] - self.m.lr * wd * p[i] - self.m.lr * (mm[i] / bc1) / (math.sqrt(vv[i] / bc2) + self.m.eps)

    # ---- phases
    def ln_rows(self, x, g, be):
        """LayerNorm on own rows (every member its rows): the bundle's expressions."""
        d, eps = self.m.d, self.m.ln_eps
        out, mean, rstd = zeros(len(x), d), [0.0] * len(x), [0.0] * len(x)
        for b in range(self.G):
            for r in self.rows[b]:
                mu = sum(x[r]) / d
                rs = 1.0 / math.sqrt(sum((v - mu) ** 2 for v in x[r]) / d + eps)
                out[r] = [(x[r][c] - mu) * rs * g[c] + be[c] for c in range(d)]
                mean[r], rstd[r] = mu, rs
        return out, mean, rstd

    def ln_bwd_rows(self, x, dy, g, mean, rstd, _):
        """LayerNorm backward on own rows; dgamma and dbeta are each member's own-row partials, summed in member order (the exchange that
        carries them rides on the next barrier)."""
        d = self.m.d
        dx = zeros(len(x), d)
        ex = Exchange('ln_params')
        for b in range(self.G):
            pg, pb = [0.0] * d, [0.0] * d
            for r in self.rows[b]:
                xh = [(x[r][c] - mean[r]) * rstd[r] for c in range(d)]
                dyg = [dy[r][c] * g[c] for c in range(d)]
                a1, a2 = sum(dyg) / d, sum(dyg[c] * xh[c] for c in range(d)) / d
                dx[r] = [rstd[r] * (dyg[c] - a1 - xh[c] * a2) for c in range(d)]
                for c in range(d):
                    pg[c] += dy[r][c] * xh[c]; pb[c] += dy[r][c]
            ex.write(b, (pg, pb))
        ex.barrier()
        dg = [sum(ex.read(b)[0][c] for b in range(self.G)) for c in range(d)]
        db = [sum(ex.read(b)[1][c] for b in range(self.G)) for c in range(d)]
        return dx, dg, db

    def all_gather(self, name, x):
        ex = Exchange(name)
        for b in range(self.G):
            ex.write(b, {r: list(x[r]) for r in self.rows[b]})
        ex.barrier()
        out = [None] * self.m.N
        for b in range(self.G):
            for r, row in ex.read(b).items():
                out[r] = row
        return out

    def reduce_scatter(self, name, parts, res, bias):
        """Each member wrote a partial of every row; each member sums its own rows' partials in member order, plus the residual and the bias."""
        ex = Exchange(name)
        for b in range(self.G):
            ex.write(b, parts[b])
        ex.barrier()
        d = self.m.d
        out = zeros(self.m.N, d)
        for b in range(self.G):
            for r in self.rows[b]:
                for c in range(d):
                    acc = sum(ex.read(bb)[r][c] for bb in range(self.G))
                    out[r][c] = acc + (res[r][c] if res is not None else 0.0) + (bias[c] if bias is not None else 0.0)
        return out

    def part_rows_by_heads(self, o, Wo, b):
        cols = [c for hh in self.heads[b] for c in range(hh * self.m.hd, (hh + 1) * self.m.hd)]
        return self.part_rows(o, Wo, cols)

    def part_rows(self, a, W, units):
        """A row-parallel GEMM's partial: a[:, units] W[units, :] (N x d)."""
        d = self.m.d
        return [[sum(a[r][c] * W[c][j] for c in units) for j in range(d)] for r in range(self.m.N)]

    def part_rows_t(self, dy, W, units):
        """A column-parallel GEMM's input-gradient partial: dy[:, units] W[:, units]^T (N x d)."""
        d = self.m.d
        return [[sum(dy[r][c] * W[j][c] for c in units) for j in range(d)] for r in range(self.m.N)]

    def rope(self, x, hh, sign):
        T, hd, half = self.m.T, self.m.hd, self.m.hd // 2
        for r in range(self.m.N):
            for i in range(half):
                ang = (r % T) * self.m.theta ** (-(2 * i) / hd)
                sn, cs = sign * math.sin(ang), math.cos(ang)
                a, b = x[r][hh * hd + i], x[r][hh * hd + i + half]
                x[r][hh * hd + i], x[r][hh * hd + i + half] = a * cs - b * sn, a * sn + b * cs

    def attn_fwd(self, q, k, v, o, lse, sq, hh):
        T, hd, sc = self.m.T, self.m.hd, self.m.scale
        cols = range(hh * hd, (hh + 1) * hd)
        for i in range(T):
            r = sq * T + i
            s = [sum(q[r][c] * k[sq * T + j][c] for c in cols) * sc for j in range(i + 1)]
            mx = max(s); p = [math.exp(x - mx) for x in s]; tot = sum(p)
            for c in cols:
                o[r][c] = sum(p[j] * v[sq * T + j][c] for j in range(i + 1)) / tot
            lse[(sq, hh, i)] = mx + math.log(tot)

    def attn_bwd(self, s, do, dq, dk, dv, sq, hh):
        T, hd, sc = self.m.T, self.m.hd, self.m.scale
        q, k, v, o, lse = s['q'], s['k'], s['v'], s['o'], s['lse']
        cols = range(hh * hd, (hh + 1) * hd)
        for i in range(T):
            r = sq * T + i
            delta = sum(o[r][c] * do[r][c] for c in cols)
            for j in range(i + 1):
                rj = sq * T + j
                p = math.exp(sum(q[r][c] * k[rj][c] for c in cols) * sc - lse[(sq, hh, i)])
                dp = sum(do[r][c] * v[rj][c] for c in cols)
                ds = p * (dp - delta)
                for c in cols:
                    dq[r][c] += ds * k[rj][c] * sc
                    dk[rj][c] += ds * q[r][c] * sc
                    dv[rj][c] += p * do[r][c]


# ---------------------------------------------------------------------------------------------- the VM's run of the same steps
def vm_run(spec: dict, data_dir: Path, steps: int):
    """The bundle's own tasks for the first `steps` steps on the workload VM (deterministic, float32 a op)."""
    keep = {f's{s}/' for s in range(steps)}
    cut = json.loads(json.dumps(spec))
    for g in cut['graphs']:
        g['tasks'] = [t for t in g['tasks'] if any(t['id'].startswith(k) for k in keep)]
        ids = {t['id'] for t in g['tasks']}
        for t in g['tasks']:
            t['after'] = [a for a in t.get('after', []) if a in ids]
    compiled = cw.compile_spec(cut, data_dir=data_dir)
    out = vm.run_golden(compiled, timeout=3600)
    return out['buffers'], compiled


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('workload'); ap.add_argument('--steps', type=int, default=2)
    ap.add_argument('--gangs', default='1,2,4'); ap.add_argument('--tol', type=float, default=3e-4)
    a = ap.parse_args()
    wdir = Path(a.workload)
    spec = json.loads((wdir / 'workload.json').read_text())
    vm_bufs, compiled = vm_run(spec, wdir, a.steps)       # init files are named relative to the workload directory (the VM's default)
    model = Model(spec, compiled, a.steps)
    print(f'model: L {model.L}, d {model.d}, heads {model.H} x {model.hd}, MLP {model.F}, V {model.V}, {model.B} x {model.T} rows; {a.steps} steps')
    gangs = [int(x) for x in a.gangs.split(',')]
    runs = {}
    for G in gangs:
        g = Gang(model, G)
        for _ in range(a.steps):
            g.step()
        runs[G] = g
        print(f'G {G}: losses {", ".join(f"{x:.6f}" for x in g.losses)}')
    fail = 0
    ref = runs[gangs[0]]
    for G in gangs[1:]:   # 1. the decomposition
        worst = max(abs(x - y) / max(1e-30, abs(y)) for x, y in zip(runs[G].losses, ref.losses))
        gw = 0.0
        for s in range(a.steps):
            for n in ref.grads[s]:
                scale = max(abs(x) for x in ref.grads[s][n]) or 1.0
                gw = max(gw, max(abs(x - y) for x, y in zip(runs[G].grads[s][n], ref.grads[s][n])) / scale)
        ok = worst < 1e-10 and gw < 1e-10
        fail += not ok
        print(f'decomposition G {G} against G {gangs[0]}: losses {worst:.2e}, gradients {gw:.2e} relative: {"ok" if ok else "FAIL"}')
    # 2. the semantics: G = gangs[0] against the VM, the last step's gradients and every step's loss
    names = {}
    for t in (t for g in spec['graphs'] for t in g['tasks'] if t['op'] == 'adamw' and t['id'].startswith(f's{a.steps - 1}/')):
        names[t['args']['p']] = t['args']['g']
    losses_vm = [vm_bufs[f'loss.{s}'][0] for s in range(a.steps)]
    lw = max(abs(x - y) / abs(y) for x, y in zip(ref.losses, losses_vm))
    # each gradient tensor's largest difference over its largest magnitude (a float32 GEMM's error scales with its operands, not its elements)
    gw, worst = 0.0, ''
    for p, gname in names.items():
        mine, theirs = ref.grads[-1][p], vm_bufs[gname]
        scale = max(abs(x) for x in theirs) or 1.0
        e = max(abs(x - y) for x, y in zip(mine, theirs)) / scale
        if e > gw:
            gw, worst = e, p
    ok = lw < a.tol and gw < a.tol
    fail += not ok
    print(f'semantics G {gangs[0]} against the VM: losses {lw:.2e}; step {a.steps - 1} gradients {gw:.2e} of each tensor\'s scale (worst {worst}): {"ok" if ok else "FAIL"}')
    print('OK' if not fail else f'FAILED: {fail}')
    return 1 if fail else 0


if __name__ == '__main__':
    sys.exit(main())
