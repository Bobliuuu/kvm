#!/usr/bin/env python3
"""Scale-out simulator: hierarchical topology (GPU / node / rack / datacenter, WAN between datacenters) + a generic task-graph
lowering of a transformer step + a discrete-event scheduler. NOTHING here runs on a GPU; every number is the output of a cost
model over stated assumptions (see results/2026-11-04-scaleout-sim/README.md for what is validated and what is not).

What is generic and what is not
  * The graph is built from a layer description (`ModelCfg.phases`: compute flops + the tensor-parallel / expert-parallel bytes
    each phase needs) and a parallelism config. There is no per-model hand-written schedule: the pipeline order (1F1B), the
    ZeRO prefetch order and the compute/communication overlap all EMERGE from dependency edges, a per-stream priority rule and a
    window edge (the in-flight microbatch limit) in the DES below. `balance_stages` (the linear-partition DP) is the one
    function reused from runtime/partition.py.
  * The communication each phase needs mirrors the layout algebra of runtime/partition.py (a COLS x ROWS matmul pair yields PART
    and needs an all-reduce, ZeRO shards parameters ROWS and gathers per use, expert parallel is an all-to-all). This file
    RE-IMPLEMENTS those rules at cost-model level; the real partitioner is exercised only on scaled-down shapes
    (tests/dist/test_scaleout.py cross-checks message volumes against its output).
  * Symmetry reduction: at full scale one representative rank per pipeline stage is simulated (all TP x DP ranks of a stage run
    the same schedule). `explicit=True` instantiates every rank with real barrier edges for collectives; a test checks that both
    give the same makespan on a small cluster. Heterogeneity (stragglers, failures) is NOT captured by the reduced graph.

    python3 dist/scaleout.py --help
"""
from __future__ import annotations

import argparse
import heapq
import itertools
import json
import math
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'runtime'), str(ROOT / 'dist')]
import partition as pt   # noqa: E402  (balance_stages: the generic linear-partition DP)

NVLINK, RACK, DC, WAN = 0, 1, 2, 3
TIER_NAMES = ['nvlink', 'rack', 'dc', 'wan']


# ------------------------------------------------------------------------------------------------------------- topology
@dataclass
class Topology:
    """GPU -> node -> rack -> datacenter. Bandwidths are per GPU, one direction, GB/s. The WAN is a shared pipe per datacenter
    pair (`wan_gbps` aggregate, per direction), however many GPUs send through it."""
    gpus_per_node: int = 8
    nodes_per_rack: int = 32
    racks_per_dc: int = 4
    n_dc: int = 4
    nvlink_gbs: float = 450.0        # H100 NVLink 4: 900 GB/s bidirectional per GPU
    nvlink_us: float = 1.0
    rack_gbs: float = 50.0           # one 400 Gb/s NIC per GPU
    rack_us: float = 5.0
    dc_gbs: float = 50.0
    dc_us: float = 15.0
    dc_oversub: float = 1.0          # spine oversubscription (2.0 = half the per-GPU bandwidth above the rack)
    wan_gbps: float = 100.0          # aggregate per datacenter pair per direction, Gb/s
    wan_ms: float = 20.0             # one-way latency

    @property
    def per_rack(self): return self.gpus_per_node * self.nodes_per_rack
    @property
    def per_dc(self): return self.per_rack * self.racks_per_dc
    @property
    def n(self): return self.per_dc * self.n_dc

    def coords(self, g):
        return (g // self.per_dc, g // self.per_rack, g // self.gpus_per_node)       # (dc, rack, node) ids

    def tier(self, a, b):
        if a // self.gpus_per_node == b // self.gpus_per_node: return NVLINK
        if a // self.per_rack == b // self.per_rack: return RACK
        if a // self.per_dc == b // self.per_dc: return DC
        return WAN

    def bw_gbs(self, tier, sharers=1):
        """Effective GB/s per GPU at a tier. On the WAN the pipe is shared by `sharers` GPUs sending at the same time."""
        if tier == NVLINK: return self.nvlink_gbs
        if tier == RACK: return self.rack_gbs
        if tier == DC: return self.dc_gbs / self.dc_oversub
        return min(self.rack_gbs, self.wan_gbps / 8.0 / max(1, sharers))

    def lat_us(self, tier):
        return [self.nvlink_us, self.rack_us, self.dc_us, self.wan_ms * 1e3][tier]

    def levels(self, ranks):
        """Hierarchical decomposition of a group from its real placement: [(tier, fanout)], innermost first. fanout = the largest
        number of distinct children any parent has among the members (exact for the regular placements used here)."""
        ranks = list(ranks)
        specs = ((NVLINK, lambda g: g // self.gpus_per_node, lambda g: g),
                 (RACK, lambda g: g // self.per_rack, lambda g: g // self.gpus_per_node),
                 (DC, lambda g: g // self.per_dc, lambda g: g // self.per_rack),
                 (WAN, lambda g: 0, lambda g: g // self.per_dc))
        out = []
        for tier, parent, child in specs:
            per_parent = {}
            for g in ranks:
                per_parent.setdefault(parent(g), set()).add(child(g))
            f = max(len(v) for v in per_parent.values())
            if f > 1:
                out.append((tier, f))
        return out


def coll_time_us(topo: Topology, levels, kind, nbytes, sharers=1):
    """Time of one collective over a group, ring within each level, bytes shrink by the fanout going outward (hierarchical).
    kind: 'rs' | 'ag' | 'ar' | 'a2a' | 'p2p'. nbytes = the full per-rank tensor (ar/rs/ag) or the per-rank send total (a2a)."""
    if not levels: return 0.0
    if kind == 'a2a':
        n = math.prod(f for _, f in levels)
        worst, inner = 0.0, 1
        for tier, f in levels:
            traffic = nbytes * (f - 1) * inner / n
            worst = max(worst, traffic / (topo.bw_gbs(tier, sharers) * 1e3) + topo.lat_us(tier))
            inner *= f
        return worst
    t, B = 0.0, float(nbytes)
    for tier, f in levels:
        t += (f - 1) / f * B / (topo.bw_gbs(tier, sharers) * 1e3) + (f - 1) * topo.lat_us(tier)
        B /= f
    return t * (2 if kind == 'ar' else 1)


# ------------------------------------------------------------------------------------------------------------- model
@dataclass
class ModelCfg:
    kind: str = 'dense'              # 'dense' | 'moe'
    layers: int = 128
    hidden: int = 25600
    seq: int = 4096
    vocab: int = 51200
    ffn_mult: int = 4                # dense: ffn = ffn_mult * hidden (2 matrices)
    experts: int = 0                 # moe
    topk: int = 0
    expert_ffn: int = 0              # moe: SwiGLU expert width (3 matrices)
    name: str = ''

    def layer_params(self):
        h = self.hidden
        if self.kind == 'dense': return 4 * h * h + 2 * self.ffn_mult * h * h
        return 4 * h * h + self.experts * 3 * h * self.expert_ffn
    def params(self): return self.layers * self.layer_params() + 2 * self.vocab * self.hidden
    def active_params(self):
        h = self.hidden
        if self.kind == 'dense': return self.params()
        return self.layers * (4 * h * h + self.topk * 3 * h * self.expert_ffn) + 2 * self.vocab * h

    def phases(self):
        """Per layer: [(name, flops_fwd_per_token, tp_bytes_per_token_factor, a2a_bytes_per_token_factor)]. A factor is multiplied
        by hidden*2 bytes (bf16). tp factor: reduce-scatter + all-gather pair per phase (sequence parallel)."""
        h, s = self.hidden, self.seq
        attn = ('attn', 8 * h * h + 4 * s * h, 1.0, 0.0)               # Megatron FLOP counts: 8h^2 projections + 4sh scores
        if self.kind == 'dense':
            return [attn, ('mlp', 4 * self.ffn_mult * h * h, 1.0, 0.0)]
        return [attn, ('moe', self.topk * 6 * h * self.expert_ffn, 1.0, float(self.topk))]


def dense_1t(): return ModelCfg('dense', 128, 25600, 4096, 51200, 4, name='dense-1T (Megatron-LM 1T shape)')
def moe_1t(): return ModelCfg('moe', 64, 7168, 4096, 160000, 0, experts=384, topk=8, expert_ffn=2048, name='moe-1T (384 experts, top-8)')


@dataclass
class Hw:
    peak_tflops: float = 989.0       # H100 SXM dense bf16
    kernel_eff: float = 0.60         # fraction of peak the GEMMs reach (assumption)
    hbm_gb: float = 80.0
    hbm_gbs: float = 3350.0
    comm_sm_tax: float = 0.05        # compute slowdown while this device also runs comm tasks on SMs (assumption, see README)
    ctx_gb: float = 2.0


@dataclass
class Par:
    tp: int = 8
    pp: int = 16
    dp: int = 32
    microbatches: int = 64
    global_seqs: int = 4096
    ep: int = 1                      # == dp for moe
    order: tuple = ('pp', 'dp', 'tp')    # slowest -> fastest in the physical rank numbering
    dp_mode: str = 'zero3'           # 'zero3' (gather once per step) | 'zero3_mb' (regather per microbatch) | 'zero1' | 'ddp'
    recompute: bool = True
    tp_chunks: int = 1               # >1: token-chunked comm/compute pipelining inside a phase (needs row-tile ops: WHAT-IF)
    lpt: int = 1                     # layers per simulated task
    inflight: str = '1f1b'           # '1f1b' | 'gpipe'
    grad_bytes: int = 2              # wire bytes per gradient element (2 = bf16, 1 = fp8 WHAT-IF)
    param_bytes: int = 2             # wire bytes per gathered parameter
    act_bytes: int = 2               # wire bytes per activation element across pipeline stages (1 = fp8 WHAT-IF)
    window_extra: int = 0            # extra microbatches allowed in flight beyond 1F1B's pp - stage (hides WAN round trips; costs activation memory)
    a2a_scale: float = 1.0           # multiplier on expert-parallel all-to-all bytes (node-limited routing / fp8 dispatch WHAT-IF)
    overlap: bool = True             # False: every communication task runs on the compute stream (a serialized strawman)

    @property
    def G(self): return self.tp * self.pp * self.dp

    def rank(self, pp, dp, tp):
        size = dict(pp=self.pp, dp=self.dp, tp=self.tp)
        idx = dict(pp=pp, dp=dp, tp=tp)
        r = 0
        for ax in self.order:
            r = r * size[ax] + idx[ax]
        return r


# ------------------------------------------------------------------------------------------------------------- DES
class Graph:
    """Generic task graph: each task runs on ONE stream for `dur` microseconds; successors see it `lat` after it ends."""
    def __init__(self):
        self.stream, self.dur, self.prio, self.lat, self.kind, self.tag, self.deps = [], [], [], [], [], [], []

    def add(self, stream, dur, deps=(), prio=0, kind='comp', lat=0.0, tag=None):
        self.stream.append(stream); self.dur.append(float(dur)); self.prio.append(prio); self.lat.append(float(lat))
        self.kind.append(kind); self.tag.append(tag); self.deps.append([d for d in deps if d is not None])
        return len(self.dur) - 1

    def __len__(self): return len(self.dur)


def simulate(g: Graph):
    """Event-driven list scheduler. A stream runs one task at a time, highest priority first, FIFO within a priority."""
    n = len(g)
    indeg = [len(d) for d in g.deps]
    succ = [[] for _ in range(n)]
    for i, ds in enumerate(g.deps):
        for d in ds:
            succ[d].append(i)
    ready_at = [0.0] * n
    start, end = [-1.0] * n, [-1.0] * n
    queues, busy = {}, {}
    ev, seq = [], itertools.count()
    for i in range(n):
        if indeg[i] == 0:
            heapq.heappush(ev, (0.0, 0, next(seq), i))        # (time, 0=ready / 1=done / 2=stream free)

    def try_start(s, now):
        q = queues.get(s)
        while q and busy.get(s, 0.0) <= now:
            _, _, _, i = heapq.heappop(q)
            start[i], end[i] = now, now + g.dur[i]
            busy[s] = end[i]
            heapq.heappush(ev, (end[i], 1, next(seq), i))
            return

    done = 0
    while ev:
        now, typ, _, i = heapq.heappop(ev)
        if typ == 2:                                  # a stream became free: successors released at this instant are queued first
            try_start(i, now)
        elif typ == 0:
            s = g.stream[i]
            heapq.heappush(queues.setdefault(s, []), (-g.prio[i], now, next(seq), i))
            heapq.heappush(ev, (now, 2, next(seq), s))
        else:
            done += 1
            s = g.stream[i]
            for j in succ[i]:
                indeg[j] -= 1
                ready_at[j] = max(ready_at[j], now + g.lat[i])
                if indeg[j] == 0:
                    heapq.heappush(ev, (ready_at[j], 0, next(seq), j))
            heapq.heappush(ev, (now, 2, next(seq), s))
    if done != n:
        raise RuntimeError(f'deadlock: {n - done} of {n} tasks never ran')
    return start, end


# ------------------------------------------------------------------------------------------------------------- lowering
@dataclass
class Result:
    makespan_us: float
    tasks: int
    stage: list
    memory: dict
    metrics: dict
    detail: dict = field(default_factory=dict)


def lower(model: ModelCfg, par: Par, topo: Topology, hw: Hw):
    """Transformer training step -> Graph (+ bookkeeping). One representative (dp=0, tp=0) per stage unless explicit."""
    P, M, S = par, par.microbatches, par.pp
    assert P.G <= topo.n, f'{P.G} GPUs do not fit a topology of {topo.n}'
    assert P.global_seqs % (P.dp * M) == 0, 'global batch must divide dp x microbatches'
    b = P.global_seqs // (P.dp * M)                              # sequences per microbatch per DP rank
    toks = b * model.seq
    ph = model.phases()
    h2 = model.hidden * 2
    # --- stage assignment: the generic linear-partition DP over per-layer cost (+ the embedding / head on the ends)
    lf = sum(p[1] for p in ph)
    head_flops = 2 * model.vocab * model.hidden
    cost = [lf] * model.layers
    cost[-1] += head_flops
    cuts = pt.balance_stages(cost, S)
    stage_layers = [hi - lo for lo, hi in cuts]
    assert len(stage_layers) == S and min(stage_layers) >= 1, f'cannot place {model.layers} layers on {S} stages'
    flops_us = lambda f: f / (hw.peak_tflops * 1e6 * hw.kernel_eff)         # flops -> microseconds
    tax = 1.0 + hw.comm_sm_tax
    members = [(0, 0)]
    # --- groups and their physical levels (from the rank mapping)
    tp_levels = topo.levels([P.rank(0, 0, t) for t in range(P.tp)]) if P.tp > 1 else []
    dp_levels = topo.levels([P.rank(0, d, 0) for d in range(P.dp)]) if P.dp > 1 else []
    ep = P.dp if model.kind == 'moe' else 1
    if model.kind == 'moe':
        assert P.ep in (0, 1, P.dp), 'ep must equal dp in this model (experts partitioned over the whole DP group)'
        assert model.experts % P.dp == 0
    ep_levels = dp_levels if model.kind == 'moe' else []
    # a stage boundary (s -> s+1): tier of the rank pair, WAN sharers = the flows crossing at once
    def boundary(s):
        a, c = P.rank(s, 0, 0), P.rank(s + 1, 0, 0)
        return topo.tier(a, c)
    # --- bytes and flops per chunk
    act_b = toks * h2 / P.tp                                              # one sequence-parallel activation shard, bytes
    pp_b = toks * model.hidden * P.act_bytes / P.tp                       # what crosses a stage boundary
    layer_p = model.layer_params()
    g = Graph()
    # a collective that crosses the WAN shares the datacenter-pair pipe with every GPU of its datacenter doing the same (all groups run in lockstep)
    ct = lambda lv, kind, nb: coll_time_us(topo, lv, kind, nb, sharers=max(1, P.G // max(1, next((f for t_, f in lv if t_ == WAN), 1))))
    PR_B, PR_R, PR_F, PR_AG = 30, 25, 20, 10
    pr = lambda base, mb: base * 1_000_000 - mb          # class first, then the older microbatch (depth-first flow)
    # per-stage chunk list
    chunks = []
    for s in range(S):
        n = stage_layers[s]
        cs = [min(P.lpt, n - i) for i in range(0, n, P.lpt)]
        chunks.append(cs)
    info = dict(b=b, toks=toks, stage_layers=stage_layers, chunks=chunks)
    ags, fwd_first, fwd_last, bwd_first, bwd_last, stored_ev = {}, {}, {}, {}, {}, []
    comm_ids, comp_ids = [], []
    recv_f, recv_b = {}, {}                                              # (s, mb, m) -> task that makes the activation/grad visible
    opt_done = []

    def tp_comm(s, m, nl, deps, prio, k):
        """The tensor-parallel reduce-scatter + all-gather of one phase (sequence parallel), nl layers, 1/k of the tokens."""
        if P.tp <= 1: return deps
        t = ct(tp_levels, 'rs', act_b / k) + ct(tp_levels, 'ag', act_b / k)
        i = g.add((s, m, 'intra'), t * nl, deps, prio, 'comm', tag='tp')
        comm_ids.append(i)
        return [i]

    def sync(group_deps):
        return g.add(('sync',), 0.0, group_deps, 0, 'sync')

    def run_pass(s, m, c, nl, passes_mul, prev, prio, mb):
        """One pass (fwd = 1x, recompute = 1x, bwd = 2x flops) of chunk c over all phases, token-chunked k ways. prev: list per token chunk."""
        k = P.tp_chunks
        cur = list(prev)
        order = ph if passes_mul != 2 else ph[::-1]
        for name, fl, tpf, a2af in order:
            for j in range(k):
                deps = list(cur[j])
                if a2af and ep_levels:
                    a2a_b = toks * a2af * h2 * P.a2a_scale / P.tp / k
                    # members of one EP group must reach the all-to-all together
                    d1 = g.add((s, m, 'net'), ct(ep_levels, 'a2a', a2a_b) * nl, deps, prio, 'comm', tag='ep')
                    comm_ids.append(d1)
                    deps = [d1]
                dur = flops_us(fl * toks / P.tp / k * passes_mul * nl) * tax
                c1 = g.add((s, m, 'comp'), dur, deps, prio, 'comp', tag=name)
                comp_ids.append(c1)
                out = [c1]
                out = tp_comm(s, m, nl, out, prio, k) if tpf else out
                if a2af and ep_levels:
                    d2 = g.add((s, m, 'net'), ct(ep_levels, 'a2a', a2a_b) * nl, out, prio, 'comm', tag='ep')
                    comm_ids.append(d2)
                    out = [d2]
                cur[j] = out
        return cur

    # --- weights / optimizer comm per chunk (stage-local, per DP rank shard)
    def chunk_param_bytes(s, c, nl):
        # bf16 bytes of the chunk's weights held by one TP rank (experts already partitioned over EP)
        if model.kind == 'dense': return nl * layer_p / P.tp
        h = model.hidden
        return nl * (4 * h * h / P.tp)                      # dense (attention) part is gathered; experts stay local

    for s in range(S):
        m = members[0]
        nch = len(chunks[s])
        # ZeRO-3 gather once per step: all chunks queued on the NIC stream in layer order (prefetch = the order itself)
        if P.dp_mode == 'zero3' and P.dp > 1:
            prev = None
            for c in range(nch):
                nb = chunk_param_bytes(s, c, chunks[s][c]) * P.param_bytes
                t = ct(dp_levels, 'ag', nb)
                ags[s, c] = g.add((s, m, 'net'), t, [prev], pr(PR_AG, 0), 'comm', tag='zero_ag')
                comm_ids.append(ags[s, c]); prev = ags[s, c]
    # microbatch flow
    comp_end_f, comp_end_b = {}, {}
    send_f, send_b = {}, {}
    done_b = {}                                   # (s, mb) -> last bwd task (the window edge)
    tail = {}
    for mb in range(M):
        for s in range(S):
            m = members[0]
            nch = len(chunks[s])
            W = ((S - s) + P.window_extra) if P.inflight == '1f1b' else M
            wdep = done_b.get((s, mb - W)) if mb - W >= 0 else None
            # ---- forward
            incoming = send_f.get((s - 1, mb)) if s > 0 else None
            cur = [[x for x in (incoming, wdep) if x is not None] for _ in range(P.tp_chunks)]
            first = None
            for c in range(nch):
                nl = chunks[s][c]
                deps_extra = []
                if P.dp_mode == 'zero3' and P.dp > 1 and mb == 0:
                    deps_extra = [ags[s, c]]
                if P.dp_mode == 'zero3_mb' and P.dp > 1:
                    nb = chunk_param_bytes(s, c, nl) * P.param_bytes
                    a = g.add((s, m, 'net'), ct(dp_levels, 'ag', nb), [wdep] if c < 2 else [comp_end_f[s, mb, c - 2]], pr(PR_AG + 1, mb), 'comm', tag='zero_ag')
                    comm_ids.append(a); deps_extra = [a]
                cur = [list(cur[j]) + deps_extra for j in range(P.tp_chunks)]
                cur = run_pass(s, m, c, nl, 1, cur, pr(PR_F, mb), mb)
                comp_end_f[s, mb, c] = cur[0][0] if cur[0] else None
                stored_ev.append((s, mb, c, nl))
            fwd_last[s, mb] = [x for l in cur for x in l]
            if s < S - 1:
                bnd = boundary(s)
                nbytes = pp_b
                sharers = P.tp * P.dp
                t = nbytes / (topo.bw_gbs(RACK if bnd == WAN else bnd) * 1e3)
                snd = g.add((s, m, 'p2p'), t, fwd_last[s, mb], pr(PR_F + 3, mb), 'comm', tag='pp_send')
                comm_ids.append(snd)
                if bnd == WAN:
                    wire = g.add(('wan', s, 'f'), nbytes * sharers / (topo.wan_gbps / 8.0 * 1e3), [snd], pr(PR_F, mb), 'wan', lat=topo.wan_ms * 1e3, tag='wan')
                    send_f[s, mb] = wire
                else:
                    send_f[s, mb] = g.add((s, m, 'p2p'), 0.0, [snd], pr(PR_F, mb), 'comm', lat=topo.lat_us(bnd), tag='pp_lat')
            # last stage runs the head: logits matmul + loss as part of its last chunk's tail
        # ---- backward (reverse stage order for this microbatch)
        for s in reversed(range(S)):
            m = members[0]
            nch = len(chunks[s])
            incoming = send_b.get((s + 1, mb)) if s < S - 1 else None
            base = [x for x in ([incoming] + fwd_last[s, mb]) if x is not None]
            if s == S - 1:
                hd = g.add((s, m, 'comp'), flops_us(3 * head_flops * toks / P.tp) * tax, base, pr(PR_B, mb), 'comp', tag='head')
                comp_ids.append(hd)
                base = [hd]
            cur = [list(base) for _ in range(P.tp_chunks)]
            for c in reversed(range(nch)):
                nl = chunks[s][c]
                if P.dp_mode == 'zero3_mb' and P.dp > 1:
                    nb = chunk_param_bytes(s, c, nl) * P.param_bytes
                    a = g.add((s, m, 'net'), ct(dp_levels, 'ag', nb), list(base), pr(PR_AG + 2, mb), 'comm', tag='zero_ag')
                    comm_ids.append(a)
                    cur = [list(x) + [a] for x in cur]
                if P.recompute:
                    cur = run_pass(s, m, c, nl, 1, cur, pr(PR_R, mb), mb)
                cur = run_pass(s, m, c, nl, 2, cur, pr(PR_B, mb), mb)
                comp_end_b[s, mb, c] = cur[0][0]
                # gradient reduction after the LAST microbatch's backward of this chunk
                if mb == M - 1 and P.dp > 1:
                    gb = chunk_param_bytes(s, c, nl) * P.grad_bytes
                    kind = 'ar' if P.dp_mode == 'ddp' else 'rs'
                    t = ct(dp_levels, kind, gb)
                    r = g.add((s, m, 'net'), t, [x for l in cur for x in l], pr(PR_AG + 5, mb), 'comm', tag='grad_rs')
                    comm_ids.append(r)
                    tail.setdefault(s, []).append((c, r))
                elif mb == M - 1:
                    tail.setdefault(s, []).append((c, [x for l in cur for x in l][0]))
            bw_last = [x for l in cur for x in l]
            done_b[s, mb] = g.add(('sync',), 0.0, bw_last, 0, 'sync')
            if s > 0:
                bnd = boundary(s - 1)
                nbytes = pp_b
                t = nbytes / (topo.bw_gbs(RACK if bnd == WAN else bnd) * 1e3)
                snd = g.add((s, m, 'p2p'), t, bw_last, pr(PR_B + 3, mb), 'comm', tag='pp_send')
                comm_ids.append(snd)
                if bnd == WAN:
                    send_b[s, mb] = g.add(('wan', s - 1, 'b'), nbytes * P.tp * P.dp / (topo.wan_gbps / 8.0 * 1e3), [snd], pr(PR_B, mb), 'wan', lat=topo.wan_ms * 1e3, tag='wan')
                else:
                    send_b[s, mb] = g.add((s, m, 'p2p'), 0.0, [snd], pr(PR_B, mb), 'comm', lat=topo.lat_us(bnd), tag='pp_lat')
    # optimizer step per chunk (HBM bound: read+write 16 B/param of the local shard) and, for zero1/ddp-sharded, the param gather
    finals = []
    for s in range(S):
        m = members[0]
        for c, r in tail.get(s, []):
            nl = chunks[s][c]
            shard = nl * layer_p / (P.tp * P.dp) if model.kind == 'dense' else nl * (4 * model.hidden ** 2 / P.tp / P.dp + model.experts * 3 * model.hidden * model.expert_ffn / P.tp / P.dp)
            o = g.add((s, m, 'comp'), shard * 32 / (hw.hbm_gbs * 1e3), [r], pr(PR_AG + 6, 0), 'comp', tag='opt')
            comp_ids.append(o)
            fin = o
            if P.dp_mode == 'zero1' and P.dp > 1:
                nb = chunk_param_bytes(s, c, nl) * P.param_bytes
                fin = g.add((s, m, 'net'), ct(dp_levels, 'ag', nb), [o], pr(PR_AG, 0), 'comm', tag='zero_ag')
                comm_ids.append(fin)
            finals.append(fin)
    info.update(finals=finals, comm_ids=comm_ids, comp_ids=comp_ids, stored_ev=stored_ev, comp_end_f=comp_end_f, comp_end_b=comp_end_b,
                b=b, toks=toks, act_b=act_b, head_flops=head_flops, stage_layers=stage_layers)
    if not P.overlap:
        g.stream = [(st[0], st[1], 'comp') if (len(st) == 3 and st[2] in ('intra', 'net', 'p2p')) else st for st in g.stream]
    return g, info


COMM_TAGS = {'tp', 'ep', 'zero_ag', 'grad_rs', 'pp_send'}
COLLECTIVE_TAGS = {'tp': 't_group', 'ep': 'd_group', 'zero_ag': 'd_group', 'grad_rs': 'd_group'}


def replicate(g: Graph, tp: int, dp: int, slow=None):
    """Explicit mode: instantiate the reduced graph for EVERY (dp, tp) rank. A collective task becomes one task per rank behind a
    zero-cost barrier over its group (tp group = same dp index; dp/ep group = same tp index); the shared WAN pipe stays one task
    whose dependencies are the union of every copy's. `slow` = {(d, t): factor} stretches that rank's compute (a straggler)."""
    slow = slow or {}
    K = [(d, t) for d in range(dp) for t in range(tp)]
    out = Graph()
    n = len(g)
    mapid = {}
    for k, (d, t) in enumerate(K):
        for i in range(n):
            st = g.stream[i]
            if g.kind[i] == 'wan' or st[0] == 'sync':
                continue
            nst = (st[0], (d, t), st[2])
            dur = g.dur[i] * (slow.get((d, t), 1.0) if g.kind[i] == 'comp' else 1.0)
            mapid[k, i] = out.add(nst, dur, [], g.prio[i], g.kind[i], g.lat[i], g.tag[i])
    for i in range(n):                                      # shared tasks (WAN pipe, zero-cost syncs): one instance
        if g.kind[i] == 'wan' or g.stream[i][0] == 'sync':
            for k in range(len(K)):
                pass
            mapid['shared', i] = out.add(g.stream[i], g.dur[i], [], g.prio[i], g.kind[i], g.lat[i], g.tag[i])
    ref = lambda k, j: mapid[k, j] if (k, j) in mapid else mapid['shared', j]
    for i in range(n):
        shared = ('shared', i) in mapid
        if shared:
            out.deps[mapid['shared', i]] = sorted({ref(k, j) for k in range(len(K)) for j in g.deps[i]})
            continue
        grp = COLLECTIVE_TAGS.get(g.tag[i])
        for k in range(len(K)):
            out.deps[mapid[k, i]] = [ref(k, j) for j in g.deps[i]]
        if grp:
            groups = {}
            for k, (d, t) in enumerate(K):
                groups.setdefault(d if grp == 't_group' else t, []).append(k)
            for members in groups.values():
                if len(members) < 2: continue
                bar = out.add(('sync',), 0.0, sorted({x for k in members for x in out.deps[mapid[k, i]]}), 0, 'sync')
                for k in members:
                    out.deps[mapid[k, i]] = [bar]
    return out


def run(model: ModelCfg, par: Par, topo: Topology, hw: Hw) -> Result:
    g, info = lower(model, par, topo, hw)
    start, end = simulate(g)
    T = max(end)
    S, M = par.pp, par.microbatches
    # ---- per-stage timelines: compute busy, exposed communication (compute idle, comm active), idle otherwise
    stage_rows = []
    for s in range(S):
        comp = sorted((start[i], end[i]) for i in range(len(g)) if g.stream[i][0] == s and g.kind[i] == 'comp' and g.dur[i] > 0)
        comm = sorted((start[i], end[i]) for i in range(len(g)) if g.stream[i][0] == s and g.tag[i] in COMM_TAGS and g.kind[i] == 'comm' and g.dur[i] > 0)
        busy = sum(b - a for a, b in comp)
        comm_total = sum(b - a for a, b in comm)
        union = _union(comm)
        cu = _union(comp)
        both = _overlap(cu, union)
        union_len = sum(b - a for a, b in union)
        exposed = union_len - both
        stage_rows.append(dict(stage=s, layers=info['stage_layers'][s], compute_us=busy, comm_busy_us=comm_total, comm_union_us=union_len,
                               hidden_comm_us=both, exposed_comm_us=exposed, idle_us=T - busy - exposed))
    crit = max(stage_rows, key=lambda r: r['compute_us'])
    busy_max = crit['compute_us']
    lb_pipe = busy_max * (M + S - 1) / M
    # MFU = 6 N_active D + attention-score flops (12 s h L per token), no credit for recomputation; HFU adds the recompute forward
    h, s_ = model.hidden, model.seq
    D = par.global_seqs * s_
    model_flops = 6 * model.active_params() * D + 12 * s_ * h * model.layers * D
    hw_flops = model_flops + ((2 * model.active_params() + 4 * s_ * h * model.layers) * D if par.recompute else 0)
    mfu = model_flops / (par.G * hw.peak_tflops * 1e12 * (T * 1e-6))
    hfu = hw_flops / (par.G * hw.peak_tflops * 1e12 * (T * 1e-6))
    mem = memory(model, par, topo, hw, info, g, start, end)
    metrics = dict(
        step_s=T * 1e-6, mfu=mfu, hfu=hfu, tflops_per_gpu_hw=hw_flops / par.G / (T * 1e-6) / 1e12, tokens_per_s=par.global_seqs * s_ / (T * 1e-6),
        compute_busy_s=busy_max * 1e-6, lb_perfect_overlap_s=busy_max * 1e-6, lb_pipeline_s=lb_pipe * 1e-6,
        bubble_theory=(S - 1) / (M + S - 1), bubble_measured=1 - busy_max / T,
        exposed_comm_frac=crit['exposed_comm_us'] / T, idle_frac=crit['idle_us'] / T,
        comm_hidden_frac=(crit['hidden_comm_us'] / crit['comm_union_us']) if crit['comm_union_us'] else 1.0,
        over_pipeline_lb=T / lb_pipe, over_perfect_lb=T / busy_max)
    return Result(T, len(g), stage_rows, mem, metrics, dict(stage_layers=info['stage_layers']))


def _union(iv):
    out = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def _overlap(u1, u2):
    i = j = 0
    tot = 0.0
    while i < len(u1) and j < len(u2):
        lo, hi = max(u1[i][0], u2[j][0]), min(u1[i][1], u2[j][1])
        if hi > lo: tot += hi - lo
        if u1[i][1] < u2[j][1]: i += 1
        else: j += 1
    return tot


def memory(model, par, topo, hw, info, g, start, end):
    """Per-GPU bytes by component for every stage (max reported). Activation storage uses the PEAK live set the DES produced."""
    S, G = par.pp, par.G
    N = model.params()
    lp = model.layer_params()
    h = model.hidden
    out = []
    for s in range(S):
        nl = info['stage_layers'][s]
        if model.kind == 'dense':
            stage_params = nl * lp + (model.vocab * h if s in (0, S - 1) else 0)
            expert_p = 0
        else:
            expert_p = nl * model.experts * 3 * h * model.expert_ffn
            stage_params = nl * 4 * h * h + expert_p + (model.vocab * h if s in (0, S - 1) else 0)
        dense_p = stage_params - expert_p
        # persistent state
        if par.dp_mode in ('zero3', 'zero3_mb'):
            state = 16 * stage_params / (par.tp * par.dp)
        elif par.dp_mode == 'zero1':
            state = (2 + 2) * stage_params / par.tp + 12 * stage_params / (par.tp * par.dp)
        else:
            state = 16 * stage_params / par.tp
        # gathered bf16 weights resident (dense part; experts are local and already counted in state as bf16 shard)
        if par.dp_mode == 'zero3':
            gathered = par.param_bytes * dense_p / par.tp
        elif par.dp_mode == 'zero3_mb':
            gathered = 2 * par.param_bytes * dense_p / par.tp / max(1, nl) * par.lpt
        else:
            gathered = 0.0
        # activations: stored chunk inputs, peak live count from the schedule
        ev = []
        for (st, mb, c, n_l) in info['stored_ev']:
            if st != s: continue
            tf = end[_find(info['comp_end_f'], st, mb, c)]
            tb = end[_find(info['comp_end_b'], st, mb, c)] if (st, mb, c) in info['comp_end_b'] else tf
            ev.append((tf, +n_l)); ev.append((tb, -n_l))
        ev.sort()
        cur = peak = 0
        for _, d in ev:
            cur += d; peak = max(peak, cur)
        stored = peak * info['toks'] * h * 2 / par.tp if par.recompute else peak * 34 * info['toks'] * h / par.tp
        workspace = 34 * info['toks'] * h / par.tp * (1 if par.recompute else 0)
        logits = info['toks'] * model.vocab * 6 / par.tp if s == S - 1 else 0.0
        moe_buf = 4 * info['toks'] * model.topk * h * 2 / par.tp if model.kind == 'moe' else 0.0
        comm_buf = 2 * info['act_b'] * 2
        parts = dict(state=state, gathered_weights=gathered, stored_activations=stored, recompute_workspace=workspace, logits=logits,
                     moe_dispatch=moe_buf, comm_buffers=comm_buf, cuda_context=hw.ctx_gb * 1e9)
        sub = sum(parts.values())
        parts['fragmentation_5pct'] = 0.05 * sub
        out.append(dict(stage=s, total_gb=(sub * 1.05) / 1e9, parts_gb={k: v / 1e9 for k, v in parts.items()}, peak_live_layers=peak))
    worst = max(out, key=lambda r: r['total_gb'])
    return dict(max_gb=worst['total_gb'], worst_stage=worst['stage'], parts_gb=worst['parts_gb'], peak_live_layers=worst['peak_live_layers'],
                fits=worst['total_gb'] <= hw.hbm_gb, per_stage_gb=[r['total_gb'] for r in out])


def _find(d, s, mb, c):
    return d[s, mb, c]


# ------------------------------------------------------------------------------------------------------------- scenarios
def make_topology(gpus, wan_gbps=100.0, wan_ms=20.0, n_dc=4, **kw):
    per_dc = gpus // n_dc
    nodes_per_rack = 32
    racks = max(1, per_dc // (8 * nodes_per_rack))
    if racks * 8 * nodes_per_rack != per_dc:
        nodes_per_rack = max(1, per_dc // (8 * racks)); racks = max(1, per_dc // (8 * nodes_per_rack))
    return Topology(8, nodes_per_rack, racks, n_dc, wan_gbps=wan_gbps, wan_ms=wan_ms, **kw)


def fmt(r: Result, par: Par, model: ModelCfg):
    m, mem = r.metrics, r.memory
    return (f"{model.name or model.kind}: {model.params()/1e12:.2f}T params ({model.active_params()/1e9:.0f}B active), {par.G} GPUs = TP{par.tp} x PP{par.pp} x DP{par.dp}, "
            f"M={par.microbatches}, {par.dp_mode}\n"
            f"  step {m['step_s']:.2f} s, MFU {m['mfu']:.1%}, {m['tokens_per_s']/1e6:.2f} Mtok/s | LB: perfect-overlap {m['lb_perfect_overlap_s']:.2f} s, "
            f"+pipeline bubble {m['lb_pipeline_s']:.2f} s | actual/LB_pipe {m['over_pipeline_lb']:.3f}\n"
            f"  critical stage: exposed comm {m['exposed_comm_frac']:.1%} of step, other idle {m['idle_frac']:.1%}, comm hidden under compute {m['comm_hidden_frac']:.1%}; "
            f"bubble measured {m['bubble_measured']:.1%} (theory {m['bubble_theory']:.1%})\n"
            f"  memory/GPU {mem['max_gb']:.1f} GB (limit 80: {'fits' if mem['fits'] else 'DOES NOT FIT'}), peak live layer-inputs {mem['peak_live_layers']}; tasks {r.tasks}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--model', choices=['dense', 'moe'], default='dense')
    ap.add_argument('--gpus', type=int, default=4096)
    ap.add_argument('--tp', type=int, default=8)
    ap.add_argument('--pp', type=int, default=16)
    ap.add_argument('--microbatches', type=int, default=64)
    ap.add_argument('--global-seqs', type=int, default=4096)
    ap.add_argument('--dp-mode', default='zero3')
    ap.add_argument('--wan-gbps', type=float, default=100.0)
    ap.add_argument('--wan-ms', type=float, default=20.0)
    ap.add_argument('--dcs', type=int, default=4)
    ap.add_argument('--tp-chunks', type=int, default=1)
    ap.add_argument('--lpt', type=int, default=2)
    ap.add_argument('--json', action='store_true')
    a = ap.parse_args(argv)
    model = dense_1t() if a.model == 'dense' else moe_1t()
    dp = a.gpus // (a.tp * a.pp)
    par = Par(a.tp, a.pp, dp, a.microbatches, a.global_seqs, ep=dp if a.model == 'moe' else 1, dp_mode=a.dp_mode, tp_chunks=a.tp_chunks, lpt=a.lpt)
    topo = make_topology(a.gpus, a.wan_gbps, a.wan_ms, a.dcs)
    r = run(model, par, topo, Hw())
    if a.json:
        print(json.dumps(dict(metrics=r.metrics, memory=r.memory, stage=r.stage), indent=1))
    else:
        print(fmt(r, par, model))
    return 0


if __name__ == '__main__':
    sys.exit(main())
