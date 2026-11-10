"""KVM graph partitioner: one ABI-v2 workload + a cluster + a strategy -> one sub-workload per device, with communication tasks
inserted and cross-device version variables assigned. No NCCL: collectives are ordinary tasks (dist/commops.py, include/
kvm_ops_comm.def) in the same DAG, so communication overlaps compute at tile granularity inside the persistent kernel.

    python3 runtime/partition.py workload.json cluster.json strategy.json out_dir [--plan]

Two planning modes (strategy["mode"]):

  shard  GSPMD-style layout algebra. Every buffer has one layout: R (replicated), ROWS / COLS (equal blocks of axis 0 / 1,
         one per device) or PART (every device holds an addend; the value is their sum). Each op has rules for which operand
         layouts it accepts and which layout it produces; where an operand is in the wrong layout the planner inserts the
         resharding collective (all-gather, reduce-scatter, all-reduce, all-to-all, local slice), choosing the cheapest legal
         operand layouts by communication volume. Data parallel, ZeRO/FSDP (parameters ROWS: gathered per use, gradients
         reduce-scattered), Megatron tensor parallel (W1 COLS, W2 ROWS -> PART -> all-reduce) and sequence/expert-style
         all-to-all (ROWS <-> COLS) are all just different initial layouts.
  place  task-to-device placement: explicit map, balanced pipeline stages (cost model, linear-partition DP) or HEFT. A buffer
         lives where it was last written; a consumer elsewhere gets a transferred copy. `microbatches` clones the graph per
         microbatch first, which is what makes pipeline stages overlap.

Dependencies are derived by replaying the program order and tracking read/write hazards per buffer row range (RAW, WAR, WAW),
which is a superset of the author's `after` edges. Collectives can be cut into `chunks` independent rings whose landings are
tracked separately, so a consumer that reads only some rows waits only for those chunks; the core registry's ops all start at
element 0, so the planner itself does not yet split compute ops into row tiles (offset-capable tile ops such as gptops'
matmul_tile are what that needs; the tracker is already row-range exact). Overlap that IS built: microbatch pipelining and
chunked collectives running beside unrelated compute. Cross-device data only ever lands in FRESH single-assignment buffers, so a
remote write can never race a local reader; each landing publishes a one-shot version variable (version 1 in the receiver's
inbox) that the consumers wait on (policy at_least). See DISTRIBUTED.md for the protocol and what is and is not tested.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'runtime'), str(ROOT / 'dist')]
import commops                       # noqa: E402
import compile_workload as cw        # noqa: E402
import ref_ops                       # noqa: E402

commops.install()
log = logging.getLogger('kvm.partition')

R, ROWS, COLS, PART = 'R', 'ROWS', 'COLS', 'PART'
LAYOUT_NAMES = {'repl': R, 'r': R, 'rows': ROWS, 'cols': COLS, 'part': PART, 'partial': PART}


class PartitionError(ValueError):
    """The workload cannot be partitioned as requested; the message names the task/buffer."""


def _need(c, m):
    if not c:
        raise PartitionError(m)


# ------------------------------------------------------------------------------------------------ cluster and cost model
@dataclass
class Link:
    kind: str            # 'p2p' (peer pointers, NVLink/PCIe) or 'proxy' (host proxy + transport, any network)
    bw_gbs: float        # GB/s one direction
    lat_us: float


@dataclass
class Cluster:
    node_of: list[int]
    intra: Link = field(default_factory=lambda: Link('p2p', 300.0, 2.0))
    inter: Link = field(default_factory=lambda: Link('proxy', 12.5, 15.0))
    tflops: float = 19.5           # per device, fp32 (A100 non-tensor); bf16/TF32 would be higher
    hbm_gbs: float = 1500.0
    task_overhead_us: float = 1.4  # measured executor dispatch (results/2026-11-03-a100-exec)
    comm_lanes: int = 4            # worker blocks a device can spend on communication at once (model)
    max_msg_bytes: int = 256 * 1024   # proxy send-queue slot payload
    gbytes: float = 40.0

    @property
    def n(self):
        return len(self.node_of)

    def link(self, a, b):
        return self.intra if self.node_of[a] == self.node_of[b] else self.inter

    def xfer_us(self, a, b, nbytes):
        l = self.link(a, b)
        return l.lat_us + nbytes / (l.bw_gbs * 1e3)

    @staticmethod
    def uniform(n, nodes=1, **kw):
        _need(n % nodes == 0, f'{n} devices do not split into {nodes} nodes')
        return Cluster([i // (n // nodes) for i in range(n)], **kw)

    @staticmethod
    def from_dict(d):
        """{"devices": [{"node": 0}, ...], "links": {"intra": {"kind": "p2p", "bandwidth_gbs": 300, "latency_us": 2},
        "inter": {"kind": "proxy", ...}}, "tflops": 19.5, "hbm_gbs": 1500, "comm_lanes": 4, "max_msg_bytes": 262144}"""
        node_of = [dv.get('node', 0) for dv in d['devices']]
        kw = {k: d[k] for k in ('tflops', 'hbm_gbs', 'task_overhead_us', 'comm_lanes', 'max_msg_bytes', 'gbytes') if k in d}
        for k, key in (('intra', 'intra'), ('inter', 'inter')):
            if key in d.get('links', {}):
                l = d['links'][key]
                kw[k] = Link(l['kind'], l.get('bandwidth_gbs', l.get('bw_gbs')), l.get('latency_us', l.get('lat_us')))
        return Cluster(node_of, **kw)


def op_cost_us(op, a, cl: Cluster):
    """Roofline-ish microseconds for one task on one device: max(flops, bytes) plus dispatch overhead."""
    f32 = 4
    if op == 'matmul':
        flops, byt = 2.0 * a['m'] * a['n'] * a['k'], f32 * (a['m'] * a['k'] + a['k'] * a['n'] + a['m'] * a['n'])
    elif op in ('put', 'put_signal', 'send_proxy', 'signal'):
        return cl.task_overhead_us
    elif op == 'noop':
        return cl.task_overhead_us
    elif op == 'adamw':                                   # 4 streams read, 3 written, ~20 flops per element
        n = a['e1'] - a['e0']
        return cl.task_overhead_us + max(20.0 * n / (cl.tflops * 1e6), 7 * f32 * n / (cl.hbm_gbs * 1e3))
    else:
        n = a.get('n', a.get('rows', 1) * a.get('cols', 1))
        flops, byt = 4.0 * n, 3 * f32 * n
    return cl.task_overhead_us + max(flops / (cl.tflops * 1e6), byt / (cl.hbm_gbs * 1e3))


# ------------------------------------------------------------------------------------------------ workload IR
OUT_ONLY = {'copy': ['dst'], 'add': ['out'], 'scale': ['out'], 'relu': ['out'], 'predict': ['out'],
            'mse_grad': ['loss', 'grad'], 'matmul': ['out'], 'fill': ['x']}
ELEMENTWISE = {'copy': ('n', ['src'], ['dst'], True), 'add': ('n', ['a', 'b'], ['out'], True),
               'scale': ('n', ['x'], ['out'], True), 'relu': ('n', ['x'], ['out'], False),
               'sgd': ('n', ['w', 'g'], ['w'], False), 'predict': ('n', ['x'], ['out'], False),
               'fill': ('n', [], ['x'], False)}      # name -> (count arg, inputs, outputs, linear in its inputs)


@dataclass
class Prog:
    buffers: dict            # name -> dict(rows, cols, values)
    variables: list
    tasks: list              # program order; dicts with id, op, args, after(list of ids), priority, publishes, reads, deadline_us


def parse_program(spec, symbolic=False) -> Prog:
    """Flatten graphs into one program-ordered task list (ids 'graph.task'). Order respects `after` and version edges."""
    compiled = cw.compile_spec(copy.deepcopy(spec))        # full validation, and gives us the canonical buffer values
    # symbolic: nothing is materialised (init metadata only), for shapes whose values do not fit in Python lists
    bufs = {b.name: dict(rows=b.rows, cols=b.cols, dtype=b.dtype, init=b.init, values=None if symbolic else b.materialize())
            for b in compiled.buffers}
    tasks, idx = [], {}
    for g in spec['graphs']:
        for t in g['tasks']:
            nt = dict(t, id=f"{g['name']}.{t['id']}", gid=g['name'], after=[f"{g['name']}.{a}" for a in t.get('after', [])],
                      args=dict(t.get('args', {})), reads=list(t.get('reads', [])))
            idx[nt['id']] = len(tasks)
            tasks.append(nt)
    succ = [set() for _ in tasks]
    for j, t in enumerate(tasks):
        for a in t['after']:
            succ[idx[a]].add(j)
    pubs = {}                                                  # (var, version) -> task index, in program (file) order
    count = {}
    for i, t in enumerate(tasks):
        if t.get('publishes'):
            count[t['publishes']] = count.get(t['publishes'], 0) + 1
            pubs[(t['publishes'], count[t['publishes']])] = i
    for j, t in enumerate(tasks):
        for r in t['reads']:
            p = pubs.get((r['var'], r['version']))
            if r['version'] > 0 and p is not None and p != j:
                succ[p].add(j)                                  # publisher before reader
            nxt = pubs.get((r['var'], r['version'] + 1))
            if r['policy'] == 'exact' and nxt is not None and nxt != j:
                succ[j].add(nxt)                                # exact reader before the next publisher
    for var in {v for v, _ in pubs}:                            # consecutive publishers keep their order
        k = 1
        while (var, k + 1) in pubs:
            succ[pubs[(var, k)]].add(pubs[(var, k + 1)])
            k += 1
    indeg = [0] * len(tasks)
    for outs in succ:
        for j in outs:
            indeg[j] += 1
    order, ready = [], sorted(i for i, d in enumerate(indeg) if d == 0)
    while ready:
        i = ready.pop(0)
        order.append(i)
        for j in sorted(succ[i]):
            indeg[j] -= 1
            if indeg[j] == 0:
                ready.append(j)
                ready.sort()
    _need(len(order) == len(tasks), 'dependency cycle')
    return Prog(bufs, list(spec.get('variables', [])), [tasks[i] for i in order])


# ------------------------------------------------------------------------------------------------ hazard tracking
class Tracker:
    """Per physical buffer: who wrote / read which row ranges. A token is ('t', task_id) or ('w', in_var_name)."""

    def __init__(self):
        self.writes, self.reads = [], []

    @staticmethod
    def _ov(a0, a1, b0, b1):
        return a0 < b1 and b0 < a1

    def query(self, r0, r1, write):
        toks = [t for (a, b, t) in self.writes if self._ov(a, b, r0, r1)]
        if write:
            toks += [t for (a, b, t) in self.reads if self._ov(a, b, r0, r1)]
        return toks

    def commit_local(self, r0, r1, tok, write):
        if write:
            self.writes = [w for w in self.writes if not (r0 <= w[0] and w[1] <= r1)] + [(r0, r1, tok)]
            self.reads = [x for x in self.reads if not (r0 <= x[0] and x[1] <= r1)]
        else:
            self.reads.append((r0, r1, tok))

    def commit_remote(self, r0, r1, tok):       # fresh buffers only: accumulate, never prune (every landing must be awaited)
        self.writes.append((r0, r1, tok))


class ShardBuilder:
    def __init__(self, rank):
        self.rank = rank
        self.buffers: dict[str, dict] = {}
        self.variables: list[str] = []
        self.in_vars: list[str] = []
        self.tasks: list[dict] = []
        self.remotes: dict[str, dict] = {}
        self.trk: dict[str, Tracker] = {}
        self.ntmp = 0
        self.mirror: dict = {}

    def buf(self, name, rows, cols, values=None, dtype='f32', gen=None, place=None):
        if name in self.buffers:
            b = self.buffers[name]
            _need((b['rows'], b['cols']) == (rows, cols), f'shard {self.rank}: buffer {name!r} redeclared with another shape')
            return name
        self.buffers[name] = dict(name=name, rows=rows, cols=cols, values=values, dtype=dtype, gen=gen, place=place)
        self.trk[name] = Tracker()
        return name

    def tmp(self, base, rows, cols):
        self.ntmp += 1
        return self.buf(f'{base}~{self.ntmp}', rows, cols)

    def index(self, name):
        return list(self.buffers).index(name)

    def var(self, name):
        if name not in self.variables:
            self.variables.append(name)
        return self.variables.index(name)

    def inbox_var(self):
        name = f'in.{len(self.in_vars)}'
        self.in_vars.append(name)
        self.var(name)
        return name

    def remote_peer(self, other, phys):
        b = other.buffers[phys]
        name = f'@{other.rank}/{phys}'
        self.buf(name, b['rows'], b['cols'])
        self.remotes[name] = dict(kind='peer', rank=other.rank, buf=phys)
        return name

    def remote_inbox(self, other):
        name = f'@inbox/{other.rank}'
        self.buf(name, 1, 1)                  # cols patched to the receiver's variable count in finish()
        self.remotes[name] = dict(kind='inbox', rank=other.rank)
        return name

    def sendq(self):
        self.buf('@sendq', 1, 1)
        self.remotes['@sendq'] = dict(kind='sendq')
        return '@sendq'

    def add(self, op, args, acc, label=None, priority=0, reads=(), publishes=None, extra_after=(), deadline_us=-1):
        toks = []
        for phys, r0, r1, mode in acc:
            toks += self.trk[phys].query(r0, r1, mode == 'w')
        after, waits = [], []
        for t in toks:
            (after if t[0] == 't' else waits).append(t[1])
        after = sorted(set(after) | set(extra_after))
        rd = [dict(r) for r in reads]
        have = {r['var'] for r in rd}
        for w in sorted(set(waits)):
            if w not in have:
                rd.append(dict(var=w, policy='at_least', version=1))
        tid = f'{label or op}#{len(self.tasks)}'
        task = dict(id=tid, op=op, args=args, after=after, priority=priority, deadline_us=deadline_us)
        if rd:
            task['reads'] = rd
        if publishes:
            task['publishes'] = publishes
        self.tasks.append(task)
        for phys, r0, r1, mode in acc:
            self.trk[phys].commit_local(r0, r1, ('t', tid), mode == 'w')
        return tid

    def spec(self, nvars_of):
        for name, rem in self.remotes.items():
            if rem['kind'] == 'inbox':
                self.buffers[name]['cols'] = max(1, nvars_of(rem['rank']))
        def bdict(b):
            d = dict(name=b['name'], rows=b['rows'], cols=b['cols'],
                     init={'kind': 'values', 'values': b['values']} if b['values'] is not None else
                     {'kind': 'fill', 'value': (b['gen'] or {}).get('value', 0) if (b['gen'] or {}).get('kind') == 'fill' else 0})
            if b.get('dtype', 'f32') != 'f32':
                d['dtype'] = b['dtype']
            return d
        return {'variables': self.variables,
                'buffers': [bdict(b) for b in self.buffers.values()],
                'graphs': [{'name': 'shard', 'tasks': self.tasks or [dict(id='idle', op='noop')]}]}   # a device may own no work


@dataclass
class Shard:
    rank: int
    spec: dict               # a normal ABI-v2 workload (compile_workload accepts it once dist/commops.install() ran)
    remotes: dict            # buffer name -> {'kind': 'peer'|'inbox'|'sendq', ...}: table entries that point off-device
    in_vars: list            # variables fed by remote publishers (the pump polls their inbox slots)
    buffer_names: list       # declaration order == KvmBuf table index
    meta: dict = field(default_factory=dict)    # buffer name -> {dtype, gen (symbolic init descriptor), place (r0, c0, GR, GC)}

    def buf_index(self, name):
        return self.buffer_names.index(name)


# ------------------------------------------------------------------------------------------------ planner core
class Planner:
    def __init__(self, prog: Prog, cluster: Cluster, strategy: dict):
        self.prog, self.cl, self.st = prog, cluster, strategy
        self.P = cluster.n
        self.sh = [ShardBuilder(d) for d in range(self.P)]
        self.prio_comm = strategy.get('comm_priority', 31)      # transfers launch as soon as their data exists
        self.wire = strategy.get('wire', 'f32')
        _need(self.wire in ('f32', 'bf16'), "strategy.wire must be 'f32' or 'bf16'")
        self.algo = strategy.get('collective', 'auto')
        _need(self.algo in ('auto', 'ring', 'tree', 'direct'), "strategy.collective must be auto|ring|tree|direct")
        self.segments = strategy.get('chunks', 1)
        self.comm_bytes = 0
        self.comm_msgs = 0

    # ---- communication primitive: one tile, any link ------------------------------------------------------------------
    def xfer(self, s, src, src_off, d, dst, dst_off, rows, cols, src_ld, dst_ld, src_rows, dst_rows):
        """Move a tile src(dev s) -> dst(dev d, a fresh buffer). Returns the arrival token(s) (committed in d's tracker)."""
        S, D = self.sh[s], self.sh[d]
        link = self.cl.link(s, d)
        per_row = cols * 4
        step = rows if link.kind == 'p2p' else max(1, min(rows, self.cl.max_msg_bytes // per_row))
        _need(link.kind == 'p2p' or per_row <= self.cl.max_msg_bytes,
              f'a single row of {per_row} B exceeds the proxy slot ({self.cl.max_msg_bytes} B); raise max_msg_bytes')
        toks = []
        for r in range(0, rows, step):
            nr = min(step, rows - r)
            v = D.inbox_var()
            acc = [(src, src_rows[0] + r, src_rows[0] + r + nr, 'r')]
            so, do = src_off + r * src_ld, dst_off + r * dst_ld
            if link.kind == 'p2p':
                S.add('put_signal', dict(src=src, dst=S.remote_peer(D, dst), src_off=so, dst_off=do, rows=nr, cols=cols,
                                         src_ld=src_ld, dst_ld=dst_ld, flag=S.remote_inbox(D), var=D.var(v), ver=1),
                      acc, label=f'put>{d}', priority=self.prio_comm)
            else:
                S.add('send_proxy', dict(src=src, q=S.sendq(), src_off=so, rows=nr, cols=cols, src_ld=src_ld, dst_rank=d,
                                         dst_buf=D.index(dst), dst_off=do, dst_ld=dst_ld, var=D.var(v), ver=1),
                      acc, label=f'send>{d}', priority=self.prio_comm)
            D.trk[dst].commit_remote(dst_rows[0] + r, dst_rows[0] + r + nr, ('w', v))
            toks.append(('w', v))
            self.comm_bytes += nr * cols * 4
            self.comm_msgs += 1
        return toks

    def local_put(self, dev, src, src_off, dst, dst_off, rows, cols, src_ld, dst_ld, src_rows, dst_rows, label='slice'):
        return self.sh[dev].add('put', dict(src=src, dst=dst, src_off=src_off, dst_off=dst_off, rows=rows, cols=cols,
                                            src_ld=src_ld, dst_ld=dst_ld),
                                [(src, *src_rows, 'r'), (dst, *dst_rows, 'w')], label=label)

    # ---- shapes ---------------------------------------------------------------------------------------------------------
    def phys_shape(self, shape, lay):
        r, c = shape
        if lay == ROWS:
            _need(r % self.P == 0, f'ROWS layout needs rows % {self.P} == 0, got {r}')
            return r // self.P, c
        if lay == COLS:
            _need(c % self.P == 0, f'COLS layout needs cols % {self.P} == 0, got {c}')
            return r, c // self.P
        return r, c

    def reshard_cost(self, shape, frm, to):
        """Bytes sent per device by the cheapest implemented resharding (the selection metric)."""
        if frm == to or (frm == R and to in (ROWS, COLS)):
            return 0                                        # a slice of a replica is local
        n, P = shape[0] * shape[1] * 4, self.P
        if to == R:
            return n * (P - 1) / P if frm in (ROWS, COLS) else 2 * n * (P - 1) / P
        if frm == PART:
            return n * (P - 1) / P if to == ROWS else 2 * n * (P - 1) / P
        return n * (P - 1) / P / P                          # ROWS <-> COLS all-to-all: one tile to every other device

    # ---- collectives (all produce FRESH buffers; return their name, same on every device) ------------------------------------
    def _alloc(self, base, shape, lay):
        """The same fresh buffer name on every device (collective outputs are addressed by name across devices)."""
        r, c = self.phys_shape(shape, lay)
        self.galloc = getattr(self, 'galloc', 0) + 1
        name = f'{base}~g{self.galloc}'
        for d in range(self.P):
            self.sh[d].buf(name, r, c)
        return name

    def blk(self, shape, lay, b):
        """(row0, nrows, col0, ncols) of block b under layout lay."""
        r, c = shape
        if lay == ROWS:
            return b * (r // self.P), r // self.P, 0, c
        return 0, r, b * (c // self.P), c // self.P

    def all_gather(self, name, src, shape, lay):
        P = self.P
        full = self._alloc(f'{name}.ag', shape, R)
        r, c = shape
        algo = self.algo if self.algo in ('ring', 'direct') else ('ring' if len(set(self.cl.node_of)) > 1 else 'direct')
        n_sh = r * c // P
        if self.wire == 'bf16' and lay == ROWS and n_sh % 2 == 0:
            # bf16 on the wire. EVERY device unpacks every block (its own too), so all replicas hold identical bf16-rounded values.
            half = n_sh // 2
            for d in range(P):
                pk = self.sh[d].tmp(f'{name}.pk', 1, half)
                self.sh[d].add('pack_bf16', dict(src=src, dst=pk, src_off=0, dst_off=0, n=n_sh), [(src, 0, r // P, 'r'), (pk, 0, 1, 'w')],
                               label='ag.pack', priority=self.prio_comm)
                for e in range(P):
                    r0 = d * (r // P)
                    if e == d:
                        land, toks = pk, []
                    else:
                        land = self.sh[e].tmp(f'{name}.pkl{d}', 1, half)
                        self.xfer(d, pk, 0, e, land, 0, 1, half, half, half, (0, 1), (0, 1))
                    self.sh[e].add('unpack_bf16', dict(src=land, dst=full, src_off=0, dst_off=d * n_sh, n=n_sh, acc=0),
                                   [(land, 0, 1, 'r'), (full, r0, r0 + r // P, 'w')], label='ag.unpack')
            return full
        for d in range(P):                                           # own block, local
            r0, nr, c0, nc = self.blk(shape, lay, d)
            self.local_put(d, src, 0, full, r0 * c + c0, nr, nc, nc, c, (0, nr), (r0, r0 + nr), label='ag.own')
        if algo == 'direct':
            for d in range(P):
                for e in range(P):
                    if e != d:
                        r0, nr, c0, nc = self.blk(shape, lay, d)
                        self.xfer(d, src, 0, e, full, r0 * c + c0, nr, nc, nc, c, (0, nr), (r0, r0 + nr))
        else:
            for s in range(P - 1):
                for d in range(P):
                    b = (d - s) % P
                    r0, nr, c0, nc = self.blk(shape, lay, b)
                    self.xfer(d, full, r0 * c + c0, (d + 1) % P, full, r0 * c + c0, nr, nc, c, c, (r0, r0 + nr), (r0, r0 + nr))
        return full

    def _flat_ranges(self, n):
        P, k = self.P, self.segments
        _need(n % (P * k) == 0, f'all-reduce/reduce-scatter of {n} elements needs n % (devices*chunks) == 0 ({P}*{k})')
        B = n // P
        return B, B // k

    def _rows_of(self, shape, lo, hi):
        c = shape[1]
        return lo // c, -(-hi // c)

    def reduce_scatter_flat(self, name, src, shape, j):
        """Ring reduce-scatter of segment j of every block of a PART buffer. Returns {dev: (buffer, flat_offset, n)} of the
        fully reduced sub-chunk each device owns (block d). Deterministic: chunk c sums x_{c+1}, x_{c+2}, ..., x_c in order."""
        P = self.P
        n = shape[0] * shape[1]
        B, sub = self._flat_ranges(n)
        off = lambda c: c * B + j * sub
        res = {}
        if P == 1:
            return {0: (src, off(0), sub)}
        prev_land = {}
        for s in range(P - 1):
            nxt_land = {}
            for d in range(P):
                c = (d - s - 1) % P
                e = (d + 1) % P
                land = self.sh[e].tmp(f'{name}.rs{j}s{s}', 1, sub)
                nxt_land[e] = land
                lo, hi = off(c), off(c) + sub
                src_rows = self._rows_of(shape, lo, hi)
                if s == 0:
                    self.xfer(d, src, lo, e, land, 0, 1, sub, sub, sub, src_rows, (0, 1))
                else:   # fused: land(prev) + local chunk -> remote landing (reduce_put_signal when peer-reachable)
                    self._reduce_forward(d, prev_land[d], src, lo, e, land, sub, src_rows)
            prev_land = nxt_land
        for d in range(P):
            c = d
            lo, hi = off(c), off(c) + sub
            out = self.sh[d].tmp(f'{name}.rsout{j}', 1, sub)
            self.sh[d].add('reduce2', dict(a=prev_land[d], b=src, out=out, a_off=0, b_off=lo, o_off=0, n=sub),
                           [(prev_land[d], 0, 1, 'r'), (src, *self._rows_of(shape, lo, hi), 'r'), (out, 0, 1, 'w')], label='rs.final')
            res[d] = (out, 0, sub)
        return res

    def _reduce_forward(self, d, land_prev, src, lo, e, land, sub, src_rows):
        S, D = self.sh[d], self.sh[e]
        link = self.cl.link(d, e)
        if link.kind == 'p2p':
            v = D.inbox_var()
            S.add('reduce_put_signal', dict(a=land_prev, b=src, dst=S.remote_peer(D, land), a_off=0, b_off=lo, d_off=0, n=sub,
                                            flag=S.remote_inbox(D), var=D.var(v), ver=1),
                  [(land_prev, 0, 1, 'r'), (src, *src_rows, 'r')], label=f'rput>{e}', priority=self.prio_comm)
            D.trk[land].commit_remote(0, 1, ('w', v))
            self.comm_bytes += sub * 4
            self.comm_msgs += 1
        else:
            acc = S.tmp('rsacc', 1, sub)
            S.add('reduce2', dict(a=land_prev, b=src, out=acc, a_off=0, b_off=lo, o_off=0, n=sub),
                  [(land_prev, 0, 1, 'r'), (src, *src_rows, 'r'), (acc, 0, 1, 'w')], label='rs.acc')
            self.xfer(d, acc, 0, e, land, 0, 1, sub, sub, sub, (0, 1), (0, 1))

    def all_reduce(self, name, src, shape):
        """PART -> fresh full R buffer on every device (identical bits on all devices)."""
        P = self.P
        n = shape[0] * shape[1]
        full = self._alloc(f'{name}.ar', shape, R)
        c = shape[1]
        algo = self.algo
        if algo == 'auto':
            algo = 'direct' if n * 4 / max(P, 1) < 32 * 1024 or P == 2 else 'ring'
        if P == 1:
            self.local_put(0, src, 0, full, 0, 1, n, n, n, (0, shape[0]), (0, shape[0]), label='ar.copy')
            return full
        if algo == 'direct':
            lands = {}
            for d in range(P):
                for e in range(P):
                    if e != d:
                        land = lands.setdefault((d, e), self.sh[e].tmp(f'{name}.dar{d}', shape[0], c))
                        self.xfer(d, src, 0, e, land, 0, shape[0], c, c, c, (0, shape[0]), (0, shape[0]))
            for e in range(P):
                parts = [src if i == e else lands[(i, e)] for i in range(P)]
                acc = parts[0]
                for i in range(1, P):
                    out = full if i == P - 1 else self.sh[e].tmp(f'{name}.dsum', shape[0], c)
                    self.sh[e].add('reduce2', dict(a=acc, b=parts[i], out=out, a_off=0, b_off=0, o_off=0, n=n),
                                   [(acc, 0, shape[0], 'r'), (parts[i], 0, shape[0], 'r'), (out, 0, shape[0], 'w')], label='ar.sum')
                    acc = out
        elif algo == 'tree':
            self._tree_all_reduce(name, src, shape, full)
        else:
            B, sub = self._flat_ranges(n)
            for j in range(self.segments):
                red = self.reduce_scatter_flat(name, src, shape, j)
                for d in range(P):                                   # all-gather the reduced sub-chunks, ring or direct
                    buf, o, ln = red[d]
                    lo = d * B + j * sub
                    rr = self._rows_of(shape, lo, lo + sub)
                    self.local_put(d, buf, o, full, lo, 1, ln, ln, ln, (0, 1), rr, label='ar.own')
                for s in range(P - 1) if self.algo != 'direct' else []:
                    for d in range(P):
                        b = (d - s) % P
                        lo = b * B + j * sub
                        rr = self._rows_of(shape, lo, lo + sub)
                        self.xfer(d, full, lo, (d + 1) % P, full, lo, 1, sub, sub, sub, rr, rr)
        return full

    def _tree_all_reduce(self, name, src, shape, full):
        P, n, c = self.P, shape[0] * shape[1], shape[1]
        rws = shape[0]
        kids = {i: [k for k in (2 * i + 1, 2 * i + 2) if k < P] for i in range(P)}
        up = {}
        for i in range(P - 1, -1, -1):                    # children before parents
            acc = src
            for k, kid in enumerate(kids[i]):
                land = self.sh[i].tmp(f'{name}.tr{kid}', rws, c)
                self.xfer(kid, up[kid], 0, i, land, 0, rws, c, c, c, (0, rws), (0, rws))
                out = full if (i == 0 and k == len(kids[i]) - 1) else self.sh[i].tmp(f'{name}.tsum', rws, c)
                self.sh[i].add('reduce2', dict(a=acc, b=land, out=out, a_off=0, b_off=0, o_off=0, n=n),
                               [(acc, 0, rws, 'r'), (land, 0, rws, 'r'), (out, 0, rws, 'w')], label='tree.sum')
                acc = out
            up[i] = acc
        if P == 1 or not kids[0]:
            self.local_put(0, src, 0, full, 0, rws, c, c, c, (0, rws), (0, rws), label='tree.copy')
        for i in range(P):                                # broadcast down: results are copied, never recomputed
            for kid in kids[i]:
                self.xfer(i, full, 0, kid, full, 0, rws, c, c, c, (0, rws), (0, rws))

    def reduce_scatter(self, name, src, shape):
        """PART -> fresh ROWS buffer (device d owns row block d)."""
        P = self.P
        out = self._alloc(f'{name}.rs', shape, ROWS)
        B, sub = self._flat_ranges(shape[0] * shape[1])
        c = shape[1]
        for j in range(self.segments):
            red = self.reduce_scatter_flat(name, src, shape, j)
            for d in range(P):
                buf, o, ln = red[d]
                rr = self._rows_of((shape[0] // P, c), j * sub, (j + 1) * sub)
                self.local_put(d, buf, o, out, j * sub, 1, ln, ln, ln, (0, 1), rr, label='rs.store')
        return out

    def all_to_all(self, name, src, shape, frm):
        """ROWS <-> COLS: every (d, e) tile moves from d to e. Fresh destination, each tile written once."""
        to = COLS if frm == ROWS else ROWS
        dst = self._alloc(f'{name}.a2a', shape, to)
        r, c = shape
        P = self.P
        sr, sc = self.phys_shape(shape, frm)
        dr, dc = self.phys_shape(shape, to)
        for d in range(P):
            for e in range(P):
                if frm == ROWS:      # src[d] rows all, cols block e -> dst[e] rows block d
                    so, nr, nc, ssrows = e * dc, sr, dc, (0, sr)
                    do, drows = d * sr * dc, (d * sr, (d + 1) * sr)
                else:                # src[d] rows block e, all cols -> dst[e] cols block d
                    so, nr, nc, ssrows = e * dr * sc, dr, sc, (e * dr, (e + 1) * dr)
                    do, drows = d * sc, (0, dr)
                if d == e:
                    self.local_put(d, src, so, dst, do, nr, nc, sc, dc, ssrows, drows, label='a2a.own')
                else:
                    self.xfer(d, src, so, e, dst, do, nr, nc, sc, dc, ssrows, drows)
        return dst

    def slice_local(self, name, src, shape, lay):
        out = self._alloc(f'{name}.sl', shape, lay)
        for d in range(self.P):
            r0, nr, c0, nc = self.blk(shape, lay, d)
            self.local_put(d, src, r0 * shape[1] + c0, out, 0, nr, nc, shape[1], nc, (r0, r0 + nr), (0, nr), label='slice')
        return out

    def reshard(self, name, src, shape, frm, to):
        """Return the name of a fresh buffer holding logical `name` in layout `to`."""
        _need(frm != to, 'internal: no-op reshard')
        if frm == R:
            return self.slice_local(name, src, shape, to)
        if to == R:
            if frm == PART:
                return self.all_reduce(name, src, shape)
            return self.all_gather(name, src, shape, frm)
        if frm == PART:
            if to == ROWS:
                return self.reduce_scatter(name, src, shape)
            return self.slice_local(name, self.all_reduce(name, src, shape), shape, COLS)
        return self.all_to_all(name, src, shape, frm)


# ------------------------------------------------------------------------------------------------ shard mode
class ShardPlanner(Planner):
    # (layout of a, layout of b) -> layout of the product, in preference order; contracting a sharded k gives PART
    MM_RULES = [((ROWS, R), ROWS), ((R, COLS), COLS), ((COLS, ROWS), PART), ((R, R), R)]

    def __init__(self, prog, cluster, strategy):
        super().__init__(prog, cluster, strategy)
        lay = {}
        for k, v in strategy.get('layouts', {}).items():
            _need(k in prog.buffers, f'strategy.layouts names unknown buffer {k!r}')
            _need(v.lower() in LAYOUT_NAMES and LAYOUT_NAMES[v.lower()] != PART, f'layout of {k!r} must be repl|rows|cols')
            lay[k] = LAYOUT_NAMES[v.lower()]
        self.init_layout = lay
        self.layout = {}                 # logical buffer -> layout, fixed at first use
        self.cache = {}                  # (name, layout) -> (physical name, epoch of the source when built)
        self.epoch = {n: 0 for n in prog.buffers}

    def shape(self, name):
        b = self.prog.buffers[name]
        return b['rows'], b['cols']

    def scalar(self, name):
        return self.shape(name) == (1, 1)

    def decide(self, name, lay):
        if name in self.layout:
            return
        if self.scalar(name) and lay in (ROWS, COLS):
            lay = R
        self.layout[name] = lay
        shape, B = self.shape(name), self.prog.buffers[name]
        r, c = self.phys_shape(shape, lay)
        if self.st.get('symbolic'):
            gen = self.st.get('gen', {}).get(name)
            for d in range(self.P):
                r0, nr, c0, nc = (0, shape[0], 0, shape[1]) if lay in (R, PART) else self.blk(shape, lay, d)
                if lay == PART:
                    desc = dict(kind='fill', value=0)
                elif gen:
                    desc = dict(kind='gen', **gen)
                else:
                    _need(B['init']['kind'] == 'fill', f'symbolic mode: {name!r} needs a fill init or a strategy.gen entry')
                    desc = dict(kind='fill', value=B['init']['value'])
                self.sh[d].buf(name, r, c, None, dtype=B['dtype'], gen=desc, place=(r0, c0, shape[0], shape[1]))
            return
        full = B['values']
        for d in range(self.P):
            if lay in (R, PART):
                vals = list(full) if lay == R else [0.0] * len(full)
            else:
                r0, nr, c0, nc = self.blk(shape, lay, d)
                vals = [full[(r0 + i) * shape[1] + c0 + j] for i in range(nr) for j in range(nc)]
            self.sh[d].buf(name, r, c, vals, dtype=B['dtype'], place=(0, 0, shape[0], shape[1]) if lay in (R, PART) else None)

    def ensure(self, name):
        self.decide(name, self.init_layout.get(name, R))
        return self.layout[name]

    def view(self, name, lay):
        """Physical buffer holding logical `name` in layout `lay`; resharded once per write epoch."""
        cur = self.ensure(name)
        if self.scalar(name) and lay in (ROWS, COLS):
            lay = R
        if cur == lay:
            return name
        key = (name, lay)
        if key not in self.cache or self.cache[key][1] != self.epoch[name]:
            self.cache[key] = (self.reshard(name, name, self.shape(name), cur, lay), self.epoch[name])
        return self.cache[key][0]

    def rows_of(self, d, phys):
        return self.sh[d].buffers[phys]['rows']

    def plan(self):
        for t in self.prog.tasks:
            try:
                self.plan_task(t)
            except PartitionError as e:
                raise PartitionError(f"{t['id']}: {e}") from None
        shards = [Shard(d, self.sh[d].spec(lambda r: len(self.sh[r].variables)), self.sh[d].remotes, self.sh[d].in_vars,
                        list(self.sh[d].buffers),
                        {n: dict(dtype=b['dtype'], gen=b['gen'], place=b['place']) for n, b in self.sh[d].buffers.items()})
                  for d in range(self.P)]
        return Plan(shards, dict(self.layout), self.comm_bytes, self.comm_msgs, self.prog, self.cl, self.st)

    def emit_all(self, t, op, args_for, acc_for, label=None, carry=True):
        """Emit `op` on every device. `carry`: the original task's version reads/publishes ride on this task."""
        for d in range(self.P):
            kw = {}
            if carry and t.get('reads'):
                kw['reads'] = t['reads']
            if carry and t.get('publishes'):
                kw['publishes'] = t['publishes']
            for r in (t.get('reads', []) if carry else []):
                self.sh[d].var(r['var'])
            if carry and t.get('publishes'):
                self.sh[d].var(t['publishes'])
            self.sh[d].add(op, args_for(d), acc_for(d), label=label or t['id'], priority=t.get('priority', 0),
                           deadline_us=t.get('deadline_us', -1), **kw)

    def acc(self, d, reads=(), writes=()):
        return [(p, 0, self.rows_of(d, p), 'r') for p in reads] + [(p, 0, self.rows_of(d, p), 'w') for p in writes]

    def plan_task(self, t):
        op = t['op']
        if op in ('noop', 'spin'):
            self.emit_all(t, op, lambda d: dict(t['args']), lambda d: [])
        elif op in ELEMENTWISE:
            self.plan_elementwise(t)
        elif op == 'mse_grad':
            self.plan_mse(t)
        elif op == 'matmul':
            self.plan_matmul(t)
        elif op == 'adamw':
            self.plan_adamw(t)
        else:
            raise PartitionError(f'op {op!r} has no sharding rule')

    def plan_adamw(self, t):
        """Sharded optimizer step (ZeRO): AdamW is elementwise, so it runs on whatever shard of p/m/v a device owns. p, m, v take
        the layout declared for p (ROWS = ZeRO-style); the gradient is brought into that layout, which for a PART gradient is a
        reduce-scatter (no all-reduce, no replicated optimizer state). Covers the whole buffer (e0 = 0, e1 = size)."""
        a = t['args']
        p, g, m, v = a['p'], a['g'], a['m'], a['v']
        full = self.shape(p)[0] * self.shape(p)[1]
        _need(a['e0'] == 0 and a['e1'] == full, f'adamw must cover its whole buffer (e0=0, e1={full}), got [{a["e0"]},{a["e1"]})')
        for nm in (g, m, v):
            _need(self.shape(nm) == self.shape(p), f'adamw: {nm!r} and {p!r} differ in shape')
        T = self.ensure(p)
        for nm in (m, v):
            _need(self.ensure(nm) == T, f'adamw: {nm!r} must have the layout of {p!r} ({T})')
        gv = self.view(g, T)
        st = self.view(a['step'], R)
        extra = [x for x in (a.get('gscale'),) if x]
        ev = [self.view(x, R) for x in extra]
        n_loc = full // self.P if T in (ROWS, COLS) else full

        def args_for(d):
            args = dict(a, g=gv, step=st, e0=0, e1=n_loc)
            if ev:
                args['gscale'] = ev[0]
            return args
        self.emit_all(t, 'adamw', args_for, lambda d: self.acc(d, reads=[gv, st] + ev, writes=[p, m, v]))
        for o in (p, m, v):
            self.epoch[o] += 1

    def common_layout(self, ins, outs, linear):
        for o in outs:                                   # a decided output (parameter, in-place operand) fixes the layout
            if o in self.layout or o in self.init_layout:
                return self.ensure(o)
        lays = [self.ensure(i) for i in ins if not self.scalar(i)]
        if not lays:
            lays = [self.ensure(i) for i in ins] or [R]
        non_part = [x for x in lays if x != PART]
        return non_part[0] if non_part else (PART if linear else R)

    def local_n(self, total, ref, lay):
        full = self.prog.buffers[ref]['rows'] * self.prog.buffers[ref]['cols']
        if lay in (ROWS, COLS):
            _need(total == full, f'a sharded op must cover its whole buffer ({total} of {full} elements of {ref!r})')
            return total // self.P
        return total

    def plan_elementwise(self, t):
        op, a = t['op'], t['args']
        cnt, ins, outs, linear = ELEMENTWISE[op]
        data_in = [a[i] for i in ins]
        out_names = [a[o] for o in outs]
        T = self.common_layout(data_in, out_names, linear)
        _need(T != PART or linear, f'{op} is not linear but its output buffer holds partial sums')
        ref = next((x for x in data_in + out_names if not self.scalar(x)), (data_in + out_names)[0])
        n_loc = self.local_n(a[cnt], ref, T)
        views = {i: self.view(a[i], T) for i in ins}
        wv = self.view(a['w'], R) if op == 'predict' else None
        for o in out_names:
            self.decide(o, T)
        _need(all(self.layout[o] == T for o in out_names), f'{op}: output layout differs from the operand layout')

        def args_for(d):
            args = dict(a, **views)
            if wv:
                args['w'] = wv
            args[cnt] = n_loc
            return args
        self.emit_all(t, op, args_for, lambda d: self.acc(d, reads=[p for k, p in views.items() if p not in out_names] + ([wv] if wv else []), writes=out_names))
        for o in out_names:
            self.epoch[o] += 1

    def plan_mse(self, t):
        a = t['args']
        T = self.ensure(a['pred'])
        T = R if T == PART else T
        views = {k: self.view(a[k], T) for k in ('pred', 'target', 'x')}
        n_loc = self.local_n(a['n'], a['pred'], T)
        res = R if T == R else PART
        for nm in (a['loss'], a['grad']):
            self.ensure(nm) if nm in self.init_layout else self.decide(nm, res)
            _need(self.layout[nm] == res, f'{nm!r} must be {res} (the result layout of a {T} mse_grad)')
        rd = list(views.values())
        if T == R:
            self.emit_all(t, 'mse_grad', lambda d: dict(a, **views), lambda d: self.acc(d, reads=rd, writes=[a['loss'], a['grad']]))
        else:
            tl = {d: self.sh[d].tmp('mse.l', 1, 1) for d in range(self.P)}
            tg = {d: self.sh[d].tmp('mse.g', 1, 1) for d in range(self.P)}
            self.emit_all(t, 'mse_grad', lambda d: dict(a, n=n_loc, loss=tl[d], grad=tg[d], **views),
                          lambda d: self.acc(d, reads=rd, writes=[tl[d], tg[d]]), label=t['id'] + '.local', carry=False)
            w = n_loc / a['n']          # weighted partial means: their sum is the global mean
            for tm, dst in ((tl, a['loss']), (tg, a['grad'])):
                self.emit_all(t, 'scale', lambda d, tm=tm, dst=dst: dict(x=tm[d], out=dst, alpha=w, n=1),
                              lambda d, tm=tm, dst=dst: self.acc(d, reads=[tm[d]], writes=[dst]), label=t['id'] + '.w')
        self.epoch[a['loss']] += 1
        self.epoch[a['grad']] += 1

    def plan_matmul(self, t):
        a = t['args']
        A, B, O = a['a'], a['b'], a['out']
        m, n, k = a['m'], a['n'], a['k']
        for nm, shp in ((A, (m, k)), (B, (k, n)), (O, (m, n))):
            _need(self.shape(nm) == shp, f'matmul needs exact shapes: {nm!r} is {self.shape(nm)}, expected {shp}')
        la0, lb0 = self.ensure(A), self.ensure(B)
        want = self.layout.get(O) or self.init_layout.get(O)
        best = None
        for (la, lb), lo in (self.MM_RULES if self.P > 1 else [((R, R), R)]):     # one device: no layout games, no slice copies
            if (lo == ROWS and m % self.P) or (lo == COLS and n % self.P) or (lo == PART and k % self.P):
                continue
            cost = self.reshard_cost((m, k), la0, la) + self.reshard_cost((k, n), lb0, lb)
            if want and want != lo:
                cost += self.reshard_cost((m, n), lo, want)
            if best is None or cost < best[0]:
                best = (cost, la, lb, lo)
        _need(best is not None, f'no legal sharding for matmul {m}x{n}x{k} on {self.P} devices')
        _, la, lb, lo = best
        va, vb = self.view(A, la), self.view(B, lb)
        lm, ln, lk = (m // self.P if lo == ROWS else m), (n // self.P if lo == COLS else n), (k // self.P if lo == PART else k)
        direct = not want or want == lo
        self.decide(O, lo if direct else want)       # an annotated layout wins; the product is resharded into it below
        if direct:
            outn = O
        else:
            outn = self._alloc(O + '.mm', (m, n), lo)
        self.emit_all(t, 'matmul', lambda d: dict(a=va, b=vb, out=outn, m=lm, n=ln, k=lk),
                      lambda d: self.acc(d, reads=[va, vb], writes=[outn]))
        if not direct:
            res = self.reshard(O, outn, (m, n), lo, want)
            for d in range(self.P):
                r, c = self.rows_of(d, O), self.sh[d].buffers[O]['cols']
                self.local_put(d, res, 0, O, 0, r, c, c, c, (0, r), (0, r), label='mm.store')
        self.epoch[O] += 1


@dataclass
class Plan:
    shards: list
    layout: dict            # shard mode: logical buffer -> final layout; place mode: logical buffer -> (device, physical name)
    comm_bytes: int
    comm_msgs: int
    prog: Prog
    cluster: Cluster
    strategy: dict

    def reconstruct(self, results):
        """results[d] = {physical buffer: values}. Returns {logical buffer: values} (R must agree on every device)."""
        out = {}
        for name, lay in self.layout.items():
            b = self.prog.buffers[name]
            if self.strategy.get('mode', 'shard') == 'place':
                dev, phys = lay
                out[name] = results[dev][phys]
                continue
            P = len(results)
            if lay == R:
                for d in range(1, P):
                    if results[d][name] != results[0][name]:
                        raise PartitionError(f'replicated buffer {name!r} differs between device 0 and {d}')
                out[name] = results[0][name]
            elif lay == PART:
                acc = list(results[0][name])
                for d in range(1, P):
                    acc = [ref_ops.f32(x + y) for x, y in zip(acc, results[d][name])]
                out[name] = acc
            elif lay == ROWS:
                out[name] = [x for d in range(P) for x in results[d][name]]
            else:
                rows, cols, cs = b['rows'], b['cols'], b['cols'] // P
                out[name] = [results[c // cs][name][r * cs + c % cs] for r in range(rows) for c in range(cols)]
        return out


# ------------------------------------------------------------------------------------------------ place mode
def microbatch(spec, M, split, accumulate):
    """Clone the per-microbatch part of the graph M times over row-slices of the `split` buffers.

    split:      buffers sliced by rows (activations, targets): `x` becomes `x.mb0..x.mb{M-1}`.
    accumulate: buffers (gradients, losses) written once per microbatch and then combined in microbatch order into the
                original buffer: {'grad': 'mean'|'sum'}.
    A task that touches a split buffer is per-microbatch (cloned); everything else (weights, the optimizer) runs once, after
    the combine. Count arguments are divided by M. This is a semantics-preserving rewrite for equal-size microbatches; it is
    also the only thing that creates pipeline overlap, so it is itself verified against the CPU VM.
    """
    prog = parse_program(spec)
    split, accumulate = list(split), dict(accumulate)
    for n in split + list(accumulate):
        _need(n in prog.buffers, f'microbatch names unknown buffer {n!r}')
    for n in split:
        _need(prog.buffers[n]['rows'] % M == 0, f"cannot split {n!r} ({prog.buffers[n]['rows']} rows) into {M} microbatches")
    touched = lambda t: {v for v in t['args'].values() if isinstance(v, str)}
    per_mb = {t['id'] for t in prog.tasks if touched(t) & set(split)}
    buffers = [dict(b) for b in spec['buffers'] if b['name'] not in split and b['name'] not in accumulate]
    buffers = [b for b in buffers if b['name'] not in ()]
    for n in split:
        b = prog.buffers[n]
        rr = b['rows'] // M
        for i in range(M):
            buffers.append(dict(name=f'{n}.mb{i}', rows=rr, cols=b['cols'],
                                init=dict(kind='values', values=b['values'][i * rr * b['cols']:(i + 1) * rr * b['cols']])))
    orig_init = {b['name']: b.get('init', dict(kind='fill', value=0)) for b in spec['buffers']}
    for n in accumulate:
        b = prog.buffers[n]
        buffers.append(dict(name=n, rows=b['rows'], cols=b['cols'], init=orig_init[n]))
        for i in range(M):
            buffers.append(dict(name=f'{n}.mb{i}', rows=b['rows'], cols=b['cols'], init=dict(kind='fill', value=0)))
    rename = lambda v, i: f'{v}.mb{i}' if v in split or v in accumulate else v
    graphs = []
    for g in spec['graphs']:
        tasks, clones, combine = [], {}, {}
        gt = [t for t in prog.tasks if t['gid'] == g['name']]
        wrote = {n: [] for n in accumulate}
        depth = {t['id']: min(30, (i * 30) // max(1, len(gt))) for i, t in enumerate(gt)}
        for t in gt:
            local = t['id'].split('.', 1)[1]
            if t['id'] in per_mb:
                for i in range(M):
                    nt = copy.deepcopy(g['tasks'][[x['id'] for x in g['tasks']].index(local)])
                    nt['id'] = f'{local}@mb{i}'
                    nt['args'] = {k: (rename(v, i) if isinstance(v, str) else v) for k, v in nt.get('args', {}).items()}
                    if t['op'] == 'matmul':       # only the row count of the product shrinks ('n' is a column count here)
                        _need(nt['args']['m'] % M == 0, f"{t['id']}: matmul m not divisible by {M}")
                        nt['args']['m'] //= M
                    elif 'n' in nt['args']:
                        _need(nt['args']['n'] % M == 0, f"{t['id']}: count {nt['args']['n']} not divisible by {M}")
                        nt['args']['n'] //= M
                    nt['after'] = []
                    nt['priority'] = min(30, nt.get('priority', 0) + depth[t['id']])     # deeper layers first: depth-first flow, so
                    clones.setdefault(t['id'], []).append(nt['id'])                      # microbatch i leaves a stage before i+1 enters
                    tasks.append(nt)
                    for n in accumulate:
                        if n in touched(t):
                            wrote[n].append(nt['id'])
            else:
                nt = copy.deepcopy(g['tasks'][[x['id'] for x in g['tasks']].index(local)])
                nt['after'] = []
                clones[t['id']] = [nt['id']]
                tasks.append(nt)
        for n in accumulate:
            if not wrote[n]:
                continue
            cnt = prog.buffers[n]['rows'] * prog.buffers[n]['cols']
            prev, chain = f'{n}.mb0', list(wrote[n])
            for i in range(1, M):
                dst = f'{n}.acc{i}'
                buffers.append(dict(name=dst, rows=prog.buffers[n]['rows'], cols=prog.buffers[n]['cols'], init=dict(kind='fill', value=0)))
                tid = f'__acc/{n}/{i}'
                tasks.append(dict(id=tid, op='add', args=dict(a=prev, b=f'{n}.mb{i}', out=dst, n=cnt), after=chain))
                prev, chain = dst, [tid]
            fin = f'__acc/{n}/final'
            tasks.append(dict(id=fin, op='scale', args=dict(x=prev, out=n, alpha=(1.0 / M) if accumulate[n] == 'mean' else 1.0, n=cnt),
                              after=chain))
            combine[n] = fin
        # dependencies: mb i of X waits on mb i of its per-mb predecessors, and on every clone of single predecessors
        by_id = {x['id']: x for x in tasks}
        for t in gt:
            local = t['id'].split('.', 1)[1]
            orig_after = g['tasks'][[x['id'] for x in g['tasks']].index(local)].get('after', [])
            for ci, cid in enumerate(clones[t['id']]):
                deps = []
                for dep in orig_after:
                    full_dep = f"{g['name']}.{dep}"
                    cl_ids = clones[full_dep]
                    if t['id'] in per_mb and full_dep in per_mb:
                        deps.append(cl_ids[ci])
                    else:
                        deps.extend(cl_ids)
                        dep_task = next(x for x in gt if x['id'] == full_dep)
                        deps.extend(combine[n] for n in accumulate if n in touched(dep_task) and n in combine)
                for n in accumulate:                  # a consumer of the combined buffer runs after the combine
                    if t['id'] not in per_mb and n in touched(t) and n in combine and combine[n] not in deps:
                        deps.append(combine[n])
                by_id[cid]['after'] = sorted(set(deps))
        graphs.append(dict(name=g['name'], tasks=tasks))
    return dict(variables=spec.get('variables', []), buffers=buffers, graphs=graphs)


class PlacePlanner(Planner):
    """Task-to-device placement with coherent buffer copies: a buffer is valid on the devices that hold its latest data; a
    reader elsewhere pulls a copy (one transfer, cached until the next write); a writer invalidates every other copy."""

    def __init__(self, prog, cluster, strategy):
        super().__init__(prog, cluster, strategy)
        self.valid = {}                  # buffer -> {device: physical name}
        self.assign = {}
        self.mirror_sig = {}

    # -- placement ---------------------------------------------------------------------------------------------------------
    def place(self):
        st, P, tasks = self.st, self.P, self.prog.tasks
        cost = {t['id']: op_cost_us(t['op'], t['args'], self.cl) for t in tasks}
        if st.get('auto') == 'pipeline':
            S = st.get('stages', P)
            _need(S <= P, 'more stages than devices')
            for s, (lo, hi) in enumerate(balance_stages([cost[t['id']] for t in tasks], S)):
                for t in tasks[lo:hi]:
                    self.assign[t['id']] = s
        elif st.get('auto') == 'heft':
            self.assign = self.heft(cost)
        for pat, dev in st.get('assign', {}).items():
            for t in tasks:
                if t['id'] == pat or (pat.endswith('*') and t['id'].startswith(pat[:-1])):
                    self.assign[t['id']] = dev
        for t in tasks:
            self.assign.setdefault(t['id'], 0)
            _need(0 <= self.assign[t['id']] < P, f"{t['id']}: device {self.assign[t['id']]} out of range")

    def buf_args(self, t):
        return [(k, v) for k, v in t['args'].items() if isinstance(v, str) and v in self.prog.buffers]

    def writes(self, t):
        return set(OUT_ONLY.get(t['op'], [])) | ({'w'} if t['op'] == 'sgd' else set())

    def heft(self, cost):
        """Heterogeneous-earliest-finish-time list scheduling in program order, with transfer cost on cross-device RAW."""
        P = self.P
        avail, fin, dev_of, last_w = [0.0] * P, {}, {}, {}
        for t in self.prog.tasks:
            best = None
            for d in range(P):
                ready = max([fin[x] for x in t['after']] + [0.0])
                for k, v in self.buf_args(t):
                    w = last_w.get(v)
                    if w is not None:
                        sz = self.prog.buffers[v]['rows'] * self.prog.buffers[v]['cols'] * 4
                        ready = max(ready, fin[w] + (0 if dev_of[w] == d else self.cl.xfer_us(dev_of[w], d, sz)))
                eft = max(avail[d], ready) + cost[t['id']]
                if best is None or eft < best[0] - 1e-9:
                    best = (eft, d)
            fin[t['id']], dev_of[t['id']] = best
            avail[best[1]] = best[0]
            for k, v in self.buf_args(t):
                if k in self.writes(t):
                    last_w[v] = t['id']
        return dev_of

    # -- coherence ---------------------------------------------------------------------------------------------------------
    def ensure(self, name, dev, need_data):
        b = self.prog.buffers[name]
        r, c = b['rows'], b['cols']
        vmap = self.valid.setdefault(name, {})
        if not vmap:
            home = self.st.get('home', {}).get(name, dev)
            _need(0 <= home < self.P, f'home of {name!r} out of range')
            self.sh[home].buf(name, r, c, list(b['values']))
            vmap[home] = name
        if dev in vmap:
            return vmap[dev]
        if not need_data:
            return self.sh[dev].tmp(name + '.o', r, c)
        src = min(vmap, key=lambda s: self.cl.xfer_us(s, dev, r * c * 4))
        phys = self.sh[dev].tmp(name + '.c', r, c)
        self.xfer_buf(src, vmap[src], dev, phys, r, c)
        vmap[dev] = phys
        return phys

    def xfer_buf(self, s, sphys, d, dphys, r, c):
        n = r * c
        if self.wire == 'bf16' and n % 2 == 0:
            packed = self.sh[s].tmp('pk', 1, n // 2)
            self.sh[s].add('pack_bf16', dict(src=sphys, dst=packed, src_off=0, dst_off=0, n=n),
                           [(sphys, 0, r, 'r'), (packed, 0, 1, 'w')], label='pack', priority=self.prio_comm)
            land = self.sh[d].tmp('pkl', 1, n // 2)
            self.xfer(s, packed, 0, d, land, 0, 1, n // 2, n // 2, n // 2, (0, 1), (0, 1))
            self.sh[d].add('unpack_bf16', dict(src=land, dst=dphys, src_off=0, dst_off=0, n=n, acc=0),
                           [(land, 0, 1, 'r'), (dphys, 0, r, 'w')], label='unpack', priority=self.prio_comm)
        else:
            self.xfer(s, sphys, 0, d, dphys, 0, r, c, c, c, (0, r), (0, r))

    # -- cross-device version variables ------------------------------------------------------------------------------------
    def mirror_var(self, var, dev):
        D = self.sh[dev]
        if ('mirror', var) not in D.mirror:
            D.mirror[('mirror', var)] = D.inbox_var()
        return D.mirror[('mirror', var)]

    def plan(self):
        self.place()
        tasks = self.prog.tasks
        pubs, rev, cnt = {}, {}, {}
        for t in tasks:
            if t.get('publishes'):
                v = t['publishes']
                cnt[v] = cnt.get(v, 0) + 1
                pubs[(v, cnt[v])], rev[t['id']] = t['id'], (v, cnt[v])
                owner = self.assign[t['id']]
                _need(all(self.assign[x] == owner for (vv, _), x in pubs.items() if vv == v), f"{t['id']}: publishers of {v!r} are on different devices")
        readers = {}                                   # var -> devices reading a version published elsewhere
        for t in tasks:
            for r in t.get('reads', []):
                if r['version'] > 0 and (r['var'], r['version']) in pubs:
                    owner = self.assign[pubs[(r['var'], r['version'])]]
                    if owner != self.assign[t['id']]:
                        _need(r['policy'] == 'at_least', f"{t['id']}: cross-device exact read of {r['var']!r} is not supported (use at_least)")
                        readers.setdefault(r['var'], set()).add(self.assign[t['id']])
        for t in tasks:
            try:
                self.place_task(t, pubs, rev, readers)
            except PartitionError as e:
                raise PartitionError(f"{t['id']}: {e}") from None
        where = {n: next(iter(vm.items())) for n, vm in self.valid.items()}
        shards = [Shard(d, self.sh[d].spec(lambda r: len(self.sh[r].variables)), self.sh[d].remotes, self.sh[d].in_vars,
                        list(self.sh[d].buffers),
                        {n: dict(dtype=b['dtype'], gen=b['gen'], place=b['place']) for n, b in self.sh[d].buffers.items()})
                  for d in range(self.P)]
        return Plan(shards, where, self.comm_bytes, self.comm_msgs, self.prog, self.cl, self.st)

    def place_task(self, t, pubs, rev, readers):
        d = self.assign[t['id']]
        S = self.sh[d]
        a, acc, wrote = dict(t['args']), [], []
        for k, v in self.buf_args(t):
            b = self.prog.buffers[v]
            is_out = k in self.writes(t)
            full = self.covers(t, k, b)
            phys = self.ensure(v, d, need_data=not (is_out and full) or (t['op'] == 'sgd'))
            a[k] = phys
            acc.append((phys, 0, b['rows'], 'w' if is_out else 'r'))
            if is_out:
                wrote.append((v, phys))
        reads = []
        for r in t.get('reads', []):
            r = dict(r)
            pt = pubs.get((r['var'], r['version'])) if r['version'] > 0 else None
            if pt is not None and self.assign[pt] != d:
                r['var'] = self.mirror_var(r['var'], d)
            S.var(r['var'])
            reads.append(r)
        if t.get('publishes'):
            S.var(t['publishes'])
        tid = S.add(t['op'], a, acc, label=t['id'], priority=t.get('priority', 0), reads=reads,
                    publishes=t.get('publishes'), deadline_us=t.get('deadline_us', -1))
        for v, phys in wrote:
            self.valid[v] = {d: phys}
        if t.get('publishes'):
            var, ver = rev[t['id']]
            for e in sorted(readers.get(var, ())):
                D = self.sh[e]
                name = self.mirror_var(var, e)
                prev = self.mirror_sig.get((var, e))
                self.mirror_sig[(var, e)] = S.add('signal', dict(flag=S.remote_inbox(D), var=D.var(name), ver=ver), [],
                                                  label=f'sig.{var}>{e}', priority=self.prio_comm,
                                                  extra_after=[tid] + ([prev] if prev else []))

    def covers(self, t, arg, b):
        full = b['rows'] * b['cols']
        a = t['args']
        if t['op'] == 'matmul':
            return a['m'] * a['n'] == full
        if t['op'] == 'mse_grad':
            return full == 1
        return a.get('n', 0) >= full


def balance_stages(costs, S):
    """Linear partition: contiguous task ranges minimising the slowest stage (classic DP). Returns [(lo, hi)] * S."""
    n = len(costs)
    S = min(S, n)
    pre = [0.0]
    for c in costs:
        pre.append(pre[-1] + c)
    INF = float('inf')
    dp = [[INF] * (n + 1) for _ in range(S + 1)]
    cut = [[0] * (n + 1) for _ in range(S + 1)]
    dp[0][0] = 0.0
    for s in range(1, S + 1):
        for i in range(s, n + 1):
            for j in range(s - 1, i):
                v = max(dp[s - 1][j], pre[i] - pre[j])
                if v < dp[s][i]:
                    dp[s][i], cut[s][i] = v, j
    out, i = [], n
    for s in range(S, 0, -1):
        j = cut[s][i]
        out.append((j, i))
        i = j
    return out[::-1]


# ------------------------------------------------------------------------------------------------ API, estimate, I/O
def partition(spec, cluster: Cluster, strategy: dict) -> Plan:
    mode = strategy.get('mode', 'shard')
    if strategy.get('microbatches', 1) > 1:
        spec = microbatch(spec, strategy['microbatches'], strategy.get('split', []), strategy.get('accumulate', {}))
    prog = parse_program(spec, bool(strategy.get('symbolic')))
    _need(mode in ('shard', 'place'), "strategy.mode must be 'shard' or 'place'")
    plan = (ShardPlanner if mode == 'shard' else PlacePlanner)(prog, cluster, strategy).plan()
    plan.strategy = dict(strategy, mode=mode)
    plan.spec = spec
    log.info('%s: %d devices, %d tasks total, %d messages, %.1f KiB moved', mode, cluster.n,
             sum(len(s.spec['graphs'][0]['tasks']) for s in plan.shards), plan.comm_msgs, plan.comm_bytes / 1024)
    return plan


def estimate(plan: Plan):
    """Modeled makespan in microseconds: discrete-event simulation of every shard with the cost model and the executor's
    dispatch rule (a free device runs the highest-priority ready task, FIFO within a priority). A compute task occupies its
    device; a comm task occupies one of the device's `comm_lanes` for issue + bytes/bandwidth, and the receiver sees the data
    latency + bytes/bandwidth after it starts. A MODEL of the design, not a measurement (DISTRIBUTED.md says what is modeled)."""
    import heapq
    import itertools
    cl, P = plan.cluster, plan.cluster.n
    task = {(s.rank, t['id']): t for s in plan.shards for t in s.spec['graphs'][0]['tasks']}
    vname = lambda dev, idx: plan.shards[dev].spec['variables'][idx]
    pend, deps_of, wait_of = {}, {}, {}
    for (d, tid), t in task.items():
        waits = [r['var'] for r in t.get('reads', []) if r['var'] in plan.shards[d].in_vars]
        pend[(d, tid)] = len(t['after']) + len(waits)
        for a in t['after']:
            deps_of.setdefault((d, a), []).append((d, tid))
        for w in waits:
            wait_of.setdefault((d, w), []).append((d, tid))
    seq = itertools.count()
    ev = []                                           # (time, phase, order, kind, payload)
    phase = {'done': 0, 'arrive': 0, 'ready': 1, 'free': 2}      # at one instant: finish things, then admit, then dispatch
    push = lambda t, kind, x: heapq.heappush(ev, (t, phase[kind], next(seq), kind, x))
    for k, n in pend.items():
        if n == 0:
            push(0.0, 'ready', k)
    dev_busy, dev_ready = [0.0] * P, [[] for _ in range(P)]
    lanes = [[0.0] * cl.comm_lanes for _ in range(P)]
    done, compute = {}, [0.0] * P
    is_comm = lambda op: op in ('put_signal', 'send_proxy', 'reduce_put_signal', 'signal')

    def release(key, at):
        pend[key] -= 1
        if pend[key] == 0:
            push(at, 'ready', key)

    def dispatch(d, now):
        if dev_busy[d] > now or not dev_ready[d]:
            return
        dev_ready[d].sort(key=lambda x: (-x[0], x[1], x[2]))
        _, _, _, key = dev_ready[d].pop(0)
        t = task[key]
        c = op_cost_us(t['op'], t['args'], cl)
        dev_busy[d] = now + c
        compute[d] += c
        push(now + c, 'done', key)
        push(now + c, 'free', d)

    while ev:
        now, _, _, kind, x = heapq.heappop(ev)
        if kind == 'ready':
            d = x[0]
            t = task[x]
            if is_comm(t['op']):
                a = t['args']
                nbytes = 4 if t['op'] == 'signal' else (a['n'] if t['op'] == 'reduce_put_signal' else a['rows'] * a['cols']) * 4
                dst = int(a['flag'].split('/')[1]) if 'flag' in a else a['dst_rank']
                ln = min(range(cl.comm_lanes), key=lambda i: lanes[d][i])
                st = max(now, lanes[d][ln])
                lanes[d][ln] = st + cl.task_overhead_us + nbytes / (cl.link(d, dst).bw_gbs * 1e3)
                push(st + cl.task_overhead_us, 'done', x)
                if a.get('var', -1) >= 0:
                    push(st + cl.xfer_us(d, dst, nbytes), 'arrive', (dst, vname(dst, a['var'])))
            else:
                dev_ready[d].append((t.get('priority', 0), now, next(seq), x))
                dispatch(d, now)
        elif kind == 'done':
            done[x] = now
            for nxt in deps_of.get(x, []):
                release(nxt, now)
        elif kind == 'arrive':
            for w in wait_of.get(x, []):
                release(w, now)
        else:
            dispatch(x, now)
    _need(len(done) == len(task), f'estimate: {len(task) - len(done)} tasks never became ready (deadlock in the plan)')
    mk = max(done.values())
    return dict(makespan_us=mk, compute_us=compute, comm_bytes=plan.comm_bytes, comm_msgs=plan.comm_msgs,
                utilisation=[c / mk if mk else 0 for c in compute], finish=done)


def compile_shard(shard: Shard):
    """Compile one shard. Variables fed by remote publishers are published OUTSIDE this batch, which the compiler would reject
    ("can never be satisfied"); telling it they are already at the highest version this shard waits for (base_versions) makes
    it keep the waits. The runtime must therefore start those variables at 0, not at the compiled base: write_plan, the
    simulator and the harness all do (the `var` lines of dist.txt carry the runtime start value)."""
    need = {}
    for t in shard.spec['graphs'][0]['tasks']:
        for r in t.get('reads', []):
            if r['var'] in shard.in_vars:
                need[r['var']] = max(need.get(r['var'], 0), r['version'])
    return cw.compile_spec(copy.deepcopy(shard.spec), need)


def write_plan(plan: Plan, out_dir, golden=None):
    """out_dir/rank<r>/{tasks,successors,args}.bin + dist.txt (+ expect lines). dist.txt is line based so the C++ harness
    needs no JSON parser; see tests/dist/test_dist.cu."""
    out = Path(out_dir)
    for s in plan.shards:
        compiled = compile_shard(s)
        d = out / f'rank{s.rank}'
        cw.write_bundle(compiled, d)
        lines = [f'rank {s.rank} world {len(plan.shards)}', f'nvars {len(compiled.variables)}']
        for b in compiled.buffers:
            rem = s.remotes.get(b.name)
            if rem is None:
                m = s.meta.get(b.name, {})
                g = m.get('gen')
                tail = ' i32' if b.dtype == 'i32' else ''
                if g and g['kind'] == 'gen':
                    pl = m['place']
                    lines.append(f'buf {b.index} local {b.rows} {b.cols} gen {g["seed"]} {g["draws"]} {g["bias"]!r} {g["coef"]!r} '
                                 f'{g["gr"]} {g["gc"]} {1 if g.get("transpose") else 0} {pl[0]} {pl[1]}{tail}')
                elif g and g['kind'] == 'fill':
                    lines.append(f'buf {b.index} local {b.rows} {b.cols} fill {g["value"]!r}{tail}')
                elif b.init['kind'] == 'fill':
                    lines.append(f'buf {b.index} local {b.rows} {b.cols} fill {b.init["value"]!r}{tail}')
                else:
                    lines.append(f'buf {b.index} local {b.rows} {b.cols} values ' + ' '.join(repr(v) for v in b.materialize()) + tail)
            elif rem['kind'] == 'peer':
                lines.append(f'buf {b.index} peer {rem["rank"]} {plan.shards[rem["rank"]].buf_index(rem["buf"])} {b.rows} {b.cols}')
            elif rem['kind'] == 'inbox':
                lines.append(f'buf {b.index} inbox {rem["rank"]} {b.cols}')
            else:
                lines.append(f'buf {b.index} sendq')
        names = [v.name for v in compiled.variables]
        for v in compiled.variables:
            lines.append(f'var {v.id} {0 if v.name in s.in_vars else v.base_version}')
        for nme in s.in_vars:
            lines.append(f'inbox_var {names.index(nme)}')
        for nme in plan.strategy.get('dump', []):
            if nme in s.meta and s.meta[nme].get('place'):
                bi = next(b.index for b in compiled.buffers if b.name == nme)
                lines.append(f'dump {bi} {nme} ' + ' '.join(str(x) for x in s.meta[nme]['place']))
        if golden is not None:
            for name, lay in plan.layout.items():
                if plan.strategy['mode'] == 'place':
                    dev, phys = lay
                    if dev != s.rank:
                        continue
                    vals, target = golden[name], phys
                else:
                    if lay == PART:
                        continue
                    vals, target = expected_shard(plan, golden, name, lay, s.rank), name
                tb = next(b for b in compiled.buffers if b.name == target)
                if tb.dtype == 'i32':
                    continue                  # the harness compares floats; integer buffers (step counters) are inputs only
                bi = tb.index
                lines.append(f'expect {bi} {len(vals)} ' + ' '.join(repr(float(v)) for v in vals))
        (d / 'dist.txt').write_text('\n'.join(lines) + '\n')
    log.info('plan written to %s', out)


def expected_shard(plan, golden, name, lay, d):
    b = plan.prog.buffers[name]
    full = golden[name]
    if lay == R:
        return full
    P = len(plan.shards)
    if lay == ROWS:
        rr = b['rows'] // P
        return full[d * rr * b['cols']:(d + 1) * rr * b['cols']]
    cs = b['cols'] // P
    return [full[r * b['cols'] + d * cs + j] for r in range(b['rows']) for j in range(cs)]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('workload')
    p.add_argument('cluster')
    p.add_argument('strategy')
    p.add_argument('out_dir')
    p.add_argument('--plan', action='store_true', help='also write per-rank bundles + dist.txt for tests/dist/test_dist')
    p.add_argument('--log-level', default='INFO')
    args = p.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    try:
        spec = json.loads(Path(args.workload).read_text())
        plan = partition(spec, Cluster.from_dict(json.loads(Path(args.cluster).read_text())), json.loads(Path(args.strategy).read_text()))
    except (PartitionError, cw.CompileError, OSError, json.JSONDecodeError) as e:
        log.error('%s', e)
        return 1
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for s in plan.shards:
        (out / f'shard{s.rank}.json').write_text(json.dumps(s.spec) + '\n')
    if args.plan:
        import workload_vm
        golden = workload_vm.run_golden(cw.compile_spec(copy.deepcopy(plan.spec)))['buffers']
        write_plan(plan, out / 'plan', golden)
    est = estimate(plan)
    log.info('modeled makespan %.1f us; comm %d messages, %.1f KiB', est['makespan_us'], est['comm_msgs'], est['comm_bytes'] / 1024)
    return 0


if __name__ == '__main__':
    sys.exit(main())
