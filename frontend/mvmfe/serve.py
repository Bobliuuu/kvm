"""Serving graph builder: prefill + greedy decode with a KV cache, one graph per request, parameterized by the prompt.

A request reads the trainer's weight buffers under a version policy of the `weights` variable (see train.py):
  exact    -> the request sees exactly the weights stable at `version`; the compiler orders it before the next overwrite
  at_least -> any version >= `version`; reads whatever the buffers hold (may mix two versions if a step lands mid-request)
Requests are ordinary graphs on the same scheduler (own priority and deadline); nothing here knows about training.

Structure: the first P-1 prompt tokens are prefilled in one batch (rows tiled like training, K/V appended to the cache), then every
step (the last prompt token, then each generated token) is a one-row decode: embed -> layers -> head -> argmax. All decode buffers
have one row, so no op needs row offsets into a larger buffer.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import log
from .emit import GraphPolicy, lower_program, to_workload
from .ir import Builder, FrontendError, Tensor
from .lower import Tiling


@dataclass(frozen=True)
class ServeRequest:
    name: str
    prompt: tuple
    max_new: int = 4
    version: int = 0                 # `weights` variable version (train.weights_version(k) = stable after k updates)
    policy: str = 'exact'            # exact | at_least
    priority: int = 10
    deadline_us: int = 2000
    context: int = 0                 # tokens the KV cache already holds (a decode step at that context length; zero-filled rows here)


def build_serving(model, req: ServeRequest) -> Builder:
    cfg = model.cfg
    P, N = len(req.prompt), req.max_new
    if P < 1 or N < 1:
        raise FrontendError(f'{req.name}: need a nonempty prompt and max_new >= 1')
    if any(not 0 <= t < cfg.vocab for t in req.prompt):
        raise FrontendError(f'{req.name}: prompt token outside [0, {cfg.vocab})')
    if req.context < 0:
        raise FrontendError(f'{req.name}: context must be >= 0, got {req.context}')
    if req.context and P > 1:   # the prefill's attention reads its own k/v only, not the cache: it would ignore the context
        raise FrontendError(f'{req.name}: a context needs a one-token prompt (the prefill does not attend to the cache)')
    if req.policy not in ('exact', 'at_least'):
        raise FrontendError(f'policy must be exact or at_least, got {req.policy!r}')
    b = Builder(req.name)
    b.phase = 'serve'
    H, Hkv, d, Tmax = cfg.heads, model.kv_heads, cfg.d_model, req.context + P - 1 + N
    hd = d // H
    w = {n: Tensor(s.name, s.rows, s.cols, 'param', buf=f'w.{s.name}') for n, s in model.specs.items()}
    pin = lambda n, r, c, init=None, dtype='f32': Tensor(f'{req.name}.{n}', r, c, 'state', buf=f'{req.name}.{n}', init=init, dtype=dtype)
    lens = pin('lens', 1, 1, ('fill', float(req.context)), 'i32')
    cdt = 'bf16' if getattr(cfg, 'precision', 'f32') == 'bf16s' else 'f32'   # bf16s: the cache is bf16 storage (k, v are bf16 values already)
    kc = [pin(f'kc{i}', Hkv * Tmax, hd, ('fill', 0.0), cdt) for i in range(cfg.layers)]
    vc = [pin(f'vc{i}', Hkv * Tmax, hd, ('fill', 0.0), cdt) for i in range(cfg.layers)]
    b.tensors += [lens, *kc, *vc]

    if P > 1:                                                    # prefill the first P-1 tokens
        n = P - 1
        ids = pin('prompt', n, 1, ('values', [float(t) for t in req.prompt[:-1]]), 'i32')
        x = model.embed(b, ids, w, n)
        for l, blk in enumerate(model.blocks):
            with b.scope(f'prefill.h{l}'):
                q, k, v = blk.attn_inputs(b, x, w, n)
                b.kv_append(k, v, kc[l], vc[l], lens, n, Hkv, Tmax)
                x = blk.ffn_block(b, blk.attn_out(b, b.attention(q, k, v, H, n, Hkv), w, x), w)
        b.counter_inc(lens, n)
        b.tensors.append(ids)

    cur = pin('cur0', 1, 1, ('values', [float(req.prompt[-1])]), 'i32')
    for i in range(N):                                           # one-row decode steps
        with b.scope(f'dec{i}'):
            x = model.embed(b, cur, w, 1)
            for l, blk in enumerate(model.blocks):
                q, k, v = blk.attn_inputs(b, x, w, 1, lens)
                b.kv_append(k, v, kc[l], vc[l], lens, 1, Hkv, Tmax)
                x = blk.ffn_block(b, blk.attn_out(b, b.attn_decode(q, kc[l], vc[l], lens, H, Tmax, Hkv), w, x), w)
            b.counter_inc(lens, 1)
            nxt = pin(f'cur{i + 1}', 1, 1, None, 'i32')
            b.tensors.append(nxt)
            b.argmax(model.logits(b, x, w), nxt)
            cur = nxt
    log.info('serve %s: prompt %d tokens, %d new, weights %s version %d', req.name, P, N, req.policy, req.version)
    return b


def compile_serving(model, req: ServeRequest, tiling: Tiling, reuse='shape', weights='master', gemv_rows=0, attn_splits=0):
    """weights='master' (default): the GEMMs read the trainer's fp32 master buffers (rounded to bf16 inside the tile in a bf16 precision).
    weights='bf16': the buffers a GEMM alone reads are stored bf16 (ABI dtype 2), the serving copy a trainer would publish once per
    weight version; the GEMV tiles then stream half the bytes. The values the tiles multiply are the same (the tile rounded them on load).
    Under precision 'bf16s' the storage pass (storage.py) also runs, without weight shadows (weights are stored bf16 directly here,
    not refreshed from a master every step): casts fold into their producers and activations are bf16 storage."""
    if weights not in ('master', 'bf16', 'bf16t'):
        raise FrontendError(f"weights must be 'master', 'bf16' or 'bf16t', got {weights!r}")
    if gemv_rows and (weights != 'bf16t' or reuse != 'none' or gemv_rows not in (8, 16, 32, 64)):
        raise FrontendError("gemv_rows needs weights='bf16t', reuse='none' and 8, 16, 32 or 64 rows a task")
    b = build_serving(model, req)
    reuse = 'shape' if reuse == 'step' else reuse            # a request is one step: 'step' would mean no reuse at all
    ops = b.ops
    if getattr(model.cfg, 'precision', 'f32') == 'bf16s':
        from . import storage
        ops, _ = storage.apply(ops, weights=False)
    low = lower_program(ops, tiling, reuse, 0, prefix=f'{req.name}.act')
    spec = to_workload(low, {req.name: GraphPolicy(req.priority, lambda t: req.deadline_us, (req.policy, req.version), rank_levels=0)})
    if weights in ('bf16', 'bf16t'):
        _store_gemm_weights_bf16(spec, transpose=weights == 'bf16t')
    if gemv_rows:
        _decode_gemv(spec, gemv_rows)
    if attn_splits:
        _decode_attention(spec, attn_splits)
    if gemv_rows or attn_splits:
        _dissolve_small_joins(spec)
    return low, spec


def _dissolve_small_joins(spec, limit=16):
    """Drop join noops whose fan-in fits one static-schedule record (MVM_S_WAITS = 16): their dependents wait on the sources directly.
    A join exists to cut fan-in x fan-out edges; after the decode passes some joins wait on a few merges, and as their own task they cost
    a dependent level (a hop) on the critical path."""
    n = 0
    for g in spec['graphs']:
        joins = {t['id']: t.get('after', []) for t in g['tasks'] if t['op'] == 'noop' and not t.get('args') and len(t.get('after', [])) <= limit
                 and not t.get('publishes') and not t.get('reads')}
        if not joins:
            continue

        def src(d, seen=()):
            return [x for e in joins[d] for x in (src(e, seen + (d,)) if e in joins and e not in seen else [e])] if d in joins else [d]
        for t in g['tasks']:
            if t.get('after') and any(d in joins for d in t['after']):
                t['after'] = list(dict.fromkeys(x for d in t['after'] for x in src(d)))
        g['tasks'] = [t for t in g['tasks'] if t['id'] not in joins]
        n += len(joins)
    if n:
        log.info('serve: %d small join noops dissolved', n)

def _decode_attention(spec, S):
    """Each decode step's attention as split-context tasks (mvm_ops_extra.def attn_decode_part / attn_decode_merge).

    The H attn_decode tasks of one step (one per query head) become, per kv head, S attn_decode_part tasks over S slices of the context
    for the whole GQA group (k and v read once, not H / Hkv times), and one attn_decode_merge task for the group's heads. Consumers of
    a head's attention wait for its group's merge. The partials live in a scratch buffer `<o>.attnpart` [S*H, hd+2] per step.

    When q is a one-row rope_fwd's output that only the attention reads, and the cache is written by a one-token kv_append whose k is a
    one-row rope_fwd's output that only the append reads, the three fold into the split tasks (flags 1, 2 when q was stored bf16): they
    rotate q themselves, and the split holding the new key appends the rotated k and v. Whatever waited on the ropes or the append then
    waits on the group's merges (the cache row is written by then)."""
    if S < 1:
        raise FrontendError(f'attn_splits must be >= 1, got {S}')
    names = {b['name'] for b in spec['buffers']}
    ngroups = 0
    for g in spec['graphs']:
        groups: dict = {}
        for t in g['tasks']:
            if t['op'] == 'attn_decode':
                a = t['args']
                groups.setdefault((a['q'], a['kc'], a['vc'], a['lens'], a['o'], a['b']), []).append(t)
        if not groups:
            continue
        repl: dict = {}
        new_of: dict = {}
        by_id = {t['id']: t for t in g['tasks']}
        readers: dict = {}
        for t in g['tasks']:
            for v in (t.get('args') or {}).values():
                if isinstance(v, str):
                    readers.setdefault(v, []).append(t['id'])
        dtype = {b['name']: b.get('dtype', 'f32') for b in spec['buffers']}

        def rope_of(buf, only):   # the one-row rope_fwd producing `buf`, read only by the tasks in `only`
            prods = [t for t in g['tasks'] if t['op'] == 'rope_fwd' and t['args'].get('out') == buf]
            if len(prods) != 1:
                return None
            r = prods[0]['args']
            if not (r['r0'] == 0 and r['r1'] == 1 and r['T'] == 1) or not set(readers.get(buf, [])) <= only | {prods[0]['id']}:
                return None
            return prods[0]
        drop: dict = {}
        for key, ts in groups.items():
            a0 = ts[0]['args']
            H, Hkv, hd = a0['H'], a0['Hkv'], a0['hd']
            G = H // Hkv
            if sorted(t['args']['h'] for t in ts) != list(range(H)):
                raise FrontendError(f"{ts[0]['id']}: a decode attention needs all {H} heads to split it")
            part = f"{a0['o']}.attnpart"
            if part not in names:
                spec['buffers'].append({'name': part, 'rows': (a0['b'] + 1) * S * H, 'cols': hd + 2, 'init': {'kind': 'fill', 'value': 0.0}})
                names.add(part)
            after = list(dict.fromkeys(d for t in ts for d in t.get('after', [])))
            extra = {k: v for k, v in ts[0].items() if k not in ('id', 'op', 'args', 'after')}
            fold = None   # (rope q, rope k, kv_append) to fold into this group's split tasks
            rq = rope_of(a0['q'], {t['id'] for t in ts})
            apps = [by_id[d] for d in after if by_id.get(d, {}).get('op') == 'kv_append']
            if rq and rq['args']['pos'] == a0['lens'] and len(apps) == 1:
                ka = apps[0]['args']
                if (ka['kc'], ka['vc'], ka['lens']) == (a0['kc'], a0['vc'], a0['lens']) and (ka['b0'], ka['b1'], ka['n_new']) == (0, 1, 1):
                    rk = rope_of(ka['knew'], {apps[0]['id']})
                    if rk and rk['args']['pos'] == a0['lens'] and rk['args']['theta'] == rq['args']['theta']:
                        fold = (rq, rk, apps[0])
            if fold:
                rq, rk, ka = fold
                gone = {rq['id'], rk['id'], ka['id']}
                after = list(dict.fromkeys([d for d in after if d not in gone] + [d for t in fold for d in t.get('after', []) if d not in gone]))
            tasks = []
            for kvh in range(Hkv):
                pids = []
                for sp in range(S):
                    tid = f"{ts[0]['id']}#attn{kvh}.{sp}"
                    args = {'q': a0['q'], 'kc': a0['kc'], 'vc': a0['vc'], 'lens': a0['lens'], 'part': part, 'knew': None, 'vnew': None,
                            'b': a0['b'], 'kvh': kvh, 's': sp, 'S': S, 'H': H, 'Hkv': Hkv, 'hd': hd, 'Tmax': a0['Tmax'], 'scale': a0['scale'],
                            'theta': 0.0, 'flags': 0}
                    if fold:
                        args.update({'q': rq['args']['x'], 'knew': rk['args']['x'], 'vnew': ka['args']['vnew'], 'theta': rq['args']['theta'],
                                     'flags': 1 | (2 if dtype.get(a0['q']) == 'bf16' else 0)})
                    tasks.append({'id': tid, 'op': 'attn_decode_part', **extra, 'after': list(after), 'args': args})
                    pids.append(tid)
                mid = f"{ts[0]['id']}#merge{kvh}"
                tasks.append({'id': mid, 'op': 'attn_decode_merge', **extra, 'after': pids,
                              'args': {'part': part, 'o': a0['o'], 'b': a0['b'], 'h0': kvh * G, 'h1': (kvh + 1) * G, 'S': S, 'H': H, 'hd': hd}})
                for t in ts:
                    if t['args']['h'] // G == kvh:
                        repl[t['id']] = [mid]
            new_of[ts[0]['id']] = tasks
            if fold:
                merges = [t['id'] for t in tasks if t['op'] == 'attn_decode_merge']
                for t in fold:
                    drop[t['id']] = merges
            ngroups += 1
        repl.update(drop)
        out = []
        for t in g['tasks']:
            if t['id'] in drop:
                continue
            out += new_of.get(t['id'], []) if t['id'] in repl else [t]
        for t in out:   # new tasks too: a later step's split tasks inherit waits on an earlier step's dropped append
            if t.get('after'):
                t['after'] = list(dict.fromkeys(x for d in t['after'] for x in repl.get(d, [d])))
        nfold = len(drop) // 3
        g['tasks'] = out
    log.info('serve: %d decode attentions split %d ways over the context; %d with their rope and kv_append folded in', ngroups, S, nfold)

def _decode_gemv(spec, rows):
    """One-row decode GEMMs as gemv_bf16 tasks (mvm_ops_extra.def) of `rows` outputs each, with their norm / SwiGLU folded in.

    A matmul_tile_bf16 task of M = 1 over a transposed bf16 weight (flag 2 alone, no bias / pre / accumulate, alpha 1) becomes one
    gemv_bf16 task per `rows` of its 64 output columns, reading the [N, K] weight rows contiguously; its consumers wait for all of them.
    Then a one-row rmsnorm_fwd or silu_mul_fwd whose output only such GEMVs read (and whose rstd nobody reads) is folded into those
    GEMVs' prologue (pro 1 / 2) and dropped: one dependent level less each, and its output is never stored. Activation buffers must be
    written once per step (reuse 'none'), so a buffer name identifies one value."""
    nsplit = nfold = 0
    for g in spec['graphs']:
        tasks = g['tasks']
        readers: dict = {}
        for t in tasks:
            for v in (t.get('args') or {}).values():
                if isinstance(v, str):
                    readers.setdefault(v, []).append(t['id'])
        repl: dict = {}
        out = []
        for t in tasks:
            a = t.get('args') or {}
            if (t['op'] == 'matmul_tile_bf16' and a.get('M') == 1 and a.get('m0') == 0 and a.get('flags') == 2 and a.get('alpha') == 1.0
                    and a.get('bias') is None and a.get('pre') is None):
                n0, n1 = a['n0'], min(a['n0'] + 64, a['N'])
                ids = []
                for j, c0 in enumerate(range(n0, n1, rows)):
                    nt = {k: v for k, v in t.items() if k not in ('op', 'args', 'id')}
                    nt['id'] = f"{t['id']}#g{j}"
                    nt['op'] = 'gemv_bf16'
                    nt['args'] = {'x': a['a'], 'x2': None, 'gamma': None, 'w': a['b'], 'out': a['out'], 'res': a.get('res'),
                                  'n0': c0, 'n1': min(c0 + rows, n1), 'K': a['K'], 'pro': 0, 'eps': 0.0}
                    out.append(nt)
                    ids.append(nt['id'])
                repl[t['id']] = ids
                nsplit += 1
            else:
                out.append(t)
        for t in out:
            if t.get('after'):
                t['after'] = [x for d in t['after'] for x in repl.get(d, [d])]
        # fold one-row norms and SwiGLUs into the GEMVs that alone read them
        by_id = {t['id']: t for t in out}
        readers = {}
        for t in out:
            for v in (t.get('args') or {}).values():
                if isinstance(v, str):
                    readers.setdefault(v, []).append(t['id'])
        drop = set()
        for t in out:
            a = t.get('args') or {}
            if t['op'] == 'rmsnorm_fwd' and a.get('r0') == 0 and a.get('r1') == 1:
                pro, extra = 1, {'gamma': a['gamma'], 'eps': a['eps']}
                src, src2, K, outs = a['x'], None, a['d'], (a['out'],)
                if readers.get(a['rstd'], []) != [t['id']]:
                    continue
            elif t['op'] == 'silu_mul_fwd' and a.get('e0') == 0:
                pro, extra = 2, {}
                src, src2, K, outs = a['gate'], a['up'], a['e1'], (a['out'],)
            else:
                continue
            cons = [by_id[c] for c in readers.get(outs[0], []) if c != t['id']]
            if not cons or any(c['op'] != 'gemv_bf16' or c['args']['x'] != outs[0] or c['args']['K'] != K or c['args']['pro'] for c in cons):
                continue
            for c in cons:
                c['args'].update({'x': src, 'x2': src2, 'pro': pro, **extra})
                c['after'] = list(dict.fromkeys([d for d in c.get('after', []) if d != t['id']] + list(t.get('after', []))))
            drop.add(t['id'])
            nfold += 1
        if drop:
            for t in out:
                if t.get('after') and any(d in drop for d in t['after']):
                    t['after'] = list(dict.fromkeys(x for d in t['after'] for x in (by_id[d].get('after', []) if d in drop else [d])))
        g['tasks'] = [t for t in out if t['id'] not in drop]
    log.info('serve: %d one-row GEMM tiles as gemv_bf16 tasks of %d rows; %d norms / SwiGLUs folded into their GEMVs', nsplit, rows, nfold)

def _store_gemm_weights_bf16(spec, transpose=False):
    """Mark every weight buffer (w.*) whose every use is the b operand of a bf16 matmul tile as bf16 storage.

    transpose: store it [N, K] instead of [K, N] (the tiles read it with flag 2, transB), so a GEMV tile's weight stream is its 64 rows of
    K contiguous elements instead of K strided 128-byte pieces. The serving copy is then the master transposed (a publish step does both)."""
    uses: dict = {}
    for g in spec['graphs']:
        for t in g['tasks']:
            for k, v in (t.get('args') or {}).items():
                if isinstance(v, str):
                    uses.setdefault(v, set()).add((t['op'], k))
    chosen = set()
    for buf in spec['buffers']:
        u = uses.get(buf['name'])
        if buf['name'].startswith('w.') and u and all(op in ('matmul_tile_bf16', 'matmul_mma_bf16') and k == 'b' for op, k in u):
            buf['dtype'] = 'bf16'
            chosen.add(buf['name'])
            if transpose:
                buf['rows'], buf['cols'] = buf['cols'], buf['rows']
    if transpose:
        for g in spec['graphs']:
            for t in g['tasks']:
                a = t.get('args') or {}
                if a.get('b') in chosen:
                    if a['flags'] & 2:
                        raise FrontendError(f"{t['id']}: weight {a['b']} is already read transposed")
                    a['flags'] |= 2
    log.info('serve: %d GEMM weight buffers stored bf16%s', len(chosen), ' [N, K]' if transpose else '')
