"""Where one 162M bf16s step goes, from kvm_exec --trace-csv plus the bundle's tasks.bin/successors.bin (slot == task index).
usage: python3 -I analyze_trace.py TRACE.csv BUNDLE_DIR"""
import sys, heapq, collections
import numpy as np, pandas as pd
trace, bd = sys.argv[1], sys.argv[2]
d = pd.read_csv(trace); d = d[d.run_us >= 0].set_index('slot').sort_index()
t = np.fromfile(f'{bd}/tasks.bin', dtype='<i4').reshape(-1, 16); s = np.fromfile(f'{bd}/successors.bin', dtype='<i4')
N = len(t); assert N == len(d); off, cnt, indeg = t[:, 1], t[:, 2], t[:, 3]
run, st, en, rdy, adm, op = (d[c].values for c in ('run_us', 'start_us', 'end_us', 'ready_us', 'admit_us', 'op'))
W = d.block.nunique(); t0, t1 = st.min(), en.max(); span = t1 - t0
print(f'tasks {N}, workers {W}, first start -> last end {span/1e3:.1f} ms, sum of task run times {run.sum()/1e3:.0f} CTA-ms, busy {run.sum()/(span*W):.1%}')
g = d.groupby('op').agg(n=('run_us', 'size'), cta_ms=('run_us', lambda x: x.sum()/1e3), mean_us=('run_us', 'mean'), qwait_us=('queue_wait_us', 'mean')).sort_values('cta_ms', ascending=False)
print(g.head(14).round(1).to_string())
def sweep(a, b):
    ev = np.concatenate([np.c_[a, np.ones(len(a))], np.c_[b, -np.ones(len(b))]]); ev = ev[np.argsort(ev[:, 0], kind='stable')]
    return ev[:, 0], np.cumsum(ev[:, 1])
def at(tt, v, x):
    i = np.searchsorted(tt, x, side='right') - 1; return np.where(i >= 0, v[np.clip(i, 0, None)], 0)
gr = np.arange(t0, t1, 500.0)
R = at(*sweep(st, en), gr); Q = at(*sweep(rdy, st), gr)
idle = W - R
print(f'worker-time: running {R.mean()/W:.1%}; idle with a ready task waiting {(idle*(Q>0)).mean()/W:.1%}; idle with nothing ready {(idle*(Q==0)).mean()/W:.1%}')
g1 = np.arange(t0, t1, 1000.0); R1 = at(*sweep(st, en), g1); Q1 = at(*sweep(rdy, st), g1); S1 = np.histogram(st, bins=np.r_[g1, g1[-1] + 1000])[0]
m = Q1 >= 100
print(f'backlog (>=100 ready): {m.sum()} ms, running {R1[m].mean():.0f}, {S1[m].mean():.0f} starts/ms = {1000/S1[m].mean():.2f} us per start; best 1 ms window {S1.max()} starts')
print(f'ready->start wait: median {np.median(st-rdy):.0f} us, mean {(st-rdy).mean():.0f} us')
# dependency structure
order = []; dg = indeg.copy(); stack = list(np.where(dg == 0)[0]); lvl = np.zeros(N, int); cp = np.zeros(N)
while stack:
    u = stack.pop(); order.append(u)
    for v in s[off[u]:off[u]+cnt[u]]:
        lvl[v] = max(lvl[v], lvl[u]+1); cp[v] = max(cp[v], cp[u]+run[u]); dg[v] -= 1
        if dg[v] == 0: stack.append(v)
assert len(order) == N
print(f'dependency levels {lvl.max()+1}; critical path with measured run times and zero hop cost {(cp+run).max()/1e3:.1f} ms')
def sched(P, r, hop=0.0, pop=0.0):
    dg = indeg.copy(); av = [0.0]*P; heapq.heapify(av); rt = np.zeros(N); h = [(0.0, int(u)) for u in np.where(dg == 0)[0]]; heapq.heapify(h); mk = 0.0; pf = 0.0
    while h:
        rtime, u = heapq.heappop(h); w = heapq.heappop(av); b = max(w, rtime, pf + pop); pf = b if pop else pf   # pop: one start per `pop` us
        e = b + r[u]; mk = max(mk, e); heapq.heappush(av, e)
        for v in s[off[u]:off[u]+cnt[u]]:
            rt[v] = max(rt[v], e + hop); dg[v] -= 1
            if dg[v] == 0: heapq.heappush(h, (rt[v], int(v)))
    return mk
mm = op == 'matmul_mma_bf16'; at_ = np.char.startswith(op.astype(str), 'attn_')
print('list schedule (greedy, tape order), same tasks and measured run times:')
for hop, pop in ((0, 0), (7, 0), (28, 0), (0, 1.6), (0, 2.2)):
    print(f'  {W} workers, hop {hop} us, one start per {pop} us: {sched(W, run, hop, pop)/1e3:.1f} ms')
for lab, sc in (('matmul tiles x0.45 (cuBLAS-class efficiency)', np.where(mm, 0.45, 1.0)), ('matmul x0.45, attention x0.5', np.where(mm, 0.45, np.where(at_, 0.5, 1.0)))):
    print(f'  {lab}: zero overhead {sched(W, run*sc)/1e3:.1f} ms, one start per 1.6 us {sched(W, run*sc, 0, 1.6)/1e3:.1f} ms')
