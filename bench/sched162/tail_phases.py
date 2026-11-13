import numpy as np, csv, collections
rows=list(csv.DictReader(open("logs/trace_162m_best.csv")))
op=np.array([r["op"] for r in rows]); S=np.array([float(r["start_us"]) for r in rows]); E=np.array([float(r["end_us"]) for r in rows])
t0=S.min(); S-=t0;E-=t0
for lo in range(60000,116000,2500):
    m=(S>=lo)&(S<lo+2500)
    c=collections.Counter(op[m])
    busy=sum(min(e,lo+2500)-max(s,lo) for s,e in zip(S,E) if e>lo and s<lo+2500)/(2500*107)
    print("%6.1f-%6.1f ms busy %3.0f%% started: %s"%(lo/1e3,(lo+2500)/1e3,100*busy,", ".join("%s:%d"%kv for kv in c.most_common(5))))
# first-start/last-end per op class for long ops
for o in ["embedding_bwd","softmax_ce_fused","norm_bwd_params","gradnorm_partial","adamw","colsum"]:
    m=op==o; print(o,m.sum(),"start %.1f..%.1f ms, end %.1f ms, run mean %.0f us max %.0f"%(S[m].min()/1e3,S[m].max()/1e3,E[m].max()/1e3,(E[m]-S[m]).mean(),(E[m]-S[m]).max()))
