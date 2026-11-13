import numpy as np, csv, collections
rows=list(csv.DictReader(open("logs/trace_162m_best.csv")))
op=np.array([r["op"] for r in rows]); S=np.array([float(r["start_us"]) for r in rows]); E=np.array([float(r["end_us"]) for r in rows])
R=np.array([float(r["ready_us"]) for r in rows]); B=np.array([int(r["block"]) for r in rows])
t0=S.min(); S-=t0;E-=t0;R-=t0
span=E.max(); W=107
# sweep: idle workers over time = W - running
ev=[(s,1,i) for i,s in enumerate(S)]+[(e,-1,i) for i,e in enumerate(E)]
ev.sort()
run=set(); last=0.0; idle_by_op=collections.defaultdict(float); idle_total=0.0
# time-weighted idle worker-us, attributed to the ops running in that moment, proportional by task
for t,k,i in ev:
    dt=t-last
    if dt>0 and len(run)<W:
        idle=(W-len(run))*dt; idle_total+=idle
        if run:
            c=collections.Counter(op[j] for j in run)
            for o,n in c.items(): idle_by_op[o]+=idle*n/len(run)
        else: idle_by_op["(none running)"]+=idle
    last=t
    if k==1: run.add(i)
    else: run.discard(i)
print("idle worker-time %.1f ms of %.1f (%.1f%%)"%(idle_total/1e3,span*W/1e3,100*idle_total/(span*W)))
for o,v in sorted(idle_by_op.items(),key=lambda x:-x[1])[:12]: print("  idle while running %-24s %.1f ms (%.1f%% of idle)"%(o,v/1e3,100*v/idle_total))
# timeline in 5ms buckets: busy fraction
nb=int(span//5000)+1; busy=np.zeros(nb)
for s,e in zip(S,E):
    b=int(s//5000)
    while b*5000<e and b<nb:
        busy[b]+=min(e,(b+1)*5000)-max(s,b*5000); b+=1
print("busy per 5ms:", " ".join("%.0f"%(100*x/(5000*W)) for x in busy))
