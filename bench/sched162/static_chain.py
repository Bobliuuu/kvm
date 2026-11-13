import numpy as np, csv, sys
rows=list(csv.DictReader(open(sys.argv[1]))); bundle=sys.argv[2]
n=int(sys.argv[3]) if len(sys.argv)>3 else None
task=np.array([int(r["task"]) for r in rows]); w=np.array([int(r["worker"]) for r in rows])
S=np.array([float(r["start_ns"]) for r in rows]);R=np.array([float(r["ready_ns"]) for r in rows]);D=np.array([float(r["done_ns"]) for r in rows]);P=np.array([float(r["published_ns"]) for r in rows])
nt=len(np.unique(task))
t=np.frombuffer(open(bundle+"/tasks.bin","rb").read(),dtype=np.int32).reshape(-1,16); succ=np.frombuffer(open(bundle+"/successors.bin","rb").read(),dtype=np.int32)
# index by task id; the task array may be longer than records (wait-only records share task id): use last record of a task
idx={}
for i,tk in enumerate(task): idx[tk]=i
Dt=np.full(len(t),np.nan);St=np.full(len(t),np.nan);Rt=np.full(len(t),np.nan);Wt=np.full(len(t),-1);Pt=np.full(len(t),np.nan)
for tk,i in idx.items(): Dt[tk]=D[i];St[tk]=S[i];Rt[tk]=R[i];Wt[tk]=w[i];Pt[tk]=P[i]
pred_best=np.full(len(t),-1);pred_t=np.full(len(t),-1.0)
for j in range(len(t)):
    if np.isnan(Dt[j]): continue
    for s in succ[t[j,1]:t[j,1]+t[j,2]]:
        if Dt[j]>pred_t[s]: pred_t[s]=Dt[j];pred_best[s]=j
hx=[];hl=[];ex=[]
for i in range(len(t)):
    j=pred_best[i]
    if j<0 or np.isnan(Dt[i]): continue
    if Wt[j]==Wt[i]: hl.append(St[i]-Pt[j] if False else max(Rt[i],St[i])-Dt[j])
    else: hx.append(max(Rt[i],St[i])-Dt[j])
    ex.append(Dt[i]-max(Rt[i],St[i]))
hx=np.array(hx)/1e3;hl=np.array(hl)/1e3
print("critical-pred done -> this task starts executing: cross-worker n=%d median %.2f mean %.2f us (p10 %.2f p90 %.2f); same-worker n=%d median %.2f mean %.2f"%(len(hx),np.median(hx),hx.mean(),np.percentile(hx,10),np.percentile(hx,90),len(hl),np.median(hl),hl.mean()))
# walk back the critical chain from the last finishing task
import collections
ops={}
for r in rows: ops[int(r["task"])]=r["op"]
cur=int(np.nanargmax(Dt)); tally=collections.defaultdict(lambda:[0,0.0,0.0]); L=0; tot_exec=0; tot_gap=0
while cur>=0:
    j=pred_best[cur]
    e=Dt[cur]-max(Rt[cur],St[cur]); g=(max(Rt[cur],St[cur])-Dt[j]) if j>=0 else 0
    tally[ops[cur]][0]+=1;tally[ops[cur]][1]+=e;tally[ops[cur]][2]+=g; L+=1; tot_exec+=e; tot_gap+=g
    cur=j
print("critical chain: %d tasks, exec %.0f us, gaps (hop+idle) %.0f us, total %.0f us  per task exec %.2f gap %.2f"%(L,tot_exec/1e3,tot_gap/1e3,(tot_exec+tot_gap)/1e3,tot_exec/L/1e3,tot_gap/L/1e3))
for o,(c,e,g) in sorted(tally.items(),key=lambda kv:-kv[1][1])[:12]: print("  %-22s n=%5d exec %.2f us gap-before %.2f us  (sum %.0f us)"%(o,c,e/c/1e3,g/c/1e3,(e+g)/1e3))
