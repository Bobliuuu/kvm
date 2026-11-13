import numpy as np, csv, sys, collections
rows=list(csv.DictReader(open(sys.argv[1])))
f=lambda k:np.array([float(r[k]) for r in rows])
w=np.array([int(r["worker"]) for r in rows]); S=f("start_ns");R=f("ready_ns");D=f("done_ns");P=f("published_ns"); op=np.array([r["op"] for r in rows]); nw=f("nwait")
ok=S>0
t0=S[ok].min(); span=(P[ok].max()-t0)
print("records",ok.sum(),"span us %.1f"%(span/1e3),"workers",len(set(w)))
run=(D-S)[ok]; pub=(P-D)[ok]; wait=(S-R)[ok]
print("mean run %.2f us, publish %.2f us, start-ready %.2f us (sum run %.0f us)"%(run.mean()/1e3,pub.mean()/1e3,wait.mean()/1e3,run.sum()/1e3))
# per worker busy
W=sorted(set(w[ok])); tot=0
for x in W:
    m=ok&(w==x); busy=(P[m]-S[m]).sum(); tot+=busy
print("busy (start..published) over %d workers: %.1f%% of worker-span"%(len(W),100*tot/(span*len(W))))
per=collections.defaultdict(lambda:[0,0.0])
for o,r,p in zip(op[ok],run,pub): per[o][0]+=1; per[o][1]+=r
for o,(c,s) in sorted(per.items(),key=lambda kv:-kv[1][1])[:8]: print("  %-24s %6d mean %.2f us"%(o,c,s/c/1e3))
print("--- split of start..done into wait (until ready) and exec")
we=np.maximum(R-S,0)[ok]; ex=(D-np.maximum(R,S))[ok]
print("mean wait-for-deps %.2f us, mean exec %.2f us, sum exec %.0f us -> exec utilisation %.1f%% of %d workers x span"%(we.mean()/1e3,ex.mean()/1e3,ex.sum()/1e3,100*ex.sum()/(span*len(W)),len(W)))
per=collections.defaultdict(lambda:[0,0.0,0.0])
for o,a,b in zip(op[ok],we,ex): per[o][0]+=1; per[o][1]+=a; per[o][2]+=b
for o,(c,a,b) in sorted(per.items(),key=lambda kv:-kv[1][2])[:10]: print("  %-22s %6d wait %.2f exec %.2f us"%(o,c,a/c/1e3,b/c/1e3))
