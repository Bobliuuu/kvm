"""Ready-queue depth / busy workers / starts per time bin from a kvm_exec --trace-csv (the analysis behind README section 1)."""
import csv,sys,numpy as np
T=sys.argv[1]   # kvm_exec --trace-csv of the 1-step gpt_162m tape
rows=list(csv.DictReader(open(T)))
a=lambda k:np.array([float(r[k]) for r in rows])
adm,rdy,st,en=a('admit_us'),a('ready_us'),a('start_us'),a('end_us')
m=(en>st); adm,rdy,st,en=adm[m],rdy[m],st[m],en[m]
print(len(st),"span ms",(en.max()-st.min())/1e3)
t0=st.min()
wait=st-rdy
print("wait us mean/median/p90/p99",wait.mean(),np.median(wait),np.percentile(wait,90),np.percentile(wait,99))
print("admit->ready mean",(rdy-adm).mean(), "median",np.median(rdy-adm))
# timeline in 20ms bins: tasks started, ready-queue depth (ready not started), busy workers
for b in range(int((en.max()-t0)//50000)+1):
    lo=t0+b*50000;hi=lo+50000
    started=((st>=lo)&(st<hi)).sum()
    busy=np.clip(np.minimum(en,hi)-np.maximum(st,lo),0,None).sum()/50000
    depth=((rdy<=hi)&(st>hi)).sum()
    w=wait[(st>=lo)&(st<hi)]
    print(f"{b*50:5d}ms started {started:6d} busy_workers {busy:6.1f} readyq_at_end {depth:6d} meanwait {w.mean() if len(w) else 0:9.1f}")
print("---- fine bins 300-312ms, 250us")
B=250
for b in range(int(300000//B),int(312000//B)):
    lo=t0+b*B;hi=lo+B
    started=((st>=lo)&(st<hi)).sum()
    busy=np.clip(np.minimum(en,hi)-np.maximum(st,lo),0,None).sum()/B
    depth=((rdy<=hi)&(st>hi)).sum()
    readied=((rdy>=lo)&(rdy<hi)).sum(); admitted=((adm>=lo)&(adm<hi)).sum()
    print(f"{b*B/1000:7.2f}ms started {started:5d} readied {readied:5d} admitted {admitted:5d} busy {busy:6.1f} q {depth:5d}")
