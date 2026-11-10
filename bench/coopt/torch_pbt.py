"""PyTorch oracle + baselines for in-kernel PBT (coopt_pbt.cu, --mode 2) with optional held-out in-kernel eval (--eval 1).
Rule at each rung boundary (every E steps, K rungs): metric = held-out MSE (eval=1) or mean train loss over the last NB steps (eval=0);
rank ascending (ties -> lower run index); bottom R/4 runs (rank >= R-R/4) copy weights of the run with rank (rank-(R-nb)) (exploit) and
set lr = lr[src] * (1.2 if run index odd else 0.8) (explore). Plain SGD has no optimizer state, so the weights are the whole state.
  ref : fp32 (TF32 off) eager oracle.
  B0  : TF32, whole run in ONE CUDA graph, no PBT (reference for kernel efficiency).
  B1  : TF32, whole run + PBT decisions in ONE CUDA graph, no host sync (strongest: static shapes allow it).
  B3  : TF32, one graph per rung segment, PBT decided on the HOST after a sync (typical PBT driver: read metrics, python sort, copy).
Each with eager or 'compiled' (torch.compile on the elementwise loss/dY piece only; no in-place mutation inside compile)."""
import json, sys, time
import numpy as np, torch

d = sys.argv[1]
meta = {}; lrs = []
for line in open(f"{d}/meta.txt"):
    k, v = line.split()
    if k == "lr": lrs.append(float(v))
    else: meta[k] = float(v) if any(c in v for c in ".e") else int(v)
B, D, H, O, NB, R, steps, E, K, win, EVAL = (meta[k] for k in "B D H O NB R steps E K win eval".split())
dev = "cuda"
ld = lambda n, shp: torch.from_numpy(np.fromfile(f"{d}/{n}.bin", dtype=np.float32).reshape(shp)).to(dev)
X = torch.stack([ld(f"X{k}", (B, D)) for k in range(NB)]); T = torch.stack([ld(f"T{k}", (B, O)) for k in range(NB)])
XV, TV = ld("XV", (B, D)), ld("TV", (B, O))
W1i, W2i = ld("W1", (D, H)), ld("W2", (H, O))
lr0 = torch.tensor(lrs, device=dev)
kloss = ld("loss_sh", (R, steps)); kev = ld("evloss_sh", (K + 1, R))
ksrc = np.fromfile(f"{d}/src_sh.bin", dtype=np.int32).reshape(64, 64)[:, :R]; klr = np.fromfile(f"{d}/lr_sh.bin", dtype=np.float32)[:R]
kw = ld("w_sh", (R, D * H + H * O))
C = 2.0 / (B * O)
idx = torch.arange(R, device=dev)
fac = torch.where(idx % 2 == 1, torch.tensor(1.2, device=dev), torch.tensor(0.8, device=dev))
NBOT = max(1, R // 4)

def fresh(n): return W1i.unsqueeze(0).repeat(n, 1, 1).contiguous(), W2i.unsqueeze(0).repeat(n, 1, 1).contiguous()

def post_y(Y, t):
    dd = Y - t
    return dd * C, (dd * dd).sum((1, 2)) / (B * O)
_cpost = None
USE_COMPILED = False
def post(Y, t):
    global _cpost
    if USE_COMPILED:
        if _cpost is None:
            import torch._dynamo as dz; dz.config.cache_size_limit = 64
            _cpost = torch.compile(post_y, dynamic=False)
        return _cpost(Y, t)
    return post_y(Y, t)

def step(W1, W2, lr, s, lossrow):
    k = s % NB; x = X[k]
    Hh = torch.relu(torch.matmul(x, W1))
    dY, l = post(torch.bmm(Hh, W2), T[k]); lossrow.copy_(l)
    dH = torch.ops.aten.threshold_backward(torch.bmm(dY, W2.transpose(1, 2)), Hh, 0)
    gW2 = torch.bmm(Hh.transpose(1, 2), dY); gW1 = torch.matmul(x.t(), dH)
    W2.addcmul_(lr, gW2, value=-1); W1.addcmul_(lr, gW1, value=-1)

def eval_loss(W1, W2):
    dd = torch.bmm(torch.relu(torch.matmul(XV, W1)), W2) - TV
    return (dd * dd).sum((1, 2)) / (B * O)

def metric_of(W1, W2, L, s):
    m = eval_loss(W1, W2) if EVAL else L[s - win + 1:s + 1].mean(0)
    return torch.where(torch.isnan(m), torch.full_like(m, 3.4e38), m)

def decide(m, lr):
    rank = ((m[None, :] < m[:, None]) | ((m[None, :] == m[:, None]) & (idx[None, :] < idx[:, None]))).sum(1)
    who = torch.argsort(rank)
    loser = rank >= R - NBOT
    src = torch.where(loser, who[(rank - (R - NBOT)).clamp(min=0)], idx)
    newlr = torch.where(loser, lr.view(R)[src] * fac, lr.view(R))
    return src, newlr

def apply(W1, W2, lr, src, newlr):
    W1.copy_(W1[src]); W2.copy_(W2[src]); lr.copy_(newlr.view(R, 1, 1))

def run_pbt(graph, tf32, pbt=True, force=None):
    torch.backends.cuda.matmul.allow_tf32 = tf32
    W1, W2 = fresh(R); lr = lr0.clone().view(R, 1, 1); L = torch.zeros(steps, R, device=dev)
    EVL = torch.zeros(K + 1, R, device=dev); SRC = torch.full((K + 1, R), -1, dtype=torch.long, device=dev)
    def body():
        for s in range(steps):
            step(W1, W2, lr, s, L[s])
            if pbt and (s + 1) % E == 0 and (s + 1) // E <= K:
                e = (s + 1) // E
                EVL[e - 1] = eval_loss(W1, W2)
                if force is None: src, nl = decide(metric_of(W1, W2, L, s), lr)
                else: src = force[e]; nl = torch.where(src != idx, lr.view(R)[src] * fac, lr.view(R))
                SRC[e] = src
                apply(W1, W2, lr, src, nl)
    def reset():
        W1.copy_(W1i.unsqueeze(0).expand(R, -1, -1)); W2.copy_(W2i.unsqueeze(0).expand(R, -1, -1)); lr.copy_(lr0.view(R, 1, 1)); L.zero_()
    if not graph:
        body(); torch.cuda.synchronize(); return L, EVL, SRC, W1, W2, lr
    st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st): body()
    torch.cuda.current_stream().wait_stream(st); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph(); reset(); torch.cuda.synchronize()
    with torch.cuda.graph(g): body()
    def go():
        reset(); torch.cuda.synchronize(); e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record(); g.replay(); e1.record(); torch.cuda.synchronize(); return e0.elapsed_time(e1)
    ts = [go() for _ in range(10)][3:]
    return L.clone(), EVL.clone(), SRC.clone(), W1.clone(), W2.clone(), lr.clone(), ts

class HostPBT:
    def __init__(self):
        torch.backends.cuda.matmul.allow_tf32 = True
        self.W1, self.W2 = fresh(R); self.lr = lr0.clone().view(R, 1, 1); self.segs = []
        self.L = torch.zeros(steps, R, device=dev); self.g = []
        for k in range(K + 1):
            a, b = k * E, ((k + 1) * E if k < K else steps)
            def body(a=a, b=b):
                for s in range(a, b): step(self.W1, self.W2, self.lr, s, self.L[s])
            st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st): body()
            torch.cuda.current_stream().wait_stream(st); torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g): body()
            self.g.append(g)
    def run(self):
        self.W1.copy_(W1i.unsqueeze(0).expand(R, -1, -1)); self.W2.copy_(W2i.unsqueeze(0).expand(R, -1, -1)); self.lr.copy_(lr0.view(R, 1, 1))
        self.src_hist = []
        for k in range(K + 1):
            self.g[k].replay()
            if k < K:
                s = (k + 1) * E - 1
                m = metric_of(self.W1, self.W2, self.L, s).cpu().tolist()          # host sync: metrics to the host
                lr = self.lr.view(R).cpu().tolist()
                rank = sorted(range(R), key=lambda r: (m[r], r)); rk = {r: i for i, r in enumerate(rank)}
                src = list(range(R)); nl = list(lr)
                for r in range(R):
                    if rk[r] >= R - NBOT: src[r] = rank[rk[r] - (R - NBOT)]; nl[r] = lr[src[r]] * (1.2 if r % 2 else 0.8)
                self.src_hist.append(src)
                srct = torch.tensor(src, device=dev); self.W1.copy_(self.W1[srct]); self.W2.copy_(self.W2[srct]); self.lr.copy_(torch.tensor(nl, device=dev).view(R, 1, 1))

def timeit(fn, n=7, warm=3):
    for _ in range(warm): fn(); torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        torch.cuda.synchronize(); e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record(); fn(); e1.record(); torch.cuda.synchronize(); ts.append(e0.elapsed_time(e1))
    return ts
stat = lambda ts: dict(best_ms=min(ts), median_ms=float(np.median(ts)), all_ms=[round(t, 3) for t in ts])

res = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "cfg": dict(B=B, D=D, H=H, O=O, NB=NB, R=R, steps=steps, E=E, K=K, eval=EVAL)}
Lr, EVr, SRCr, W1r, W2r, lrr = run_pbt(False, False)
Lnp, _, _, _, _, _ = run_pbt(False, False, pbt=False)
kforce = torch.where(torch.from_numpy(ksrc.astype(np.int64)).to(dev) < 0, idx[None, :].expand(64, R), torch.from_numpy(ksrc.astype(np.int64)).to(dev))
Lf_, EVf_, SRCf_, W1f_, W2f_, lrf_ = run_pbt(False, False, True, force=kforce)
rel = lambda a, b: float(((a - b).abs() / (b.abs() + 1e-6)).max())
agree = [bool((torch.where(torch.from_numpy(ksrc[e].astype(np.int64)).to(dev) < 0, idx, torch.from_numpy(ksrc[e].astype(np.int64)).to(dev)) == SRCr[e]).all()) for e in range(1, K + 1)]
same_src = all(agree)
c = dict(same_pbt_decisions_all_rungs=same_src, rungs_where_kvm_decision_equals_fp32_oracle=sum(agree), rungs_total=K,
         forced_replay_max_rel_train_loss_err=rel(kloss.T, Lf_), forced_replay_max_rel_eval_loss_err=(rel(kev[:K], EVf_[:K]) if EVAL else None),
         forced_replay_max_rel_final_lr_err=float(((torch.from_numpy(klr).to(dev) - lrf_.view(R)).abs() / lrf_.view(R).abs()).max()),
         forced_replay_max_rel_weight_l2=max(float((kw[r] - torch.cat([W1f_[r].flatten(), W2f_[r].flatten()])).norm() / torch.cat([W1f_[r].flatten(), W2f_[r].flatten()]).norm()) for r in range(R)),
         max_rel_train_loss_err=rel(kloss.T, Lr), max_rel_eval_loss_err=(rel(kev[:K], EVr[:K]) if EVAL else None),
         max_rel_final_lr_err=float(((torch.from_numpy(klr).to(dev) - lrr.view(R)).abs() / lrr.view(R).abs()).max()),
         max_rel_weight_l2=max(float((kw[r] - torch.cat([W1r[r].flatten(), W2r[r].flatten()])).norm() / torch.cat([W1r[r].flatten(), W2r[r].flatten()]).norm()) for r in range(R)),
         final_best_train_loss_pbt_ref=float(Lr[-1].min()), final_best_train_loss_no_pbt_ref=float(Lnp[-1].min()),
         final_median_train_loss_pbt_ref=float(Lr[-1].median()), final_median_train_loss_no_pbt_ref=float(Lnp[-1].median()),
         final_best_heldout_pbt_ref=float(eval_loss(W1r, W2r).min()), kvm_final_best_train_loss=float(kloss[:, -1].min()),
         final_lrs_ref=[round(float(x), 5) for x in lrr.view(R)])
res["correctness"] = c
res["torch_ms"] = {}
def baselines(tag):
    global USE_COMPILED
    USE_COMPILED = tag == "compiled"; sfx = "_compiled" if USE_COMPILED else ""
    out = run_pbt(True, True, pbt=False); res["torch_ms"]["B0_graph_no_pbt" + sfx] = stat(out[-1])
    L1, EV1, SRC1, W11, W21, lr1, ts1 = run_pbt(True, True, pbt=True); res["torch_ms"]["B1_graph_in_graph_pbt" + sfx] = stat(ts1)
    c["B1_decisions_match_ref" + sfx] = all(torch.equal(SRC1[e].cpu(), SRCr[e].cpu()) for e in range(1, K + 1))
    c["B1_max_rel_train_loss_err_vs_ref" + sfx] = rel(L1, Lr)
    hp = HostPBT(); hp.run(); torch.cuda.synchronize(); hp.run(); torch.cuda.synchronize()
    ok = all(hp.src_hist[e - 1] == SRCr[e].cpu().tolist() for e in range(1, K + 1)); c["B3_decisions_match_ref" + sfx] = ok
    c["B3_max_rel_train_loss_err_vs_ref" + sfx] = rel(hp.L, Lr)
    res["torch_ms"]["B3_host_synced_pbt" + sfx] = stat(timeit(hp.run))
    return min(ts1), c["B1_max_rel_train_loss_err_vs_ref" + sfx] < 0.1 and c["B3_max_rel_train_loss_err_vs_ref" + sfx] < 0.1
b1E, vE = baselines("eager")
try: b1C, vC = baselines("compiled")
except Exception as ex: res["compiled_error"] = repr(ex)[:300]; b1C, vC = b1E, False
tm = res["torch_ms"]
valid = [tm["B1_graph_in_graph_pbt"]["best_ms"], tm["B3_host_synced_pbt"]["best_ms"]]
if vC and "B1_graph_in_graph_pbt_compiled" in tm: valid += [tm["B1_graph_in_graph_pbt_compiled"]["best_ms"], tm["B3_host_synced_pbt_compiled"]["best_ms"]]
res["kvm_ms"] = dict(no_pbt=meta["kvm_full_ms"], pbt=meta["kvm_sh_ms"])
res["speedup"] = dict(kvm_pbt_vs_B1_eager=tm["B1_graph_in_graph_pbt"]["best_ms"] / meta["kvm_sh_ms"], kvm_pbt_vs_B3_eager=tm["B3_host_synced_pbt"]["best_ms"] / meta["kvm_sh_ms"],
    kvm_pbt_vs_best_valid_torch=min(valid) / meta["kvm_sh_ms"], kvm_nopbt_vs_B0_eager=tm["B0_graph_no_pbt"]["best_ms"] / meta["kvm_full_ms"])
print(json.dumps(res, indent=1)); json.dump(res, open(f"{d}/torch_result.json", "w"), indent=1)
