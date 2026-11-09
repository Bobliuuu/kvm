"""PyTorch reference + baselines for the KVM successive-halving population sweep (coopt_sh.cu).

Same model (2-layer ReLU MLP, MSE, SGD, R learning rates, same data and init, loaded from the files coopt_sh dumps),
same selection rule (every E steps keep the better half by mean loss over the last NB steps; ties -> lower index).
  ref      : fp32 (TF32 off), eager, batched over runs. Correctness oracle.
  B0       : TF32, whole run in ONE CUDA graph, full population, no halving (what KVM 'no halving' compares to).
  B1       : TF32, whole run in ONE CUDA graph, halving done in-graph by masking (lr=0 for retired runs; no compute saved).
  B2       : TF32, one pre-captured CUDA graph per rung with the surviving runs only (compute shrinks), selection by async eager torch ops, no host sync.
Manual fwd/bwd with bmm (no autograd) so the graph holds the minimum kernels.
"""
import json, sys, time
import numpy as np, torch

d = sys.argv[1]
meta = {}
lrs = []
for line in open(f"{d}/meta.txt"):
    k, v = line.split()
    if k == "lr": lrs.append(float(v))
    else: meta[k] = float(v) if any(c in v for c in ".e") else int(v)
B, D, H, O, NB, R, steps, E, K, win = (meta[k] for k in "B D H O NB R steps E K win".split())
MODE, TARGET, DIVTHR, PRUNE = meta["mode"], meta["target"], meta["divthr"], meta["prune"]
dev = "cuda"
ld = lambda n, shp: torch.from_numpy(np.fromfile(f"{d}/{n}.bin", dtype=np.float32).reshape(shp)).to(dev)
X = torch.stack([ld(f"X{k}", (B, D)) for k in range(NB)]); T = torch.stack([ld(f"T{k}", (B, O)) for k in range(NB)])
W1i, W2i = ld("W1", (D, H)), ld("W2", (H, O))
lr0 = torch.tensor(lrs, device=dev)
kloss = ld("loss_sh", (R, steps)); kalive = np.fromfile(f"{d}/alive_sh.bin", dtype=np.int32)[:R]
kw = ld("w_sh", (R, D * H + H * O))
kloss_full = ld("loss_full", (R, steps))
C = 2.0 / (B * O)

def fresh(n):
    return W1i.unsqueeze(0).repeat(n, 1, 1).contiguous(), W2i.unsqueeze(0).repeat(n, 1, 1).contiguous()

def post_y(Y, t):
    dd = Y - t
    return dd * C, (dd * dd).sum((1, 2)) / (B * O)

_cpost = None
USE_COMPILED = False
def post(Y, t):
    global _cpost
    if USE_COMPILED:   # compile only the pure elementwise/reduction piece; in-place weight updates stay outside (compiling them gave wrong results)
        if _cpost is None:
            import torch._dynamo as _dz; _dz.config.cache_size_limit = 64
            _cpost = torch.compile(post_y, dynamic=False)
        return _cpost(Y, t)
    return post_y(Y, t)

def step(W1, W2, lr, s, lossrow):
    k = s % NB; x = X[k]; t = T[k]
    Hh = torch.relu(torch.matmul(x, W1))
    dY, l = post(torch.bmm(Hh, W2), t); lossrow.copy_(l)
    dH = torch.ops.aten.threshold_backward(torch.bmm(dY, W2.transpose(1, 2)), Hh, 0)
    gW2 = torch.bmm(Hh.transpose(1, 2), dY)
    gW1 = torch.matmul(x.t(), dH)
    W2.addcmul_(lr, gW2, value=-1)
    W1.addcmul_(lr, gW1, value=-1)

def metric(L, s):   # L [steps, n]; mean of last win steps ending at s (inclusive); NaN -> huge
    m = L[s - win + 1:s + 1].mean(0)
    return torch.where(torch.isnan(m), torch.full_like(m, 3.4e38), m)

def select_mask(m, alive):  # exactly the kernel's rule (mode 0: keep better half; mode 1: retire diverged / converged / hopeless)
    n = m.numel(); idx = torch.arange(n, device=dev)
    if MODE == 1:
        best = torch.where(alive, m, torch.full_like(m, 3.4e38)).min()
        return alive & (m < DIVTHR) & (m >= TARGET) & (m <= best * PRUNE)
    mm = torch.where(alive, m, torch.full_like(m, float("inf")))
    lt = (mm[None, :] < mm[:, None]) & alive[None, :]
    eq = (mm[None, :] == mm[:, None]) & (idx[None, :] < idx[:, None]) & alive[None, :]
    rank = (lt | eq).sum(1)
    keep = torch.clamp(alive.sum() // 2, min=1)
    return alive & (rank < keep)

def run_masked(graph_whole, halving, tf32):
    """whole run, full population. returns loss [steps,R], alive, W1, W2 ; optionally capture as one graph."""
    torch.backends.cuda.matmul.allow_tf32 = tf32
    W1, W2 = fresh(R); lr = lr0.clone().view(R, 1, 1); alive = torch.ones(R, dtype=torch.bool, device=dev)
    L = torch.zeros(steps, R, device=dev)
    def body():
        nonlocal alive
        for s in range(steps):
            step(W1, W2, lr, s, L[s])
            if halving and (s + 1) % E == 0 and (s + 1) // E <= K:
                alive_new = select_mask(metric(L, s), alive)
                alive.copy_(alive_new)
                lr.copy_((lr0 * alive).view(R, 1, 1))
    def reset():
        W1.copy_(W1i.unsqueeze(0).expand(R, -1, -1)); W2.copy_(W2i.unsqueeze(0).expand(R, -1, -1))
        lr.copy_(lr0.view(R, 1, 1)); alive.fill_(True); L.zero_()
    if not graph_whole:
        body(); torch.cuda.synchronize(); return L, alive, W1, W2
    s_ = torch.cuda.Stream()
    s_.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s_): body()          # warmup compiles kernels
    torch.cuda.current_stream().wait_stream(s_); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    reset(); torch.cuda.synchronize()
    with torch.cuda.graph(g): body()
    def go():
        reset(); g.replay()
    def go_timed():
        reset(); torch.cuda.synchronize(); e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record(); g.replay(); e1.record(); torch.cuda.synchronize(); return e0.elapsed_time(e1)
    ts = [go_timed() for _ in range(3)] + [go_timed() for _ in range(7)]
    ts = ts[3:]
    return L.clone(), alive.clone(), W1.clone(), W2.clone(), ts

class RungGraphs:
    """B2: one graph per rung, surviving runs only."""
    def __init__(self):
        torch.backends.cuda.matmul.allow_tf32 = True
        self.sizes = [R >> k for k in range(K + 1)]
        self.segs = [(k * E, (k + 1) * E if k < K else steps) for k in range(K + 1)]
        self.buf = []
        for k, n in enumerate(self.sizes):
            W1, W2 = fresh(n); lr = torch.zeros(n, 1, 1, device=dev); a, b = self.segs[k]
            L = torch.zeros(b - a, n, device=dev)
            def body(W1=W1, W2=W2, lr=lr, L=L, a=a, b=b):
                for s in range(a, b): step(W1, W2, lr, s, L[s - a])
            st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st): body()
            torch.cuda.current_stream().wait_stream(st); torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g): body()
            self.buf.append(dict(W1=W1, W2=W2, lr=lr, L=L, g=g))
        self.ids = None
    def run(self):
        b0 = self.buf[0]
        b0["W1"].copy_(W1i.unsqueeze(0).expand(R, -1, -1)); b0["W2"].copy_(W2i.unsqueeze(0).expand(R, -1, -1)); b0["lr"].copy_(lr0.view(R, 1, 1))
        ids = torch.arange(R, device=dev); self.hist = []
        for k in range(K + 1):
            b = self.buf[k]; b["g"].replay(); self.hist.append(ids)
            if k < K:
                n = self.sizes[k]
                m = b["L"][-win:].mean(0); m = torch.where(torch.isnan(m), torch.full_like(m, 3.4e38), m)
                alive = torch.ones(n, dtype=torch.bool, device=dev)
                idx = torch.arange(n, device=dev)
                rank = ((m[None, :] < m[:, None]) | ((m[None, :] == m[:, None]) & (idx[None, :] < idx[:, None]))).sum(1)
                sel = torch.argsort(rank)[: n // 2]      # sync-free: halving keeps a static count
                nb = self.buf[k + 1]
                nb["W1"].copy_(b["W1"][sel]); nb["W2"].copy_(b["W2"][sel]); nb["lr"].copy_(b["lr"][sel]); ids = ids[sel]
        self.final_ids = ids


class Polled:
    """B3: host-polled compaction. After each rung the host reads which runs survive (sync), compacts the weights and replays a graph
    for exactly that many runs. Graphs are cached per (rung, n alive) and the cache is warmed by an untimed first pass (generous to torch)."""
    def __init__(self):
        torch.backends.cuda.matmul.allow_tf32 = True
        self.segs = [(k * E, (k + 1) * E if k < K else steps) for k in range(K + 1)]
        self.cache = {}
    def get(self, k, n):
        if (k, n) not in self.cache:
            W1, W2 = fresh(n); lr = torch.zeros(n, 1, 1, device=dev); a, b = self.segs[k]; L = torch.zeros(b - a, n, device=dev)
            def body():
                for s in range(a, b): step(W1, W2, lr, s, L[s - a])
            st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st): body()
            torch.cuda.current_stream().wait_stream(st); torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g): body()
            self.cache[(k, n)] = dict(W1=W1, W2=W2, lr=lr, L=L, g=g)
        return self.cache[(k, n)]
    def run(self):
        ids = torch.arange(R, device=dev); cW1, cW2 = fresh(R); lr = lr0.view(R, 1, 1).clone()
        self.Lfull = torch.zeros(steps, R, device=dev); self.kill = [0] * R
        for k in range(K + 1):
            n = ids.numel()
            if n == 0: break
            b = self.get(k, n); a, e = self.segs[k]
            b["W1"].copy_(cW1[ids]); b["W2"].copy_(cW2[ids]); b["lr"].copy_(lr[ids])
            b["g"].replay()
            self.Lfull[a:e, ids] = b["L"]
            cW1[ids] = b["W1"]; cW2[ids] = b["W2"]
            if k < K:
                m = metric_seg(b["L"]); keep = select_mask(m, torch.ones(n, dtype=torch.bool, device=dev))
                kl = keep.tolist()                       # host sync: the host must learn the new population size
                for i, kp in zip(ids.tolist(), kl):
                    if not kp: self.kill[i] = k + 1
                ids = ids[keep]
        self.final_ids = ids; self.cW1, self.cW2 = cW1, cW2

def metric_seg(Lseg):
    m = Lseg[-win:].mean(0)
    return torch.where(torch.isnan(m), torch.full_like(m, 3.4e38), m)

def timeit(fn, n=7, warm=3):
    for _ in range(warm): fn(); torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        torch.cuda.synchronize(); e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter(); e0.record(); fn(); e1.record(); torch.cuda.synchronize(); ts.append(e0.elapsed_time(e1))
    return ts

res = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "cfg": dict(B=B, D=D, H=H, O=O, NB=NB, R=R, steps=steps, E=E, K=K)}
# --- reference, fp32
Lr, ar, W1r, W2r = run_masked(False, True, False)
Lf, _, _, _ = run_masked(False, False, False)
surv_ref = set(int(i) for i in torch.nonzero(ar).flatten().tolist())
surv_kvm = set(int(i) for i in np.nonzero(kalive)[0])
# kill epoch per run from the ref: run alive iff survives all; reconstruct via rerun with record
def kill_epochs():
    alive = torch.ones(R, dtype=torch.bool, device=dev); ep = [0] * R
    for e in range(1, K + 1):
        s = e * E - 1; new = select_mask(metric(Lr, s), alive)
        for r in range(R):
            if alive[r] and not new[r]: ep[r] = e
        alive = new
    return ep
ep = kill_epochs()
# compare loss curves while alive (ref vs kvm), KVM-reported loss is the loss of the step in progress
rel = 0.0; relk = 0.0
for r in range(R):
    last = steps if ep[r] == 0 else ep[r] * E
    a = kloss[r, :last]; b = Lr[:last, r]
    rel = max(rel, float(((a - b).abs() / (b.abs() + 1e-6)).max()))
# full-population loss curves (no halving) vs fp32 ref
relf = 0.0
for r in range(R):
    a = kloss_full[r]; b = Lf[:, r]
    ok = torch.isfinite(b)
    if ok.any(): relf = max(relf, float(((a[ok] - b[ok]).abs() / (b[ok].abs() + 1e-6)).max()))
# weights of survivors: relative L2 diff kvm vs ref
wd = 0.0
for r in range(R):
    a = kw[r]; b = torch.cat([W1r[r].flatten(), W2r[r].flatten()])
    if torch.isfinite(b).all() and torch.isfinite(a).all():
        wd = max(wd, float((a - b).norm() / b.norm()))
# loss at end of survivors
res["correctness"] = dict(survivors_ref=sorted(surv_ref), survivors_kvm=sorted(surv_kvm), same_survivors=surv_ref == surv_kvm,
    max_rel_loss_err_sh_alive_steps=rel, max_rel_loss_err_full_population=relf, max_rel_weight_l2_survivors=wd, kill_epoch_ref=ep,
    final_loss_ref={str(r): float(Lr[-1, r]) for r in sorted(surv_ref)}, final_loss_kvm={str(r): float(kloss[r, -1]) for r in sorted(surv_kvm)})
nan_runs = [r for r in range(R) if not torch.isfinite(Lf[:, r]).all()]
res["correctness"]["runs_diverged_in_full_pop_ref"] = nan_runs

# --- baselines, TF32
kill_kvm = [int(x) for x in np.fromfile(f"{d}/kill_sh.bin", dtype=np.int32)[:R]]
res["correctness"]["kill_rung_kvm"] = kill_kvm
res["correctness"]["same_kill_rungs"] = kill_kvm == ep
def baselines(tag):
    global USE_COMPILED
    USE_COMPILED = (tag == "compiled")
    sfx = "_compiled" if USE_COMPILED else ""
    L0, _, _, _, ts0 = run_masked(True, False, True)
    L1, a1, W11, W21, ts1 = run_masked(True, True, True)
    res["correctness"]["B1_survivors_match_ref" + sfx] = set(torch.nonzero(a1).flatten().tolist()) == surv_ref
    res["torch_ms"]["B0_graph_full_population" + sfx] = stat(ts0)
    res["torch_ms"]["B1_graph_masked" + sfx] = stat(ts1)
    pg = Polled(); pg.run(); torch.cuda.synchronize(); pg.run(); torch.cuda.synchronize()   # warm graph cache
    ts3 = timeit(pg.run)
    res["torch_ms"]["B3_host_polled_compaction_cached_graphs" + sfx] = stat(ts3)
    res["correctness"]["B3_kill_rungs_match_ref" + sfx] = pg.kill == ep
    res["correctness"]["B3_kill_rungs" + sfx] = pg.kill
    res["correctness"]["B3_max_rel_loss_err_vs_ref" + sfx] = max(float(((pg.Lfull[:(steps if ep[r] == 0 else ep[r] * E), r] - Lr[:(steps if ep[r] == 0 else ep[r] * E), r]).abs() / (Lr[:(steps if ep[r] == 0 else ep[r] * E), r].abs() + 1e-6)).max()) for r in range(R))
    cands = [min(ts1)]
    if pg.kill == ep: cands.append(min(ts3))          # a baseline that makes different retire decisions than the oracle is not counted
    else: res["correctness"]["B3_INVALID_not_counted" + sfx] = True
    if MODE == 0:
        rg = RungGraphs(); ts2 = timeit(rg.run)
        res["torch_ms"]["B2_rung_graphs_static_halving" + sfx] = stat(ts2)
        res["correctness"]["B2_survivors_match_ref" + sfx] = set(rg.final_ids.tolist()) == surv_ref
        if set(rg.final_ids.tolist()) == surv_ref: cands.append(min(ts2))
    return cands, min(ts0), min(ts1), min(ts3)
stat = lambda ts: dict(best_ms=min(ts), median_ms=float(np.median(ts)), all_ms=[round(t, 3) for t in ts])
res["torch_ms"] = {}
cE, b0E, b1E, b3E = baselines("eager")
try:
    cC, b0C, b1C, b3C = baselines("compiled")
except Exception as ex:
    res["compiled_error"] = repr(ex)[:300]; cC, b0C, b1C, b3C = cE, b0E, b1E, b3E
res["kvm_ms"] = dict(full_population=meta["kvm_full_ms"], controlled=meta["kvm_sh_ms"])
res["speedup"] = dict(kvm_ctrl_vs_best_torch_ctrl_eager=min(cE) / meta["kvm_sh_ms"], kvm_ctrl_vs_best_torch_ctrl_compiled=min(cC) / meta["kvm_sh_ms"],
    kvm_ctrl_vs_best_torch_any=min(cE + cC) / meta["kvm_sh_ms"], kvm_full_vs_B0_eager=b0E / meta["kvm_full_ms"], kvm_full_vs_B0_compiled=b0C / meta["kvm_full_ms"])
res["torch_per_run_gpu_mem_note"] = "all torch variants use manual fwd/bwd with bmm, TF32, CUDA graphs"
print(json.dumps(res, indent=1))
json.dump(res, open(f"{d}/torch_result.json", "w"), indent=1)
