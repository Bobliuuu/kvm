#!/usr/bin/env python3
"""kvm.cu's MLP, trained by PyTorch (eager, whole-run CUDA graph, torch.compile, torch.func.vmap, hand-batched bmm) and by the VM,
all IN ONE PROCESS so that every ratio is measured on the same GPU state.

  python3 torch_mlp.py --lib ./kvm_lib.so --runs 8 --steps 300 --rounds 3 --reps 5 --out result.json

kvm_lib.so is kvm.cu built with `nvcc -shared -Xcompiler -fPIC -DKVM_LIB` (exports kvm_main). The VM call dumps the exact data
(X, T, W1, W2, per-run learning rates) and its own final weights/losses; PyTorch trains from those very bits.

Model (identical to kvm.cu): y = relu(x W1) W2, loss = mean((y - t)^2), plain SGD with a per-run learning rate, batch k = step % NB,
R independent runs (an LR sweep) from the same init. Matmuls use TF32 (the VM's wmma path): the fastest conventional precision.
Timing: reset the weights, then CUDA-event time one whole R-run training (R*steps SGD steps); best of --reps, repeated --rounds times,
alternating torch variants and the VM so a drifting GPU hits both.
"""
import argparse, ctypes, json, os, statistics, sys, tempfile, time
import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument('--lib', required=True)
p.add_argument('--runs', type=int, default=8)
p.add_argument('--steps', type=int, default=300)
p.add_argument('--reps', type=int, default=5)
p.add_argument('--rounds', type=int, default=3)
p.add_argument('--progs', default='tiles,fused', help='VM programs to time (kvm.cu --prog)')
p.add_argument('--timeline-prefix', default='', help='first VM call per program also writes PREFIX_<prog>.csv (per-instruction timestamps)')
p.add_argument('--tl-steps', default='100,103')
p.add_argument('--variants', default='all', help='comma list of class names, or all')
p.add_argument('--out', required=True)
p.add_argument('--precision', default='tf32', choices=['tf32', 'fp32'])
p.add_argument('--precs', default='tf32,fp16,bf16', help='torch precision classes: tf32 = fp32 tensors + TF32 matmul; fp16/bf16 = autocast (half matmul inputs, fp32 accumulate, fp32 master weights, gradients carried x1024, same SGD semantics)')
a = p.parse_args()
R, S = a.runs, a.steps
torch.backends.cuda.matmul.allow_tf32 = a.precision == 'tf32'
torch.backends.cudnn.allow_tf32 = a.precision == 'tf32'
torch.set_float32_matmul_precision('high' if a.precision == 'tf32' else 'highest')
dev = 'cuda'

# ------------------------------------------------------------------------------------------------ the VM, in this process
lib = ctypes.CDLL(os.path.abspath(a.lib))
lib.kvm_main.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]


def vm_call(args):
    argv = [b'kvm'] + [x.encode() for x in args]
    arr = (ctypes.c_char_p * len(argv))(*argv)
    sys.stdout.flush()
    rc = lib.kvm_main(len(argv), arr)
    ctypes.CDLL(None).fflush(None)
    return rc


tmp = tempfile.mkdtemp(prefix='kvmmlp')
vm_base = ['--runs', str(R), '--steps', str(S), '--reps', str(a.reps), '--ref-steps', '20']
PROGS = a.progs.split(',')
vm_rc = {}
for k, prog in enumerate(PROGS):
    extra = ['--dump-data', tmp] if k == 0 else []
    if a.timeline_prefix:
        extra += ['--timeline', f'{a.timeline_prefix}_{prog}.csv', '--tl-steps'] + a.tl_steps.split(',')
    vm_rc[prog] = vm_call(vm_base + ['--prog', prog, '--json', f'{tmp}/vm_{prog}.json'] + extra)
meta = json.load(open(f'{tmp}/meta.json'))
B, D, H, O, NB = meta['B'], meta['D'], meta['H'], meta['O'], meta['NB']
ld = lambda n, shp: torch.from_numpy(np.fromfile(f'{tmp}/{n}.f32', dtype=np.float32).reshape(shp)).to(dev)
X = [ld(f'X{k}', (B, D)) for k in range(NB)]
T = [ld(f'T{k}', (B, O)) for k in range(NB)]
W1_0, W2_0 = ld('W1', (D, H)), ld('W2', (H, O))
LRS = ld('lrs', (R,))
vm_loss = np.fromfile(f'{tmp}/vm_loss.f32', dtype=np.float32).reshape(R, S)
vm_w = np.fromfile(f'{tmp}/vm_weights.f32', dtype=np.float32).reshape(R, D * H + H * O)
SC = 2.0 / (B * O)

# ------------------------------------------------------------------------------------------------ the training programs
# Every variant owns W1 [R,D,H], W2 [R,H,O], L [R,S]; reset() restores them; run() trains all R runs for S steps.


import contextlib
DT, GSC = None, 1.0     # current autocast dtype and gradient scale; set per variant (read at trace/run time)
GS_HALF = 1024.0


def ac():
    return torch.autocast('cuda', dtype=DT, cache_enabled=False) if DT is not None else contextlib.nullcontext()


def set_prec(prec):
    global DT, GSC
    DT = {'tf32': None, 'fp16': torch.float16, 'bf16': torch.bfloat16}[prec]
    GSC = 1.0 if prec == 'tf32' else GS_HALF


class Variant:
    name = '?'
    prec = 'tf32'

    def __init__(self):
        self.W1 = W1_0.expand(R, D, H).clone()
        self.W2 = W2_0.expand(R, H, O).clone()
        self.L = torch.zeros(R, S, device=dev)

    def reset(self):
        self.W1.copy_(W1_0.expand(R, D, H)); self.W2.copy_(W2_0.expand(R, H, O)); self.L.zero_()


def step_manual(W1, W2, x, t, lr, Lslot):
    """one SGD step for one run, hand-written backward (what the VM's tape does); in-place weight update, loss into Lslot"""
    with ac():
        h = torch.relu(x @ W1)
        d = h @ W2 - t
        loss = (d * d).mean()
        Lslot.copy_(loss)
        dy = d * (SC * GSC)
        dh = (dy @ W2.t()) * (h > 0)
        W2.sub_((h.t() @ dy).float() * (lr / GSC))
        W1.sub_((x.t() @ dh).float() * (lr / GSC))
    return loss


class EagerAutograd(Variant):
    name = 'torch eager (autograd)'

    def run(self):
        for r in range(R):
            w1 = self.W1[r].clone().requires_grad_(True); w2 = self.W2[r].clone().requires_grad_(True)
            lr = LRS[r]
            for s in range(S):
                k = s % NB
                loss = ((torch.relu(X[k] @ w1) @ w2 - T[k]) ** 2).mean()
                loss.backward()
                with torch.no_grad():
                    self.L[r, s] = loss
                    w1.sub_(w1.grad * lr); w2.sub_(w2.grad * lr)
                    w1.grad = None; w2.grad = None
            self.W1[r].copy_(w1.detach()); self.W2[r].copy_(w2.detach())


class EagerManual(Variant):
    name = 'torch eager (manual backward)'

    def run(self):
        for r in range(R):
            for s in range(S):
                k = s % NB
                step_manual(self.W1[r], self.W2[r], X[k], T[k], LRS[r], self.L[r, s])


def compiled(fn):
    """torch.compile(fn) with the per-step loss delivered by return value (an in-place write to a view argument is not reliable under compile)"""
    def nolast(*args):
        return fn(*args, torch.empty_like(args[0][..., 0, 0]) if args[0].dim() == 3 else torch.empty((), device=dev))
    cf = torch.compile(nolast, dynamic=False)

    def run(*args):
        args[-1].copy_(cf(*args[:-1]))
    return run


def capture(fn):
    """capture fn() into a CUDA graph (after a warm-up on a side stream); fn must only use static tensors"""
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(st)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g


class GraphManual(Variant):
    """each run's S steps captured as ONE CUDA graph (the strongest per-run baseline); R graph launches"""
    name = 'torch CUDA graph, whole run (manual backward)'
    step = staticmethod(step_manual)

    def __init__(self):
        super().__init__()
        self.graphs = []
        for r in range(R):
            def body(r=r):
                for s in range(S):
                    k = s % NB
                    self.step(self.W1[r], self.W2[r], X[k], T[k], LRS[r], self.L[r, s])
            self.graphs.append(capture(body))

    def run(self):
        for g in self.graphs:
            g.replay()


class CompileGraph(GraphManual):
    name = 'torch.compile step + whole-run CUDA graph'

    def __init__(self):
        type(self).step = staticmethod(compiled(step_manual))
        super().__init__()


class CompileLoop(Variant):
    name = 'torch.compile step, python loop'

    def __init__(self):
        super().__init__()
        self.f = compiled(step_manual)
        self.f(self.W1[0], self.W2[0], X[0], T[0], LRS[0], self.L[0, 0])
        torch.cuda.synchronize()

    def run(self):
        for r in range(R):
            for s in range(S):
                k = s % NB
                self.f(self.W1[r], self.W2[r], X[k], T[k], LRS[r], self.L[r, s])


def step_bmm(W1, W2, x, t, lr, Ls):
    """one SGD step for ALL runs at once on stacked weights: W1 [R,D,H], W2 [R,H,O], lr [R,1,1], Ls [R]"""
    with ac():
        h = torch.relu(torch.matmul(x, W1))                  # [R,B,H]
        d = torch.bmm(h, W2) - t                             # [R,B,O]
        loss = (d * d).mean(dim=(1, 2))
        Ls.copy_(loss)
        dy = d * (SC * GSC)
        dh = torch.bmm(dy, W2.transpose(1, 2)) * (h > 0)
        W2.sub_(torch.bmm(h.transpose(1, 2), dy).float() * (lr / GSC))
        W1.sub_(torch.matmul(x.t(), dh).float() * (lr / GSC))
    return loss


class BmmEager(Variant):
    name = 'torch hand-batched bmm, eager loop'
    step = staticmethod(step_bmm)

    def run(self):
        lr = LRS.view(R, 1, 1)
        for s in range(S):
            k = s % NB
            Ls = torch.empty(R, device=dev)
            self.step(self.W1, self.W2, X[k], T[k], lr, Ls)
            self.L[:, s] = Ls


class BmmGraph(Variant):
    """all runs batched with bmm, all S steps in ONE CUDA graph (the strongest hand baseline)"""
    name = 'torch hand-batched bmm + whole CUDA graph'
    step = staticmethod(step_bmm)
    lrs = None

    def __init__(self):
        super().__init__()
        lr = LRS.view(R, 1, 1) if self.lrs is None else self.lrs
        self.Ls = torch.zeros(S, R, device=dev)

        def body():
            for s in range(S):
                k = s % NB
                self.step(self.W1, self.W2, X[k], T[k], lr, self.Ls[s])
        self.g = capture(body)

    def run(self):
        self.g.replay()
        self.L.copy_(self.Ls.t())


class BmmCompileGraph(BmmGraph):
    name = 'torch.compile bmm step + whole CUDA graph'

    def __init__(self):
        type(self).step = staticmethod(compiled(step_bmm))
        super().__init__()


# torch.func: vmap over runs of a per-run functional step whose gradient comes from autograd (torch.func.grad)
def _loss_fn(W1, W2, x, t):
    with ac():
        d = torch.relu(x @ W1) @ W2 - t
        l = (d * d).mean()
    return l * GSC, l


_grad = torch.func.grad(_loss_fn, argnums=(0, 1), has_aux=True)


def step_vmap_one(W1, W2, x, t, lr):
    (g1, g2), l = _grad(W1, W2, x, t)
    return W1 - (lr / GSC) * g1.float(), W2 - (lr / GSC) * g2.float(), l


_vstep = torch.func.vmap(step_vmap_one, in_dims=(0, 0, None, None, 0))


def step_vmap(W1, W2, x, t, lr, Ls):
    n1, n2, l = _vstep(W1, W2, x, t, lr)
    W1.copy_(n1); W2.copy_(n2); Ls.copy_(l)
    return l


class VmapEager(BmmEager):
    name = 'torch.func.vmap(grad), eager loop'
    step = staticmethod(step_vmap)

    def run(self):
        for s in range(S):
            k = s % NB
            Ls = torch.empty(R, device=dev)
            self.step(self.W1, self.W2, X[k], T[k], LRS, Ls)
            self.L[:, s] = Ls


class VmapGraph(BmmGraph):
    name = 'torch.func.vmap(grad) + whole CUDA graph'
    step = staticmethod(step_vmap)
    lrs = LRS


class VmapCompileGraph(VmapGraph):
    name = 'torch.compile(vmap(grad)) + whole CUDA graph'

    def __init__(self):
        type(self).step = staticmethod(compiled(step_vmap))
        super().__init__()


VARIANTS = [EagerAutograd, EagerManual, CompileLoop, GraphManual, CompileGraph, BmmEager, BmmGraph, BmmCompileGraph, VmapEager, VmapGraph,
            VmapCompileGraph]

# ------------------------------------------------------------------------------------------------ run + check
ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


def timed(v):
    set_prec(v.prec)
    v.reset(); torch.cuda.synchronize()
    ev0.record(); v.run(); ev1.record(); ev1.synchronize()
    return ev0.elapsed_time(ev1)


def check(v):
    """loss curve vs the VM's (relative, first 20 steps, before TF32 rounding decorrelates), final weights vs the VM's, loss decreasing"""
    set_prec(v.prec)
    v.reset(); v.run(); torch.cuda.synchronize()
    L = v.L.cpu().numpy()
    n = min(20, S)
    le = float(np.max(np.abs(L[:, :n] - vm_loss[:, :n]) / (np.abs(vm_loss[:, :n]) + 1e-6)))
    w = np.concatenate([v.W1.cpu().numpy().reshape(R, -1), v.W2.cpu().numpy().reshape(R, -1)], 1)
    we = float(np.max(np.abs(w - vm_w)) / (np.max(np.abs(vm_w)) + 1e-12))
    dec = bool(np.all(L[:, -1] < L[:, 0]))
    return le, we, dec


sel = None if a.variants == 'all' else set(a.variants.split(','))
HALF_SKIP = {'EagerAutograd', 'CompileLoop', 'VmapEager'}   # half precision: autograd / python-loop eager variants add nothing beyond EagerManual
vs = []
for prec in a.precs.split(','):
    for C in VARIANTS:
        if sel and C.__name__ not in sel:
            continue
        if prec != 'tf32' and C.__name__ in HALF_SKIP:
            continue
        t0 = time.time()
        try:
            set_prec(prec)
            v = C()
            v.prec = prec
            v.name = f'{C.name} [{prec}]'
            le, we, dec = check(v)
            ok = le < (2e-2 if prec == 'tf32' else 5e-2) and dec and we < 5e-2
            print(f'[build] {v.name:62s} build+check {time.time()-t0:5.1f}s  loss relerr(20 steps) {le:.1e}  weights relerr {we:.1e}  decreasing {dec}  {"ok" if ok else "MISMATCH"}', flush=True)
            vs.append((v, le, we, ok))
        except Exception as e:  # keep going: a baseline that fails to build is reported, not hidden
            print(f'[build] {C.name} [{prec}] FAILED: {type(e).__name__}: {str(e)[:300]}', flush=True)

res = {v.name: [] for v, *_ in vs}
vmres = {prog: [] for prog in PROGS}
for rd in range(a.rounds):
    for v, *_ in vs:
        res[v.name].append([timed(v) for _ in range(a.reps)])
    for prog in PROGS:
        vm_call(vm_base + ['--prog', prog, '--json', f'{tmp}/vm_{prog}{rd}.json'])
        j = json.load(open(f'{tmp}/vm_{prog}{rd}.json'))
        vmres[prog].append(j)
        print(f'[round {rd}] VM prog={prog} {j["modes"][3]["best_ms"]:.3f} ms  pass={j["pass"]}', flush=True)

tot = R * S
rows = []
for v, le, we, ok in vs:
    allr = [x for rr in res[v.name] for x in rr]
    rows.append(dict(name=v.name, best_ms=min(allr), median_ms=statistics.median(allr), max_ms=max(allr), round_best=[min(rr) for rr in res[v.name]],
                     us_per_step_run=min(allr) * 1000 / tot, loss_relerr=le, weights_relerr=we, matches_vm=ok, prec=v.prec))
vm_modes = {}
for prog in PROGS:
    vm_modes[prog] = []
    for i in range(4):
        allb = [j['modes'][i]['best_ms'] for j in vmres[prog]]
        vm_modes[prog].append(dict(name=f'{vmres[prog][0]["modes"][i]["name"]} [{prog}]', best_ms=min(allb), round_best=allb, us_per_step_run=min(allb) * 1000 / tot,
                                   max_ms=max(j['modes'][i]['max_ms'] for j in vmres[prog])))
good = [r for r in rows if r['matches_vm']]
best_torch = min(good, key=lambda r: r['best_ms'])
best_by_prec = {q: min([r for r in good if r['prec'] == q], key=lambda r: r['best_ms']) for q in sorted({r['prec'] for r in good})}
best_prog = min(PROGS, key=lambda q: vm_modes[q][3]['best_ms'])
vmb = vm_modes[best_prog][3]['best_ms']
out = dict(runs=R, steps=S, reps=a.reps, rounds=a.rounds, precision=a.precision, torch=torch.__version__, cuda=torch.version.cuda,
           gpu=torch.cuda.get_device_name(), sms=torch.cuda.get_device_properties(0).multi_processor_count,
           vm_json={q: vmres[q][0] for q in PROGS}, torch_variants=rows, vm_modes=vm_modes, vm_pass={q: all(j['pass'] for j in vmres[q]) for q in PROGS},
           best_torch=best_torch['name'], best_torch_by_prec={q: dict(name=r['name'], best_ms=r['best_ms'], us_per_step_run=r['us_per_step_run']) for q, r in best_by_prec.items()}, best_vm_prog=best_prog, vm_vs_best_torch=best_torch['best_ms'] / vmb,
           vm_vs_best_hand_baseline={q: min(m['best_ms'] for m in vm_modes[q][:3]) / vm_modes[q][3]['best_ms'] for q in PROGS})
json.dump(out, open(a.out, 'w'), indent=1)
hdr = f'VM({best_prog}) speedup'
print(f'\n{"variant":66s} {"best ms":>9s} {"us/step/run":>12s} {"max/min":>8s} {hdr:>20s}')
allrows = [(r['name'], r['best_ms'], r['us_per_step_run'], r['max_ms'] / r['best_ms'], '' if r['matches_vm'] else '  (numerics MISMATCH)') for r in rows]
for q in PROGS:
    allrows += [(m['name'], m['best_ms'], m['us_per_step_run'], m['max_ms'] / m['best_ms'], '') for m in vm_modes[q]]
for n, b, u, spread, note in sorted(allrows, key=lambda r: r[1]):
    print(f'{n:66s} {b:9.3f} {u:12.3f} {spread:7.2f}x {b / vmb:10.2f}x{note}')
print(f'\nR={R} S={S}: best VM = {best_prog} {vmb:.3f} ms; vs best torch ({best_torch["name"]}) = {out["vm_vs_best_torch"]:.2f}x ; pass={out["vm_pass"]}')
print('\nbest torch per precision class vs VM programs (VM speedup = torch ms / VM ms):')
for q, r in best_by_prec.items():
    print(f'  torch {q:5s} best {r["us_per_step_run"]:7.3f} us/step/run ({r["name"]})')
    for g in PROGS:
        print(f'      vs VM {g:8s} {vm_modes[g][3]["us_per_step_run"]:7.3f} us -> {r["best_ms"] / vm_modes[g][3]["best_ms"]:.2f}x')
