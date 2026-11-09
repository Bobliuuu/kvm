"""python3 -m kvmfe.demo --all --out DIR [--train-steps N] [--window S] [--serve-every K] [--game-frames F]

Emits the three-workload demo (training + serving + the game) as compiled ABI bundles plus MANIFEST.json, which a host runner
consumes without any model- or game-specific code (see `MANIFEST.json` below and FRONTEND.md).

DIR/workloads/*.json   workload JSON per distinct program (+ data/ files referenced by `file` inits)
DIR/bundles/<name>/    compile_workload output (tasks.bin, successors.bin, args.bin, metadata.json, data)
DIR/MANIFEST.json      bundles, shared buffers, admission schedule

Training is cut into windows of S steps: one program (a template), admitted N/S times. Everything that differs between steps lives
in device buffers (the step counter selects the batch), so the template is identical; only the version numbers it publishes move,
which the manifest states as `base_versions` of each admission (recompile with them, or let the executor add the offset).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from . import log
from .emit import Lowered
from .game import GameConfig, compile_game
from .inspect import add_common
from .ir import FrontendError
from .lower import Tiling
from .nn import GPT, ModelConfig
from .pipeline import compile_training
from .serve import ServeRequest, compile_serving
from .optim import OptimConfig
from .train import Schedule, TrainConfig, weights_version

MANIFEST_VERSION = 1
F_PRECISE = 16                 # matmul_tile flag: 3xTF32 (fp32 accuracy); the default is 1xTF32 (see tests/gptdiff)
# Scheduler parameters per workload class (include/kvm_sched.h KvmClassParams; omitted fields keep the executor default). The frontend
# knows what it built: the game class has per-frame deadlines and gets a one-CTA reservation plus the earliest-deadline-first window that
# the A100 scheduler results validated (results/2026-11-04-a100-sched-integrated); the others share what is left by weight.
CLASS_POLICY = {'train': dict(weight=1), 'serve': dict(weight=1), 'game': dict(weight=1, min_cta=1, urgent_us=20000)}
GAME_STACK_BYTES = 16 * 1024   # include/kvm_ops_game.cuh: per-thread stack the Doom engine's BSP recursion needs
GAME_WAD = 'doom1.wad'         # file name only: the host runner resolves it in its --assets directory (doom/setup.sh fetches it)


def prompt_for(cfg, n=3):
    t, out = 1, []
    for _ in range(n):
        out.append(t % cfg.vocab)
        t = (5 * t + 3) % cfg.vocab
    return tuple(out)


@dataclass
class Workload:
    name: str                    # file stem under workloads/
    kind: str                    # scheduler class: train | serve | game
    spec: dict
    data_files: dict
    tasks: int
    edges: int


def summarize(spec):
    tasks = [t for g in spec['graphs'] for t in g['tasks']]
    dls = [t['deadline_us'] for t in tasks if 'deadline_us' in t]
    return dict(priority=max((t.get('priority', 0) for t in tasks), default=0),
                deadline_us=[min(dls), max(dls)] if dls else None, variables=spec['variables'],
                graphs=[g['name'] for g in spec['graphs']])


def requirements(w: Workload) -> dict:
    """Host resources a workload needs beyond buffers (the runner dispatches on `kind`, never on the workload's name)."""
    if w.kind != 'game':
        return {}
    return dict(requirements=dict(stack_bytes=GAME_STACK_BYTES, resources=[dict(kind='doom_wad', file=GAME_WAD)]))


def write_workload(root: Path, w: Workload) -> Path:
    path = root / 'workloads' / f'{w.name}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(w.spec, separators=(',', ':')) + '\n')
    for rel, data in w.data_files.items():
        (path.parent / rel).parent.mkdir(parents=True, exist_ok=True)
        (path.parent / rel).write_bytes(data)
    log.info('wrote %s (%.2f MiB, %d data files)', path, path.stat().st_size / 2**20, len(w.data_files))
    return path


def compile_admission(root: Path, w: Workload, name: str, base_versions: dict) -> str:
    """Compile one admission to root/bundles/<name>; returns the directory relative to root."""
    from . import _runtime  # noqa: F401  (puts the compiler on sys.path)
    import compile_workload as cw
    compiled = cw.compile_spec(w.spec, base_versions=base_versions, data_dir=root / 'workloads')
    out = root / 'bundles' / name
    cw.write_bundle(compiled, out)
    log.info('bundle %s: %d tasks, %d edges (base %s)', name, compiled.metadata['task_count'], compiled.metadata['edge_count'],
             base_versions or '{}')
    return f'bundles/{name}'


def build_manifest(root: Path, workloads: dict[str, Workload], admissions: list[dict], max_bundles: int, notes: list[str]):
    buffers: dict = {}
    for w in workloads.values():
        for b in w.spec['buffers']:
            cur = buffers.get(b['name'])
            real = b['init']['kind'] != 'fill'
            if cur is None or (real and cur['init_kind'] == 'fill'):
                buffers[b['name']] = dict(rows=b['rows'], cols=b['cols'], dtype=b.get('dtype', 'f32'), owner=w.name, init_kind=b['init']['kind'])
            elif (cur['rows'], cur['cols']) != (b['rows'], b['cols']):
                raise FrontendError(f"buffer {b['name']!r} has different shapes in workloads {cur['owner']} and {w.name}")
    shared = sorted(n for n in buffers if sum(any(b['name'] == n for b in w.spec['buffers']) for w in workloads.values()) > 1)
    compiled = 0
    for adm in admissions:
        w = workloads[adm['workload']]
        adm['compile'] = dict(workload=f'workloads/{w.name}.json', data_dir='workloads', base_versions=adm['base_versions'])
        adm['bundle_dir'] = None
        is_train = adm['workload'].startswith('train')
        if not is_train or compiled < max_bundles:
            adm['bundle_dir'] = compile_admission(root, w, adm['name'], adm['base_versions'])
            compiled += is_train
    kinds = sorted({w.kind for w in workloads.values()}, key=list(CLASS_POLICY).index)
    manifest = dict(
        manifest_version=2 if any('guard' in a for a in admissions) else MANIFEST_VERSION,     # 2 = guarded admissions (run_manifest.validate_guard)
        semantics=[
            "Allocate every buffer in `buffers` once and initialize it from its owner workload's init; later bundles that name "
            "the same buffer share it (state persists across admissions).",
            "An admission may start (list order breaks ties) when: time since start >= at_us (if set), every bundle in "
            "after_bundles has completed, and for each entry of `when` the live version of that variable is in [min, max] "
            "(max omitted = unbounded). Several admissions may be in flight at once; priority and deadline_us are inside the tasks.",
            "base_versions are the live versions of the variables at admission and bundle_dir was compiled with them. For a variable the "
            "admission does not publish (publishes lacks it) the value is only a minimum: use the live version if it is higher "
            "(a reader of weights >= v compiles against whatever is live). When bundle_dir is null, or the live version differs, compile "
            "`compile.workload` with the live versions (compile_workload.py --base-versions) or add the offset in the executor.",
            "A bundle's publishes advance the variable by `publishes[var]`; the live version after completion = base + that.",
            "manifest_version 2 only: an admission with a `guard` {buffer, index, cmp, value} additionally needs the comparison of that buffer "
            "element (read when it would otherwise start) to hold; otherwise it is skipped, as is every admission naming a skipped one in "
            "after_bundles. A guarded admission publishes nothing."],
        notes=notes,
        classes={k: CLASS_POLICY[k] for k in kinds},
        workloads={n: dict(file=f'workloads/{n}.json', tasks=w.tasks, edges=w.edges, **{'class': w.kind}, **requirements(w),
                           **summarize(w.spec)) for n, w in workloads.items()},
        buffers={n: {k: v for k, v in b.items() if k != 'init_kind'} for n, b in buffers.items()},
        shared_buffers=shared,
        admissions=admissions)
    (root / 'MANIFEST.json').write_text(json.dumps(manifest, indent=1) + '\n')
    return manifest


def make_precise(spec):
    """3xTF32 on every GEMM (the exactness mode of tests/gptdiff: fp32-accurate, so CPU and GPU trajectories stay comparable)."""
    for g in spec['graphs']:
        for t in g['tasks']:
            if t['op'] == 'matmul_tile':
                t['args']['flags'] |= F_PRECISE
    return spec


def train_config(args, steps, total, first_step):
    """AdamW lr (--lr) and a device-side linear warmup (--warmup steps, then constant); the defaults keep the library defaults."""
    optim = OptimConfig(lr=args.lr) if args.lr else OptimConfig()
    return TrainConfig(steps=steps, total_steps=total, first_step=first_step, optim=optim,
                       schedule=Schedule('const', warmup=args.warmup) if args.warmup else None)


def plan_train(cfg, tiling, args):
    n, s = args.train_steps, args.window or args.train_steps
    if s > n:
        raise FrontendError(f'--window {s} exceeds --train-steps {n}')
    return plan_windows(lambda start, steps, total: compile_training(cfg, train_config(args, steps, total, start), tiling, args.reuse, args.gap),
                        n, s, args.precise)


def plan_windows(compile_window, n, s, precise):
    """Training of n steps cut into windows of s steps. compile_window(start, steps, total) -> pipeline.Compiled for one window.
    Returns (workloads, admissions, segs): a window of s steps is one template admitted n/s times, the remainder its own 'train_tail'."""
    if not 1 <= s <= n:
        raise FrontendError(f'window {s} must be between 1 and the number of training steps {n}')
    segs = [(a, min(s, n - a)) for a in range(0, n, s)]
    workloads, admissions = {}, []
    for i, (start, steps) in enumerate(segs):
        key = 'train' if steps == s else 'train_tail'
        if key not in workloads:
            c = compile_window(start, steps, n)
            workloads[key] = Workload(key, 'train', make_precise(c.spec) if precise else c.spec, c.low.data_files, len(c.low.tasks), c.low.edges)
        admissions.append(dict(name=f'train_w{i}', workload=key, steps=[start, start + steps],
                               after_bundles=[f'train_w{i - 1}'] if i else [], when=[], at_us=0 if i == 0 else None,
                               base_versions={'weights': weights_version(start)}, publishes={'weights': 2 * steps}))
    return workloads, admissions, segs


def write_training_demo(out, module, *, loss, data, steps, optimizer=None, window=None, tiling='a100', reuse='step', gap=0, precise=False,
                        max_bundles=6, tc: TrainConfig | None = None, seed=0, name=None):
    """Compile ANY traced module (see generic.py) into a MANIFEST demo directory the host runner (demo/kvm_run) executes with no
    model-specific code: training only, `steps` optimizer steps in windows of `window` (default: one window). The per-step loss is
    the buffer `loss.<k>` of window-local step k. Returns the manifest dict."""
    root, til = Path(out), Tiling.preset(tiling) if isinstance(tiling, str) else tiling
    root.mkdir(parents=True, exist_ok=True)
    base = tc or TrainConfig()
    workloads, admissions, segs = plan_windows(
        lambda start, n, total: compile_training(module, replace(base, steps=n, total_steps=total, first_step=start), til, reuse, gap,
                                                 loss=loss, data=data, optimizer=optimizer, seed=seed),
        steps, window or steps, precise)
    notes = [f'generic module {name or getattr(module, "name", type(module).__name__)}: loss {type(loss).__name__}, data '
             f'{type(data).__name__}, optimizer {(optimizer or base.optim).kind}; tiling {tiling}; {steps} steps in {len(segs)} window(s)']
    for w in workloads.values():
        write_workload(root, w)
    m = build_manifest(root, workloads, admissions, max_bundles, notes)
    log.info('generic demo: %s', {n: (w.tasks, w.edges) for n, w in workloads.items()})
    return m


def main(argv=None):
    p = argparse.ArgumentParser(prog='kvmfe.demo', description=__doc__)
    add_common(p)
    p.add_argument('--all', action='store_true', help='train + serving + game with defaults (serve every 2 steps, 8 frames)')
    p.add_argument('--train-steps', type=int, default=4)
    p.add_argument('--window', type=int, default=0, help='steps per training bundle (default: serve-every, else all steps)')
    p.add_argument('--lr', type=float, default=None, help='AdamW learning rate (default: the library default, 3e-3)')
    p.add_argument('--warmup', type=int, default=0, help='device-side linear LR warmup steps, then constant (0: none)')
    p.add_argument('--serve-every', type=int, default=None, help='one serving request per K steps, K a multiple of the window (0: none)')
    p.add_argument('--serve-policy', choices=['exact', 'at_least'], default='at_least',
                   help='exact makes the next training window wait for the request (a host-level barrier)')
    p.add_argument('--game-frames', type=int, default=None)
    p.add_argument('--game-fb-slots', type=int, default=2, help='frame buffers (0: one per frame, so every frame can be compared byte for byte)')
    p.add_argument('--precise', action='store_true', help='3xTF32 GEMMs in training and serving (fp32-accurate; default 1xTF32)')
    p.add_argument('--defs-dir', default=None, help='directory with kvm_ops*.def (default: <repo>/include or $KVM_INCLUDE)')
    p.add_argument('--max-bundles', type=int, default=6, help='precompile at most this many training admissions (the rest carry compile info)')
    p.add_argument('--out', required=True)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO, format='%(levelname)s %(message)s')
    if a.defs_dir:
        os.environ['KVM_INCLUDE'] = a.defs_dir
    if a.serve_every is None:
        a.serve_every = 2 if a.all else 0
    if a.game_frames is None:
        a.game_frames = 8 if a.all else 0
    if not a.window:
        a.window = a.serve_every or a.train_steps
    cfg, tiling, root = ModelConfig.load(a.model), Tiling.preset(a.tiling), Path(a.out)
    root.mkdir(parents=True, exist_ok=True)

    workloads, admissions, segs = plan_train(cfg, tiling, a)
    notes = [f'model {cfg.name}: {cfg.params / 1e6:.1f}M parameters; tiling {a.tiling}; training {a.train_steps} steps in '
             f'{len(segs)} window(s) of {a.window}']
    if a.serve_every:
        starts = {s for s, _ in segs} | {a.train_steps}
        for k in range(0, a.train_steps + 1, a.serve_every):
            if k not in starts:
                raise FrontendError(f'serving after {k} steps falls inside a training window; make --serve-every a multiple of --window')
            req = ServeRequest(f'serve_s{k}', prompt_for(cfg), max_new=3, version=weights_version(k), policy=a.serve_policy)
            low, spec = compile_serving(GPT(cfg), req, tiling, a.reuse)
            workloads[req.name] = Workload(req.name, 'serve', make_precise(spec) if a.precise else spec, low.data_files, len(low.tasks), low.edges)
            admissions.append(dict(name=req.name, workload=req.name, steps=None, after_bundles=[],
                                   when=[dict(var='weights', min=req.version, **({'max': req.version} if a.serve_policy == 'exact' else {}))],
                                   at_us=None, base_versions={'weights': req.version}, publishes={}))
            if a.serve_policy == 'exact':                   # barrier: the next window may not overwrite the weights meanwhile
                nxt = next((x for x in admissions if x['name'].startswith('train_w') and x['steps'][0] == k), None)
                if nxt:
                    nxt['after_bundles'].append(req.name)
    if a.game_frames:
        low, spec = compile_game(GameConfig(frames=a.game_frames, fb_slots=a.game_fb_slots), tiling)
        workloads['game'] = Workload('game', 'game', spec, low.data_files, len(low.tasks), low.edges)
        admissions.append(dict(name='game', workload='game', steps=None, after_bundles=[], when=[], at_us=0,
                               base_versions={'frame': 0}, publishes={'frame': a.game_frames}))
        notes.append('game ops are opaque to the CPU reference (run the CPU VM with --allow-opaque)')
    key = lambda x: (x['base_versions'].get('weights', -1) - (0.5 if x['workload'].startswith('serve') else 0), x['name'])
    admissions.sort(key=key)                     # list order = a valid serial order (readers of version v before the window starting at v)
    for w in workloads.values():
        write_workload(root, w)
    m = build_manifest(root, workloads, admissions, a.max_bundles, notes)
    log.info('demo: %s', {n: (w.tasks, w.edges) for n, w in workloads.items()})
    log.info('manifest: %d admissions, %d buffers (%d shared) -> %s', len(m['admissions']), len(m['buffers']), len(m['shared_buffers']),
             root / 'MANIFEST.json')
    return 0


if __name__ == '__main__':
    sys.exit(main())
