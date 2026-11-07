#!/usr/bin/env python3
"""Verify one kvm_run output against independent references. Every check prints PASS / FAIL / SKIP with the measured numbers; the exit
status is 0 only when no check FAILed.

  check_demo.py DEMO_DIR RUN_DIR [--doom-oracle DIR] [--golden FILE] [--report OUT.json] [--loss-tol R] [--no-golden --model M [--precise]]

DEMO_DIR  the directory kvm_run ran (MANIFEST.json, workloads/, bundles/)
RUN_DIR   kvm_run --out directory (run.json, trace.csv, observations.jsonl, buffers/)

Checks (nothing here is specific to a model; names come from the manifest):
  executor      the kernel finished without a device error and every admission completed
  versions      the executor's trace shows every version-carrying task saw the requested version: `exact` readers saw the same version at
                start and end (a torn mix of two weight versions is impossible), `at_least` readers saw >= the version
  training      observed per-window loss buffers against the CPU golden (kvmfe.run_manifest, float32 reference VM), and that the loss falls
  serving       served token buffers equal the golden decode of the weights snapshot at the version the request read: the CPU VM's, or
                with --no-golden --model M a PyTorch float64 decode (serve_golden.py) of the requests whose weights are known (version 0, final)
  game          per-frame fingerprints, game state and (when every frame was kept) raw frame bytes equal the CPU build of the same engine
  cadence       tick-gap percentiles, frames per second against the 35 Hz period (from the executor's own per-task timestamps)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'frontend'))
sys.path.insert(0, str(ROOT / 'demo'))
PERIOD_US = 1e6 / 35

results: list[tuple[str, str, str]] = []


def verdict(name, status, detail):
    results.append((name, status, detail))
    print(f'{status:5s} {name}: {detail}', flush=True)


def load_json(p):
    return json.loads(Path(p).read_text())


def pct(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))] if v else float('nan')


# ---------------------------------------------------------------- golden: the CPU VM runs the same MANIFEST
def golden_observations(demo: Path, observe: str, cache: Path | None):
    if cache and cache.exists():
        return load_json(cache)
    from kvmfe.run_manifest import run_manifest
    rx = re.compile(observe)
    manifest = load_json(demo / 'MANIFEST.json')
    names = {w: [b['name'] for b in json.loads((demo / info['file']).read_text())['buffers']] for w, info in manifest['workloads'].items()}
    out = {}

    def grab(adm, vm):
        out[adm['name']] = {n: list(vm.buffers[n]) for n in names[adm['workload']] if rx.search(n)}
    run_manifest(demo, allow_opaque=True, after_each=grab)
    if cache:
        cache.write_text(json.dumps(out))
    return out


def rel_err(a, b):
    num = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    den = math.sqrt(sum(y * y for y in b))
    return num / den if den else num


# ---------------------------------------------------------------- checks
def check_executor(run):
    bad = [a['name'] for a in run['admissions'] if a['state'] != 3]
    ok = run['ok'] and run['err_code'] == 0 and not bad and run['tasks_done'] == run['tasks_admitted']
    verdict('executor', 'PASS' if ok else 'FAIL',
            f"{run['tasks_done']}/{run['tasks_admitted']} tasks in {run['kernel_ms']:.1f} ms on {run['device']} ({run['sms']} SMs, grid {run['grid']}), "
            f"sched {run['sched']}, err_code {run['err_code']}" + (f", not done: {bad}" if bad else '') + (f", errors: {run['errors']}" if run['errors'] else ''))


def check_versions(run):
    v = run['version_check']
    if v['checked'] < 0:
        verdict('versions', 'SKIP', 'slots were recycled: the trace holds only the last occupants')
    elif v['checked'] == 0:
        verdict('versions', 'SKIP', 'no admission reads a versioned variable')
    else:
        verdict('versions', 'PASS' if v['violations'] == 0 else 'FAIL',
                f"{v['checked']} version-carrying tasks checked against the device trace ({v['exact']} exact), {v['violations']} violations; "
                f"{v['at_least_advanced_during_read']} at_least readers saw the version advance while running")


def observations(run_dir):
    p = Path(run_dir) / 'observations.jsonl'
    return {o['adm']: o for o in map(json.loads, p.read_text().splitlines())} if p.exists() else {}


def check_training(manifest, obs, golden, tol):
    train = [a for a in manifest['admissions'] if a['steps'] is not None]
    if not train:
        return verdict('training', 'SKIP', 'no training admissions')
    series, worst, missing = [], 0.0, []
    for a in train:
        o = obs.get(a['name'])
        if not o:
            missing.append(a['name'])
            continue
        losses = sorted((int(re.search(r'(\d+)$', n).group(1)), v['values']) for n, v in o['buffers'].items() if n.startswith('loss.'))
        for k, vals in losses:
            ref = golden.get(a['name'], {}).get(f'loss.{k}')
            if ref is not None:
                worst = max(worst, rel_err(vals, ref))
            series.append(vals[0])
    if missing:
        return verdict('training', 'FAIL', f'no loss observation for {missing[:4]}')
    if len(series) < 2:
        return verdict('training', 'SKIP', 'fewer than two observed losses')
    k = max(1, len(series) // 4)
    head, tail = statistics.mean(series[:k]), statistics.mean(series[-k:])
    falls = tail < head
    close = worst <= tol if golden else True
    verdict('training', 'PASS' if falls and close else 'FAIL',
            f'{len(series)} step losses, first {series[0]:.4f} -> last {series[-1]:.4f} (mean of first/last {k}: {head:.4f} -> {tail:.4f}, '
            f'{"falls" if falls else "DOES NOT FALL"}); ' + (f'max relative deviation from the CPU golden {worst:.2e} (tolerance {tol:.0e})' if golden else 'no CPU golden at this size'))
    return None


def check_serving_torch(serve, obs, demo, run_dir, run, model, precise):
    """Teacher-forced PyTorch float64 decode (demo/serve_golden.py) of every request whose weights are known: version 0 (the initial
    values, rebuilt from the config) and the final version (RUN/buffers/w.*.bin). Other versions leave no snapshot: reported unchecked."""
    import serve_golden as sg
    cfg = sg.ModelConfig.load(model)
    final = run['final_versions'].get('weights', 0)
    wdir = Path(run_dir) / 'buffers'
    has_dump = (wdir / f'w.{next(iter(sg.GPT(cfg).specs))}.bin').exists()
    weights, checked, skipped, exact_reqs, bad = {}, 0, [], 0, []
    for a in serve:
        o, v = obs.get(a['name']), a['base_versions'].get('weights', 0)
        if not o:
            bad.append(f"{a['name']}: not observed")
            continue
        src = 'init' if v == 0 else 'dump' if v == final and has_dump else None
        if src is None:
            skipped.append(f"{a['name']}@v{v}")
            continue
        if src not in weights:
            weights[src] = sg.init_weights(cfg) if src == 'init' else sg.load_dumped(cfg, wdir)
        wl = load_json(demo / a['compile']['workload'])
        init = {b['name']: b['init'] for b in wl['buffers'] if b['name'].startswith(a['name'] + '.') and 'init' in b}
        prompt = [int(x) for x in init.get(f"{a['name']}.prompt", {'values': []})['values']] + [int(init[f"{a['name']}.cur0"]['values'][0])]
        gen = sorted((int(re.search(r'(\d+)$', n).group(1)), n) for n in o['buffers'] if re.search(r'\.cur[1-9]\d*$', n))
        toks = [int(o['buffers'][n]['values'][0]) for _, n in gen]
        r = sg.check_request(cfg, weights[src], prompt, toks, precise)
        checked += 1
        exact_reqs += r['exact'] == r['steps']
        print(f"  serving {a['name']} (weights version {v}, {'initial values' if src == 'init' else 'dumped final weights'}): tokens {toks} vs float64 "
              f"argmax {[t['ref_argmax'] for t in r['tokens']]}, min top1-top2 margin {min(t['top1_top2_margin'] for t in r['tokens']):.3f}, "
              f"tie tolerance {max(t['tie_tol'] for t in r['tokens']):.4f}", flush=True)
        if not r['ok']:
            bad.append(f"{a['name']}: tokens {toks} vs float64 argmax {[t['ref_argmax'] for t in r['tokens']]} (logit gaps {[round(t['gap'], 4) for t in r['tokens']]})")
    if bad:
        return verdict('serving', 'FAIL', '; '.join(bad[:3]))
    if not checked:
        return verdict('serving', 'SKIP', f'no request reads a known weights version (unchecked: {skipped})')
    return verdict('serving', 'PASS', f'{checked} of {len(serve)} requests equal the PyTorch float64 decode ({exact_reqs} token-exact on every step, the rest within the '
                   f'TF32 near-tie tolerance); unchecked, no weights snapshot survives: {skipped or "none"}')


def check_serving(manifest, obs, golden, torch_ctx=None):
    """Served tokens against a golden decode: the CPU VM's, or when it was skipped (--no-golden) PyTorch's (demo/serve_golden.py)."""
    serve = [a for a in manifest['admissions'] if a['steps'] is None and a['when'] and not a['publishes']]   # version-gated readers
    if not serve:
        return verdict('serving', 'SKIP', 'no serving admissions')
    if not golden:
        if torch_ctx is None:
            return verdict('serving', 'SKIP', 'no golden decode (--no-golden without --model); coherence is covered by the version check')
        return check_serving_torch(serve, obs, **torch_ctx)
    same = total = 0
    detail = []
    for a in serve:
        o = obs.get(a['name'])
        ref = golden.get(a['name'], {})
        if not o:
            detail.append(f"{a['name']}: not observed")
            continue
        for n, v in o['buffers'].items():
            total += 1
            if n in ref and [int(x) for x in v['values']] == [int(x) for x in ref[n]]:
                same += 1
            else:
                detail.append(f"{n}: gpu {v['values']} golden {ref.get(n)}")
    verdict('serving', 'PASS' if total and same == total else 'FAIL',
            f'{same}/{total} served token buffers equal the golden decode of the same weights version ({len(serve)} requests)' + (' ' + '; '.join(detail[:3]) if detail else ''))


def read_buf(run_dir, name):
    idx = load_json(Path(run_dir) / 'buffers' / 'index.json')
    return idx[name], (Path(run_dir) / 'buffers' / f'{name}.bin').read_bytes()


def check_game(run_dir, oracle):
    d = Path(run_dir) / 'buffers'
    if not (d / 'index.json').exists():
        return verdict('game', 'SKIP', 'no buffers dumped')
    idx = load_json(d / 'index.json')
    state = [n for n in idx if n.endswith('state') and idx[n]['rows'] == 1 and 'doom' in n]
    if not state:
        return verdict('game', 'SKIP', 'no game state buffer')
    raw = (d / f'{state[0]}.bin').read_bytes()
    words = struct.unpack(f'<{len(raw) // 4}I', raw)
    frames = (len(words) - 16) // 16
    hashes = [[words[16 + 16 * f], words[17 + 16 * f]] for f in range(frames)]
    states = [[struct.unpack('<i', struct.pack('<I', w))[0] for w in words[18 + 16 * f:32 + 16 * f]] for f in range(frames)]
    fbs = {int(m.group(1)): n for n in idx if (m := re.fullmatch(r'doom_frame\.(\d+)', n))}
    cand = Path(run_dir) / 'doom_candidate'
    cand.mkdir(exist_ok=True)
    kept_all = len(fbs) >= frames
    if kept_all:
        with open(cand / 'frames.bin', 'wb') as fh:
            for f in range(frames):
                fh.write((d / f'{fbs[f]}.bin').read_bytes())
    (cand / 'summary.json').write_text(json.dumps(dict(frames=frames, hashes=hashes, states=states,
                                                       state_names=["booted", "tics", "gamestate", "health", "armor", "kills", "monsters", "items", "secrets", "deaths", "level_finished_tic", "x", "y", "waypoint"])))
    if not oracle:
        return verdict('game', 'SKIP', f'{frames} frames recorded, no --doom-oracle given')
    r = subprocess.run([sys.executable, str(ROOT / 'doom/compare.py'), str(oracle), str(cand), '--frames', str(frames)], capture_output=True, text=True)
    first = r.stdout.splitlines()[0] if r.stdout else r.stderr[:200]
    final = [ln for ln in r.stdout.splitlines() if ln.startswith('final cand')]
    verdict('game', 'PASS' if r.returncode == 0 else 'FAIL',
            f'{first}' + ('' if kept_all else f' (frame buffers reused: raw bytes only for the last {len(fbs)}; fingerprints cover every frame)') + (f'; {final[0]}' if final else ''))
    return None


def check_cadence(run_dir, run, manifest):
    p = Path(run_dir) / 'trace.csv'
    if not p.exists():
        return verdict('cadence', 'SKIP', 'no trace.csv')
    rows = list(csv.DictReader(p.open()))
    ticks = [r for r in rows if r['op'] == 'doom_tick' and float(r['end_us']) >= 0]
    if len(ticks) < 4:
        return verdict('cadence', 'SKIP', f'{len(ticks)} doom_tick rows')
    s = [float(r['start_us']) for r in ticks]
    e = [float(r['end_us']) for r in ticks]
    gaps = [s[f] - s[f - 1] for f in range(2, len(s))]           # frame 1's gap contains the 170-330 ms engine boot (frame 0)
    fps = (len(s) - 2) / ((s[-1] - s[1]) / 1e6)
    over = sum(g > PERIOD_US for g in gaps)
    waits = [float(r['queue_wait_us']) for r in ticks]
    runs = [float(r['run_us']) for r in ticks[1:]]
    missed = sum(int(r['missed']) for r in ticks)
    ok = over == 0 and fps >= 35
    verdict('cadence', 'PASS' if ok else 'FAIL',
            f'{len(ticks)} ticks: {fps:.1f} frames/s (35 needed), tick gap p50/p99/max {pct(gaps, .5) / 1e3:.2f}/{pct(gaps, .99) / 1e3:.2f}/{max(gaps) / 1e3:.2f} ms, '
            f'{over} of {len(gaps)} gaps over 28.57 ms; tick run p50 {statistics.median(runs) / 1e3:.2f} ms, queue wait p99 {pct(waits, .99):.0f} us; '
            f'boot tick {e[0] - s[0]:.0f} us; {missed} ticks past their absolute deadline (frames before the boot finishes are late by construction)')
    # concurrency evidence: how much training / serving ran while Doom ticked
    classes = run['classes']
    span = (s[0], e[-1])
    per = {}
    for r in rows:
        if float(r['end_us']) < 0 or r['op'] in ('noop',):
            continue
        c = classes[int(r['workload'])] if int(r['workload']) < len(classes) else r['workload']
        a, b = float(r['start_us']), float(r['end_us'])
        d = per.setdefault(c, dict(tasks=0, busy_us=0.0, in_span_tasks=0, in_span_busy=0.0, first=a, last=b))
        d['tasks'] += 1
        d['busy_us'] += b - a
        d['first'], d['last'] = min(d['first'], a), max(d['last'], b)
        lo, hi = max(a, span[0]), min(b, span[1])
        if hi > lo:
            d['in_span_tasks'] += 1
            d['in_span_busy'] += hi - lo
    line = '; '.join(f"{c}: {d['tasks']} tasks, {d['busy_us'] / 1e6:.3f} CTA-s ({d['in_span_busy'] / 1e6:.3f} while Doom ran, {d['in_span_tasks']} tasks overlapping)" for c, d in per.items())
    verdict('overlap', 'INFO', f'Doom ran {(span[1] - span[0]) / 1e6:.2f} s; {line}')
    return dict(ticks=len(ticks), fps=fps, gap_p50_us=pct(gaps, .5), gap_p99_us=pct(gaps, .99), gap_max_us=max(gaps), gaps_over_period=over,
                tick_wait_p99_us=pct(waits, .99), per_class=per, doom_span_s=(span[1] - span[0]) / 1e6)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('demo')
    ap.add_argument('run')
    ap.add_argument('--doom-oracle', default=None, help='cpu_run.py output directory (frames.bin + summary.json)')
    ap.add_argument('--golden', default=None, help='cache file for the CPU golden observations (computed once with the CPU VM)')
    ap.add_argument('--observe', default=r'^(loss\.|.*\.cur[0-9]+$)', help='must equal the regex kvm_run observed with')
    ap.add_argument('--no-golden', action='store_true', help='no CPU reference (model too large): loss must fall, tokens are not compared')
    ap.add_argument('--model', default=None, help='the demo\'s model (preset, JSON file or string): without a CPU golden, serving is checked against a PyTorch decode '
                    '(needs torch; weights: initial values for version 0, RUN/buffers/w.*.bin from kvm_run --dump \'^w\\.\' for the final version)')
    ap.add_argument('--precise', action='store_true', help='the demo ran 3xTF32 tiles (tokens must equal the float64 argmax, no near-tie slack)')
    ap.add_argument('--loss-tol', type=float, default=2e-3)
    ap.add_argument('--report', default=None)
    a = ap.parse_args(argv)
    demo, run_dir = Path(a.demo), Path(a.run)
    manifest, run = load_json(demo / 'MANIFEST.json'), load_json(run_dir / 'run.json')
    check_executor(run)
    check_versions(run)
    obs = observations(run_dir)
    golden = golden_observations(demo, a.observe, Path(a.golden) if a.golden else None) if obs and not a.no_golden else {}
    check_training(manifest, obs, golden, a.loss_tol)
    torch_ctx = dict(demo=demo, run_dir=run_dir, run=run, model=a.model, precise=a.precise) if a.model and not golden else None
    check_serving(manifest, obs, golden, torch_ctx)
    check_game(run_dir, a.doom_oracle)
    cad = check_cadence(run_dir, run, manifest)
    if a.report:
        Path(a.report).write_text(json.dumps(dict(checks=[dict(name=n, status=s, detail=d) for n, s, d in results], cadence=cad), indent=1, default=str))
    bad = [n for n, s, _ in results if s == 'FAIL']
    print(('FAIL: ' + ', '.join(bad)) if bad else 'ALL CHECKS PASSED (SKIPs are listed above)')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
