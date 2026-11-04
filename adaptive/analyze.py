#!/usr/bin/env python3
"""Aggregate adaptive_run outputs: per configuration and load phase, median [min, max] over repetitions.

  analyze.py RUNS_DIR --workers 106 --phases "base:1-8,burst:8-12,calm:12-14,pause:14-17,post:17-18,scene:18-22,tail:22-24" [--md out.md] [--json out.json]

RUNS_DIR/<config>/rep<k>/{summary.json,doom_frames.csv,serve.csv,train.csv,timeline.csv}. Times in the CSVs are microseconds from the start of the run.
Per phase: Doom frame time p50/p99/p999/max (release to completion) and frames past their deadline (completion later than one period after
the nominal release), serve latency p50/p99 (arrival to completion), training steps per second, SM utilisation (sum of per-class SM time
over workers x time) and stranded CTA fraction (idle workers while at least one task was ready; sampled every 2 ms).
"""
import argparse
import csv
import json
import statistics
import sys
from pathlib import Path


def pct(v, p):
    if not v:
        return float('nan')
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))]


def read(path):
    try:
        with open(path) as f:
            return list(csv.DictReader(f))
    except FileNotFoundError:
        return []


def parse_phases(text):
    out = []
    for item in text.split(','):
        name, rng = item.split(':')
        a, b = rng.split('-')
        out.append((name, float(a) * 1e6, float(b) * 1e6))
    return out


def phase_metrics(rep, phases, workers, period_us):
    doom = [(float(r['release_us']), float(r['submit_us']), float(r['done_us'])) for r in read(rep / 'doom_frames.csv')]
    serve = [(float(r['arrival_us']), float(r['done_us'])) for r in read(rep / 'serve.csv')]
    train = [float(r['done_us']) for r in read(rep / 'train.csv')]
    tl = read(rep / 'timeline.csv')
    ncls = len([k for k in (tl[0].keys() if tl else []) if k.endswith('_running')])
    res = {}
    for name, a, b in phases:
        m = {}
        d = [x for x in doom if a <= x[0] < b]
        ft = [x[2] - x[1] for x in d]
        m['doom_n'] = len(d)
        m['doom_p50_ms'] = pct(ft, .5) / 1e3
        m['doom_p99_ms'] = pct(ft, .99) / 1e3
        m['doom_p999_ms'] = pct(ft, .999) / 1e3
        m['doom_max_ms'] = pct(ft, 1) / 1e3
        m['doom_miss'] = sum(1 for x in d if x[2] > x[0] + period_us)
        sv = [x[1] - x[0] for x in serve if a <= x[0] < b]
        m['serve_n'] = len(sv)
        m['serve_p50_ms'] = pct(sv, .5) / 1e3
        m['serve_p99_ms'] = pct(sv, .99) / 1e3
        tr = [x for x in train if a <= x < b]
        m['train_steps_s'] = len(tr) / ((b - a) / 1e6)
        rows = [r for r in tl if a / 1e3 <= float(r['t_ms']) < b / 1e3]
        if len(rows) > 2 and ncls:
            t0, t1 = float(rows[0]['t_ms']), float(rows[-1]['t_ms'])
            sm = sum(int(rows[-1][f'c{c}_sm_ns']) - int(rows[0][f'c{c}_sm_ns']) for c in range(ncls))
            m['util_pct'] = 100 * sm / (workers * (t1 - t0) * 1e6) if t1 > t0 else float('nan')
            for c in range(ncls):
                m[f'share_c{c}_pct'] = 100 * (int(rows[-1][f'c{c}_sm_ns']) - int(rows[0][f'c{c}_sm_ns'])) / (workers * (t1 - t0) * 1e6) if t1 > t0 else float('nan')
            idle = 0.0
            n = 0
            for r in rows:
                run = sum(int(r[f'c{c}_running']) for c in range(ncls))
                rdy = sum(max(0, int(r[f'c{c}_ready'])) for c in range(ncls))
                n += 1
                if rdy > 0:
                    idle += max(0, workers - run) / workers
            m['stranded_pct'] = 100 * idle / n if n else float('nan')
        res[name] = m
    return res


def summ(vals):
    vals = [v for v in vals if v == v]
    if not vals:
        return None
    return {'med': statistics.median(vals), 'min': min(vals), 'max': max(vals)}


def flat_table(root, a):
    rows = ['| config | reps | Doom fps/instance | Doom frame p50 ms | p99 ms | max ms | frames late (of) | serve p50 ms | serve p99 ms | serve dropped | train steps/s | sweep GB/s |',
            '|---|---|---|---|---|---|---|---|---|---|---|---|']

    def g(js, *path):
        for k in path:
            js = js[k]
        return js
    def f(vals, d=2):
        vals = [v for v in vals if v == v]
        return 'n/a' if not vals else f"{statistics.median(vals):.{d}f} [{min(vals):.{d}f}, {max(vals):.{d}f}]"
    def fps_of(rep):   # frames per second per instance from the event log (older drivers wrote a wrong summary fps)
        ev = read(rep / 'doom_frames.csv')
        if not ev:
            return float('nan')
        inst = len({e['inst'] for e in ev})
        return len(ev) / inst / ((max(float(e['done_us']) for e in ev) - min(float(e['submit_us']) for e in ev)) / 1e6)
    for cfg in sorted(p for p in root.iterdir() if p.is_dir()):
        sums = []
        for r in sorted(cfg.glob('rep*/summary.json')):
            js = json.load(open(r))
            js['doom']['fps'] = fps_of(r.parent)
            sums.append(js)
        if not sums:
            continue
        ok = sum(1 for r in cfg.glob('rep*/status') if r.read_text().split()[:2] == ['0', '1'])
        rows.append(f"| {cfg.name} | {len(sums)} ({ok} oracle-exact) | {f([g(x, 'doom', 'fps') for x in sums], 1)} | {f([g(x, 'doom', 'frame_time_us', 'p50') / 1e3 for x in sums])} | "
                    f"{f([g(x, 'doom', 'frame_time_us', 'p99') / 1e3 for x in sums])} | {f([g(x, 'doom', 'frame_time_us', 'max') / 1e3 for x in sums])} | "
                    f"{f([g(x, 'doom', 'deadline_misses') for x in sums], 0)} of {sums[0]['doom']['frames']} | {f([g(x, 'serve', 'latency_us', 'p50') / 1e3 for x in sums])} | "
                    f"{f([g(x, 'serve', 'latency_us', 'p99') / 1e3 for x in sums])} | {f([g(x, 'serve', 'dropped') for x in sums], 0)} | {f([g(x, 'train', 'steps_per_s') for x in sums], 1)} | "
                    f"{f([g(x, 'sweep', 'gb_per_s') for x in sums], 0)} |")
    text = '\n'.join(rows) + '\n'
    print(text)
    if a.md:
        Path(a.md).write_text(text)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs')
    ap.add_argument('--workers', type=int, required=True)
    ap.add_argument('--phases')
    ap.add_argument('--flat', action='store_true', help='one table from summary.json (median [min,max] over repetitions), no phases')
    ap.add_argument('--fps', type=float, default=35)
    ap.add_argument('--md')
    ap.add_argument('--json')
    a = ap.parse_args()
    if a.flat:
        return flat_table(Path(a.runs), a)
    if not a.phases:
        ap.error('--phases is required without --flat')
    phases = parse_phases(a.phases)
    period = 1e6 / a.fps
    root = Path(a.runs)
    out = {}
    pooled = {}
    for cfg in sorted(p for p in root.iterdir() if p.is_dir()):
        reps = sorted(p for p in cfg.iterdir() if p.is_dir() and (p / 'summary.json').exists())
        if not reps:
            continue
        per = [phase_metrics(r, phases, a.workers, period) for r in reps]
        out[cfg.name] = {'reps': len(reps), 'phases': {}}
        for name, _, _ in phases:
            out[cfg.name]['phases'][name] = {k: summ([p[name].get(k, float('nan')) for p in per]) for k in per[0][name]}
        # pooled across repetitions: frame times and serve latencies of ALL reps (p999 needs the samples)
        ft, sl = [], []
        miss = 0
        for r in reps:
            d = [(float(x['release_us']), float(x['submit_us']), float(x['done_us'])) for x in read(r / 'doom_frames.csv')]
            ft += [x[2] - x[1] for x in d if x[0] >= phases[0][1]]
            miss += sum(1 for x in d if x[0] >= phases[0][1] and x[2] > x[0] + period)
            sl += [float(x['done_us']) - float(x['arrival_us']) for x in read(r / 'serve.csv') if float(x['arrival_us']) >= phases[0][1]]
        out[cfg.name]['pooled'] = {'doom_frames': len(ft), 'doom_p50_ms': pct(ft, .5) / 1e3, 'doom_p99_ms': pct(ft, .99) / 1e3, 'doom_p999_ms': pct(ft, .999) / 1e3,
                                   'doom_max_ms': pct(ft, 1) / 1e3, 'doom_miss': miss, 'serve_n': len(sl), 'serve_p50_ms': pct(sl, .5) / 1e3, 'serve_p99_ms': pct(sl, .99) / 1e3,
                                   'serve_p999_ms': pct(sl, .999) / 1e3}
        pooled[cfg.name] = out[cfg.name]['pooled']
    lines = []

    def f(s, d=2):
        return 'n/a' if s is None else f"{s['med']:.{d}f} [{s['min']:.{d}f}, {s['max']:.{d}f}]"
    for name, _, _ in phases:
        lines.append(f"\n### phase `{name}`\n")
        lines.append('| config | reps | Doom frame p99 ms | Doom max ms | frames late | serve p50 ms | serve p99 ms | train steps/s | util % | stranded % |')
        lines.append('|---|---|---|---|---|---|---|---|---|---|')
        for cfg, v in out.items():
            ph = v['phases'][name]
            lines.append(f"| {cfg} | {v['reps']} | {f(ph['doom_p99_ms'])} | {f(ph['doom_max_ms'])} | {f(ph['doom_miss'], 0)} | {f(ph['serve_p50_ms'])} | {f(ph['serve_p99_ms'])} | "
                         f"{f(ph['train_steps_s'], 1)} | {f(ph.get('util_pct'), 1)} | {f(ph.get('stranded_pct'), 1)} |")
    lines.append('\n### pooled over all repetitions (after the first phase starts)\n')
    lines.append('| config | Doom frames | p50 ms | p99 ms | p999 ms | max ms | late | serve n | serve p50 | serve p99 | serve p999 |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|---|')
    for cfg, p in pooled.items():
        lines.append(f"| {cfg} | {p['doom_frames']} | {p['doom_p50_ms']:.2f} | {p['doom_p99_ms']:.2f} | {p['doom_p999_ms']:.2f} | {p['doom_max_ms']:.2f} | {p['doom_miss']} | "
                     f"{p['serve_n']} | {p['serve_p50_ms']:.2f} | {p['serve_p99_ms']:.2f} | {p['serve_p999_ms']:.2f} |")
    text = '\n'.join(lines) + '\n'
    print(text)
    if a.md:
        Path(a.md).write_text(text)
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
