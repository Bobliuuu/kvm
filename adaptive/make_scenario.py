#!/usr/bin/env python3
"""Compile the GPT training windows and the serving-request pool for adaptive_run (the Doom frames are built by the driver itself).

  make_scenario.py --out DIR [--model PRESET|JSON] [--train-runs 3] [--train-window 1] [--train-total 600] [--serve-pool 48]
                   [--serve-prompt 3] [--serve-new 4] [--tiling a100]

DIR/train_r<r>/       one bundle per independent training run: a template window of --train-window optimizer steps. The step counter lives in a
                      device buffer, so the SAME bundle is re-admitted for every window (the host admits window k+1 after window k completed).
                      Weight-version publishes are removed: the runs are independent and ordered by the host, nothing else reads their weights.
DIR/serve_<i>/        a pool of identical serving requests (prefill + KV-cache decode), on private copies of the weights (sw.*), each with its
                      own KV cache; a first task resets the cache length so a request slot can be reused.
DIR/scenario.json     what the driver needs: counts, tokens per request, buffer names to verify.
Runs are independent: buffer names are prefixed t<r>. so the executor's single buffer table holds them side by side.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'frontend'))

from kvmfe import _runtime  # noqa: E402,F401  (puts runtime/ on sys.path)
from kvmfe.lower import Tiling  # noqa: E402
from kvmfe.nn import GPT, ModelConfig  # noqa: E402
from kvmfe.pipeline import compile_training  # noqa: E402
from kvmfe.serve import ServeRequest, compile_serving  # noqa: E402
from kvmfe.train import TrainConfig  # noqa: E402
import compile_workload as cw  # noqa: E402

log = logging.getLogger('make_scenario')


def rename_buffers(spec, fn):
    """Apply fn(name) -> name to every buffer and to every task argument that names one."""
    old = {b['name'] for b in spec['buffers']}
    for b in spec['buffers']:
        b['name'] = fn(b['name'])
    for g in spec['graphs']:
        for t in g['tasks']:
            t['args'] = {k: (fn(v) if isinstance(v, str) and v in old else v) for k, v in t.get('args', {}).items()}


def strip_versions(spec):
    """Independent workload: no version variables, publishes or waits (the host orders windows)."""
    spec['variables'] = []
    for g in spec['graphs']:
        for t in g['tasks']:
            t.pop('publishes', None)
            t.pop('reads', None)


def write_bundle(spec, files, root: Path, name: str):
    wdir = root / 'workloads' / name
    wdir.mkdir(parents=True, exist_ok=True)
    for rel, data in files.items():
        (wdir / rel).parent.mkdir(parents=True, exist_ok=True)
        (wdir / rel).write_bytes(data)
    compiled = cw.compile_spec(spec, base_versions={}, data_dir=wdir)
    cw.write_bundle(compiled, root / name)
    return compiled.metadata


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--model', default='{"name":"adp","vocab":2048,"d_model":256,"layers":4,"heads":8,"seq":128,"batch":8}')
    ap.add_argument('--tiling', default='a100')
    ap.add_argument('--train-runs', type=int, default=3)
    ap.add_argument('--train-window', type=int, default=1, help='optimizer steps per training bundle')
    ap.add_argument('--train-total', type=int, default=600, help='length of each run\'s token stream in steps (windows available)')
    ap.add_argument('--serve-pool', type=int, default=48, help='request slots that can be in flight at once')
    ap.add_argument('--serve-prompt', type=int, default=3)
    ap.add_argument('--serve-new', type=int, default=4)
    ap.add_argument('-v', action='store_true')
    a = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if a.v else logging.INFO, format='%(levelname)s %(message)s')
    cfg, tiling, root = ModelConfig.load(a.model), Tiling.preset(a.tiling), Path(a.out)
    root.mkdir(parents=True, exist_ok=True)

    comp = compile_training(cfg, TrainConfig(steps=a.train_window, total_steps=a.train_total, first_step=0), tiling)
    weight_init = {b['name']: b['init'] for b in comp.spec['buffers'] if b['name'].startswith('w.')}
    train = {}
    for r in range(a.train_runs):
        spec = json.loads(json.dumps(comp.spec))
        strip_versions(spec)
        rename_buffers(spec, lambda n, r=r: f't{r}.{n}')
        for g in spec['graphs']:
            g['name'] = f'train{r}'
        m = write_bundle(spec, comp.low.data_files, root, f'train_r{r}')
        train = dict(tasks=m['task_count'], edges=m['edge_count'])
    log.info('train: %d runs x window of %d step(s): %s', a.train_runs, a.train_window, train)

    model = GPT(cfg)
    prompt = tuple((1 + 7 * i) % cfg.vocab for i in range(a.serve_prompt))
    serve = {}
    for i in range(a.serve_pool):
        req = ServeRequest(f'sv{i}', prompt, max_new=a.serve_new, version=0, policy='at_least', priority=0, deadline_us=0)
        low, spec = compile_serving(model, req, tiling)
        strip_versions(spec)
        for g in spec['graphs']:
            for t in g['tasks']:
                t.pop('deadline_us', None)     # request latency is measured end to end by the driver; no per-task deadline
        have = {b['name'] for b in spec['buffers']}
        for b in spec['buffers']:
            if b['name'] in weight_init:
                b['init'] = weight_init[b['name']]       # real weights, not the placeholder zero fill
        rename_buffers(spec, lambda n: 's' + n if n.startswith('w.') else n)
        g = spec['graphs'][0]
        roots = [t for t in g['tasks'] if not t.get('after')]
        g['tasks'].append({'id': 'reset_lens', 'op': 'zero_buf', 'args': {'x': f'{req.name}.lens', 'e0': 0, 'e1': 1}})
        for t in roots:
            t['after'] = ['reset_lens']
        m = write_bundle(spec, low.data_files, root, f'serve_{i}')
        serve = dict(tasks=m['task_count'], edges=m['edge_count'], buffers=len(have))
    log.info('serve: pool of %d requests: %s', a.serve_pool, serve)

    json.dump(dict(model=json.loads(json.dumps(cfg.__dict__)), params=cfg.params, tiling=a.tiling,
                   train=dict(runs=a.train_runs, window_steps=a.train_window, total_steps=a.train_total, tasks_per_window=train['tasks'],
                              loss_buffer_suffix='loss'),
                   serve=dict(pool=a.serve_pool, prompt=list(prompt), max_new=a.serve_new, tasks_per_request=serve['tasks'],
                              token_buffers=[f'cur{i + 1}' for i in range(a.serve_new)])),
              open(root / 'scenario.json', 'w'), indent=1)
    log.info('scenario -> %s', root / 'scenario.json')


if __name__ == '__main__':
    main()
