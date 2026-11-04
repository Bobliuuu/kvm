"""python3 -m kvmfe.inspect: task counts, critical path (tasks and us), parallelism profile and memory plan of a config."""
from __future__ import annotations

import argparse
import json
import logging
import sys

from .analyze import CostModel, analyze
from .lower import Tiling
from .nn import ModelConfig
from .pipeline import compile_training
from .train import TrainConfig


def add_common(p):
    p.add_argument('--model', default='gpt_tiny', help='preset name, JSON file or JSON string (ModelConfig fields)')
    p.add_argument('--tiling', default='cpu', help='preset: ' + ', '.join(sorted(Tiling.PRESETS)))
    p.add_argument('--reuse', default='shape', choices=['none', 'shape'], help='activation memory reuse policy')
    p.add_argument('--gap', type=int, default=0, help='do not reuse a buffer within this many logical ops of its last use')
    p.add_argument('-v', '--verbose', action='store_true', help='DEBUG logging')


def report(r: dict, out=sys.stdout):
    m = r['memory']
    w = r['widths']
    print(f"tasks {r['tasks']:,}   edges {r['edges']:,}   critical path {r['critical_path_tasks']:,} tasks / "
          f"{r['critical_path_us']:,.0f} us (model)", file=out)
    print(f"work {r['total_work_us'] / 1e6:,.3f} s   ideal parallelism {r['avg_parallelism']:,.1f}   max width {r['max_width']:,}   "
          f"mean width {r['mean_width']:,.1f}", file=out)
    print(f"list schedule on {r['workers']} workers: {r['makespan_us']:,.0f} us, utilization {r['utilization']:.1%}", file=out)
    if w:
        k = max(1, len(w) // 12)
        prof = [max(w[i:i + k]) for i in range(0, len(w), k)]
        print(f'parallelism profile (max width per {k}-level bucket): {prof[:24]}{" ..." if len(prof) > 24 else ""}', file=out)
    print('tasks by phase:', r['by_phase'], file=out)
    print('tasks by device op:', dict(sorted(r['by_op'].items(), key=lambda kv: -kv[1])), file=out)
    print(f"memory: persistent {m['persistent_bytes'] / 2**20:,.1f} MiB, activation pool {m['pool_bytes'] / 2**20:,.1f} MiB in "
          f"{m['slots']} slots ({m['transient_tensors']} tensors, {m['no_reuse_bytes'] / 2**20:,.1f} MiB without reuse, "
          f"x{m['reuse_factor']:.1f} reuse, gap {m['gap']})", file=out)
    c = r['cost_model']
    print(f"cost model: per-SM {c['sm_flops']:.2e} FLOP/s, {c['sm_bw']:.2e} B/s, hop {c['hop_us']} us", file=out)


def main(argv=None):
    p = argparse.ArgumentParser(prog='kvmfe.inspect', description=__doc__)
    add_common(p)
    p.add_argument('--steps', type=int, default=1)
    p.add_argument('--workers', type=int, default=108, help='SMs for the list-schedule estimate')
    p.add_argument('--json', action='store_true')
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO, format='%(levelname)s %(message)s')
    c = compile_training(ModelConfig.load(a.model), TrainConfig(steps=a.steps), Tiling.preset(a.tiling), a.reuse, a.gap, with_spec=False)
    r = analyze(c.low, CostModel(), a.workers)
    if a.json:
        r.pop('widths')
        print(json.dumps(r, indent=1))
    else:
        report(r)


if __name__ == '__main__':
    main()
