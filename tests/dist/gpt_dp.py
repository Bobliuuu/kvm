#!/usr/bin/env python3
"""Data-parallel training plans for frontend models (GPT, Llama, MoE, ...) and their single-device reference.

    gpt_dp.py OUT_DIR --model gpt_tiny --ranks 2 --steps 2 [--precise] [--tiling cpu|a100] [--golden]

DP over P ranks with per-rank batch b is gradient accumulation over P microbatches of b: the reference is the SAME frontend model
compiled with accum=P on one device. The per-rank program is the accum=1 compile; its data streams are replaced by the accum=P
compile's streams (P x the windows), and the dp planner (runtime/partition.py, DPPlanner) points rank r at window r of each group.
Writes OUT_DIR/plan (per-rank bundles, dist.txt with `expect` lines from the CPU golden when --golden), OUT_DIR/ref (the accum=P
workload + bundle, for a single-GPU kvm_exec reference run) and OUT_DIR/info.json.
"""
import argparse
import copy
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / 'runtime'), str(ROOT / 'dist'), str(ROOT / 'frontend'), str(ROOT / 'examples/models')]
import commops            # noqa: E402
import partition as pt    # noqa: E402
import gen_bundle as zoo  # noqa: E402  (examples/models: frontend build helper)


def build(model, ranks, steps, tiling='cpu', precise=False, reuse='none'):
    rank_c, _ = zoo.build(model, steps, tiling, precise, accum=1, reuse=reuse)
    ref_c, _ = zoo.build(model, steps, tiling, precise, accum=ranks, reuse=reuse)
    rank_spec, ref_spec = copy.deepcopy(rank_c.spec), copy.deepcopy(ref_c.spec)
    build.data_files = dict(ref_c.low.data_files, **{k: v for k, v in rank_c.low.data_files.items() if k not in ref_c.low.data_files})
    ref_bufs = {b['name']: b for b in ref_spec['buffers']}
    streams = [b['name'] for b in rank_spec['buffers'] if b['name'].startswith('stream_')]
    for i, b in enumerate(rank_spec['buffers']):
        if b['name'] in streams:
            rank_spec['buffers'][i] = copy.deepcopy(ref_bufs[b['name']])
    return rank_spec, ref_spec, streams


def golden(spec, timeout=3600):
    """CPU golden buffers (the VM's default 60 s timeout is too short for gpt_small on a busy host)."""
    import compile_workload as cw
    import workload_vm
    return workload_vm.run_golden(cw.compile_spec(copy.deepcopy(spec)), timeout=timeout)['buffers']


def _need(c, m):
    if not c:
        raise SystemExit(f'gpt_dp: {m}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('out')
    ap.add_argument('--model', default='gpt_tiny')
    ap.add_argument('--ranks', type=int, default=2)
    ap.add_argument('--steps', type=int, default=2)
    ap.add_argument('--tiling', default='cpu')
    ap.add_argument('--precise', action='store_true')
    ap.add_argument('--golden', action='store_true', help='CPU golden of the accum=P reference -> expect lines (small models only)')
    ap.add_argument('--collective', default='auto')
    ap.add_argument('--nodes', type=int, default=1, help='ranks spread over this many nodes (cross-node messages go through the proxy)')
    ap.add_argument('--reuse', default='none', help="frontend buffer reuse (none keeps every activation; 'shape' for model-sized runs)")
    ap.add_argument('--symbolic', action='store_true', help='do not materialise buffer values in Python (model-sized runs; no --golden)')
    ap.add_argument('--pipeline', action='store_true', help='instead of dp: the accum=1 program cut into --ranks pipeline stages (place mode)')
    a = ap.parse_args()
    out = Path(a.out)
    if a.pipeline:
        c, _ = zoo.build(a.model, a.steps, a.tiling, a.precise, accum=1)
        spec = copy.deepcopy(c.spec)
        plan = pt.partition(spec, pt.Cluster.uniform(a.ranks, a.nodes), {'mode': 'place', 'auto': 'pipeline', 'stages': a.ranks})
        gold = golden(spec) if a.golden else None
        pt.write_plan(plan, out / 'plan', gold)
        one = pt.partition(copy.deepcopy(spec), pt.Cluster.uniform(1), {'mode': 'place', 'auto': 'pipeline', 'stages': 1})
        pt.write_plan(one, out / 'ref_plan', gold)       # the same program unpartitioned on one GPU: the GPU-vs-CPU noise floor
        print(f"{a.model} pipeline x{a.ranks}: tasks/rank {[len(s.spec['graphs'][0]['tasks']) for s in plan.shards]}, {plan.comm_msgs} messages, "
              f"{plan.comm_bytes / 2**20:.2f} MiB")
        return
    rank_spec, ref_spec, streams = build(a.model, a.ranks, a.steps, a.tiling, a.precise, a.reuse)
    plan = pt.partition(rank_spec, pt.Cluster.uniform(a.ranks, a.nodes), {'mode': 'dp', 'streams': streams, 'collective': a.collective,
                                                                          'dump_state': True, 'symbolic': a.symbolic})
    data_dir = None
    if getattr(build, 'data_files', None):                 # big streams travel as raw files (frontend FILE_MIN)
        data_dir = out / 'data'
        for rel, raw in build.data_files.items():
            (data_dir / rel).parent.mkdir(parents=True, exist_ok=True)
            (data_dir / rel).write_bytes(raw)
    gold = None
    if a.golden:
        gold = golden(ref_spec)
    pt.write_plan(plan, out / 'plan', gold, data_dir)
    # the reference on ONE GPU through the same harness (dp planner with P = 1 over the accum=P workload: identity apart from
    # a copy per gradient), so step times and dumped state compare like for like
    ref_plan = pt.partition(copy.deepcopy(ref_spec), pt.Cluster.uniform(1), {'mode': 'dp', 'streams': streams, 'dump_state': True, 'symbolic': a.symbolic})
    pt.write_plan(ref_plan, out / 'ref_plan', gold, data_dir)
    (out / 'ref').mkdir(parents=True, exist_ok=True)
    (out / 'ref' / 'workload.json').write_text(json.dumps(ref_spec, separators=(',', ':')))
    info = dict(model=a.model, ranks=a.ranks, nodes=a.nodes, steps=a.steps, tiling=a.tiling, precise=a.precise, streams=streams,
                tasks_per_rank=[len(s.spec['graphs'][0]['tasks']) for s in plan.shards], ref_tasks=sum(len(g['tasks']) for g in ref_spec['graphs']),
                messages=plan.comm_msgs, comm_bytes=plan.comm_bytes, reductions=plan.reductions, state_buffers=sorted(plan.layout))
    (out / 'info.json').write_text(json.dumps(info, indent=1) + '\n')
    print(f"{a.model} dp x{a.ranks} ({a.nodes} node(s)): {info['tasks_per_rank'][0]} tasks/rank (reference {info['ref_tasks']}), "
          f"{plan.reductions} gradient all-reduces, {plan.comm_msgs} messages, {plan.comm_bytes / 2**20:.2f} MiB")


if __name__ == '__main__':
    main()
