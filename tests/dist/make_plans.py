#!/usr/bin/env python3
"""Write partitioner plans (per-rank bundles + dist.txt + golden expectations) for the GPU harness `test_dist plan DIR`.

    make_plans.py OUT_DIR [--devices N]

Plans whose name ends in `_net` put every device on its own node, so every cross-device message is a send_proxy message: they
run as separate processes (dist/launch.py) and across hosts. The others mix peer-to-peer puts (same node) and the proxy path.
`--devices` is the rank count of the `*_gpus` plans (set it to the GPU count of the machine: one rank per GPU).
"""
import argparse
import copy
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
import verify            # noqa: E402
import workloads as wl   # noqa: E402
import partition as pt   # noqa: E402

TP = {'W1': 'cols', 'W2': 'rows', 'gW1': 'cols', 'gW2': 'rows'}
ZERO = {'x': 'rows', 'W1': 'rows', 'W2': 'rows', 'gW1': 'rows', 'gW2': 'rows'}


def plans(devices):
    mlp, regress = wl.mlp(), wl.regress(16)
    chain, _ = wl.chain(L=4, m=8, d=8)
    split = [f'p{l}' for l in range(4)] + [f'h{l}' for l in range(4)] + ['x']
    sh = lambda lay, **kw: dict({'mode': 'shard', 'layouts': lay}, **kw)
    U = pt.Cluster.uniform
    P = max(2, devices)
    out = {
        'tp_direct':   (mlp, U(4), sh(TP, collective='direct'), 0),
        'tp_ring':     (mlp, U(4), sh(TP, collective='ring', chunks=2), 0),
        'tp_tree':     (mlp, U(4), sh(TP, collective='tree'), 0),
        'zero_mixed':  (mlp, U(4, 2), sh(ZERO, collective='ring'), 0),           # 2 nodes x 2 devices: p2p inside, proxy across
        'zero_net':    (mlp, U(4, 4), sh(dict(ZERO, **TP)), 0),
        'zero_bf16_net': (mlp, U(4, 4), sh(ZERO, wire='bf16'), 0),
        'tp_ring_net': (mlp, U(4, 4), sh(TP, collective='ring', chunks=2), 0),
        'dp_fp_net':   (regress, U(4, 4), sh({'x': 'rows', 'target': 'rows', 'pred': 'rows'}), 1e-5),
        'pipe_net':    (chain, U(4, 4), {'mode': 'place', 'auto': 'pipeline', 'stages': 4, 'microbatches': 4, 'split': split}, 0),
        'pipe_p2p':    (chain, U(4, 1), {'mode': 'place', 'auto': 'pipeline', 'stages': 4, 'microbatches': 4, 'split': split}, 0),
        'tp_gpus':     (mlp, U(P, 1) if 8 % P == 0 else U(4, 1), sh(TP, collective='ring', chunks=2), 0),
        'zero_gpus':   (mlp, U(P, 1) if 8 % P == 0 else U(4, 1), sh(ZERO), 0),
    }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('out_dir')
    ap.add_argument('--devices', type=int, default=2)
    ap.add_argument('--print-proxy-only', type=int, metavar='WORLD', help='print names of the plans with WORLD ranks that run across processes/hosts')
    args = ap.parse_args()
    out = Path(args.out_dir)
    index = {}
    for name, (spec, cluster, strategy, tol) in plans(args.devices).items():
        plan = pt.partition(copy.deepcopy(spec), cluster, strategy)
        gold = verify.golden(plan.spec)
        # the simulator is the oracle for the plan that goes to the GPUs
        for seed in range(3):
            results, _ = verify.sim.run(plan, seed)
            got = plan.reconstruct(results)
            bad = [n for n in got if n in gold and got[n] != gold[n] and (tol == 0 or any(abs(a - b) > tol * max(1, abs(b)) for a, b in zip(got[n], gold[n])))]
            if bad:
                raise SystemExit(f'plan {name}: simulator disagrees with the unpartitioned run on {bad}')
        pt.write_plan(plan, out / name, gold)
        est = pt.estimate(plan)
        index[name] = dict(ranks=len(plan.shards), tol=tol, messages=plan.comm_msgs, bytes=plan.comm_bytes, modeled_us=round(est['makespan_us'], 2),
                           proxy_only=all(not r or r['kind'] != 'peer' for s in plan.shards for r in s.remotes.values()) and
                                      not any(r['kind'] == 'inbox' for s in plan.shards for r in s.remotes.values()))
        print(f"plan {name}: {len(plan.shards)} ranks, {plan.comm_msgs} messages, {plan.comm_bytes / 1024:.1f} KiB, proxy_only={index[name]['proxy_only']}")
    (out / 'index.json').write_text(json.dumps(index, indent=1) + '\n')
    if args.print_proxy_only:
        print(' '.join(n for n, v in index.items() if v['proxy_only'] and v['ranks'] == args.print_proxy_only))


if __name__ == '__main__':
    main()
