#!/usr/bin/env python3
"""Straggler mitigation (work stealing across data-parallel replicas as dynamic admission) and a failure/goodput model.
Both are SIMULATIONS of a design; neither runs in the executor. The determinism rule is what the design adds:

  Admission unit = a BUNDLE of microbatches. Its output (a partial gradient, tagged with the bundle id) is a pure function of
  (weights version, bundle data), so WHICH replica computes it does not change the bits. Partial gradients are then reduced in
  bundle-id order (a fixed tree), never in completion order. A replica that finishes early admits (steals) the next unclaimed
  bundle of the slowest replica's queue; the claim is one atomic on the owner's admission counter, i.e. what the executor's
  admitter already does for its own task queue, extended across devices.
"""
import argparse
import heapq
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'runtime')]
import ref_ops   # noqa: E402


def step_time(R, bundles, t_us, speed=None, steal=True, fetch_us=0.0, claim_us=0.0, seed=0, jitter=0.0):
    """One step of data-parallel compute: `bundles` bundles of t_us each, R replicas, replica r runs at speed[r] (<1 = slow).
    Static: bundle i belongs to replica i % R. Steal: a replica that drains its own queue claims from the replica with the most
    unclaimed bundles (paying claim_us + fetch_us once per stolen bundle). Returns (makespan_us, owner_of_bundle)."""
    rng = random.Random(seed)
    speed = speed or [1.0] * R
    queues = [[i for i in range(bundles) if i % R == r] for r in range(R)]
    free = [(0.0, r) for r in range(R)]
    heapq.heapify(free)
    owner = {}
    end = 0.0
    while free:
        t, r = heapq.heappop(free)
        if queues[r]:
            b = queues[r].pop(0)
            extra = 0.0
        elif steal:
            v = max(range(R), key=lambda x: len(queues[x]))
            if not queues[v]:
                continue
            b = queues[v].pop()                      # take from the tail: the victim keeps its own order
            extra = claim_us + fetch_us
        else:
            continue
        dur = t_us * (1 + jitter * rng.random()) / speed[r] + extra
        owner[b] = r
        end = max(end, t + dur)
        heapq.heappush(free, (t + dur, r))
    return end, owner


def bundle_grad(b, n=64, seed=11):
    """Deterministic fp32 'partial gradient' of bundle b (a pure function of the bundle)."""
    rng = random.Random(seed * 100003 + b)
    return [ref_ops.f32(rng.uniform(-1, 1) * 10 ** rng.randint(-3, 3)) for _ in range(n)]


def reduce_fixed(bundles, n=64):
    """Sum in bundle-id order (independent of who computed what, or when)."""
    acc = [0.0] * n
    for b in range(bundles):
        g = bundle_grad(b, n)
        acc = [ref_ops.f32(x + y) for x, y in zip(acc, g)]
    return acc


def reduce_by_completion(bundles, owner, R, n=64):
    """The tempting alternative: each replica sums what it computed, then replicas are summed. Order depends on the assignment."""
    parts = []
    for r in range(R):
        acc = [0.0] * n
        for b in sorted(b for b, o in owner.items() if o == r):
            acc = [ref_ops.f32(x + y) for x, y in zip(acc, bundle_grad(b, n))]
        parts.append(acc)
    out = [0.0] * n
    for p in parts:
        out = [ref_ops.f32(x + y) for x, y in zip(out, p)]
    return out


def goodput(gpus, step_s, gpu_mtbf_h, ckpt_s, restart_s, interval_steps=None, lost_work_frac=0.5):
    """Expected fraction of wall time that is useful training under periodic checkpoints (first-order, Young/Daly style).
    cluster failure rate = gpus / gpu_mtbf; a failure loses half an interval of work on average plus restart_s. If
    interval_steps is None the Young optimum sqrt(2 C M) is used."""
    M = gpu_mtbf_h * 3600.0 / gpus                           # cluster MTBF, s
    if interval_steps is None:
        tau = math.sqrt(2 * ckpt_s * M)
        interval_steps = max(1, round(tau / step_s))
    tau = interval_steps * step_s
    overhead = ckpt_s / tau + (lost_work_frac * tau + restart_s) / M
    return dict(cluster_mtbf_h=M / 3600.0, interval_steps=interval_steps, interval_s=tau, goodput=1.0 / (1.0 + overhead))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpus', type=int, default=4096)
    ap.add_argument('--step-s', type=float, default=90.0)
    a = ap.parse_args(argv)
    # 419 unexpected interruptions / 54 days / 16384 GPUs (Llama 3 report, via secondary summaries): per-GPU MTBF
    mtbf = 54 * 24 * 16384 / 419
    print(f'per-GPU MTBF from that rate: {mtbf:.0f} h')
    for ck in (5, 30, 120):
        for rs in (60, 600):
            print(a.gpus, 'ckpt', ck, 's restart', rs, 's', goodput(a.gpus, a.step_s, mtbf, ck, rs))
    return 0


if __name__ == '__main__':
    sys.exit(main())
