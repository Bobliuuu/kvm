#!/usr/bin/env python3
"""Longest dependency chain (tasks) of a compiled bundle, per step: the executor's lower bound is depth x (one dependency hop + task run).

  python3 dag_depth.py BUNDLE_DIR STEPS
Edges are the successor CSR (indegree-style), plus the version-variable ordering (wait_var / pub_var) between steps, which is how the
steps of a training run are chained; both are followed. Prints the depth of the whole run and per step.
"""
import json
import struct
import sys
from pathlib import Path


def main():
    d, steps = Path(sys.argv[1]), int(sys.argv[2])
    m = json.loads((d / 'metadata.json').read_text())
    nt, ne = m['task_count'], m['edge_count']
    t = struct.unpack(f'<{nt * 16}i', (d / 'tasks.bin').read_bytes())
    succ = struct.unpack(f'<{ne}i', (d / 'successors.bin').read_bytes())
    pubs, waits = {}, {}
    for i in range(nt):
        o = i * 16
        if t[o + 10] >= 0:
            pubs.setdefault((t[o + 10], t[o + 11]), []).append(i)
        if t[o + 8] >= 0:
            waits.setdefault((t[o + 8], t[o + 9]), []).append(i)
    edges = [[] for _ in range(nt)]
    for i in range(nt):
        o = i * 16
        edges[i] += succ[t[o + 1]:t[o + 1] + t[o + 2]]
    for key, ws in waits.items():
        for p in pubs.get(key, []):
            edges[p] += ws
    indeg = [0] * nt
    for i in range(nt):
        for j in edges[i]:
            indeg[j] += 1
    depth, order = [1] * nt, [i for i in range(nt) if indeg[i] == 0]
    for i in order:
        for j in edges[i]:
            depth[j] = max(depth[j], depth[i] + 1)
            indeg[j] -= 1
            if indeg[j] == 0:
                order.append(j)
    assert len(order) == nt, 'cycle'
    print(f'tasks {nt}, longest chain {max(depth)} tasks over {steps} steps = {max(depth) / steps:.1f} per step')


if __name__ == '__main__':
    main()
