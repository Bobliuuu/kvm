"""Generic dataflow analysis over buffer regions: RAW, WAR and WAW hazards -> task dependency edges.

Tasks are given in sequential program order with the rectangles they read and write. Per buffer we keep the *last writer* and the
*readers since that write* of every cell as disjoint rectangles (rect subtraction splits entries on partial overwrite), so the
edge set is exact for the declared regions and does not grow with history. Nothing here knows what a task computes.
"""
from __future__ import annotations

from collections import defaultdict

from . import log


def _sub(e, w):
    """e minus w for half-open rects (r0, r1, c0, c1) -> up to 4 disjoint rects."""
    if e[0] >= w[1] or w[0] >= e[1] or e[2] >= w[3] or w[2] >= e[3]:
        return [e]
    out = []
    if e[0] < w[0]:
        out.append((e[0], w[0], e[2], e[3]))
    if w[1] < e[1]:
        out.append((w[1], e[1], e[2], e[3]))
    r0, r1 = max(e[0], w[0]), min(e[1], w[1])
    if e[2] < w[2]:
        out.append((r0, r1, e[2], w[2]))
    if w[3] < e[3]:
        out.append((r0, r1, w[3], e[3]))
    return out


def _hit(a, b):
    return a[0] < b[1] and b[0] < a[1] and a[2] < b[3] and b[2] < a[3]


def build_edges(accesses):
    """accesses: list (program order) of (reads, writes), each a list of (buffer, r0, r1, c0, c1).
    Returns preds: list[set[int]] with every predecessor index < the task's own."""
    writers = defaultdict(list)     # buffer -> [(rect, task)]
    readers = defaultdict(list)
    preds = []
    for i, (reads, writes) in enumerate(accesses):
        p = set()
        for buf, *rect in reads:
            rect = tuple(rect)
            p.update(j for e, j in writers[buf] if _hit(e, rect))
        for buf, *rect in writes:
            rect = tuple(rect)
            p.update(j for e, j in writers[buf] if _hit(e, rect))
            p.update(j for e, j in readers[buf] if _hit(e, rect))
        p.discard(i)
        preds.append(p)
        for buf, *rect in reads:
            readers[buf].append((tuple(rect), i))
        for buf, *rect in writes:
            rect = tuple(rect)
            for table in (writers, readers):
                table[buf] = [(s, j) for e, j in table[buf] for s in _sub(e, rect)]
            writers[buf].append((rect, i))
    edges = sum(len(p) for p in preds)
    log.info('dataflow: %d tasks, %d edges', len(preds), edges)
    return preds
