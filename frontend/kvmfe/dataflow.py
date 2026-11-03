"""Generic dataflow analysis over buffer regions: RAW, WAR and WAW hazards -> task dependency edges.

Tasks are given in sequential program order with the rectangles they read and write. Per buffer we keep the *last writer* and the
*readers since that write* of every cell as disjoint rectangles (rect subtraction splits entries on a partial overwrite), so the
edge set is exact for the declared regions and does not grow with history. Rectangles are indexed by 64-row bucket, so a query
touches only the entries near it (the logits of a 162M model are 50k tiles). Nothing here knows what a task computes.
"""
from __future__ import annotations

from collections import defaultdict

from . import log

BUCKET = 64


def _sub(e, w):
    """e minus w for half-open rects (r0, r1, c0, c1) -> up to 4 disjoint rects."""
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


class RectTable:
    """rect -> tasks that last wrote (or have read since the last write) exactly that rect, with a row-bucket index."""

    def __init__(self):
        self.tasks: dict = {}
        self.index: dict = defaultdict(set)

    def _buckets(self, r):
        return range(r[0] // BUCKET, (r[1] - 1) // BUCKET + 1)

    def hits(self, rect):
        seen = set()
        for b in self._buckets(rect):
            seen.update(self.index.get(b, ()))
        return [e for e in seen if _hit(e, rect)]

    def tasks_hit(self, rect):
        out = set()
        for e in self.hits(rect):
            out.update(self.tasks[e])
        return out

    def add(self, rect, task):
        lst = self.tasks.get(rect)
        if lst is None:
            self.tasks[rect] = [task]
            for b in self._buckets(rect):
                self.index[b].add(rect)
        else:
            lst.append(task)

    def overwrite(self, w):
        """Remove the part of every entry covered by w (the remainder keeps its tasks)."""
        for e in self.hits(w):
            lst = self.tasks.pop(e)
            for b in self._buckets(e):
                self.index[b].discard(e)
            for piece in _sub(e, w):
                if piece in self.tasks:
                    self.tasks[piece].extend(lst)
                else:
                    self.tasks[piece] = list(lst)
                    for b in self._buckets(piece):
                        self.index[b].add(piece)


def build_edges(accesses):
    """accesses: list (program order) of (reads, writes), each a list of (buffer, r0, r1, c0, c1).
    Returns preds: list[set[int]] with every predecessor index < the task's own."""
    writers, readers = defaultdict(RectTable), defaultdict(RectTable)
    preds = []
    for i, (reads, writes) in enumerate(accesses):
        p = set()
        for buf, *rect in reads:
            p |= writers[buf].tasks_hit(tuple(rect))
        for buf, *rect in writes:
            rect = tuple(rect)
            p |= writers[buf].tasks_hit(rect)
            p |= readers[buf].tasks_hit(rect)
        p.discard(i)
        preds.append(p)
        for buf, *rect in reads:
            readers[buf].add(tuple(rect), i)
        for buf, *rect in writes:
            rect = tuple(rect)
            writers[buf].overwrite(rect)
            readers[buf].overwrite(rect)
            writers[buf].add(rect, i)
    log.info('dataflow: %d tasks, %d edges', len(preds), sum(map(len, preds)))
    return preds
