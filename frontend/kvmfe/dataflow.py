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


class Group:
    """Tasks that last wrote / have read since the last write a rect. Pieces of a split rect share one group; a shared group is
    never mutated (appends copy it), so cached joins keyed by (id, ver) stay valid."""
    __slots__ = ('tasks', 'shared', 'ver')

    def __init__(self, tasks):
        self.tasks, self.shared, self.ver = tasks, False, 0


class RectTable:
    """rect -> Group, with a row-bucket index. Reader entries may be added for identical rects repeatedly (merged)."""

    def __init__(self):
        self.groups: dict = {}
        self.index: dict = defaultdict(set)

    def _buckets(self, r):
        return range(r[0] // BUCKET, (r[1] - 1) // BUCKET + 1)

    def hits(self, rect):
        seen = set()
        for b in self._buckets(rect):
            seen.update(self.index.get(b, ()))
        return [e for e in seen if _hit(e, rect)]

    def groups_hit(self, rect):
        return [self.groups[e] for e in self.hits(rect)]

    def _put(self, rect, g):
        old = self.groups.get(rect)
        if old is None:
            for b in self._buckets(rect):
                self.index[b].add(rect)
            self.groups[rect] = g
        elif old is not g:
            self.groups[rect] = Group(old.tasks + g.tasks)

    def add(self, rect, task):
        g = self.groups.get(rect)
        if g is None:
            self._put(rect, Group([task]))
        elif g.shared:
            self.groups[rect] = Group(g.tasks + [task])
        else:
            g.tasks.append(task)
            g.ver += 1

    def overwrite(self, w):
        """Remove the part of every entry covered by w (the remainder keeps its group)."""
        for e in self.hits(w):
            g = self.groups.pop(e)
            for b in self._buckets(e):
                self.index[b].discard(e)
            pieces = _sub(e, w)
            g.shared = g.shared or len(pieces) > 1
            for piece in pieces:
                self._put(piece, g)


JOIN_MIN = 32


def build_edges(accesses):
    """accesses: list (program order) of (reads, writes), each a list of (buffer, r0, r1, c0, c1).
    Returns (seq, preds): seq[k] is the original index of the k-th task or -1 for a synthetic join (a no-op that collapses an
    R x W complete bipartite WAR/RAW fan into R + W edges, created when the same large predecessor set is needed twice);
    preds[k] are the predecessor positions in seq (always < k)."""
    writers, readers = defaultdict(RectTable), defaultdict(RectTable)
    seq, preds, new_of = [], [], {}
    set_cache, group_cache = {}, {}

    def make_join(tasks):
        seq.append(-1)
        preds.append({new_of[t] for t in tasks})
        return len(seq) - 1

    def sources(groups):
        if sum(len(g.tasks) for g in groups) < JOIN_MIN:
            return {new_of[t] for g in groups for t in g.tasks}
        key = frozenset((id(g), g.ver) for g in groups)
        ent = set_cache.get(key)
        if ent is not None:
            ent[0] += 1
            if ent[1] is None:
                ent[1] = make_join({t for g in groups for t in g.tasks})
            return {ent[1]}
        set_cache[key] = [1, None, groups]                    # keeps the groups alive so ids stay unique
        out = set()
        for g in groups:
            if len(g.tasks) < JOIN_MIN:
                out.update(new_of[t] for t in g.tasks)
                continue
            ent = group_cache.get((id(g), g.ver))
            if ent is None:
                group_cache[(id(g), g.ver)] = [1, None, g]
                out.update(new_of[t] for t in g.tasks)
            else:
                ent[0] += 1
                if ent[1] is None:
                    ent[1] = make_join(g.tasks)
                out.add(ent[1])
        return out

    for i, (reads, writes) in enumerate(accesses):
        gs = {}
        for buf, *rect in reads:
            for g in writers[buf].groups_hit(tuple(rect)):
                gs[id(g)] = g
        for buf, *rect in writes:
            rect = tuple(rect)
            for g in writers[buf].groups_hit(rect) + readers[buf].groups_hit(rect):
                gs[id(g)] = g
        p = sources(list(gs.values()))
        new_of[i] = len(seq)
        seq.append(i)
        preds.append(p)
        for buf, *rect in reads:
            readers[buf].add(tuple(rect), i)
        for buf, *rect in writes:
            rect = tuple(rect)
            writers[buf].overwrite(rect)
            readers[buf].overwrite(rect)
            writers[buf].add(rect, i)
    log.info('dataflow: %d tasks + %d joins, %d edges', len(accesses), seq.count(-1), sum(map(len, preds)))
    return seq, preds
