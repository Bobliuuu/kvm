"""Static analysis of a lowered workload: cost model, critical path, parallelism profile, list-schedule makespan."""
from __future__ import annotations

import heapq
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .ir import FrontendError
from .opset import op as device_op

if TYPE_CHECKING:          # emit imports this module (step_ranks), so Lowered is named here for annotations only
    from .emit import Lowered


@dataclass(frozen=True)
class CostModel:
    """Per-task duration = max(flops / sm_flops, bytes / sm_bw) + hop_us (one dependency hop through global memory).
    Defaults: one A100 SM (fp32 CUDA cores 19.5 TFLOP/s / 108; 1.5 TB/s HBM / 108), hop 5 us (AGENTS.md). Ops declaring
    `fixed_us` (opaque engines) use it. This is a model for ranking designs, not a prediction."""
    sm_flops: float = 1.8e11
    sm_bw: float = 1.4e10
    hop_us: float = 5.0

    def task_us(self, t) -> float:
        d = device_op(t.dev)
        if d.fixed_us is not None:
            return d.fixed_us + self.hop_us
        if t.dev == 'noop':
            return self.hop_us
        fl, by = t.flops_bytes()
        return max(fl / self.sm_flops, by / self.sm_bw) * 1e6 + self.hop_us


def step_ranks(low: Lowered, model: CostModel = CostModel()) -> list[float]:
    """Each task's longest remaining path inside its own step, in the model's microseconds: its cost plus the largest rank among its
    successors of the same graph and step (HEFT's upward rank, Topcuoglu, Hariri and Wu 2002, cut at the step's end). Cut there, a
    run's steps rank alike however many there are; ranked to the run's end, each step of a 50-step run would hold one or two levels of
    the executor's 32. lower_program lists every task after its predecessors; one that does not is refused."""
    preds, rank = low.preds, [model.task_us(t) for t in low.tasks]
    key = [(t.op.graph, t.op.step) for t in low.tasks]
    longest = [0.0] * len(rank)            # the largest rank among each task's same-step successors seen so far
    for i in range(len(rank) - 1, -1, -1):
        rank[i] += longest[i]
        for j in preds[i]:
            if j >= i:
                raise FrontendError(f'task {low.tasks[i].name}: predecessor {low.tasks[j].name} is not listed before it')
            if key[j] == key[i] and rank[i] > longest[j]:
                longest[j] = rank[i]
    return rank


def analyze(low: Lowered, model: CostModel = CostModel(), workers: int = 108) -> dict:
    n, preds = len(low.tasks), low.preds
    cost = [model.task_us(t) for t in low.tasks]
    level, finish = [0] * n, [0.0] * n                         # ASAP depth (tasks) and ASAP finish time (us)
    for i in range(n):
        p = preds[i]
        level[i] = 1 + max((level[j] for j in p), default=0)
        finish[i] = cost[i] + max((finish[j] for j in p), default=0.0)
    width = Counter(level)
    # blevel: longest path to a sink, to prioritize the critical path in the list schedule
    succ = [[] for _ in range(n)]
    for i, p in enumerate(preds):
        for j in p:
            succ[j].append(i)
    blevel = [0.0] * n
    for i in range(n - 1, -1, -1):
        blevel[i] = cost[i] + max((blevel[j] for j in succ[i]), default=0.0)
    indeg, ready_at = [len(p) for p in preds], [0.0] * n
    ready = [(0.0, -blevel[i], i) for i in range(n) if indeg[i] == 0]
    heapq.heapify(ready)
    free = [0.0] * workers
    makespan, busy = 0.0, 0.0
    while ready:
        r, _, i = heapq.heappop(ready)
        w = heapq.heappop(free)
        s = max(w, r)
        e = s + cost[i]
        heapq.heappush(free, e)
        makespan, busy = max(makespan, e), busy + cost[i]
        for j in succ[i]:
            ready_at[j] = max(ready_at[j], e)
            indeg[j] -= 1
            if indeg[j] == 0:
                heapq.heappush(ready, (ready_at[j], -blevel[j], j))
    widths = [width[k] for k in sorted(width)]
    by_phase = Counter(t.op.phase for t in low.tasks)
    by_dev = Counter(t.dev for t in low.tasks)
    return dict(
        tasks=n, edges=low.edges, critical_path_tasks=max(level, default=0), critical_path_us=max(finish, default=0.0),
        total_work_us=sum(cost), avg_parallelism=sum(cost) / max(max(finish, default=1.0), 1e-9), max_width=max(widths, default=0),
        mean_width=n / max(len(widths), 1), widths=widths, workers=workers, makespan_us=makespan,
        utilization=busy / (makespan * workers) if makespan else 0.0, by_phase=dict(by_phase), by_op=dict(by_dev),
        cost_model=dict(sm_flops=model.sm_flops, sm_bw=model.sm_bw, hop_us=model.hop_us), memory=low.plan.stats)
