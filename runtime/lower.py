"""Compile monotonic counter waits/signals into an explicit task DAG.

Input is a list of instantiated instructions, each with waits [(counter, threshold)]
and signals [counter]. A wait at threshold t requires the first t signal events
on that counter to have completed. Duplicate signals are counted independently.
The resulting DAG is a conservative partial order: the first t producers must
finish before the consumer. This is correct only if producers are meant to be
ordered by the supplied logical event order (e.g. instruction-tape order).
"""
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Sequence

@dataclass(frozen=True)
class Task:
    waits: tuple[tuple[int, int], ...] = ()
    signals: tuple[int, ...] = ()
    name: str = ""

@dataclass
class Graph:
    successors: list[list[int]]
    indegree: list[int]
    edges: int

def lower(tasks: Sequence[Task]) -> Graph:
    producers = defaultdict(list)
    for i, task in enumerate(tasks):
        for counter in task.signals:
            if counter < 0:
                raise ValueError("negative signal counter")
            producers[counter].append(i)

    successors = [[] for _ in tasks]
    indegree = [0] * len(tasks)
    edges = set()
    for consumer, task in enumerate(tasks):
        for counter, threshold in task.waits:
            if counter < 0 or threshold < 0:
                raise ValueError("negative wait counter or threshold")
            if threshold == 0:
                continue
            source = producers.get(counter, [])
            if threshold > len(source):
                raise ValueError(f"counter {counter} needs {threshold} signals; only {len(source)} available")
            for producer in source[:threshold]:
                if producer == consumer:
                    raise ValueError("self-dependency")
                edges.add((producer, consumer))
    for a, b in sorted(edges):
        successors[a].append(b)
        indegree[b] += 1

    # Reject cyclic counter dependencies rather than hanging the GPU.
    remaining = indegree.copy()
    ready = [i for i, d in enumerate(remaining) if d == 0]
    count = 0
    while ready:
        i = ready.pop()
        count += 1
        for j in successors[i]:
            remaining[j] -= 1
            if remaining[j] == 0:
                ready.append(j)
    if count != len(tasks):
        raise ValueError("cyclic counter dependencies")
    return Graph(successors, indegree, len(edges))

def validate_schedule(graph: Graph, order: Sequence[int]) -> None:
    if sorted(order) != list(range(len(graph.indegree))):
        raise ValueError("schedule must execute every task exactly once")
    position = {task: i for i, task in enumerate(order)}
    for a, targets in enumerate(graph.successors):
        for b in targets:
            if position[a] >= position[b]:
                raise ValueError(f"dependency violated: {a} -> {b}")


def graph_to_csr(graph: Graph) -> tuple[list[int], list[int]]:
    """GPU-friendly compressed sparse row successor representation."""
    offsets = [0]
    flat = []
    for successors in graph.successors:
        flat.extend(successors)
        offsets.append(len(flat))
    return offsets, flat

def lower_json(input_path: str, output_path: str) -> None:
    """Convert an instantiated counter-tape JSON into a GPU-ready DAG JSON.

    Format: {"tasks":[{"waits":[[counter,threshold]],"signals":[counter]}]}.
    This deliberately requires an instantiated tape, not a step template.
    """
    import json
    from pathlib import Path
    payload = json.loads(Path(input_path).read_text())
    tasks = [Task(tuple(tuple(w) for w in t.get("waits", [])),
                  tuple(t.get("signals", [])), t.get("name",""))
             for t in payload["tasks"]]
    graph = lower(tasks)
    offsets, successors = graph_to_csr(graph)
    Path(output_path).write_text(json.dumps({
        "task_count":len(tasks),
        "edge_count":graph.edges,
        "indegree":graph.indegree,
        "successor_offsets":offsets,
        "successors":successors,
        "initial_ready":[i for i,d in enumerate(graph.indegree) if d==0]
    }, indent=2) + "\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Lower instantiated counter tape to CSR DAG")
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()
    lower_json(args.input, args.output)
