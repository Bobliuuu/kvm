"""Compile declarative task DAGs to deterministic, packed GPU-facing descriptors.

This is a host compiler, not a CUDA execution engine. Operators are supplied
by an external manifest: no game/model-specific logic lives in this compiler.
"""
import argparse
import json
import struct
from dataclasses import dataclass
from pathlib import Path

# ABI: op_id, first_successor, successor_count, initial_indegree,
#      priority, deadline_us, payload_offset, payload_length (8 x int32).
DESC = struct.Struct('<8i')

class CompileError(ValueError):
    pass


def compile_spec(spec, manifest):
    """Return (metadata, descriptor bytes, successor bytes, payload bytes)."""
    if not isinstance(manifest, dict) or not manifest:
        raise CompileError('operator manifest must be nonempty')
    ids = list(manifest.values())
    if any(type(x) is not int or x < 0 or x > 2**31-1 for x in ids) or len(ids) != len(set(ids)):
        raise CompileError('operator ids must be unique nonnegative int32')
    graphs = spec.get('graphs', [])
    tasks = []
    seen = set()
    for graph in graphs:
        name = graph.get('name')
        if not isinstance(name, str) or not name:
            raise CompileError('invalid graph name')
        for task in graph.get('tasks', []):
            key = (name, task['id'])
            if key in seen:
                raise CompileError(f'duplicate task {key}')
            seen.add(key)
            tasks.append((name, task))
    if not tasks:
        raise CompileError('empty workload')
    index = {(name, t['id']): i for i, (name, t) in enumerate(tasks)}
    edges = [set() for _ in tasks]
    indegree = [0] * len(tasks)
    for i, (name, t) in enumerate(tasks):
        if t['op'] not in manifest:
            raise CompileError(f'unregistered operator {t["op"]}')
        deps = t.get('after', [])
        if len(deps) != len(set(deps)):
            raise CompileError('duplicate dependency')
        for dep in deps:
            j = index.get((name, dep))
            if j is None:
                raise CompileError(f'unknown dependency {name}/{dep}')
            edges[j].add(i)
            indegree[i] += 1
    remaining = indegree.copy()
    ready = [i for i, d in enumerate(remaining) if d == 0]
    visited = 0
    while ready:
        i = ready.pop()
        visited += 1
        for j in edges[i]:
            remaining[j] -= 1
            if remaining[j] == 0:
                ready.append(j)
    if visited != len(tasks):
        raise CompileError('dependency cycle')
    # No implicit cross-graph ordering: caller must declare synchronization.
    # All payloads are opaque JSON; the CUDA operator registry owns decoding.
    payload = bytearray()
    descriptors = bytearray()
    successors = []
    names = []
    for i, (name, t) in enumerate(tasks):
        op_args = t.get('args', {})
        if not isinstance(op_args, dict):
            raise CompileError('operator args must be a dictionary')
        encoded = json.dumps(op_args, sort_keys=True, separators=(',', ':')).encode()
        if len(payload) + len(encoded) > 2**31-1:
            raise CompileError('payload too large')
        offset = len(payload)
        payload.extend(encoded)
        priority = t.get('priority', 0)
        if type(priority) is not int or not -2**31 <= priority < 2**31:
            raise CompileError('invalid priority')
        deadline = t.get('deadline_us', -1)
        if type(deadline) is not int or not -1 <= deadline < 2**31:
            raise CompileError('deadline_us must be int32 microseconds or -1')
        targets = sorted(edges[i])
        descriptors.extend(DESC.pack(manifest[t['op']], len(successors), len(targets),
                                     indegree[i], priority, deadline, offset, len(encoded)))
        successors.extend(targets)
        names.append(f'{name}/{t["id"]}')
    succ_bytes = struct.pack(f'<{len(successors)}i', *successors)
    metadata = {'abi_version': 1, 'descriptor_bytes': DESC.size,
                'task_count': len(tasks), 'edge_count': len(successors),
                'task_names': names, 'initial_ready': [i for i, d in enumerate(indegree) if d == 0],
                'operator_ids': manifest}
    return metadata, bytes(descriptors), succ_bytes, bytes(payload)


def write_compiled(spec, manifest, output_dir):
    metadata, descriptors, successors, payload = compile_spec(spec, manifest)
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (dest / 'tasks.bin').write_bytes(descriptors)
    (dest / 'successors.bin').write_bytes(successors)
    (dest / 'payload.bin').write_bytes(payload)
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('workload')
    parser.add_argument('operator_manifest')
    parser.add_argument('output_dir')
    args = parser.parse_args()
    result = write_compiled(json.loads(Path(args.workload).read_text()),
                            json.loads(Path(args.operator_manifest).read_text()), args.output_dir)
    print(json.dumps({'tasks': result['task_count'], 'edges': result['edge_count']}))
