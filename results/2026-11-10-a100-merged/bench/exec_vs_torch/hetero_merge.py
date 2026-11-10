#!/usr/bin/env python3
"""Merge R DIFFERENT compiled bundles into one bundle (ABI v2): the panel's hetero row, R independent runs that do not share a shape.

  python3 hetero_merge.py OUT_BUNDLE SRC_BUNDLE [SRC_BUNDLE ...]

Run k keeps its own buffers (indices shifted by the earlier runs' counts, names "r<k>/<name>"), its own version variables and its own
tasks (successor and argument offsets shifted). No edge joins two runs, so the executor may interleave them freely. replicate.py does the
same for R copies of one bundle; a merge of R identical sources equals it byte for byte in tasks, successors and arguments.
Pure host code, no GPU.
"""
import copy
import json
import shutil
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'runtime'))
import abi  # noqa: E402

NT_WORDS = 16
MAX_VARIABLES = 64      # the executor's default number of version variables


class _Bundle:
    """One source bundle: its metadata and its three int32 sections, plus which argument words hold buffer indices."""

    def __init__(self, path: Path, ops: dict):
        self.path = path
        self.meta = json.loads((path / 'metadata.json').read_text())
        self.nt, self.ne, self.nw = self.meta['task_count'], self.meta['edge_count'], self.meta['args_words']
        self.nb, self.nv = len(self.meta['buffers']), len(self.meta.get('variables', []))
        self.tasks = list(struct.unpack(f'<{self.nt * NT_WORDS}i', (path / 'tasks.bin').read_bytes()))
        self.succ = list(struct.unpack(f'<{self.ne}i', (path / 'successors.bin').read_bytes())) if self.ne else []
        self.args = list(struct.unpack(f'<{self.nw}i', (path / 'args.bin').read_bytes())) if self.nw else []
        self.buf_words = []             # positions in self.args that hold a buffer index
        for i in range(self.nt):
            op = ops[self.tasks[i * NT_WORDS + 0]]
            arg_off = self.tasks[i * NT_WORDS + 6]
            self.buf_words += [arg_off + j for j, a in enumerate(op.args) if a.type == 'buf']


def merge(sources: list[Path], dst: Path) -> dict:
    """Write the merged bundle at dst and return its metadata. Refuses sources that disagree on the operator table or the variable limit."""
    if not sources:
        raise ValueError('nothing to merge')
    ops = {o.id: o for o in abi.load_ops().values()}
    bundles = [_Bundle(p, ops) for p in sources]
    for b in bundles[1:]:
        if b.meta.get('operator_ids') != bundles[0].meta.get('operator_ids'):
            raise SystemExit(f'{b.path}: operator table differs from {bundles[0].path}; bundles from one tree only')
    total_vars = sum(b.nv for b in bundles)
    if total_vars > MAX_VARIABLES:
        raise SystemExit(f'{total_vars} variables exceed the executor default of {MAX_VARIABLES}')

    out_t, out_s, out_a, names, ready, buffers, variables = [], [], [], [], [], [], []
    task_off = edge_off = arg_off = buf_off = var_off = 0
    dst.mkdir(parents=True, exist_ok=True)
    for k, b in enumerate(bundles):
        for i in range(b.nt):
            w = b.tasks[i * NT_WORDS:(i + 1) * NT_WORDS]
            w[1] += edge_off    # succ_off
            w[6] += arg_off     # args_off
            for field in (8, 10):   # wait_var, pub_var
                if w[field] >= 0:
                    w[field] += var_off
            out_t.extend(w)
        out_s.extend(s + task_off for s in b.succ)
        args = list(b.args)
        for p in b.buf_words:
            if args[p] >= 0:
                args[p] += buf_off
        out_a.extend(args)
        names += [f'r{k}/{n}' for n in b.meta.get('task_names', [])]
        ready += [task_off + i for i in b.meta.get('initial_ready', [])]
        for src_buffer in b.meta['buffers']:
            c = copy.deepcopy(src_buffer)
            c['name'], c['index'] = f'r{k}/{src_buffer["name"]}', src_buffer['index'] + buf_off
            init = c.get('init') or {}
            if init.get('kind') == 'file':      # data files are per run: they keep their own folder so two runs cannot collide
                shutil.copy2(b.path / init['path'], _ensure_parent(dst / f'r{k}' / init['path']))
                init['path'] = f'r{k}/{init["path"]}'
            buffers.append(c)
        for v in b.meta.get('variables', []):
            c = copy.deepcopy(v)
            c['name'], c['id'] = f'r{k}/{v["name"]}', v['id'] + var_off
            variables.append(c)
        task_off += b.nt; edge_off += b.ne; arg_off += b.nw; buf_off += b.nb; var_off += b.nv

    merged = copy.deepcopy(bundles[0].meta)
    merged['task_count'], merged['edge_count'], merged['args_words'] = task_off, edge_off, arg_off
    merged['task_names'], merged['initial_ready'], merged['buffers'], merged['variables'] = names, ready, buffers, variables
    for key in ('publishers', 'reads', 'final_versions'):       # not used by the device or by --dump-buffers comparisons (replicate.py drops them too)
        merged.pop(key, None)
    (dst / 'tasks.bin').write_bytes(struct.pack(f'<{len(out_t)}i', *out_t))
    (dst / 'successors.bin').write_bytes(struct.pack(f'<{len(out_s)}i', *out_s))
    (dst / 'args.bin').write_bytes(struct.pack(f'<{len(out_a)}i', *out_a))
    (dst / 'metadata.json').write_text(json.dumps(merged))
    return merged


def _ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    meta = merge([Path(p) for p in sys.argv[2:]], Path(sys.argv[1]))
    print(f'merged {len(sys.argv) - 2} bundles: {meta["task_count"]} tasks, {meta["edge_count"]} edges, {len(meta["buffers"])} buffers')
