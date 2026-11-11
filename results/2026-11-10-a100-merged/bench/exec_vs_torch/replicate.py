#!/usr/bin/env python3
"""Bundle-level "vmap for the VM": R independent copies of one compiled bundle in a single bundle (ABI v2).

  python3 replicate.py SRC_BUNDLE OUT_BUNDLE R [--seed-stride S]

Copy k gets its own buffers (indices shifted by k * nbufs, names "r<k>/<name>"), its own version variables, and its own tasks
(successor and argument offsets shifted); the copies share no edge, so the executor may interleave them freely. With
--seed-stride S, seeded buffers of copy k use seed + k * S (a seed sweep); otherwise every copy is bitwise the same run, which is
what the correctness gate uses (each copy's final buffers must equal the single-run result byte for byte).
Pure host code, no GPU. Nothing in the repo's frontend/compiler/executor is touched.
"""
import argparse, copy, json, shutil, struct, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'runtime'))
import abi  # noqa: E402

NT_WORDS = 16


def replicate(src: Path, dst: Path, copies: int, seed_stride: int = 0) -> dict:
    meta = json.loads((src / 'metadata.json').read_text())
    ops = {o.id: o for o in abi.load_ops().values()}
    nt, ne, nw = meta['task_count'], meta['edge_count'], meta['args_words']
    tasks = list(struct.unpack(f'<{nt * NT_WORDS}i', (src / 'tasks.bin').read_bytes()))
    succ = list(struct.unpack(f'<{ne}i', (src / 'successors.bin').read_bytes())) if ne else []
    args = list(struct.unpack(f'<{nw}i', (src / 'args.bin').read_bytes())) if nw else []
    bufs, vars_ = meta['buffers'], meta.get('variables', [])
    nb, nv = len(bufs), len(vars_)
    if nv * copies > 64:
        raise SystemExit(f'{nv} variables x {copies} copies exceeds the executor default of 64 version variables')
    # which argument words are buffer indices (op registry), per task
    buf_words = []   # absolute positions in args[] that hold a buffer index
    for i in range(nt):
        op = ops[tasks[i * NT_WORDS + 0]]
        ao = tasks[i * NT_WORDS + 6]
        for j, a in enumerate(op.args):
            if a.type == 'buf':
                buf_words.append(ao + j)
    out_t, out_s, out_a = [], [], []
    for k in range(copies):
        for i in range(nt):
            w = tasks[i * NT_WORDS:(i + 1) * NT_WORDS]
            w[1] += k * ne      # succ_off
            w[6] += k * nw      # args_off
            for f in (8, 10):   # wait_var, pub_var
                if w[f] >= 0:
                    w[f] += k * nv
            out_t.extend(w)
        out_s.extend(s + k * nt for s in succ)
        a = list(args)
        for p in buf_words:
            if a[p] >= 0:
                a[p] += k * nb
        out_a.extend(a)
    m = copy.deepcopy(meta)
    m['task_count'], m['edge_count'], m['args_words'] = nt * copies, ne * copies, nw * copies
    m['task_names'] = [f'r{k}/{n}' for k in range(copies) for n in meta.get('task_names', [])]
    m['initial_ready'] = [k * nt + i for k in range(copies) for i in meta.get('initial_ready', [])]
    m['buffers'], m['variables'] = [], []
    for k in range(copies):
        for b in bufs:
            c = copy.deepcopy(b)
            c['name'], c['index'] = f'r{k}/{b["name"]}', b['index'] + k * nb
            if seed_stride and c.get('init', {}).get('kind') == 'seeded':
                c['init']['seed'] = str(int(c['init'].get('seed', 0)) + k * seed_stride)
            m['buffers'].append(c)
        for v in vars_:
            c = copy.deepcopy(v); c['name'], c['id'] = f'r{k}/{v["name"]}', v['id'] + k * nv
            m['variables'].append(c)
    for key in ('publishers', 'reads', 'final_versions'):   # not used by the device or by --dump-buffers comparisons
        m.pop(key, None)
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.iterdir():   # data files for file-initialised buffers
        if p.is_file() and p.name not in ('metadata.json', 'tasks.bin', 'successors.bin', 'args.bin', 'golden.json'):
            shutil.copy2(p, dst / p.name)
        elif p.is_dir():
            shutil.copytree(p, dst / p.name, dirs_exist_ok=True)
    (dst / 'tasks.bin').write_bytes(struct.pack(f'<{len(out_t)}i', *out_t))
    (dst / 'successors.bin').write_bytes(struct.pack(f'<{len(out_s)}i', *out_s))
    (dst / 'args.bin').write_bytes(struct.pack(f'<{len(out_a)}i', *out_a))
    (dst / 'metadata.json').write_text(json.dumps(m))
    return m


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('src'); ap.add_argument('dst'); ap.add_argument('copies', type=int)
    ap.add_argument('--seed-stride', type=int, default=0)
    a = ap.parse_args()
    m = replicate(Path(a.src), Path(a.dst), a.copies, a.seed_stride)
    print(f'{a.copies} copies: {m["task_count"]} tasks, {m["edge_count"]} edges, {len(m["buffers"])} buffers')
