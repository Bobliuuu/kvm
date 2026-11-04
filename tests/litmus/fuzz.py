#!/usr/bin/env python3
"""Corpus generator + fuzz driver for the kvm_exec CLI.

Goal: a corrupted bundle (tasks.bin / successors.bin / args.bin / metadata.json / golden.json) or a hostile CLI flag must be
REJECTED CLEANLY: exit code 0 (accepted, ran) or 1 (rejected with a message) or 2 (usage), no signal, no hang, and the GPU must
still run a known-good bundle afterwards. Anything else is a finding.

    python3 tests/litmus/fuzz.py --exe ./kvm_exec --out fuzz_out [--seed 1] [--cases 400] [--timeout 40]

Seeds come from runtime/compile_workload.py (examples + one generated workload with versions, matmul, spin).
Classes: OK (exit 0), REJECT (exit 1, message), USAGE (2), SLOW_REJECT (device TIMEOUT: rejected, but only after the timeout),
CRASH (signal / abnormal exit), HANG (killed by the driver timeout). CRASH and HANG are findings; so is a post-case sanity failure.
Run it under the GPU lock; it never needs the network.
"""
import argparse, json, os, random, shutil, struct, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INTS = [0, 1, 2, -1, -2, 7, 8, 63, 64, 255, 256, 65535, 65536, 1 << 20, 1 << 24, (1 << 30), 2**31 - 1, 2**31 - 2, -2**31, -2**31 + 1]
FIELDS = ['op_id', 'succ_off', 'succ_cnt', 'indegree', 'priority', 'deadline_us', 'args_off', 'args_words', 'wait_var', 'wait_ver',
          'pub_var', 'pub_ver', 'workload_id', 'flags', 'reserved0', 'reserved1']


def generated_workload():
    """Buffers 8x8 + scalars, a versioned weight, readers, matmul, spin: touches every executor path."""
    bufs = [{'name': n, 'rows': 8, 'cols': 8, 'init': {'kind': 'fill', 'value': v}} for n, v in
            (('a', 1.0), ('b', 2.0), ('c', 0.0), ('d', 0.0), ('e', 0.0))]
    bufs += [{'name': 'w', 'init': {'kind': 'fill', 'value': 0.5}}, {'name': 'g', 'init': {'kind': 'fill', 'value': 0.25}},
             {'name': 'snap'}]
    t = [
        {'id': 'f', 'op': 'fill', 'args': {'x': 'c', 'value': 3.0, 'n': 64}},
        {'id': 'add', 'op': 'add', 'after': ['f'], 'args': {'a': 'a', 'b': 'c', 'out': 'd', 'n': 64}},
        {'id': 'mm', 'op': 'matmul', 'after': ['add'], 'args': {'a': 'd', 'b': 'b', 'out': 'e', 'm': 8, 'n': 8, 'k': 8}},
        {'id': 'rl', 'op': 'relu', 'after': ['mm'], 'args': {'x': 'e', 'out': 'c', 'n': 64}},
        {'id': 'sp', 'op': 'spin', 'priority': 3, 'args': {'us': 50}},
        {'id': 'u1', 'op': 'sgd', 'after': ['sp'], 'publishes': 'wv', 'args': {'w': 'w', 'g': 'g', 'lr': 0.5, 'n': 1}},
        {'id': 'u2', 'op': 'sgd', 'after': ['u1'], 'publishes': 'wv', 'args': {'w': 'w', 'g': 'g', 'lr': 0.5, 'n': 1}},
        {'id': 'rd', 'op': 'copy', 'priority': 9, 'reads': [{'var': 'wv', 'policy': 'exact', 'version': 1}],
         'args': {'src': 'w', 'dst': 'snap', 'n': 1}},
    ]
    return {'variables': ['wv'], 'buffers': bufs, 'graphs': [{'name': 'g', 'tasks': t}]}


def compile_seed(workload_json, dest):
    dest.mkdir(parents=True, exist_ok=True)
    src = dest / 'workload.json'
    src.write_text(json.dumps(workload_json))
    subprocess.run([sys.executable, str(ROOT / 'runtime' / 'compile_workload.py'), str(src), str(dest / 'bundle'), '--emit-golden'],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return dest / 'bundle'


def run(exe, bundle, extra, timeout, golden=False, cli_timeout=3):
    cmd = [exe, '--timeout-s', str(cli_timeout), '--max-tasks', '4096', '--live-cap', '256', '--ring-words', '4096'] + extra
    if golden:
        cmd.append('--golden')
    cmd.append(str(bundle))
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=timeout, text=True, errors='replace')
        dt = time.time() - t0
        err = (p.stderr or '') + (p.stdout or '')
        if p.returncode < 0:
            return 'CRASH', f'signal {-p.returncode}', dt, err
        if p.returncode == 0:
            return 'OK', '', dt, err
        if p.returncode == 2:
            return 'USAGE', '', dt, err
        if p.returncode == 1:
            return ('SLOW_REJECT' if 'TIMEOUT' in err else 'REJECT'), '', dt, err
        return 'CRASH', f'exit {p.returncode}', dt, err
    except subprocess.TimeoutExpired as e:
        return 'HANG', f'>{timeout}s', time.time() - t0, (e.stderr or b'').decode(errors='replace') if isinstance(e.stderr, bytes) else (e.stderr or '')


def words(b):
    return list(struct.unpack(f'<{len(b) // 4}i', b[: len(b) // 4 * 4]))


def pack(w):
    return struct.pack(f'<{len(w)}i', *[((x + 2**31) % 2**32) - 2**31 for x in w])


def mutate(rng, seed_dir, case_dir):
    """Copy the seed bundle and apply one random corruption. Returns (description, extra_cli_args, golden_flag)."""
    if case_dir.exists():
        shutil.rmtree(case_dir)
    shutil.copytree(seed_dir, case_dir)
    kind = rng.choice(['field', 'field', 'field', 'bitflip', 'succ', 'args', 'trunc', 'meta', 'metajson', 'golden', 'cli', 'swap'])
    tasks = words((case_dir / 'tasks.bin').read_bytes())
    nt = len(tasks) // 16
    extra, golden = [], False
    if kind == 'field':
        i, f, v = rng.randrange(nt), rng.randrange(16), rng.choice(INTS + [nt, nt + 1, nt - 1])
        tasks[i * 16 + f] = v
        (case_dir / 'tasks.bin').write_bytes(pack(tasks))
        return f'task[{i}].{FIELDS[f]}={v}', extra, golden
    if kind == 'bitflip':
        which = rng.choice(['tasks.bin', 'successors.bin', 'args.bin'])
        b = bytearray((case_dir / which).read_bytes())
        if not b:
            return f'{which} empty (skip)', extra, golden
        for _ in range(rng.choice([1, 1, 2, 8])):
            b[rng.randrange(len(b))] ^= 1 << rng.randrange(8)
        (case_dir / which).write_bytes(bytes(b))
        return f'bitflip {which}', extra, golden
    if kind == 'succ':
        w = words((case_dir / 'successors.bin').read_bytes())
        if w:
            k = rng.randrange(len(w)); w[k] = rng.choice(INTS + [nt, nt - 1, 0])
            (case_dir / 'successors.bin').write_bytes(pack(w))
            return f'succ[{k}]={w[k]}', extra, golden
        return 'succ empty (skip)', extra, golden
    if kind == 'args':
        w = words((case_dir / 'args.bin').read_bytes())
        if w:
            k = rng.randrange(len(w)); w[k] = rng.choice(INTS + [2_000_000_000, 7, 8, 9])
            (case_dir / 'args.bin').write_bytes(pack(w))
            return f'args[{k}]={w[k]}', extra, golden
        return 'args empty (skip)', extra, golden
    if kind == 'trunc':
        which = rng.choice(['tasks.bin', 'successors.bin', 'args.bin', 'metadata.json', 'golden.json'])
        b = (case_dir / which).read_bytes()
        n = rng.choice([0, 1, 3, len(b) // 2, max(0, len(b) - 1), max(0, len(b) - 4)])
        if rng.random() < 0.3:
            (case_dir / which).write_bytes(b + bytes(rng.choice([1, 4, 64])))
            return f'extend {which}', extra, True
        (case_dir / which).write_bytes(b[:n])
        return f'truncate {which} to {n}', extra, which == 'golden.json'
    if kind == 'swap':
        a, b = rng.randrange(nt), rng.randrange(nt)
        for f in range(16):
            tasks[a * 16 + f], tasks[b * 16 + f] = tasks[b * 16 + f], tasks[a * 16 + f]
        (case_dir / 'tasks.bin').write_bytes(pack(tasks))
        return f'swap task {a}<->{b} (indices in successors now stale)', extra, golden
    if kind == 'meta':
        m = json.loads((case_dir / 'metadata.json').read_text())
        key = rng.choice(['abi_version', 'task_count', 'edge_count', 'args_words', 'buffers', 'variables', 'reads'])
        choice = rng.choice(['missing', 'null', 'huge', 'neg', 'str', 'list', 'nan'])
        if choice == 'missing': m.pop(key, None)
        elif choice == 'null': m[key] = None
        elif choice == 'huge': m[key] = 1e30
        elif choice == 'neg': m[key] = -1
        elif choice == 'str': m[key] = 'x'
        elif choice == 'list': m[key] = []
        else: m[key] = float('nan')
        if m.get('buffers') and isinstance(m['buffers'], list) and rng.random() < 0.5 and isinstance(m['buffers'][0], dict):
            fld = rng.choice(['rows', 'cols', 'index', 'init', 'name'])
            m['buffers'][0][fld] = rng.choice([0, -1, 2**31, 2**31 - 1, 46341, 1e30, 'x', None, {'kind': 'values'}, {'kind': 'fill'}, {'kind': 'bogus'}])
        (case_dir / 'metadata.json').write_text(json.dumps(m))
        return f'metadata {key} -> {choice}', extra, True
    if kind == 'metajson':
        s = (case_dir / 'metadata.json').read_text()
        mode = rng.choice(['deep', 'garbage', 'unterminated', 'bom', 'dupkey', 'empty'])
        if mode == 'deep': s = '[' * 200000
        elif mode == 'garbage': s = s[: len(s) // 2] + '\x00\xff' + s[len(s) // 2:]
        elif mode == 'unterminated': s = s[:-3]
        elif mode == 'bom': s = '﻿' + s
        elif mode == 'dupkey': s = s.replace('"abi_version": 2', '"abi_version": 2, "abi_version": 3', 1)
        else: s = ''
        (case_dir / 'metadata.json').write_text(s)
        return f'metadata.json {mode}', extra, False
    if kind == 'golden':
        g = json.loads((case_dir / 'golden.json').read_text())
        mode = rng.choice(['shortbuf', 'unknownbuf', 'nullvals', 'badver', 'strvals', 'arrayroot', 'nokind'])
        if mode == 'shortbuf' and g.get('buffers'): k = next(iter(g['buffers'])); g['buffers'][k] = g['buffers'][k][:-1]
        elif mode == 'unknownbuf': g.setdefault('buffers', {})['nope'] = [1.0]
        elif mode == 'nullvals' and g.get('buffers'): k = next(iter(g['buffers'])); g['buffers'][k] = [None]
        elif mode == 'badver': g['versions'] = {'wv': 'x', 'zzz': 1}
        elif mode == 'strvals' and g.get('buffers'): k = next(iter(g['buffers'])); g['buffers'][k] = 'abc'
        elif mode == 'arrayroot': g = [1, 2]
        else: g['buffers'] = 5
        (case_dir / 'golden.json').write_text(json.dumps(g))
        m = json.loads((case_dir / 'metadata.json').read_text())
        if rng.random() < 0.5 and m.get('reads'):
            r = m['reads'][0]; r.pop(rng.choice(['task', 'policy', 'name', 'version']), None)
            if rng.random() < 0.5: r['task'] = rng.choice([-1, 10**6, 2**31, 'x'])
            (case_dir / 'metadata.json').write_text(json.dumps(m))
        return f'golden/reads {mode}', extra, True
    # cli
    flag = rng.choice(['--blocks', '--live-cap', '--max-tasks', '--ring-words', '--aging-us', '--timeout-s'])
    val = rng.choice(['0', '1', '2', '3', '-1', '-5', '7', '63', '64', '65', '1000', '4096', '2147483647', '2147483648', '4294967297', '99999999999', 'abc', ''])
    return f'cli {flag} {val}', [flag, val], False


def targeted_cases():
    """Deterministic probes, one suspected defect each: (id, description, mutator(case_dir, nt), golden, extra_cli)."""
    I32 = 2**31 - 1
    def setfield(i, f, v):
        def m(d, nt):
            w = words((d / 'tasks.bin').read_bytes()); w[i * 16 + FIELDS.index(f)] = v; (d / 'tasks.bin').write_bytes(pack(w))
        return m
    def meta(fn):
        def m(d, nt):
            j = json.loads((d / 'metadata.json').read_text()); fn(j); (d / 'metadata.json').write_text(json.dumps(j))
        return m
    def reads(fn):
        def m(d, nt):
            j = json.loads((d / 'metadata.json').read_text())
            j.setdefault('reads', [{'name': 'r', 'task': 0, 'policy': 'exact', 'version': 1}]); fn(j['reads']); (d / 'metadata.json').write_text(json.dumps(j))
        return m
    def spin_arg(v):
        def m(d, nt):
            tw = words((d / 'tasks.bin').read_bytes()); aw = words((d / 'args.bin').read_bytes())
            for i in range(nt):
                if tw[i * 16] == 10: aw[tw[i * 16 + 6]] = v
            (d / 'args.bin').write_bytes(pack(aw))
        return m
    def raw(name, text):
        def m(d, nt): (d / name).write_text(text)
        return m
    return [
        ('T01', 'task0.succ_off=INT_MAX, succ_cnt=1 (sum wraps negative, passes the range check)', lambda d, nt: (setfield(0, 'succ_off', I32)(d, nt), setfield(0, 'succ_cnt', 1)(d, nt)), False, []),
        ('T02', 'task0.succ_cnt=INT_MAX (succ_off+succ_cnt wraps)', setfield(0, 'succ_cnt', I32), False, []),
        ('T03', 'task0.args_off=INT_MAX (args_off+args_words wraps negative)', setfield(0, 'args_off', I32), False, []),
        ('T04', 'task0.args_off=INT_MAX-1', setfield(0, 'args_off', I32 - 1), False, []),
        ('T05', 'task0.succ_off=-1 succ_cnt=2', lambda d, nt: (setfield(0, 'succ_off', -1)(d, nt), setfield(0, 'succ_cnt', 2)(d, nt)), False, []),
        ('T06', 'reads[0] has no "policy" (golden)', reads(lambda r: r[0].pop('policy', None)), True, []),
        ('T07', 'reads[0] has no "name" (golden)', reads(lambda r: r[0].pop('name', None)), True, []),
        ('T08', 'reads[0].task=-1, no gate (golden)', reads(lambda r: r[0].update(task=-1)), True, []),
        ('T09', 'reads[0].task=1000000 (golden)', reads(lambda r: r[0].update(task=10**6)), True, []),
        ('T10', 'variables[0] has no "name" (golden)', meta(lambda j: j['variables'][0].pop('name', None)), True, []),
        ('T11', 'buffer init kind "values" without "values"', meta(lambda j: j['buffers'][0].__setitem__('init', {'kind': 'values'})), False, []),
        ('T12', 'metadata.json nested 200000 deep', raw('metadata.json', '[' * 200000), False, []),
        ('T13', 'spin us=100000000 (100 s task) with --timeout-s 3', spin_arg(100_000_000), False, []),
        ('T14', 'task0.wait_var=0 wait_ver=INT_MAX (never satisfied)', lambda d, nt: (setfield(0, 'wait_var', 0)(d, nt), setfield(0, 'wait_ver', I32)(d, nt)), False, []),
        ('T15', 'cli --timeout-s -1 (wraps to ~infinite) + wait_ver never satisfied', lambda d, nt: (setfield(0, 'wait_var', 0)(d, nt), setfield(0, 'wait_ver', I32)(d, nt)), False, ['--timeout-s', '-1']),
        ('T16', 'cli --timeout-s 0', lambda d, nt: None, False, ['--timeout-s', '0']),
        ('T17', 'cli --aging-us -2000000 (negative aging)', lambda d, nt: None, False, ['--aging-us', '-2000000']),
        ('T18', 'cli --blocks 100000', lambda d, nt: None, False, ['--blocks', '100000']),
        ('T19', 'cli --max-tasks 536870913 (8*max_tasks wraps the int pool caps)', lambda d, nt: None, False, ['--max-tasks', '536870913']),
        ('T20', 'cli --live-cap 1073741824', lambda d, nt: None, False, ['--live-cap', '1073741824']),
        ('T21', 'buffers[0].rows=46341 cols=46341 (8.6 GB buffer)', meta(lambda j: j['buffers'][0].update(rows=46341, cols=46341)), False, []),
        ('T22', 'task_count in metadata larger than file', meta(lambda j: j.update(task_count=j['task_count'] + 1)), False, []),
    ]


def sanity(exe, good, timeout):
    st, why, dt, err = run(exe, good, [], timeout, golden=True)
    return st == 'OK', err


def run_targeted(a, out, seeds):
    findings = []
    for cid, desc, mut, golden, extra in targeted_cases():
        cdir = out / f'case_{cid}'
        if cdir.exists(): shutil.rmtree(cdir)
        shutil.copytree(seeds['gen'], cdir)
        nt = len(words((cdir / 'tasks.bin').read_bytes())) // 16
        mut(cdir, nt)
        st, why, dt, err = run(a.exe, cdir, extra, a.timeout, golden=golden)
        last = [l for l in err.strip().splitlines() if l][-1:] or ['']
        print(f'{cid} {st:11s} {why:11s} {dt:5.1f}s  {desc}\n      -> {last[0][:160]}', flush=True)
        if st in ('CRASH', 'HANG'): findings.append(cid)
        ok, msg = sanity(a.exe, seeds['gen'], a.timeout)
        if not ok: print(f'      SANITY FAILED afterwards: {msg[-200:]}', flush=True); findings.append(cid + '+sanity')
    print(f'targeted findings: {findings}')
    return 1 if findings else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exe', required=True)
    ap.add_argument('--out', default='fuzz_out')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--cases', type=int, default=400)
    ap.add_argument('--timeout', type=int, default=40, help='driver kill timeout per case (s)')
    ap.add_argument('--sanity-every', type=int, default=25)
    ap.add_argument('--targeted', action='store_true', help='run the deterministic probes instead of random mutation')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    seeds = {
        'live': compile_seed(json.loads((ROOT / 'examples' / 'live_training_inference.json').read_text()), out / 'seed_live'),
        'gen': compile_seed(generated_workload(), out / 'seed_gen'),
    }
    ok, msg = sanity(a.exe, seeds['gen'], a.timeout)
    print(f'baseline gen seed: {"OK" if ok else "FAILED: " + msg[-300:]}')
    ok2, msg2 = sanity(a.exe, seeds['live'], a.timeout)
    print(f'baseline live seed: {"OK" if ok2 else "FAILED: " + msg2[-300:]}')
    if a.targeted:
        return run_targeted(a, out, seeds)
    rng = random.Random(a.seed)
    tally, findings = {}, []
    t_start = time.time()
    for n in range(a.cases):
        name = rng.choice(list(seeds))
        cdir = out / 'case'
        desc, extra, golden = mutate(rng, seeds[name], cdir)
        st, why, dt, err = run(a.exe, cdir, extra, a.timeout, golden=golden)
        tally[st] = tally.get(st, 0) + 1
        if st in ('CRASH', 'HANG'):
            keep = out / f'FIND_{n:04d}_{st}'
            shutil.copytree(cdir, keep, dirs_exist_ok=True)
            (keep / 'DESC.txt').write_text(f'{desc}\nseed={name} extra={extra} golden={golden}\n{st} {why}\n--- output tail ---\n{err[-1500:]}\n')
            findings.append((n, st, why, name, desc))
            print(f'[{n:4d}] {st:11s} {why:12s} {name}: {desc}', flush=True)
        elif st == 'SLOW_REJECT':
            print(f'[{n:4d}] SLOW_REJECT {dt:5.1f}s {name}: {desc}', flush=True)
        if (n + 1) % a.sanity_every == 0 or st in ('CRASH', 'HANG'):
            ok, msg = sanity(a.exe, seeds['gen'], a.timeout)
            if not ok:
                print(f'[{n:4d}] SANITY FAILED after case ({desc}): {msg[-300:]}', flush=True)
                findings.append((n, 'SANITY', '', name, desc))
    print(f'\n{a.cases} cases in {time.time() - t_start:.0f}s: {tally}')
    print(f'findings (CRASH/HANG/SANITY): {len(findings)}')
    for f in findings:
        print('  ', f)
    return 1 if findings else 0


if __name__ == '__main__':
    sys.exit(main())
