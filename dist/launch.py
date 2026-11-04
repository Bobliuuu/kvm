#!/usr/bin/env python3
"""Launch one command per rank across nodes with plain ssh + environment variables. No MPI, no scheduler, no daemon.

    dist/launch.py --nodes host1:2,host2:2 [--port-base 29500] [--timeout 600] [--rsync SRC DEST] [--setup CMD] -- CMD ARG...
    KVM_NODES=host1,host2 dist/launch.py -- ./test_dist plan plan_dir --rank {rank} --peers {peers}
    dist/launch.py --local 2 -- CMD ...            # ranks as local subprocesses (tests, single box)
    dist/launch.py --dry-run ...                   # print every command, run nothing

Nodes are `[user@]host[:slots]` (slots = ranks on that node, default 1); `localhost` runs without ssh. Hosts come from the
command line or $KVM_NODES, never from a file in the repository. Each rank gets
    KVM_RANK, KVM_WORLD, KVM_LOCAL_RANK, KVM_NODE, KVM_PORT (= port_base + rank), KVM_PEERS ("host:port,host:port,...")
and `{rank} {world} {local_rank} {peers} {port}` in the command are substituted. Rank order = node order, then slot order.

Output is prefixed [rank N]. The launcher waits for every rank, kills all of them on the first failure or on --timeout (remote
processes run under `timeout` and ssh -tt so that a dead launcher does not leave them running), prints one PASS/FAIL line per
rank and exits non-zero if any rank failed. Logging is verbose by default; --quiet suppresses the per-line output.
"""
import argparse
import os
import shlex
import subprocess
import sys
import threading
import time

LOCAL_NAMES = ('localhost', '127.0.0.1', 'local')


def log(msg):
    print(f'[launch] {msg}', file=sys.stderr, flush=True)


def qdir(path):
    """shlex.quote but keeps a leading ~/ expandable by the remote shell."""
    path = path or '~'
    return '~/' + shlex.quote(path[2:]) if path.startswith('~/') else (path if path == '~' else shlex.quote(path))


def parse_nodes(text):
    nodes = []
    for item in filter(None, (x.strip() for x in text.split(','))):
        host, slots = item, 1
        head, sep, tail = item.rpartition(':')
        if sep and tail.isdigit():
            host, slots = head, int(tail)
        if slots < 1 or not host:
            raise SystemExit(f'bad node spec {item!r}; expected [user@]host[:slots]')
        nodes.append((host, slots))
    if not nodes:
        raise SystemExit('no nodes: pass --nodes or set KVM_NODES')
    return nodes


def hostname_of(host):
    return host.split('@')[-1]


def layout(nodes, port_base):
    ranks = []
    for ni, (host, slots) in enumerate(nodes):
        for s in range(slots):
            ranks.append(dict(rank=len(ranks), node=ni, local=s, host=host, port=port_base + len(ranks)))
    peers = ','.join(f"{'127.0.0.1' if hostname_of(r['host']) in LOCAL_NAMES else hostname_of(r['host'])}:{r['port']}" for r in ranks)
    return ranks, peers


def rank_env(r, world, peers):
    return {'KVM_RANK': str(r['rank']), 'KVM_WORLD': str(world), 'KVM_LOCAL_RANK': str(r['local']), 'KVM_NODE': str(r['node']),
            'KVM_PORT': str(r['port']), 'KVM_PEERS': peers}


def subst(arg, r, world, peers):
    return (arg.replace('{rank}', str(r['rank'])).replace('{world}', str(world)).replace('{local_rank}', str(r['local']))
               .replace('{peers}', peers).replace('{port}', str(r['port'])))


SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=accept-new', '-o', 'ServerAliveInterval=10', '-tt']


def build_cmd(r, world, peers, cmd, workdir, timeout, extra_env):
    env = dict(rank_env(r, world, peers), **extra_env)
    argv = [subst(a, r, world, peers) for a in cmd]
    line = shlex.join(argv)
    envs = ' '.join(f'{k}={shlex.quote(v)}' for k, v in env.items())
    if hostname_of(r['host']) in LOCAL_NAMES:
        return ['bash', '-c', f"cd {qdir(workdir or '.')} && exec env {envs} timeout {timeout} {line}"], env
    remote = f"cd {qdir(workdir or '~')} && exec env {envs} timeout {timeout} {line}"
    return SSH + [r['host'], remote], env


def pump(proc, prefix, quiet, sink):
    for line in iter(proc.stdout.readline, b''):
        text = line.decode(errors='replace').rstrip('\r\n')
        sink.append(text)
        if not quiet:
            print(f'{prefix} {text}', flush=True)


def run_all(ranks, world, peers, cmd, args):
    procs, threads, outputs = {}, [], {}
    t0 = time.monotonic()
    for r in ranks:
        argv, _ = build_cmd(r, world, peers, cmd, args.workdir, args.timeout, dict(kv.split('=', 1) for kv in args.env))
        if args.dry_run:
            print(f"DRY rank {r['rank']} @ {r['host']}: {shlex.join(argv)}")
            continue
        log(f"rank {r['rank']}/{world} on {hostname_of(r['host'])} (slot {r['local']}, port {r['port']})")
        p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        procs[r['rank']], outputs[r['rank']] = p, []
        th = threading.Thread(target=pump, args=(p, f"[rank {r['rank']}]", args.quiet, outputs[r['rank']]), daemon=True)
        th.start(); threads.append(th)
    if args.dry_run:
        return {}
    codes, failed = {}, False
    deadline = t0 + args.timeout + 30
    while len(codes) < len(procs):
        for rk, p in procs.items():
            if rk not in codes and p.poll() is not None:
                codes[rk] = p.returncode
                if p.returncode != 0 and not failed:
                    failed = True
                    log(f'rank {rk} failed with exit code {p.returncode}: stopping the others')
        if failed or time.monotonic() > deadline:
            if not failed:
                log(f'timeout after {args.timeout}s: stopping all ranks')
            for rk, p in procs.items():
                if rk not in codes:
                    p.terminate()
            time.sleep(1)
            for rk, p in procs.items():
                if rk not in codes:
                    if p.poll() is None:
                        p.kill()
                    codes[rk] = p.wait() if p.returncode is None else p.returncode
                    codes[rk] = codes[rk] if codes[rk] else 124
            break
        time.sleep(0.05)
    for th in threads:
        th.join(timeout=2)
    return codes


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument('--nodes', default=os.environ.get('KVM_NODES', ''))
    ap.add_argument('--local', type=int, help='N ranks as local subprocesses')
    ap.add_argument('--port-base', type=int, default=int(os.environ.get('KVM_PORT_BASE', '29500')))
    ap.add_argument('--workdir', default='', help='directory on every node (default: $HOME remotely, cwd locally)')
    ap.add_argument('--timeout', type=int, default=600)
    ap.add_argument('--env', action='append', default=[], metavar='K=V')
    ap.add_argument('--rsync', nargs=2, metavar=('SRC', 'DEST'), action='append', default=[], help='copy SRC to DEST on every remote node first')
    ap.add_argument('--setup', action='append', default=[], help='shell command run once on every node (after --rsync) in --workdir')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--quiet', action='store_true')
    ap.add_argument('cmd', nargs=argparse.REMAINDER)
    args = ap.parse_args(argv)
    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == '--' else args.cmd
    if not cmd:
        ap.error('no command (put it after --)')
    nodes = [('localhost', args.local)] if args.local else parse_nodes(args.nodes)
    ranks, peers = layout(nodes, args.port_base)
    world = len(ranks)
    log(f'world {world}: ' + ', '.join(f'{hostname_of(h)}x{s}' for h, s in nodes) + f'; peers {peers}')
    remote = [h for h, _ in nodes if hostname_of(h) not in LOCAL_NAMES]
    for src, dest in args.rsync:
        for h in remote:
            line = ['rsync', '-az', '-e', 'ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new', src, f'{h}:{dest}']
            if args.dry_run:
                print('DRY ' + shlex.join(line))
            else:
                log(f'rsync {src} -> {hostname_of(h)}:{dest}')
                subprocess.run(line, check=True, timeout=args.timeout)
    for setup in args.setup:
        for h, _ in nodes:
            local = hostname_of(h) in LOCAL_NAMES
            line = ['bash', '-c', f"cd {qdir(args.workdir or '.')} && {setup}"] if local else SSH[:-1] + [h, f"cd {qdir(args.workdir or '~')} && {setup}"]
            if args.dry_run:
                print('DRY ' + shlex.join(line))
            else:
                log(f'setup on {hostname_of(h)}: {setup}')
                if subprocess.run(line, timeout=args.timeout).returncode != 0:
                    log(f'setup failed on {hostname_of(h)}')
                    return 1
    codes = run_all(ranks, world, peers, cmd, args)
    if args.dry_run:
        return 0
    bad = 0
    for r in ranks:
        c = codes.get(r['rank'], 1)
        bad += c != 0
        print(f"[launch] rank {r['rank']} @ {hostname_of(r['host'])}: {'PASS' if c == 0 else f'FAIL (exit {c})'}", flush=True)
    print(f"[launch] {'PASS' if not bad else 'FAIL'}: {world - bad}/{world} ranks succeeded", flush=True)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
