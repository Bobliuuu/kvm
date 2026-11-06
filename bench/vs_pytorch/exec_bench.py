#!/usr/bin/env python3
"""Time `kvm_exec` on a bundle made by gen.py: REPS launches (one process each), the kernel's own cudaEvent time per launch.

  python3 exec_bench.py WORKLOAD_DIR ./kvm_exec [--reps 7] [--out kvm.json] [--timeout 3600] [-- extra kvm_exec args]

One launch runs all N steps of the bundle; persistent-kernel start and host admission of the whole tape are inside the time.
Writes {"steps", "tasks", "ms_total": [...], "ms_per_step_best|median|worst", "tokens_per_s_best", "exit_codes"}.
A launch that does not end in `OK: n/n tasks` is recorded as failed and excluded from the statistics.
"""
import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path


def main():
    argv = sys.argv[1:]
    extra = argv[argv.index('--') + 1:] if '--' in argv else []
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('workload_dir')
    ap.add_argument('exe')
    ap.add_argument('--reps', type=int, default=7)
    ap.add_argument('--out')
    ap.add_argument('--timeout', type=float, default=3600)
    a = ap.parse_args(argv[:argv.index('--')] if '--' in argv else argv)
    wl = Path(a.workload_dir)
    info = json.loads((wl / 'info.json').read_text())
    ms, failed = [], []
    for r in range(a.reps):
        t0 = time.time()
        p = subprocess.run([a.exe, '--quiet', *extra, str(wl / 'bundle')], capture_output=True, text=True, timeout=a.timeout)
        m = re.search(r'^(OK|FAIL): (\d+)/(\d+) tasks, ([\d.]+) ms', p.stdout, re.M)
        if p.returncode == 0 and m and m.group(1) == 'OK':
            ms.append(float(m.group(4)))
            print(f'rep {r + 1}/{a.reps}: {ms[-1]:.1f} ms ({ms[-1] / info["steps"]:.2f} ms/step), wall {time.time() - t0:.1f} s', flush=True)
        else:
            failed.append(dict(rep=r, rc=p.returncode, stdout=p.stdout[-300:], stderr=p.stderr[-300:]))
            print(f'rep {r + 1}/{a.reps}: FAILED rc={p.returncode} {p.stdout[-200:]!r} {p.stderr[-200:]!r}', flush=True)
    n = info['steps']
    res = dict(steps=n, tasks=info['tasks'], extra_args=extra, ms_total=ms, failed=failed)
    if ms:
        res.update(ms_per_step_best=min(ms) / n, ms_per_step_median=statistics.median(ms) / n, ms_per_step_worst=max(ms) / n,
                   tokens_per_s_best=info['tokens_per_step'] / (min(ms) / n / 1e3))
        print(f'executor: best {min(ms) / n * 1e3:.0f} us/step  median {statistics.median(ms) / n * 1e3:.0f}  worst {max(ms) / n * 1e3:.0f}')
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
    sys.exit(0 if ms and not failed else 1)


if __name__ == '__main__':
    main()
