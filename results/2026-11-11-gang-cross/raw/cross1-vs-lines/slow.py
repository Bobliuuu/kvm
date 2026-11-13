"""The read's verdicts from its run logs (us a step from each bed's own clock): per row, each arm's median, the paired ratio's median
(control / candidate, round by round), the slow runs of each arm against its own arm's fastest, and Fisher's two-sided p. `--extend`
prints the rows that need 4 more rounds instead."""
import math, re, statistics, sys
from collections import defaultdict
from pathlib import Path
W, mode = Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else ''
SLOW = 1.10
EXPECT = {}   # row -> the arms it runs (rows.txt: `candidate-only` rows run on the flag-line beds alone)
for line in (W / 'rows.txt').read_text().splitlines():
    if line.strip():
        EXPECT[line.split()[0]] = ('new',) if line.split()[-1] == 'candidate-only' else ('new', 'ctrl')
t = defaultdict(dict)   # (row, arm) -> {round: us}
ok = defaultdict(list)
for f in W.glob('*_r*.log'):
    m = re.match(r'(.+)_(new|ctrl)_r(\d+)\.log$', f.name)
    if not m:
        continue
    s = f.read_text(errors='replace')
    u = re.search(r'([0-9.]+) us a step in the kernel', s)
    ok[m.group(1), m.group(2)].append('check ok' in s)
    if u:
        t[m.group(1), m.group(2)][int(m.group(3))] = float(u.group(1))
def fisher(a, b, c, d):
    n, r1, c1 = a + b + c + d, a + b, a + c
    pr = lambda x: math.comb(r1, x) * math.comb(n - r1, c1 - x) / math.comb(n, c1)
    p0 = pr(a)
    return min(1.0, sum(pr(x) for x in range(max(0, c1 - (n - r1)), min(r1, c1) + 1) if pr(x) <= p0 * (1 + 1e-9)))
rows = list(EXPECT)
slow = lambda v: sum(x > SLOW * min(v) for x in v)
if mode == '--extend':
    print(' '.join(r for r in rows if any(t.get((r, a)) and slow(list(t[r, a].values())) for a in ('new', 'ctrl'))))
    sys.exit(0)
bad = 0
ctrl_on = any(a == 'ctrl' for _, a in ok)   # a read with CTRL_BIN runs the control arm on every row not candidate-only
for r in rows:
    k, c = t.get((r, 'new'), {}), t.get((r, 'ctrl'), {})
    slow_any = any(slow(list(v.values())) for v in (k, c) if v)
    want = set(range(1, 10 if slow_any else 6))
    lacking = [a for a in EXPECT[r] if (a == 'new' or ctrl_on) and set(t.get((r, a), {})) != want]
    if lacking:
        print(f'{r:22s} MISSING ROUNDS: ' + ', '.join(f"{a} has {sorted(t.get((r, a), {}))}, wants {sorted(want)}" for a in lacking)); bad += 1; continue
    checks = all(ok.get((r, 'new'), [False])) and all(ok.get((r, 'ctrl'), [True]))
    if not k:
        print(f'{r:22s} NO TIME (see its logs)'); bad += 1; continue
    kv = list(k.values()); line = f'{r:22s} candidate median {statistics.median(kv):8.2f} us (n {len(kv)}, slow {slow(kv)})'
    if c:
        cv = list(c.values()); both = sorted(set(k) & set(c))
        pr = statistics.median(c[i] / k[i] for i in both) if both else float('nan')
        sk, sc = slow(kv), slow(cv); p = fisher(sk, len(kv) - sk, sc, len(cv) - sc)
        slow_bad = sk > sc and p < 0.05
        line += f'  control median {statistics.median(cv):8.2f} us (n {len(cv)}, slow {sc})  paired x{pr:.3f}  Fisher p {p:.3f}'
        line += '  SLOW MODE' if slow_bad else ''
        bad += slow_bad
        lines = [[l for l in (W / f'{r}_{a}_r1.log').read_text(errors='replace').splitlines() if l.startswith('check')]
                 if (W / f'{r}_{a}_r1.log').exists() else [] for a in ('new', 'ctrl')]
        same = bool(lines[0]) and lines[0] == lines[1]   # the barrier changes no arithmetic: round 1's check lines, equal and not empty
        line += f'  check lines {len(lines[0])} {"identical" if same else "DIFFER"}'
        bad += not same
    line += '' if checks else '  CHECK FAILED'
    bad += not checks
    print(line)
if not (W / 'gpu.end').exists():
    print('the GPU part never reached its end (no gpu.end)'); bad += 1
print(f"verdict: {'PASS' if not bad else 'FAIL'}: {bad} problem(s) (a row with a failed check, missing rounds or a slow mode; a missing end), {len(rows)} rows expected")
sys.exit(1 if bad else 0)
