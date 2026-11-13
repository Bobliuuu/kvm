"""The form each hold-out MLP shape takes at 108 workers (from the pass test's log): a width x depth table, and the reasons the unfused cells stay so."""
import re
import sys
from collections import Counter, defaultdict

lines = open(sys.argv[1]).read().splitlines()
form = {}
why = {}
for i, l in enumerate(lines):
    m = re.match(r'^h(\d+)-d(\d)/bundle W=108:', l)
    if not m:
        continue
    h, d = int(m.group(1)), int(m.group(2))
    rep = lines[i + 1] if i + 1 < len(lines) else ''
    if 'grid form' in rep and 'fused' in rep:
        form[h, d] = 'G'
    elif 'split form' in rep and 'fused' in rep:
        s = re.search(r'x (\d+) shares', rep)
        form[h, d] = 'S' + (s.group(1) if s else '?')
    elif 'unchanged' in rep:
        form[h, d] = '.'
        why[h, d] = rep.split('unchanged: ', 1)[1]
    elif 'fused' in rep:
        form[h, d] = 'B'
    else:
        form[h, d] = '?'
widths = sorted({h for h, _ in form})
print('form at 108 workers (B one-block, S<n> split at n shares, G grid, . unfused):')
print('  width  d1   d2   d3   d4')
for h in widths:
    print(f'  {h:5d} ' + ' '.join(f'{form.get((h, d), "-"):4s}' for d in (1, 2, 3, 4)))
c = Counter(v[0] for v in form.values())
n = len(form)
print(f'cells: {n}; one-block {c["B"]}, split {c["S"]}, grid {c["G"]}, unfused {c["."]} ({100 * c["."] / n:.0f}%)')
byd = defaultdict(Counter)
for (h, d), v in form.items():
    byd[d][v[0]] += 1
for d in (1, 2, 3, 4):
    print(f'  depth {d}: ' + ', '.join(f'{k} {byd[d][k]}' for k in 'BSG.') + f' of {sum(byd[d].values())}')
reasons = Counter()
for (h, d), r in why.items():
    parts = [p.strip() for p in r.split(';')]
    key = ' | '.join(re.sub(r'[\d,]+ floats', 'N floats', p) for p in parts[1:])
    reasons[key] += 1
print('why the unfused cells stay so (the split and grid forms\' reasons; counts):')
for k, v in reasons.most_common():
    print(f'  {v:3d}  {k[:230]}')
