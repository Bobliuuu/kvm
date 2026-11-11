"""Each MMA loop of a kernel, attributed to the function whose address range holds it: live_across.py's loop rows (report_loops' rule), with
the cubin's function ranges (cubin_functions). Prints per function: loops, and LDL a pass (median, max)."""
import sys
from collections import defaultdict
sys.path.insert(0, sys.argv[1])
import live_across as la
sass, cubin, kernel = sys.argv[2:5]
name, ins = la.parse(sass, kernel)
A = la.analyse(ins, ins[0]["addr"], every=True)
funcs = la.cubin_functions(cubin, name)            # [start, end, symbol]
rows = []
import io, contextlib
orig = la.report_loops
def collect(A, show):                               # report_loops' rows, kept instead of printed
    ins, blocks, bid, ud = A['ins'], A['blocks'], A['bid'], A['ud']
    for n, (s, e) in enumerate(blocks):
        last = ins[e - 1]
        if not last['op'].startswith('BRA') or A['tgt'](last) not in A['at']: continue
        h = A['at'][A['tgt'](last)]
        if h > s or h not in bid: continue
        body = [k for k in range(h, e) if k in ud]
        hmma = sum(ins[k]['op'].startswith('HMMA') for k in body)
        if not hmma: continue
        rows.append((ins[h]['addr'], hmma, sum(ins[k]['op'].startswith('LDL') for k in body)))
collect(A, 0)
by = defaultdict(list)
for addr, hmma, ldl in rows:
    f = next((sym for s, e, sym in funcs if s <= addr < e), '?')
    by[la.short_name(f, name) if f != '?' else '?'].append((ldl, hmma))
for f, rs in sorted(by.items()):
    ld = sorted(r[0] for r in rs)
    small = sorted(r[0] for r in rs if r[1] <= 64)
    print(f"  {f:28s} {len(rs):3d} MMA loops; LDL a pass median {ld[len(ld)//2]}, max {ld[-1]}; loops of <= 64 HMMA a pass: max {small[-1] if small else '-'}")
