#!/usr/bin/env python3
"""SASS identity of two kvm_exec binaries, function by function: each function's instruction text (cuobjdump -sass, addresses and encodings
dropped), hashed. Prints, for every function in either binary: identical, differs (instruction counts), or only in one.
    sass_ident.py BIN_A BIN_B [--skip SUBSTRING]"""
import argparse, collections, hashlib, re, subprocess, sys
INS = re.compile(r'^\s+/\*[0-9a-f]{4,}\*/\s+(.*?)\s*;')
def funcs(path):
    out = subprocess.run(['nice', '-n', '19', 'cuobjdump', '-sass', path], capture_output=True, text=True, check=True).stdout
    d, f = collections.defaultdict(list), None
    for line in out.splitlines():
        if 'Function : ' in line: f = line.split('Function : ', 1)[1].strip(); continue
        m = INS.match(line)
        if f and m: d[f].append(m.group(1))
    return {k: (len(v), hashlib.sha256('\n'.join(v).encode()).hexdigest()[:16]) for k, v in d.items()}
ap = argparse.ArgumentParser(); ap.add_argument('a'); ap.add_argument('b'); ap.add_argument('--skip', default=None)
x = ap.parse_args()
fa, fb = funcs(x.a), funcs(x.b)
names = sorted(set(fa) | set(fb))
same = [n for n in names if fa.get(n) == fb.get(n)]
diff = [n for n in names if n in fa and n in fb and fa[n] != fb[n]]
only = [n for n in names if (n in fa) != (n in fb)]
skip = lambda n: x.skip and x.skip in n
print(f'functions: {len(fa)} in A, {len(fb)} in B; identical {len(same)}; differ {len(diff)}; in one only {len(only)}')
print(f'excluding names with {x.skip!r}: identical {sum(not skip(n) for n in same)}, differ {sum(not skip(n) for n in diff)}, in one only {sum(not skip(n) for n in only)}')
for n in diff: print(f'  DIFFERS {n}: {fa[n][0]} -> {fb[n][0]} instructions')
for n in only: print(f'  ONLY IN {"A" if n in fa else "B"} {n}: {(fa.get(n) or fb.get(n))[0]} instructions')
