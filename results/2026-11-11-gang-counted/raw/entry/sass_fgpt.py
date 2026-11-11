#!/usr/bin/env python3
"""The fused GPT kernel's instances, hashed (cuobjdump -sass, addresses and encodings dropped): which instance of one binary is the other's code.
    sass_fgpt.py BIN [BIN ...]"""
import collections, hashlib, re, subprocess, sys
INS = re.compile(r'^\s+/\*[0-9a-f]{4,}\*/\s+(.*?)\s*;')
for path in sys.argv[1:]:
    out = subprocess.run(['nice', '-n', '19', 'cuobjdump', '-sass', '-fun', 'kvm_fgpt_kernel', path], capture_output=True, text=True).stdout
    if 'Function' not in out:
        out = subprocess.run(['nice', '-n', '19', 'cuobjdump', '-sass', path], capture_output=True, text=True, check=True).stdout
    d, f = collections.defaultdict(list), None
    for line in out.splitlines():
        if 'Function : ' in line: f = line.split('Function : ', 1)[1].strip(); continue
        m = INS.match(line)
        if f and m: d[f].append(m.group(1))
    for k, v in sorted(d.items()):
        if 'fgpt' in k: print(f'{path.split("/")[-2][:8]} {k}: {len(v)} instructions, sha256 {hashlib.sha256(chr(10).join(v).encode()).hexdigest()[:16]}')
