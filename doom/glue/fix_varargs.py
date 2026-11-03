#!/usr/bin/env python3
"""Lower C variadic functions in textual LLVM IR for the NVPTX backend (clang/LLVM 14 cannot select llvm.va_start).
Every variadic function f(fixed..., ...) becomes f(fixed..., i8* va.buf); every call site packs its extra arguments into a stack
buffer (each argument at its natural alignment, the layout tinylibc/stdarg.h's va_arg walks) and passes the buffer pointer."""
import re, sys

def split_top(s):
    out, depth, cur = [], 0, ''
    for ch in s:
        if ch in '([{': depth += 1
        elif ch in ')]}': depth -= 1
        if ch == ',' and depth == 0: out.append(cur.strip()); cur = ''
        else: cur += ch
    if cur.strip(): out.append(cur.strip())
    return out

def parse_type(a):
    """split 'i32 noundef %5' -> ('i32', '%5'); handles types like [9 x i8]*, %struct.x*, i8*, double, { i32, i32 }"""
    depth, i = 0, 0
    while i < len(a):
        ch = a[i]
        if ch in '([{<': depth += 1
        elif ch in ')]}>': depth -= 1
        elif ch == ' ' and depth == 0 and not a[i + 1:].startswith('*'):
            break
        i += 1
    ty = a[:i]; rest = a[i:].strip()
    attrs = ('noundef', 'signext', 'zeroext', 'nonnull', 'inreg', 'nocapture', 'readonly', 'returned')
    while True:
        w = rest.split(' ', 1)
        if w[0] in attrs: rest = w[1] if len(w) > 1 else ''
        elif re.match(r'(dereferenceable|align)\(', w[0]) or w[0].startswith('align') and w[0][5:].isdigit(): rest = w[1] if len(w) > 1 else ''
        else: break
    return ty, rest

def size_align(ty):
    if ty.endswith('*'): return 8, 8
    if ty == 'double': return 8, 8
    if ty == 'float': return 4, 4
    m = re.fullmatch(r'i(\d+)', ty)
    if m:
        b = int(m.group(1)); n = 1 if b <= 8 else 2 if b <= 16 else 4 if b <= 32 else 8
        return max(n, 4) if b < 32 else n, max(n, 4) if b < 32 else n   # sub-int types are promoted before they reach a vararg call
    raise SystemExit('unsupported vararg type: ' + ty)

lines = open(sys.argv[1]).read().split('\n')
vf = set()
for ln in lines:
    m = re.match(r'^(?:define|declare)\b.*?@([\w.$]+)\((.*?)(,\s*)?\.\.\.\)', ln)
    if m: vf.add(m.group(1))
print('variadic functions:', sorted(vf), file=sys.stderr)

out = []; uid = 0; pending = []; fn_start = -1
call_re = re.compile(r'^(\s*(?:%[\w.$]+ = )?(?:tail |musttail |notail )?call\b)(.*?)\s\((.*?)(?:,\s*)?\.\.\.\)\s+@([\w.$]+)\((.*)\)((?:\s*#\d+|\s*,\s*![\w.]+ ![\d]+)*)\s*$')
for ln in lines:
    s = ln
    if re.match(r'^define\b', s):
        fn_start = len(out); pending = []
    m = re.match(r'^((?:define|declare)\b.*?@([\w.$]+)\()(.*?)(,\s*)?\.\.\.\)(.*)$', s)
    if m and m.group(2) in vf:
        head, name, params, comma, tail = m.groups()
        sep = ', ' if params.strip() else ''
        s = head + params.rstrip().rstrip(',') + sep + ('i8* %va.buf' if s.startswith('define') else 'i8*') + ')' + tail
    elif 'llvm.va_start' in s and 'call void' in s:
        mm = re.search(r'@llvm\.va_start\(i8\* (?:[a-z]+ )*([^) ]+)\)', s)
        if mm:
            uid += 1
            out.append(f'  %vas.{uid} = bitcast i8* {mm.group(1)} to i8**'); s = f'  store i8* %va.buf, i8** %vas.{uid}'
    elif 'llvm.va_end' in s and 'call void' in s:
        continue
    else:
        mm = call_re.match(s)
        if mm and mm.group(4) in vf:
            pre, rtype, fixed_t, name, args, post = mm.groups()
            alist = split_top(args); nfixed = len(split_top(fixed_t)) if fixed_t.strip() else 0
            fixed_args, var_args = alist[:nfixed], alist[nfixed:]
            off = 0; stores = []
            for a in var_args:
                ty, val = parse_type(a); sz, al = size_align(ty)
                off = (off + al - 1) // al * al
                stores.append((ty, val, off)); off += sz
            total = max(off, 8); uid += 1; buf = f'%vab.{uid}'
            pending.append(f'  {buf} = alloca [{total} x i8], align 16')
            for k, (ty, val, o) in enumerate(stores):
                out.append(f'  {buf}.p{k} = getelementptr inbounds [{total} x i8], [{total} x i8]* {buf}, i64 0, i64 {o}')
                out.append(f'  {buf}.c{k} = bitcast i8* {buf}.p{k} to {ty}*')
                out.append(f'  store {ty} {val}, {ty}* {buf}.c{k}, align 1')
            out.append(f'  {buf}.b = getelementptr inbounds [{total} x i8], [{total} x i8]* {buf}, i64 0, i64 0')
            nargs = ', '.join(fixed_args + [f'i8* {buf}.b'])
            ft = (fixed_t.strip().rstrip(',') + ', ' if fixed_t.strip() else '') + 'i8*'
            s = f'{pre} {rtype} ({ft}) @{name}({nargs}){post}'
    out.append(s)
    if s.startswith('}') and fn_start >= 0:
        if pending:                                   # hoist the buffers into the entry block (static allocas)
            j = fn_start
            while not out[j].rstrip().endswith('{'): j += 1
            out[j + 1:j + 1] = pending
        fn_start = -1; pending = []
open(sys.argv[2], 'w').write('\n'.join(out))
