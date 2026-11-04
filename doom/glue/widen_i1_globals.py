#!/usr/bin/env python3
"""Widen `i1` globals to `i8` in textual LLVM IR.

LLVM's GlobalOpt shrinks an int global that only ever holds 0/1 (Doom's boolean flags) to `i1`; llc NVPTX then loads it with a 16-bit
`ld.u16` from a 1-byte symbol (an out-of-bounds read, and misaligned for align-1 symbols). An i8 global with trunc/zext at the uses
is what the C program meant. Fails loudly if an i1 global is used in any way other than a plain load or store.
  widen_i1_globals.py in.ll out.ll
"""
import re
import sys

src = open(sys.argv[1]).read()
names = set(re.findall(r"^@([\w.$]+) = [^\n]*?\bglobal i1 ", src, re.M))
uid = 0


def sub_def(m):
    return f"{m.group(1)}i8 {'1' if m.group(2) == 'true' else '0'}"


src = re.sub(r"(^@(?:%s) = [^\n]*?\bglobal )i1 (true|false)" % "|".join(map(re.escape, sorted(names))), sub_def, src, flags=re.M) if names else src
out = []
for ln in src.split("\n"):
    m = re.match(r"^(\s*)(%[\w.$]+) = load (atomic |volatile )*i1, i1\* @([\w.$]+)(.*)$", ln)
    if m and m.group(4) in names:
        uid += 1
        flags = "".join(g for g in re.findall(r"(?:atomic |volatile )", ln.split("i1,")[0]))
        out.append(f"{m.group(1)}%i1w.{uid} = load {flags}i8, i8* @{m.group(4)}{m.group(5)}")
        out.append(f"{m.group(1)}{m.group(2)} = trunc i8 %i1w.{uid} to i1")
        continue
    m = re.match(r"^(\s*)store (atomic |volatile )*i1 ([^,]+), i1\* @([\w.$]+)(.*)$", ln)
    if m and m.group(4) in names:
        uid += 1
        flags = "".join(re.findall(r"(?:atomic |volatile )", ln.split("i1 ")[0]))
        v = m.group(3).strip()
        if v in ("true", "false"):
            out.append(f"{m.group(1)}store {flags}i8 {'1' if v == 'true' else '0'}, i8* @{m.group(4)}{m.group(5)}")
        else:
            out.append(f"{m.group(1)}%i1z.{uid} = zext i1 {v} to i8")
            out.append(f"{m.group(1)}store {flags}i8 %i1z.{uid}, i8* @{m.group(4)}{m.group(5)}")
        continue
    out.append(ln)
res = "\n".join(out)
for ln in out:                                        # any remaining use of the old type is a pass bug, not something to ignore
    if "i1*" in ln:
        for n in re.findall(r"i1\*\s+@([\w.$]+)", ln):
            if n in names:
                sys.exit(f"widen_i1_globals: unhandled use of @{n}: {ln.strip()}")
open(sys.argv[2], "w").write(res)
print(f"widened {len(names)} i1 globals to i8, {uid} loads/stores rewritten", file=sys.stderr)
