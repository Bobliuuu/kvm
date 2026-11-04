#!/usr/bin/env python3
"""Give an LLVM IR copy of the Doom engine its own symbol namespace, so N engines can be linked into one CUDA program.
  prefix_symbols.py in.ll out.ll PREFIX
Every symbol DEFINED in the module (functions, globals incl. shared/constant/internal, aliases) gets PREFIX; declarations
(intrinsics, llvm.*, anything resolved by another object) are left alone, so all instances share them. Self-check: at least one
symbol renamed, no `@name` left that is neither defined (renamed) nor declared in the module."""
import re
import sys

NAME = r'"[^"]*"|[-a-zA-Z$._][-a-zA-Z$._0-9]*'
DEF_FN = re.compile(r'^define\b[^@]*@(%s)\(' % NAME)
DEF_GV = re.compile(r'^@(%s)\s*=\s*(?!external\b)(?!extern_weak\b)' % NAME)
DECL_FN = re.compile(r'^declare\b[^@]*@(%s)\(' % NAME)
DECL_GV = re.compile(r'^@(%s)\s*=\s*(?:external|extern_weak)\b' % NAME)
STR = re.compile(r'c"[^"]*"')   # constant byte strings may contain '@'
REF = re.compile(r'@(%s)' % NAME)


def rename(name, prefix):
    return '"%s%s"' % (prefix, name[1:-1]) if name.startswith('"') else prefix + name


def main(src, dst, prefix):
    lines = open(src).read().split('\n')
    defined, declared = set(), set()
    for ln in lines:
        for rx, into in ((DEF_FN, defined), (DEF_GV, defined), (DECL_FN, declared), (DECL_GV, declared)):
            m = rx.match(ln)
            if m:
                into.add(m.group(1))
    defined = {n for n in defined if not n.strip('"').startswith('llvm.')}
    if not defined:
        sys.exit('prefix_symbols: no defined symbols found in %s' % src)
    sub = lambda ln: REF.sub(lambda m: '@' + rename(m.group(1), prefix) if m.group(1) in defined else m.group(0), ln)
    out = [''.join(p if i % 2 else sub(p) for i, p in enumerate(re.split('(%s)' % STR.pattern, ln))) for ln in lines]
    # self-check: every remaining reference is a declaration (shared) or an llvm intrinsic
    new_defined = {rename(n, prefix) for n in defined}
    bad = set()
    for ln in out:
        for m in REF.finditer(STR.sub('', ln)):
            n = m.group(1)
            if n not in new_defined and n not in declared and not n.strip('"').startswith('llvm.'):
                bad.add(n)
    if bad:
        sys.exit('prefix_symbols: dangling references: %s' % sorted(bad)[:10])
    open(dst, 'w').write('\n'.join(out))
    print('prefix_symbols: %s: %d symbols renamed with %r, %d shared declarations kept' % (dst, len(defined), prefix, len(declared)), file=sys.stderr)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
